import Foundation
import Metal

/// Compatible Qwen3.5-MoE real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<40:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = v_proj(a)
///     per-head q/k_norm (bf16w)
///     NeoX RoPE on Q + K (full-attention layers; linear layers use
///     Gated-DeltaNet recurrent state instead of K/V slots)
///     write K and V into the cache slots
///     attn = attention(scale, full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)  // sigmoid-gated by shared_expert_gate
///     xr = rmsnorm_no_scale(h)
///     idx, w = router_topk(xr, effective_scale[L], per_expert_scale[L])
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=h1, routedBlobs=fetch(idx), w)
///     h = h + h2
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), lm_head^T)
///   // final softmax happens in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    /// The fixed threshold the runner's parsed default (16) and the A/B
    /// (`SHRIKE_PREFILL_MATRIX_MIN_ROWS=32`) both derive from.
    static let fixedMinimumRows = 32

    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int,
                                 minimumRows: Int = fixedMinimumRows) -> PrefillProjectionDispatch {
        guard chunkTokens >= minimumRows else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o:
            return .qmm
        }
    }
}

/// `SHRIKE_PREFILL_SWEEP` mode: `alternate` reverses the sweep on odd-parity
/// chunks; `fixed` keeps every chunk ascending; `carry` starts each
/// request's first chunk opposite the previous request's last chunk and
/// alternates from there (v13 T0's mini A/B winner, kept as the A/B);
/// `recency` splits the chunk's experts into a head and a tail of the last
/// `SHRIKE_PREFILL_SWEEP_TAIL` by last-row-in-chunk, then packs each
/// group's own tiles by row weight; it consults neither the direction nor
/// the carry state, and honours `participatesInCarry: false` the same as
/// the direction switch, so a chunk outside the carry keeps index
/// tiling under the knob (v13 T4 step 2, fix-up 1; fix round 1); `resident`
/// sweeps the chunk's pool-resident experts first, then the absent ones by
/// the same recency split, all three groups packed by row weight and tiled
/// flat (v13 T5 step 2, today's default).
internal enum PrefillSweepMode: String, Sendable, Equatable, CaseIterable {
    case alternate
    case fixed
    case carry
    case recency
    case resident

    /// `.recency` and `.resident` build their own per-layer order in
    /// `buildPrefillRoutes`, never reading the direction switch or the
    /// carry state.
    var usesComputedOrder: Bool {
        self == .recency || self == .resident
    }
}

/// The chunk's still-needed set of routed experts: a per-layer
/// `[Bool]` of `expertsPerLayer` entries, `true` until that expert's tile is
/// planned, so the streamer tests it with one array read and no hashing,
/// and no allocation happens per tile.
struct PrefillChunkExpertProtection {
    private(set) var remaining: [Bool]

    init(routedExperts: [Int], expertsPerLayer: Int) {
        remaining = [Bool](repeating: false, count: expertsPerLayer)
        for expert in routedExperts where expert >= 0 && expert < expertsPerLayer {
            remaining[expert] = true
        }
    }

    /// Reads `PrefillMoEGroup.expert` directly, so the chunk's routed
    /// experts do not need a separate `[Int]` array first: one allocation
    /// per layer-chunk, the `[Bool]` itself.
    init(routedGroups: [PrefillMoEGroup], expertsPerLayer: Int) {
        remaining = [Bool](repeating: false, count: expertsPerLayer)
        for group in routedGroups {
            let expert = Int(group.expert)
            if expert >= 0, expert < expertsPerLayer {
                remaining[expert] = true
            }
        }
    }

    /// Clears the tile about to be planned, since its own experts are hits
    /// or this plan's own misses, never a victim to protect against itself.
    mutating func planning(_ tileExperts: [Int]) {
        for expert in tileExperts where expert >= 0 && expert < remaining.count {
            remaining[expert] = false
        }
    }
}

/// unchecked-invariant: exclusively owned by one caller for its lifetime and
/// never shared. In the server it is a `private let` on the `ServerModelSession`
/// actor, so every entry point is already actor-isolated; the CLI and the
/// decode service each drive one runner from a single task. Its ~19 mutable
/// properties are decode cursors and scratch handles with no internal locking,
/// so two concurrent callers would corrupt them -- the ownership is the whole
/// safety argument, not an implementation detail.
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Qwen3.5-MoE [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let affineEmbed: AffineQuantEmbeddingLookup?
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let affine: AffineQuantGEMV?
    private let attention: Attention
    private let kvQuantizer: KVCacheQuantizer?
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let mla: MLA?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let int8ScalarGate: DequantInt8GEMV?

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    public var prefillProjectionPath: String {
        guard let mpp = prefillMPPAffineInt4 else { return "unavailable" }
        return "affine-threadgroup-f16 tile_n=\(mpp.tileN) tile_k=\(mpp.variant.tileK)"
            + " buffers=\(mpp.variant.dequantBuffers) loads=\(mpp.weightLoads.rawValue)"
    }

    public var prefillAttentionPathDescription: String {
        let available = prefillAttention.matrixPathAvailable
        var description = "\(prefillAttentionPath.rawValue) matrix_available=\(available)"
            + " tile=\(prefillAttention.tile.rawValue)"
        if !available {
            description += " reason=\(prefillAttention.matrixUnavailableReason)"
        }
        return description
    }

    /// `chunked` applies to chunks of 64+ rows; shorter chunks take the serial
    /// kernel regardless.
    public var prefillGDNScanPathDescription: String {
        guard let gdn else { return "none" }
        guard gdn.chunkedScanAvailable else {
            return "serial reason=\(gdn.chunkedScanUnavailableReason ?? "unavailable")"
        }
        return gdnPrefillScanChunked ? "chunked" : "serial"
    }

    /// Requested routed tiles per command buffer, the width actually in force
    /// once the streamed cache is fitted, the per-tile expert count that
    /// fitting leaves (`width=0 experts=0` = the cache admits neither), and the
    /// requested pending-tile depth — `fitting` never narrows it, so one value
    /// suffices.
    public var prefillTileBatchDescription: String {
        let requested = prefillRoutedTileSchedulerConfig
        let depth = Self.prefillTileDepthDescription(requested)
        let fetch = Self.prefillFetchDepthDescription(requested)
        guard let slotCount = model.routedExpertCacheSlotCount() else {
            return "tiles=\(requested.tilesPerCommandBuffer)"
                + " width=\(requested.tilesPerCommandBuffer) experts=\(requested.tileExperts) \(depth) \(fetch)"
        }
        guard let fitted = requested.fitting(slotCount: slotCount) else {
            return "tiles=\(requested.tilesPerCommandBuffer) width=0 experts=0 \(depth) \(fetch)"
        }
        return "tiles=\(requested.tilesPerCommandBuffer)"
            + " width=\(fitted.tilesPerCommandBuffer) experts=\(fitted.tileExperts) \(depth) \(fetch)"
    }

    static func prefillTileDepthDescription(_ config: PrefillRoutedTileSchedulerConfig) -> String {
        "depth=\(config.maxPendingDepth)"
    }

    public var prefillMatrixMinRowsDescription: String {
        Self.prefillMatrixMinRowsDescription(prefillMatrixMinRows)
    }

    static func prefillMatrixMinRowsDescription(_ rows: Int) -> String {
        "prefill_matrix_min_rows=\(rows)"
    }

    static func prefillFetchDepthDescription(_ config: PrefillRoutedTileSchedulerConfig) -> String {
        "fetch=\(config.fetchLookahead + 1)"
    }

    /// Which routed-expert GEMM a prefill chunk on the matrix path takes:
    /// `grouped`, `per-expert` (P3) or `scalar`; a chunk of
    /// `matrixPathMinimumRows` tokens or fewer takes the scalar path regardless.
    public var prefillRoutedGEMMDescription: String {
        let cfg = model.config
        guard let mpp = prefillGroupedMoE.matrixPath(
            for: prefillMPPAffineInt4,
            d: cfg.hiddenSize,
            intermediate: cfg.moeIntermediateSize) else {
            return "scalar"
        }
        guard prefillRoutedGEMMGrouped else { return "per-expert" }
        guard prefillGroupedMoE.groupedPathAvailable(for: mpp) else { return "per-expert reason=grouped-unavailable" }
        return "grouped tail_tile=\(tailTileInForce(for: mpp) == 0 ? "off" : "32")"
    }

    /// The tail tile in force for a GEMM instance: the knob, unless the
    /// selected variant has no 32-row instantiation for this model's K's.
    private func tailTileInForce(for mpp: MPPPrefillInt4QMM) -> Int {
        guard prefillTailTile != 0,
              mpp.groupedRowTile32Available(forK: cfg.hiddenSize),
              mpp.groupedRowTile32Available(forK: cfg.moeIntermediateSize) else { return 0 }
        return prefillTailTile
    }

    /// The P12 gap levers in force: the shared expert committed before the
    /// router wait, and the expert pools held in a queue residency set, with
    /// `allocations=` reported rather than inferred from the holder's mere
    /// existence; `sweep=recency` also prints `tail=` (v13 T4 step 2 fix-up 1).
    public var prefillGapLeversDescription: String {
        Self.prefillGapLeversDescription(
            overlap: prefillRouteOverlap,
            residencyAllocationCount: poolResidency?.allocationCount,
            poolResidencyUnavailableReason: poolResidencyUnavailableReason,
            sweepMode: prefillSweepMode,
            sweepTail: prefillSweepTail,
            prefetchTrace: prefetchTraceFD >= 0)
    }

    static func prefillGapLeversDescription(
        overlap: Bool,
        residencyAllocationCount: Int?,
        poolResidencyUnavailableReason: String?,
        sweepMode: PrefillSweepMode,
        sweepTail: Int = prefillSweepTailDefault,
        prefetchTrace: Bool = false
    ) -> String {
        let residency: String
        if let residencyAllocationCount {
            residency = "set allocations=\(residencyAllocationCount)"
        } else if let reason = poolResidencyUnavailableReason {
            residency = "unavailable reason=\(reason)"
        } else {
            residency = "none"
        }
        let sweep = sweepMode == .recency
            ? "sweep=\(sweepMode.rawValue) tail=\(sweepTail)"
            : "sweep=\(sweepMode.rawValue)"
        return "overlap=\(overlap ? "on" : "off") residency=\(residency)"
            + " \(sweep)"
            + (prefetchTrace ? " prefetch_trace=on" : "")
    }

    /// The prefill router kernel in force (`block` or `tiled tokens=N`) and its
    /// weight bits.
    public var prefillRouterDescription: String {
        prefillRouter.description
    }

    // Scratch — preallocated per spec'd D / F / vocab.
    private let decodeScratch: DecodeScratchBuffers
    private var hidden: MTLBuffer { decodeScratch.hidden }          // [D] FP16
    private var normed: MTLBuffer { decodeScratch.normed }          // [D] FP16
    private var attnOut: MTLBuffer { decodeScratch.attnOut }        // [N_HEADS * head_dim] FP16
    private var qScratch: MTLBuffer { decodeScratch.qScratch }      // [N_HEADS * head_dim] FP16
    private var kStage: MTLBuffer { decodeScratch.kStage }          // [max KV heads * head_dim] FP16, current token
    private var vStage: MTLBuffer { decodeScratch.vStage }          // [max KV heads * head_dim] FP16, current token
    private var oOut: MTLBuffer { decodeScratch.oOut }              // [D] FP16
    private var h1Buf: MTLBuffer { decodeScratch.h1Buf }            // [D] FP16 (dense MLP output)
    private var h2Buf: MTLBuffer { decodeScratch.h2Buf }            // [D] FP16 (routed output)
    private var routedX: MTLBuffer { decodeScratch.routedX }        // [D] FP16 (pre_feedforward_layernorm_2 output)
    private var denseX: MTLBuffer { decodeScratch.denseX }          // [D] FP16 (pre_feedforward_layernorm output)
    private var denseScratchGate: MTLBuffer { decodeScratch.denseScratchGate } // [F=2112] FP16
    private var denseScratchUp: MTLBuffer { decodeScratch.denseScratchUp }     // [F=2112] FP16
    private var denseScratchAct: MTLBuffer { decodeScratch.denseScratchAct }   // [F=2112] FP16
    private var routerInput: MTLBuffer { decodeScratch.routerInput } // [D] FP16 (rmsnorm_no_scale(h))
    private var zeroResidual: MTLBuffer { decodeScratch.zeroResidual } // [D] FP16 zeros — for routed branch base
    private var outIndices: MTLBuffer { decodeScratch.outIndices }  // [topK] UInt32
    private var outWeights: MTLBuffer { decodeScratch.outWeights }  // [topK] FP16
    /// Trace-only next-layer router result. It is never read by inference.
    private var prefetchPredictionIndices: MTLBuffer { decodeScratch.prefetchPredictionIndices }
    private var prefetchPredictionWeights: MTLBuffer { decodeScratch.prefetchPredictionWeights }
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private var moeActs: MTLBuffer { decodeScratch.moeActs }        // [topK * FmoE] FP16
    // v9 S2 cross-check scratch, ping-ponged by layer parity: layer L's pair
    // is compared only after layer L+1's spec CB has already run, so a single
    // pair would be overwritten before the comparison reads it.
    private let specDispatchArguments: MoE.SpeculativeDispatchArguments
    private let residencyReadback: ResidencyReadbackBuffers
    private var moeHitActiveSlots: MTLBuffer { residencyReadback.moeHitActiveSlots } // [topK] UInt32
    private var moeMissActiveSlots: MTLBuffer { residencyReadback.moeMissActiveSlots } // [topK] UInt32
    private var residencyHitCount: MTLBuffer { residencyReadback.hitCount }
    private var residencyHitPositions: MTLBuffer { residencyReadback.hitPositions }
    private var residencyMissCount: MTLBuffer { residencyReadback.missCount }
    private var residencyMissPositions: MTLBuffer { residencyReadback.missPositions }
    private var residencyMissExperts: MTLBuffer { residencyReadback.missExperts }
    private var residencyResolvedSlots: MTLBuffer { residencyReadback.resolvedSlots }
    private var residencyResolvedGenerations: MTLBuffer { residencyReadback.resolvedGenerations }
    private var routerHostReadback: MTLBuffer { residencyReadback.hostReadback }
    private var greedyTokenBuf: MTLBuffer { decodeScratch.greedyTokenBuf } // 4 B UInt32 fused-head output
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private var qPackedScratch: MTLBuffer? { decodeScratch.qPackedScratch } // [2 * N_HEADS * head_dim] packed [q ; gate]
    private var attnGateScratch: MTLBuffer? { decodeScratch.attnGateScratch } // [N_HEADS * head_dim]
    private let gdnScratch: GDNScratchBuffers?
    private var gdnQKVRaw: MTLBuffer? { gdnScratch?.qkvRaw }        // [qkvDim] raw in_proj_qkv output
    private var gdnConvOut: MTLBuffer? { gdnScratch?.convOut }      // [qkvDim] conv + SiLU output
    private var gdnZ: MTLBuffer? { gdnScratch?.z }                  // [valueDim]
    private var gdnA: MTLBuffer? { gdnScratch?.a }                  // [numVHeads]; KDA [Hv * Dk]
    private var gdnB: MTLBuffer? { gdnScratch?.b }                  // [numVHeads]
    private var gdnY: MTLBuffer? { gdnScratch?.y }                  // [valueDim] delta-rule output
    private var gdnOut: MTLBuffer? { gdnScratch?.out }              // [valueDim] gated-norm output
    private var gdnLowRank: MTLBuffer? { gdnScratch?.lowRank }      // [keyHeadDim] KDA f_a/g_a stage
    // Kimi MLA decode scratch (mask-3 layers only).
    private let mlaScratch: MLAScratchBuffers?
    private var mlaQRaw: MTLBuffer? { mlaScratch?.qRaw }            // [H * (nope + rope)] q_proj out
    private var mlaQ: MTLBuffer? { mlaScratch?.q }                  // [H * (latent + rope)] absorbed Q
    private var mlaAttnOut: MTLBuffer? { mlaScratch?.attnOut }      // [H * latent] attention out
    private var mlaUnembedOut: MTLBuffer? { mlaScratch?.unembedOut } // [H * vHeadDim] o_proj input
    private var sharedScalarGateBuf: MTLBuffer? { decodeScratch.sharedScalarGateBuf } // [1] shared-expert gate logit
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    /// Per-layer additive router logit bias views (gpt-oss); a shared BF16
    /// zeros buffer for families without one. The selector always reads it.
    private var routerLogitBias: [(buffer: MTLBuffer, offset: Int)] = []
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    /// Reusable UInt32 token-ID buffer for chunked prefill (R23): sized to the
    /// largest chunk seen so far and grown on demand, so the prefill hot path
    /// never allocates an MTLBuffer per chunk.
    private var prefillTokenBuffer: MTLBuffer?

    /// Host scratch reused across prefill chunks (R38) and decode layers (R16).
    /// The runner is single-flight per generation (guarded by
    /// `prefillChunkState` and the callers' serial decode loop), so these
    /// never alias concurrent work.
    private var routeIDScratch: [UInt32] = []
    private var routeWeightScratch: [Float16] = []
    private var prefillResidentExpertScratch: [Bool] = []
    private var decodeExpertsScratch: [Int] = []
    private var decodeHitSlotsScratch: [UInt32] = []
    private var decodeMissSlotsScratch: [UInt32] = []
    private var decodeHitSplitRoutedBufsScratch: [MTLBuffer] = []
    private var decodeHitSplitRoutedOffsetsScratch: [Int] = []
    private var decodeRoutedBufsScratch: [MTLBuffer] = []
    private var decodeRoutedOffsetsScratch: [Int] = []

    private let prefillRoutedTileSchedulerConfig: PrefillRoutedTileSchedulerConfig
    /// `SHRIKE_PREFILL_ROUTED_GEMM=per-expert` keeps the P3 per-expert GEMMs
    /// for the same-binary A/B; anything else takes the grouped dispatch.
    private let prefillRoutedGEMMGrouped: Bool
    /// `SHRIKE_PREFILL_ROUTE_OVERLAP=off` keeps the shared expert after the
    /// host routing for the same-binary A/B; anything else commits it before.
    private let prefillRouteOverlap: Bool
    /// `SHRIKE_PREFILL_POOL_RESIDENCY=none` leaves the expert pools to
    /// per-buffer residency for the same-binary A/B.
    private let poolResidency: ExpertPoolResidency?
    private let poolResidencyUnavailableReason: String?
    /// `SHRIKE_PREFILL_SWEEP=alternate|fixed|carry|recency|resident` selects the
    /// `PrefillSweepMode`; unset takes `prefillSweepModeDefault`, `resident`,
    /// v13 T5's measured winner, with `carry` kept as the A/B; an unknown
    /// value fails at launch like the knob's siblings.
    private let prefillSweepMode: PrefillSweepMode
    private static let prefillSweepModeDefault = PrefillSweepMode.resident
    /// `SHRIKE_PREFILL_SWEEP_TAIL=<n>` sizes `recency`'s tail group, clamped
    /// to 8 ... the layer's expert count; unset or unparsable takes
    /// `prefillSweepTailDefault` (96), itself clamped the same way.
    private let prefillSweepTail: Int
    private static let prefillSweepTailDefault = 96
    /// The last direction a carry-participating prefill chunk swept, read by
    /// `carry` mode's next request; `reset()` must not clear it, since the
    /// expert pool it describes lives on `ModelExpertIO`, not on this
    /// runner. Written before the chunk executes, so a chunk that throws
    /// still records its direction (a lost optimisation, never a wrong
    /// result).
    private var prefillLastChunkDescending: Bool?
    /// `SHRIKE_PREFILL_TAIL_TILE=32` packs each expert block's remainder of
    /// ≤ 32 rows into a 32-row tile instead of a padded 64-row one; `=off`
    /// keeps every block on 64-row tiles; unset takes `prefillTailTileDefault`.
    private let prefillTailTile: Int
    private static let prefillTailTileDefault = 32
    private let prefillMatrixMinRows: Int

    static func parsePrefillSweepMode(_ raw: String?) throws -> PrefillSweepMode {
        guard let raw, !raw.isEmpty else { return prefillSweepModeDefault }
        guard let mode = PrefillSweepMode(rawValue: raw) else {
            throw ModelError.internalInconsistency(
                detail: "unsupported SHRIKE_PREFILL_SWEEP '\(raw)'; allowed: "
                    + PrefillSweepMode.allCases.map(\.rawValue).joined(separator: ", "))
        }
        return mode
    }

    private static func environmentPrefillSweepMode() throws -> PrefillSweepMode {
        try parsePrefillSweepMode(ProcessInfo.processInfo.environment["SHRIKE_PREFILL_SWEEP"])
    }

    private static func environmentPrefillAttentionPath(
        default fallback: RuntimePrefillAttentionPath
    ) -> RuntimePrefillAttentionPath {
        switch ProcessInfo.processInfo.environment["SHRIKE_PREFILL_ATTENTION"] {
        case "tiled": return .causalTiled
        case "matrix": return .causalMatrix
        default: return fallback
        }
    }

    static func parsePrefillSweepTail(_ raw: String?, expertCount: Int) throws -> Int {
        let upperBound = max(8, expertCount)
        guard let raw else { return min(prefillSweepTailDefault, upperBound) }
        guard let n = Int(raw.trimmingCharacters(in: .whitespaces)), n >= 8, n <= upperBound else {
            throw ModelError.internalInconsistency(
                detail: "unsupported SHRIKE_PREFILL_SWEEP_TAIL '\(raw)'; allowed: 8...\(upperBound)")
        }
        return n
    }

    private static func environmentPrefillSweepTail(expertCount: Int) throws -> Int {
        try parsePrefillSweepTail(ProcessInfo.processInfo.environment["SHRIKE_PREFILL_SWEEP_TAIL"],
                                  expertCount: expertCount)
    }

    private static func environmentPrefillTailTile() -> Int {
        switch ProcessInfo.processInfo.environment["SHRIKE_PREFILL_TAIL_TILE"] {
        case "32": return 32
        case "off": return 0
        default: return prefillTailTileDefault
        }
    }

    /// `SHRIKE_PREFILL_MATRIX_MIN_ROWS=<n>` (3…32) lowers the row-count floor
    /// below which the attention, projection and shared-expert matrix kernels
    /// fall back to their scalar paths; unset or unparsable takes
    /// `prefillMatrixMinRowsDefault` (16 — the matrix attention, projection
    /// and shared-expert paths down to 16 rows, so the 21-row follow-up turn
    /// runs on them), and `=32` restores today's fixed thresholds as the A/B.
    /// The floor of 3 keeps the prompt cache's settle on today's kernels.
    private static let prefillMatrixMinRowsDefault = 16

    static func parsePrefillMatrixMinRows(_ raw: String?) -> Int {
        guard let raw, let rows = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            return prefillMatrixMinRowsDefault
        }
        return max(3, min(32, rows))
    }

    private static func environmentPrefillMatrixMinRows() -> Int {
        parsePrefillMatrixMinRows(ProcessInfo.processInfo.environment["SHRIKE_PREFILL_MATRIX_MIN_ROWS"])
    }

    private static func makePoolResidency(context: MetalContext)
        -> (holder: ExpertPoolResidency?, unavailableReason: String?) {
        guard ProcessInfo.processInfo.environment["SHRIKE_PREFILL_POOL_RESIDENCY"] != "none" else {
            return (nil, nil)
        }
        do {
            return (try ExpertPoolResidency(device: context.device, queue: context.queue), nil)
        } catch {
            return (nil, "\(error)")
        }
    }

    private static func environmentPrefillTileBatch() -> Int {
        guard let raw = ProcessInfo.processInfo.environment["SHRIKE_PREFILL_TILE_BATCH"],
              let width = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            return 1
        }
        return max(1, min(16, width))
    }

    /// `SHRIKE_PREFILL_TILE_DEPTH=<n>` (1…8) sets the routed tile pipeline's
    /// pending-tile depth; unset or unparsable takes `prefillTileDepthDefault`
    /// (P16 sweep: depth 2 beat depth 1 on the mini's 12k wall, hits unmoved).
    private static let prefillTileDepthDefault = 2

    static func parsePrefillTileDepth(_ raw: String?) -> Int {
        guard let raw, let depth = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            return prefillTileDepthDefault
        }
        return max(1, min(8, depth))
    }

    private static func environmentPrefillTileDepth() -> Int {
        parsePrefillTileDepth(ProcessInfo.processInfo.environment["SHRIKE_PREFILL_TILE_DEPTH"])
    }

    /// `SHRIKE_PREFILL_FETCH_DEPTH=<n>` (1…2) sets how many routed tile
    /// fetches run in flight; unset or unparsable takes
    /// `prefillFetchDepthDefault` (2, the next tile's fetch begun before the
    /// current one is awaited); `=1` is the A/B that restores the
    /// single-fetch loop.
    private static let prefillFetchDepthDefault = 2

    static func parsePrefillFetchDepth(_ raw: String?) -> Int {
        guard let raw, let depth = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            return prefillFetchDepthDefault
        }
        return max(1, min(2, depth))
    }

    private static func environmentPrefillFetchDepth() -> Int {
        parsePrefillFetchDepth(ProcessInfo.processInfo.environment["SHRIKE_PREFILL_FETCH_DEPTH"])
    }

    /// `PrefillChunkPlanner.spans` lays chunks out contiguously from
    /// `startPosition`, so this is the chunk index's parity within the prompt.
    static func prefillChunkSweepIsDescending(startPosition: Int, chunkTokens: Int) -> Bool {
        (startPosition / chunkTokens) % 2 == 1
    }

    /// `.fixed` is always ascending, `.alternate` ignores `carried` and
    /// matches the two-argument overload above, and `.carry` flips the
    /// previous chunk's direction (`nil` meaning ascending) — the per-chunk
    /// write already tracks position, so no further parity term belongs
    /// here; `participatesInCarry: false` forces `.alternate` behaviour
    /// regardless of `mode`, for a chunk that must neither read nor
    /// influence the request-level carry.
    static func prefillChunkSweepIsDescending(mode: PrefillSweepMode, carried: Bool?,
                                              startPosition: Int, chunkTokens: Int,
                                              participatesInCarry: Bool = true) -> Bool {
        guard participatesInCarry else {
            return prefillChunkSweepIsDescending(startPosition: startPosition, chunkTokens: chunkTokens)
        }
        switch mode {
        case .fixed:
            return false
        case .alternate:
            return prefillChunkSweepIsDescending(startPosition: startPosition, chunkTokens: chunkTokens)
        case .carry:
            return carried == false
        case .recency, .resident:
            // Both build their own per-layer order in `buildPrefillRoutes`
            // and never read this value; kept only for exhaustiveness.
            return false
        }
    }

    /// Whether a chunk should sweep by a computed order: `.recency` or
    /// `.resident`, and (matching the direction switch's own isolation)
    /// only when the chunk participates in carry, so a chunk outside the
    /// carry keeps index tiling under the knob.
    static func prefillChunkUsesComputedSweepOrder(mode: PrefillSweepMode,
                                                   participatesInCarry: Bool) -> Bool {
        mode.usesComputedOrder && participatesInCarry
    }

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head mode. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    private let gdnPrefillScanChunked: Bool
    private var routerReadbackTag: UInt32 = 0
    /// Bookkeeping that needs a command's GPU stamps, which the word wake reads before they exist.
    private var deferredGPURecords: [DeferredGPURecord] = []
    private let predictivePrefetch: ExpertPrefetchRing
    private let anePrefill: ANEPrefillAttention?
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        try runtimeConfiguration.validate(maxContext: maxContext)
        let yarnParameters = try Self.resolveYaRNParameters(
            model: model, runtimeConfiguration: runtimeConfiguration)
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
            && model.lmHeadWeightBits == 4
            && model.attentionWeightBits == 4
        self.prefillAttentionPath = Self.environmentPrefillAttentionPath(
            default: runtimeConfiguration.prefillAttentionPath)
        self.gdnPrefillScanChunked =
            ProcessInfo.processInfo.environment["SHRIKE_GDN_PREFILL_SCAN"] != "serial"
        self.prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: Self.environmentPrefillTileDepth(),
            tilesPerCommandBuffer: Self.environmentPrefillTileBatch(),
            fetchLookahead: Self.environmentPrefillFetchDepth() - 1)
        self.prefillRoutedGEMMGrouped =
            ProcessInfo.processInfo.environment["SHRIKE_PREFILL_ROUTED_GEMM"] != "per-expert"
        self.prefillRouteOverlap =
            ProcessInfo.processInfo.environment["SHRIKE_PREFILL_ROUTE_OVERLAP"] != "off"
        self.prefillSweepMode = try Self.environmentPrefillSweepMode()
        self.prefillSweepTail = try Self.environmentPrefillSweepTail(expertCount: self.cfg.numExperts)
        self.prefillTailTile = Self.environmentPrefillTailTile()
        self.prefillMatrixMinRows = Self.environmentPrefillMatrixMinRows()
        let residency = Self.makePoolResidency(context: context)
        self.poolResidency = residency.holder
        self.poolResidencyUnavailableReason = residency.unavailableReason
        self.predictivePrefetch = try Self.makePredictivePrefetch(model: model)
        self.prefetchTraceFD = try Self.openPrefetchTrace(runtimeConfiguration.prefetchTracePath)
        self.anePrefill = try Self.makeANEPrefill(
            model: model, device: context.device)
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
                                     precision: runtimeConfiguration.kvCachePrecision,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: runtimeConfiguration.prefillChunkTokens)

        let silu = cfg.hiddenActivation == "silu"
        let kernels = try Self.makeDecodeKernels(
            model: model, context: context, silu: silu, yarn: yarnParameters,
            runtimeConfiguration: runtimeConfiguration)
        self.embedInt4 = kernels.embedInt4
        self.affineEmbed = kernels.affineEmbed
        self.rms = kernels.rms
        self.int4 = kernels.int4
        self.affine = kernels.affine
        self.attention = kernels.attention
        self.kvQuantizer = kernels.kvQuantizer
        self.shared = kernels.shared
        self.moe = kernels.moe
        self.fusionHead = kernels.fusionHead
        self.fusedQKVGEMV = kernels.fusedQKVGEMV
        self.fusedQKVEpilogue = kernels.fusedQKVEpilogue
        self.elementwise = kernels.elementwise
        self.gdn = kernels.gdn
        self.gdnState = kernels.gdnState
        self.mla = kernels.mla
        self.rope = kernels.rope
        self.int8ScalarGate = kernels.int8ScalarGate
        let prefill = try Self.makePrefillKernels(
            model: model, context: context, silu: silu, yarn: yarnParameters)
        self.prefillEmbed = prefill.embed
        self.prefillRMS = prefill.rms
        self.prefillQMM = prefill.qmm
        self.prefillMPPAffineInt4 = prefill.mppAffineInt4
        self.prefillQKVEpilogue = prefill.qkvEpilogue
        self.prefillAttention = prefill.attention
        self.prefillRouter = prefill.router
        self.prefillSharedExpert = prefill.sharedExpert
        self.prefillGroupedMoE = prefill.groupedMoE
        self.prefillMoE = prefill.moe
        self.prefillFinalRowHead = prefill.finalRowHead

        self.decodeScratch = try Self.makeDecodeScratchBuffers(
            cfg: cfg, device: context.device)
        self.specDispatchArguments = try Self.makeSpeculativeDispatchArguments(
            cfg: cfg, device: context.device)
        self.residencyReadback = try Self.makeResidencyReadbackBuffers(
            cfg: cfg, device: context.device)
        self.gdnScratch = try Self.makeGDNScratchBuffers(
            cfg: cfg, device: context.device)
        self.mlaScratch = try Self.makeMLAScratchBuffers(
            cfg: cfg, device: context.device)
        self.sharedExpertProjections = try Self.makeSharedExpertProjections(
            model: model, cfg: cfg)
        let routerScales = try Self.makeRouterScaleBuffers(
            cfg: cfg, device: context.device)
        self.effectiveScaleBuffers = routerScales.effectiveScale
        self.onesPerExpertScale = routerScales.onesPerExpertScale
        self.routerLogitBias = try Self.makeRouterLogitBias(
            model: model, cfg: cfg, device: context.device)
    }

    // MARK: - Init factories

    private static func resolveYaRNParameters(
        model: Model, runtimeConfiguration: RuntimeConfiguration
    ) throws -> YaRNRoPEParameters? {
        if let archYaRN = model.config.archYaRN {
            // The architecture mandates its own YaRN; the user
            // context-extension mode would silently fight it.
            guard runtimeConfiguration.ropeScalingMode != .yarn else {
                throw RuntimeConfigurationError.yaRNUnsupportedArchitecture
            }
            return archYaRN
        }
        guard runtimeConfiguration.ropeScalingMode == .yarn else { return nil }
        guard model.config.ropeNeoxSubdim else {
            throw RuntimeConfigurationError.yaRNUnsupportedArchitecture
        }
        return YaRNRoPEParameters(
            headDim: model.config.fullHeadDim,
            partialRotaryFactor: model.config.partialRotaryFactor,
            theta: model.config.fullRopeTheta,
            targetContextTokens: runtimeConfiguration.yarnContextTokens)
    }

    private static func makePredictivePrefetch(model: Model) throws -> ExpertPrefetchRing {
        // In-flight reads hold cells beside the completed ones a later plan
        // may still take, so the ring is sized for both.
        try model.configurePrefetchCells(model.config.topKExperts + ExpertPrefetchRing.inFlightBudget)
        let cells = try model.prefetchCells()
        return try ExpertPrefetchRing(cells: cells) { layer, expert, cell in
            model.dropRoutedExpertLanding(layer: layer, expert: expert, cell: cell)
        }
    }

    static func openPrefetchTrace(_ path: String?) throws -> Int32 {
        guard let path else { return -1 }
        let descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        guard descriptor >= 0 else {
            throw RuntimeConfigurationError.invalidPrefetch(
                "trace path \(path) cannot be opened: \(String(cString: strerror(errno)))")
        }
        return descriptor
    }

    struct PrefetchRaceSplit: Equatable {
        var before: UInt64 = 0
        var during: UInt64 = 0
        var duringLastFifty: UInt64 = 0
        var duringFiftyToOneFifty: UInt64 = 0
        var duringEarlier: UInt64 = 0
        var after: UInt64 = 0
        var unknown: UInt64 = 0
    }

    /// The classifier is the tail command's last kernel, so a read completed before
    /// the command's GPU start surely beat it and one completed after its end surely lost.
    static func prefetchRaceSplit(completions: [Int: UInt64], landed: [Int],
                                  gpuStartNanos: UInt64, gpuEndNanos: UInt64) -> PrefetchRaceSplit {
        var split = PrefetchRaceSplit()
        let known = gpuStartNanos > 0 && gpuEndNanos > 0
        for expert in landed {
            guard let completed = completions[expert], known else {
                split.unknown &+= 1
                continue
            }
            if completed < gpuStartNanos {
                split.before &+= 1
            } else if completed < gpuEndNanos {
                split.during &+= 1
                let margin = gpuEndNanos - completed
                if margin < 50_000 {
                    split.duringLastFifty &+= 1
                } else if margin < 150_000 {
                    split.duringFiftyToOneFifty &+= 1
                } else {
                    split.duringEarlier &+= 1
                }
            } else {
                split.after &+= 1
            }
        }
        return split
    }

    /// The word wake reaches the plan before the tail command reports its GPU
    /// times, so a race it cannot settle waits with the deferred records.
    private func recordPrefetchRace(landed: [Int], layer L: Int, tailCB: MTLCommandBuffer) {
        guard !landed.isEmpty else { return }
        let completions = predictivePrefetch.completionNanos(layer: L, experts: Set(landed))
        if tailCB.status == .completed {
            countPrefetchRace(completions: completions, landed: landed, tailCB: tailCB)
        } else {
            deferredGPURecords.append(
                .prefetchRace(completions: completions, landed: landed, tailCB: tailCB))
        }
    }

    private func countPrefetchRace(completions: [Int: UInt64], landed: [Int],
                                   tailCB: MTLCommandBuffer) {
        let split = Self.prefetchRaceSplit(
            completions: completions, landed: landed,
            gpuStartNanos: UInt64(max(0, tailCB.gpuStartTime) * 1_000_000_000),
            gpuEndNanos: UInt64(max(0, tailCB.gpuEndTime) * 1_000_000_000))
        totalPrefetchBeforeClassify &+= split.before
        totalPrefetchDuringTail &+= split.during
        totalPrefetchDuringLastFifty &+= split.duringLastFifty
        totalPrefetchDuringFiftyToOneFifty &+= split.duringFiftyToOneFifty
        totalPrefetchDuringEarlier &+= split.duringEarlier
        totalPrefetchAfterClassify &+= split.after
        totalPrefetchRaceUnknown &+= split.unknown
    }

    /// A refused speculative read is counted, never thrown into the decode.
    private func schedulePredictivePrefetch(layer L: Int, predicted: [Int],
                                            demand: ExpertLoadOperation?) {
        guard L + 1 < cfg.numLayers else { return }
        let target = L + 1
        let model = self.model
        let predictivePrefetch = self.predictivePrefetch
        Self.schedulePrefetchIssue(demand: demand) { deferred in
            do {
                let resident = Set(try model.routedExpertResidentIDs(layer: target))
                try predictivePrefetch.begin(layer: target, experts: predicted,
                                             resident: resident, deferred: deferred) { experts, cells in
                    try model.beginRoutedExpertPrefetch(layer: target, experts: experts, cells: cells)
                }
            } catch {
                if predictivePrefetch.noteHookFailure() {
                    print("Shrike prefetch: a speculative read was refused and counted, not retried: \(error)")
                }
            }
        }
    }

    /// Waits for the layer's demand batch so the ring's reads never share
    /// the drive with it.
    static func schedulePrefetchIssue(demand: ExpertLoadOperation?,
                                      _ issue: @escaping (_ deferred: Bool) -> Void) {
        guard let demand else {
            issue(false)
            return
        }
        switch demand.state {
        case .completed, .failed:
            issue(false)
        case .submitted, .inFlight:
            demand.onCompletion { issue(true) }
        }
    }

    // Track A: the ANE prefill sidecar, opt-in. Only the qwen36 target
    // family qualifies; with the switch on and the sidecar missing,
    // construction fails closed with the export command.
    private static func makeANEPrefill(
        model: Model, device: MTLDevice
    ) throws -> ANEPrefillAttention? {
        guard try RuntimePrefillANE.environmentValue() == .on,
              model.config.family == .qwen36 else { return nil }
        return try ANEPrefillAttention(
            modelDirectory: model.directoryURL,
            device: device,
            hiddenSize: model.config.hiddenSize,
            kvDim: model.config.numFullKVHeads * model.config.fullHeadDim,
            weightsSha256: model.weightsDigestFromManifest)
    }

    private struct DecodeKernels {
        let embedInt4: EmbedLookupInt4
        let affineEmbed: AffineQuantEmbeddingLookup?
        let rms: RMSNorm
        let int4: DequantInt4GEMV
        let affine: AffineQuantGEMV?
        let attention: Attention
        let kvQuantizer: KVCacheQuantizer?
        let shared: SharedExpertRuntime
        let moe: MoE
        let fusionHead: LMHeadChainInt4
        let fusedQKVGEMV: FusedQKVGEMV
        let fusedQKVEpilogue: FusedQKVEpilogue
        let elementwise: Elementwise?
        let gdn: GDN?
        let gdnState: GDNStateManager?
        let mla: MLA?
        let rope: RoPE?
        let int8ScalarGate: DequantInt8GEMV?
    }

    private static func makeDecodeKernels(
        model: Model, context: MetalContext, silu: Bool,
        yarn: YaRNRoPEParameters?,
        runtimeConfiguration: RuntimeConfiguration
    ) throws -> DecodeKernels {
        let cfg = model.config
        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise = cfg.attnOutputGate
            || cfg.sharedExpertGated
            || cfg.hasLinearAttentionLayers
            || cfg.hasAttentionBiases
        let gdn: GDN?
        let gdnState: GDNStateManager?
        if cfg.hasLinearAttentionLayers {
            gdn = try GDN(context: context, config: cfg.linearAttention,
                          perChannelDecay: cfg.linearAttentionPerChannelDecay,
                          sigmoidGatedNormEps: cfg.linearAttentionSigmoidGateNormEps,
                          specializedHiddenSize: cfg.hiddenSize)
            gdnState = try GDNStateManager(
                device: context.device,
                config: cfg)
        } else {
            gdn = nil
            gdnState = nil
        }
        let mla: MLA?
        if let mlaCfg = cfg.mla, cfg.hasMLALayers {
            mla = try MLA(context: context, config: mlaCfg,
                          numHeads: cfg.numHeads)
        } else {
            mla = nil
        }
        return DecodeKernels(
            embedInt4: try EmbedLookupInt4(context: context),
            affineEmbed: model.embeddingWeightBits == 4 ? nil
                : try AffineQuantEmbeddingLookup(context: context,
                                                 weightBits: model.embeddingWeightBits),
            rms: try RMSNorm(context: context),
            int4: try DequantInt4GEMV(
                context: context,
                additionalShapes: cfg.decodeInt4GEMVShapes),
            affine: model.attentionWeightBits == 4 ? nil
                : try AffineQuantGEMV(context: context,
                                      weightBits: model.attentionWeightBits),
            // kvSharedApplicable gates per shape at encode time, so the
            // kv-shared default is safe for archs its kernel never serves.
            attention: try Attention(context: context,
                                     maxQHeads: cfg.numHeads,
                                     maxHeadDim: max(cfg.headDim, cfg.fullHeadDim),
                                     supportsSinks: cfg.hasAttentionSinks,
                                     supportsMLA: cfg.hasMLALayers,
                                     partialLoopVariant: .kvShared),
            kvQuantizer: runtimeConfiguration.kvCachePrecision.isQuantized
                ? try KVCacheQuantizer(context: context) : nil,
            shared: try SharedExpertRuntime(
                context: context,
                weightBits: model.sharedExpertWeightBits,
                siluActivation: silu,
                decodeShapes: cfg.hasSharedExpert
                    ? [(m: cfg.intermediateSize, n: cfg.hiddenSize),
                       (m: cfg.hiddenSize, n: cfg.intermediateSize)]
                    : []),
            moe: try MoE(context: context,
                         siluActivation: silu,
                         routedWeightBits: model.routedExpertWeightBits,
                         routerWeightBits: model.routerWeightBits,
                         eventGatedIO: true,
                         specializedD: UInt32(cfg.hiddenSize),
                         specializedF: UInt32(cfg.moeIntermediateSize),
                         specializedNumExperts: UInt32(cfg.numExperts),
                         specializedTopK: UInt32(cfg.topKExperts),
                         expertAdditiveBiases: cfg.expertsHaveAdditiveBiases,
                         clampedSwiGLU: cfg.usesClampedSwiGLU,
                         sigmoidRouterScores: cfg.routerUsesSigmoidScores,
                         routedScalingFactor: Float(cfg.routedScalingFactor)),
            fusionHead: try LMHeadChainInt4(context: context,
                                            maxD: cfg.hiddenSize,
                                            maxVocab: cfg.vocabSize),
            fusedQKVGEMV: try FusedQKVGEMV(context: context),
            fusedQKVEpilogue: try FusedQKVEpilogue(context: context),
            elementwise: needsElementwise ? try Elementwise(context: context) : nil,
            gdn: gdn,
            gdnState: gdnState,
            mla: mla,
            rope: cfg.ropeNeoxSubdim
                ? try RoPE(context: context, yarn: yarn) : nil,
            int8ScalarGate: cfg.sharedExpertGated
                ? try DequantInt8GEMV(context: context,
                                      additionalShapes: cfg.decodeInt8GEMVShapes)
                : nil)
    }

    private struct PrefillKernels {
        let embed: PrefillEmbedLookupInt4
        let rms: PrefillRMSNorm
        let qmm: PrefillInt4QMM
        let mppAffineInt4: MPPPrefillInt4QMM?
        let qkvEpilogue: PrefillQKVEpilogue
        let attention: PrefillAttention
        let router: PrefillRouter
        let sharedExpert: PrefillSharedExpert
        let groupedMoE: PrefillGroupedRoutedMoE
        let moe: PrefillMoE
        let finalRowHead: PrefillFinalRowHeadInt4
    }

    private static func makePrefillKernels(
        model: Model, context: MetalContext, silu: Bool,
        yarn: YaRNRoPEParameters?
    ) throws -> PrefillKernels {
        let cfg = model.config
        return PrefillKernels(
            embed: try PrefillEmbedLookupInt4(
                context: context,
                weightBits: model.embeddingWeightBits),
            rms: try PrefillRMSNorm(context: context),
            qmm: try PrefillInt4QMM(
                context: context,
                weightBits: model.attentionWeightBits),
            mppAffineInt4: MPPPrefillInt4QMM(
                context: context,
                weightBits: model.attentionWeightBits),
            qkvEpilogue: try PrefillQKVEpilogue(context: context,
                                                yarn: yarn),
            attention: try PrefillAttention(context: context,
                                            supportsMLA: cfg.hasMLALayers),
            router: try PrefillRouter(
                context: context,
                weightBits: model.routerWeightBits,
                sigmoidRouterScores: cfg.routerUsesSigmoidScores,
                routedScalingFactor: Float(cfg.routedScalingFactor)),
            sharedExpert: try PrefillSharedExpert(
                context: context,
                weightBits: model.sharedExpertWeightBits,
                siluActivation: silu),
            groupedMoE: try PrefillGroupedRoutedMoE(
                context: context,
                siluActivation: silu,
                weightBits: model.routedExpertWeightBits,
                expertAdditiveBiases: cfg.expertsHaveAdditiveBiases,
                clampedSwiGLU: cfg.usesClampedSwiGLU),
            moe: try PrefillMoE(context: context),
            finalRowHead: try PrefillFinalRowHeadInt4(
                context: context,
                maxD: cfg.hiddenSize,
                weightBits: model.lmHeadWeightBits))
    }

    private static func scratchBuffer(
        device: MTLDevice, _ count: Int,
        _ stride: Int = MemoryLayout<Float16>.size,
        label: String
    ) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                        options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        b.label = label
        return b
    }

    private struct DecodeScratchBuffers {
        let hidden: MTLBuffer
        let normed: MTLBuffer
        let attnOut: MTLBuffer
        let qScratch: MTLBuffer
        let kStage: MTLBuffer
        let vStage: MTLBuffer
        let oOut: MTLBuffer
        let h1Buf: MTLBuffer
        let h2Buf: MTLBuffer
        let routedX: MTLBuffer
        let denseX: MTLBuffer
        let denseScratchGate: MTLBuffer
        let denseScratchUp: MTLBuffer
        let denseScratchAct: MTLBuffer
        let routerInput: MTLBuffer
        let zeroResidual: MTLBuffer
        let outIndices: MTLBuffer
        let outWeights: MTLBuffer
        let prefetchPredictionIndices: MTLBuffer
        let prefetchPredictionWeights: MTLBuffer
        let moeActs: MTLBuffer
        let greedyTokenBuf: MTLBuffer
        let qPackedScratch: MTLBuffer?
        let attnGateScratch: MTLBuffer?
        let sharedScalarGateBuf: MTLBuffer?
    }

    private static func makeDecodeScratchBuffers(
        cfg: ArchConfig, device: MTLDevice
    ) throws -> DecodeScratchBuffers {
        func buf(_ count: Int,
                 _ stride: Int = MemoryLayout<Float16>.size,
                 label: String) throws -> MTLBuffer {
            try scratchBuffer(device: device, count, stride, label: label)
        }
        let D = cfg.hiddenSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)
        let kvStageCount = max(cfg.numKVHeads * cfg.headDim,
                               cfg.numFullKVHeads * cfg.fullHeadDim)
        let sharedScratchF = max(cfg.intermediateSize, cfg.denseIntermediateSize)
        let zeroResidual = try buf(D, label: "decode.zeroResidual")
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(zeroResidual.contents(), 0, zeroResidual.length)
        return DecodeScratchBuffers(
            hidden: try buf(D, label: "decode.hidden"),
            normed: try buf(D, label: "decode.normed"),
            attnOut: try buf(maxQ, label: "decode.attnOut"),
            qScratch: try buf(maxQ, label: "decode.qScratch"),
            kStage: try buf(kvStageCount, label: "decode.kStage"),
            vStage: try buf(kvStageCount, label: "decode.vStage"),
            oOut: try buf(D, label: "decode.oOut"),
            h1Buf: try buf(D, label: "decode.h1"),
            h2Buf: try buf(D, label: "decode.h2"),
            routedX: try buf(D, label: "decode.routedX"),
            denseX: try buf(D, label: "decode.denseX"),
            denseScratchGate: try buf(sharedScratchF, label: "decode.denseScratchGate"),
            denseScratchUp: try buf(sharedScratchF, label: "decode.denseScratchUp"),
            denseScratchAct: try buf(sharedScratchF, label: "decode.denseScratchAct"),
            routerInput: try buf(D, label: "decode.routerInput"),
            zeroResidual: zeroResidual,
            outIndices: try buf(cfg.topKExperts, MemoryLayout<UInt32>.size,
                                label: "decode.outIndices"),
            outWeights: try buf(cfg.topKExperts, label: "decode.outWeights"),
            prefetchPredictionIndices: try buf(
                cfg.topKExperts, MemoryLayout<UInt32>.size,
                label: "decode.prefetchPredictionIndices"),
            prefetchPredictionWeights: try buf(
                cfg.topKExperts, label: "decode.prefetchPredictionWeights"),
            moeActs: try buf(cfg.topKExperts * cfg.moeIntermediateSize,
                             label: "decode.moeActs"),
            greedyTokenBuf: try buf(1, MemoryLayout<UInt32>.size,
                                    label: "decode.greedyToken"),
            // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
            qPackedScratch: cfg.attnOutputGate
                ? try buf(2 * maxQ, label: "decode.qPackedScratch") : nil,
            attnGateScratch: cfg.attnOutputGate
                ? try buf(maxQ, label: "decode.attnGateScratch") : nil,
            sharedScalarGateBuf: cfg.sharedExpertGated
                ? try buf(1, label: "decode.sharedScalarGate") : nil)
    }

    private struct ResidencyReadbackBuffers {
        let moeHitActiveSlots: MTLBuffer
        let moeMissActiveSlots: MTLBuffer
        let hitCount: MTLBuffer
        let hitPositions: MTLBuffer
        let missCount: MTLBuffer
        let missPositions: MTLBuffer
        let missExperts: MTLBuffer
        let resolvedSlots: MTLBuffer
        let resolvedGenerations: MTLBuffer
        let hostReadback: MTLBuffer
    }

    private static func makeResidencyReadbackBuffers(
        cfg: ArchConfig, device: MTLDevice
    ) throws -> ResidencyReadbackBuffers {
        func buf(_ count: Int, _ stride: Int, label: String) throws -> MTLBuffer {
            try scratchBuffer(device: device, count, stride, label: label)
        }
        let topK = cfg.topKExperts
        let u32 = MemoryLayout<UInt32>.size
        return ResidencyReadbackBuffers(
            moeHitActiveSlots: try buf(topK, u32, label: "decode.moeHitActiveSlots"),
            moeMissActiveSlots: try buf(topK, u32, label: "decode.moeMissActiveSlots"),
            hitCount: try buf(1, u32, label: "decode.residencyHitCount"),
            hitPositions: try buf(topK, u32, label: "decode.residencyHitPositions"),
            missCount: try buf(1, u32, label: "decode.residencyMissCount"),
            missPositions: try buf(topK, u32, label: "decode.residencyMissPositions"),
            missExperts: try buf(topK, u32, label: "decode.residencyMissExperts"),
            resolvedSlots: try buf(topK, u32, label: "decode.residencyResolvedSlots"),
            resolvedGenerations: try buf(topK, MemoryLayout<UInt64>.size,
                                         label: "decode.residencyResolvedGenerations"),
            hostReadback: try buf(RouterHostReadback.wordCount(topK: topK), u32,
                                  label: "decode.routerHostReadback"))
    }

    private struct GDNScratchBuffers {
        let qkvRaw: MTLBuffer
        let convOut: MTLBuffer
        let z: MTLBuffer
        let a: MTLBuffer
        let b: MTLBuffer
        let y: MTLBuffer
        let out: MTLBuffer
        let lowRank: MTLBuffer?
    }

    private static func makeGDNScratchBuffers(
        cfg: ArchConfig, device: MTLDevice
    ) throws -> GDNScratchBuffers? {
        guard cfg.hasLinearAttentionLayers else { return nil }
        let la = cfg.linearAttention
        func buf(_ count: Int, label: String) throws -> MTLBuffer {
            try scratchBuffer(device: device, count, label: label)
        }
        return GDNScratchBuffers(
            qkvRaw: try buf(la.qkvDim, label: "decode.gdnQKVRaw"),
            convOut: try buf(la.qkvDim, label: "decode.gdnConvOut"),
            z: try buf(la.valueDim, label: "decode.gdnZ"),
            a: try buf(cfg.linearAttentionPerChannelDecay
                           ? la.numVHeads * la.keyHeadDim : la.numVHeads,
                       label: "decode.gdnA"),
            b: try buf(la.numVHeads, label: "decode.gdnB"),
            y: try buf(la.valueDim, label: "decode.gdnY"),
            out: try buf(la.valueDim, label: "decode.gdnOut"),
            lowRank: cfg.linearAttentionPerChannelDecay
                ? try buf(la.keyHeadDim, label: "decode.gdnLowRank") : nil)
    }

    private struct MLAScratchBuffers {
        let qRaw: MTLBuffer
        let q: MTLBuffer
        let attnOut: MTLBuffer
        let unembedOut: MTLBuffer
    }

    private static func makeMLAScratchBuffers(
        cfg: ArchConfig, device: MTLDevice
    ) throws -> MLAScratchBuffers? {
        guard let mlaCfg = cfg.mla, cfg.hasMLALayers else { return nil }
        let H = cfg.numHeads
        func buf(_ count: Int, label: String) throws -> MTLBuffer {
            try scratchBuffer(device: device, count, label: label)
        }
        return MLAScratchBuffers(
            qRaw: try buf(H * (mlaCfg.qkNopeDim + mlaCfg.qkRopeDim),
                          label: "decode.mlaQRaw"),
            q: try buf(H * (mlaCfg.latentDim + mlaCfg.qkRopeDim),
                       label: "decode.mlaQ"),
            attnOut: try buf(H * mlaCfg.latentDim, label: "decode.mlaAttnOut"),
            unembedOut: try buf(H * mlaCfg.valueHeadDim,
                                label: "decode.mlaUnembedOut"))
    }

    private static func makeSharedExpertProjections(
        model: Model, cfg: ArchConfig
    ) throws -> [LayerSharedExpertProjections] {
        guard cfg.hasSharedExpert else { return [] }
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            // Leading dense layers (Kimi layer 0) carry a plain MLP in
            // this slot: same SwiGLU kernels, per-layer intermediate.
            let isDense = L < cfg.numLeadingDenseLayers
            let FL = isDense ? cfg.denseIntermediateSize : F
            let gate = isDense ? try model.denseMLPGate(layer: L)
                               : try model.sharedExpertGate(layer: L)
            let up = isDense ? try model.denseMLPUp(layer: L)
                             : try model.sharedExpertUp(layer: L)
            let down = isDense ? try model.denseMLPDown(layer: L)
                               : try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(FL), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(FL), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(FL)),
                scalarGate: (cfg.sharedExpertGated && !isDense)
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        return sharedViews
    }

    private static func makeRouterScaleBuffers(
        cfg: ArchConfig, device: MTLDevice
    ) throws -> (effectiveScale: [MTLBuffer], onesPerExpertScale: MTLBuffer) {
        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }
        // Plain linear router (Qwen): one shared BF16 ones buffer keeps
        // the router kernel's effective_scale multiply neutral, and a ones
        // per_expert_scale keeps the top-k weights untouched. (Softmax
        // over top-k then renormalize equals Qwen's softmax over all
        // experts then renormalize the selected top-k.)
        let ones = try bf16OnesBuffer(count: cfg.hiddenSize,
                                      label: "effective_scale.ones")
        return ([MTLBuffer](repeating: ones, count: cfg.numLayers),
                try bf16OnesBuffer(count: cfg.numExperts,
                                   label: "per_expert_scale.ones"))
    }

    private static func makeRouterLogitBias(
        model: Model, cfg: ArchConfig, device: MTLDevice
    ) throws -> [(buffer: MTLBuffer, offset: Int)] {
        if cfg.family == .gptOss20b {
            return try (0..<cfg.numLayers).map { layer in
                guard let view = try model.routerBias(layer: layer) else {
                    throw ModelError.tensorNotFound(
                        name: "language_model.model.layers.\(layer).mlp.router.bias")
                }
                return (view.buffer, Int(view.offset))
            }
        }
        guard let zeros = device.makeBuffer(
            length: cfg.numExperts * MemoryLayout<UInt16>.size,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        memset(zeros.contents(), 0, zeros.length)
        zeros.label = "router_logit_bias.zeros"
        if cfg.routerHasCorrectionBias {
            // Kimi: the sigmoid selector reads the correction bias
            // through the logit-bias slot; dense layers keep the zeros
            // placeholder (their router never runs).
            return try (0..<cfg.numLayers).map { layer in
                guard layer >= cfg.numLeadingDenseLayers else {
                    return (zeros, 0)
                }
                let view = try model.routerCorrectionBias(layer: layer)
                return (view.buffer, Int(view.offset))
            }
        }
        return [(buffer: MTLBuffer, offset: Int)](
            repeating: (zeros, 0), count: cfg.numLayers)
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    /// Whether `rewind(to:)` can seat the cursor on a shorter prefix. False when
    /// state cannot follow it back: recurrent GDN has absorbed every token it was
    /// advanced over, and a ring stores position `p` at slot `p % capacity`, so a
    /// shorter window's rows may already be overwritten. Both are settled at init,
    /// so a caller can decide before doing work it would have to discard.
    public var supportsPartialRewind: Bool {
        guard let kv, gdnState == nil else { return false }
        return !(0..<cfg.numLayers).contains { kv.ringCapacity(layer: $0) > 0 }
    }

    /// Seat the continuation cursor on a shorter prefix of the KV already held.
    public func rewind(to position: Int) throws {
        guard supportsPartialRewind, let kv else {
            throw PrefillError.prefillCursorMismatch(
                "this runner's state cannot follow the cursor back to \(position)")
        }
        try kv.rewind(to: position)
        resetTransientState()
    }

    public func captureInferenceState(
        maximumBytes: Int? = nil
    ) throws -> InferenceStateSnapshot {
        guard let kv, kv.position > 0 else {
            throw InferenceStateSnapshotError.invalidPosition(kv?.position ?? 0)
        }
        let kvLengths = try kv.snapshotSegmentLengths(at: kv.position)
        let gdnLengths = gdnState?.snapshotSegmentLengths() ?? []
        var payloadBytes = 0
        for length in kvLengths + gdnLengths {
            let (next, overflow) = payloadBytes.addingReportingOverflow(length)
            guard !overflow else { throw InferenceStateSnapshotError.integerOverflow }
            payloadBytes = next
        }
        if let maximumBytes, payloadBytes > maximumBytes {
            throw InferenceStateSnapshotError.exceedsLimit(
                bytes: payloadBytes,
                limit: maximumBytes)
        }
        var payload = Data()
        payload.reserveCapacity(payloadBytes)
        try kv.appendSnapshotPayload(to: &payload, segmentLengths: kvLengths)
        try gdnState?.appendSnapshotPayload(to: &payload, segmentLengths: gdnLengths)
        guard payload.count == payloadBytes else {
            throw InferenceStateSnapshotError.invalidPayloadSize(
                expected: payloadBytes,
                actual: payload.count)
        }
        return InferenceStateSnapshot(
            descriptor: InferenceStateSnapshotDescriptor(
                position: kv.position,
                kvSegmentLengths: kvLengths,
                gdnSegmentLengths: gdnLengths,
                payloadBytes: payloadBytes),
            payload: payload)
    }

    public func restoreInferenceState(_ snapshot: InferenceStateSnapshot) throws {
        do {
            let descriptor = snapshot.descriptor
            guard descriptor.version == InferenceStateSnapshotDescriptor.currentVersion else {
                throw InferenceStateSnapshotError.unsupportedVersion(descriptor.version)
            }
            guard descriptor.position > 0, descriptor.position <= maxContext else {
                throw InferenceStateSnapshotError.invalidPosition(descriptor.position)
            }
            let expectedBytes = try descriptor.validatedPayloadBytes()
            guard snapshot.payload.count == expectedBytes else {
                throw InferenceStateSnapshotError.invalidPayloadSize(
                    expected: expectedBytes,
                    actual: snapshot.payload.count)
            }
            guard let kv else { throw InferenceStateSnapshotError.invalidLayout }
            try snapshot.payload.withUnsafeBytes { bytes in
                var offset = 0
                try kv.restoreSnapshot(
                    position: descriptor.position,
                    segmentLengths: descriptor.kvSegmentLengths,
                    bytes: bytes,
                    offset: &offset)
                if let gdnState {
                    try gdnState.restoreSnapshot(
                        segmentLengths: descriptor.gdnSegmentLengths,
                        bytes: bytes,
                        offset: &offset)
                } else if !descriptor.gdnSegmentLengths.isEmpty {
                    throw InferenceStateSnapshotError.invalidLayout
                }
                guard offset == bytes.count else {
                    throw InferenceStateSnapshotError.invalidLayout
                }
            }
            resetTransientState()
        } catch {
            reset()
            throw error
        }
    }

    private func resetTransientState() {
        prefillChunkState.reset()
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (SHRIKE_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + io + cb2.
    public private(set) var totalWaitNanos: UInt64 = 0
    public private(set) var totalBodyNanos: UInt64 = 0
    public private(set) var totalMissIoNanos: UInt64 = 0
    public private(set) var totalExposedIoNanos: UInt64 = 0
    /// I/O-event signal to fixup-CB GPU execution start — the wake latency
    /// the GPU pays on top of the read itself (both clocks are mach-based).
    public private(set) var totalFixupWakeNanos: UInt64 = 0
    public private(set) var totalHitFixupLayers: UInt64 = 0
    public private(set) var totalRouterReadbackNanos: UInt64 = 0
    /// Mean routing-weight mass per rank (E0): summed normalized top-K weights
    /// by descending-score position, over `totalRankWeightLayers` layer-steps.
    public private(set) var totalRankWeightMass: [Double] = []
    public private(set) var totalRankWeightLayers: UInt64 = 0
    public private(set) var totalLoopSampleNanos: UInt64 = 0
    public private(set) var totalLoopDetokNanos: UInt64 = 0
    public private(set) var totalLoopProgressNanos: UInt64 = 0
    public private(set) var totalLoopProduceNanos: UInt64 = 0
    public private(set) var totalCachePlanNanos: UInt64 = 0
    public var totalPrefetchBeginNanos: UInt64 { predictivePrefetch.statistics.beginNanos }
    /// Landed predictions the classifier counted resident: the landing's prize.
    public private(set) var totalPrefetchLandedHits: UInt64 = 0
    public private(set) var totalPrefetchBeforeClassify: UInt64 = 0
    public private(set) var totalPrefetchDuringTail: UInt64 = 0
    public private(set) var totalPrefetchDuringLastFifty: UInt64 = 0
    public private(set) var totalPrefetchDuringFiftyToOneFifty: UInt64 = 0
    public private(set) var totalPrefetchDuringEarlier: UInt64 = 0
    public private(set) var totalPrefetchAfterClassify: UInt64 = 0
    public private(set) var totalPrefetchRaceUnknown: UInt64 = 0
    public private(set) var totalRoutedPinNanos: UInt64 = 0
    public private(set) var totalRoutedSubmitNanos: UInt64 = 0
    public private(set) var totalHitSplitArgBufNanos: UInt64 = 0
    public private(set) var totalHitSplitEncodeNanos: UInt64 = 0
    public private(set) var totalFixupBuildNanos: UInt64 = 0
    public private(set) var totalHitCommitToKernelNanos: UInt64 = 0
    public private(set) var totalHitKernelToGPUNanos: UInt64 = 0
    public private(set) var totalFixupCommitToKernelNanos: UInt64 = 0
    public private(set) var totalRouterWakeNanos: UInt64 = 0
    public private(set) var totalRouterWakeFallbacks: UInt64 = 0
    public var prefetchStatistics: ExpertPrefetchStatistics { predictivePrefetch.statistics }
    public private(set) var totalIOQueueNanos: UInt64 = 0
    public private(set) var totalExpertIOHostWaitsAvoided: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }

    public func expertStreamingStatistics() -> ExpertStreamingStatistics {
        model.routedExpertStatistics()
    }

    /// Decode-loop phase walls (SHRIKE_RUNNER_STATS): accumulated once per
    /// token by runRawCompletion's scalar loop.
    func recordDecodeLoopPhases(sample: UInt64, detok: UInt64,
                                progress: UInt64, produce: UInt64) {
        guard runnerStatsEnabled else { return }
        totalLoopSampleNanos &+= sample
        totalLoopDetokNanos &+= detok
        totalLoopProgressNanos &+= progress
        totalLoopProduceNanos &+= produce
    }

    // MARK: - Per-command-buffer GPU timing (SHRIKE_KERNEL_STATS)

    public struct KernelGPUGap: Sendable, Equatable {
        public let transition: String
        public var millis: Double = 0
        public var count: Int = 0
        public var hostMillis: Double = 0
        public var driverMillis: Double = 0
        public var queueMillis: Double = 0
    }

    /// One command buffer's GPU span for a named kernel role. The decode path
    /// is synchronous (commit + wait), so `gpuStartTime`/`gpuEndTime` are
    /// valid right after completion and cost nothing to read.
    private struct KernelGPUTiming {
        let role: String
        let start: TimeInterval
        let end: TimeInterval
        let kernelStart: TimeInterval
        let kernelEnd: TimeInterval
    }
    private var kernelGPUTimings: [KernelGPUTiming] = []
    private let kernelGPUTimingsEnabled =
        ProcessInfo.processInfo.environment["SHRIKE_KERNEL_STATS"] != nil
    private let runnerStatsEnabled =
        ProcessInfo.processInfo.environment["SHRIKE_RUNNER_STATS"] != nil
    private let layerTraceEnabled =
        ProcessInfo.processInfo.environment["SHRIKE_LAYER_TRACE"] != nil
    /// SHRIKE_GPU_CAPTURE_DIR: write one programmatic .gputrace of decode
    /// tokens 8–10 of the process's first generation (requires launching with
    /// METAL_CAPTURE_ENABLED=1; the bundle opens in Xcode's Metal debugger).
    private let gpuCaptureDir =
        ProcessInfo.processInfo.environment["SHRIKE_GPU_CAPTURE_DIR"]
    private var gpuCaptureProduceCalls = 0
    private var gpuCaptureActive = false
    private var gpuCaptureDone = false

    private func updateGPUCaptureWindow() {
        guard let gpuCaptureDir, !gpuCaptureDone else { return }
        gpuCaptureProduceCalls += 1
        let manager = MTLCaptureManager.shared()
        if !gpuCaptureActive, gpuCaptureProduceCalls == 8 {
            let descriptor = MTLCaptureDescriptor()
            descriptor.captureObject = ctx.device
            descriptor.destination = .gpuTraceDocument
            descriptor.outputURL = URL(fileURLWithPath: gpuCaptureDir)
                .appendingPathComponent("shrike-decode-\(Int(Date().timeIntervalSince1970)).gputrace")
            do {
                try manager.startCapture(with: descriptor)
                gpuCaptureActive = true
                print("Shrike gpu-capture started: \(descriptor.outputURL?.path ?? "?")")
            } catch {
                gpuCaptureDone = true
                print("Shrike gpu-capture failed to start: \(error)")
            }
        } else if gpuCaptureActive, gpuCaptureProduceCalls >= 10 {
            manager.stopCapture()
            gpuCaptureActive = false
            gpuCaptureDone = true
            print("Shrike gpu-capture stopped")
        }
    }

    /// Open file descriptor for SHRIKE_ROUTE_TRACE, or -1. Opened once and
    /// never closed: the runner lives as long as the process, and a decode
    /// loop is the wrong place to manage a diagnostic file's lifetime.
    private let routeTraceFD: Int32 = {
        guard let path = ProcessInfo.processInfo.environment["SHRIKE_ROUTE_TRACE"],
              !path.isEmpty else { return -1 }
        return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    }()

    /// JSONL trace for the v4.3 predictive-prefetch qualification probe.
    /// It deliberately records only exact routing and authoritative cache
    /// residency before planning; enabling it cannot submit I/O or alter cache
    /// decisions. Kept separate from SHRIKE_ROUTE_TRACE for compatibility.
    private let prefetchTraceFD: Int32

    static let prefetchJoinNanos: UInt64 = 400_000

    public func resetKernelGPUTimings() {
        kernelGPUTimings.removeAll(keepingCapacity: true)
        deferredGPURecords.removeAll()
    }

    func recordKernelGPU(role: String, _ cb: MTLCommandBuffer) {
        guard kernelGPUTimingsEnabled, cb.gpuEndTime > 0 else { return }
        kernelGPUTimings.append(
            KernelGPUTiming(role: role,
                            start: cb.gpuStartTime,
                            end: cb.gpuEndTime,
                            kernelStart: cb.kernelStartTime,
                            kernelEnd: cb.kernelEndTime))
    }

    /// Aggregated per-role GPU milliseconds for the current generation,
    /// largest first.
    public func kernelGPUTimingSummary() -> [(role: String, millis: Double, count: Int)] {
        var acc: [String: (millis: Double, count: Int)] = [:]
        for t in kernelGPUTimings {
            let millis = (t.end - t.start) * 1000
            acc[t.role, default: (0, 0)].millis += millis
            acc[t.role]!.count += 1
        }
        return acc.map { (role: $0.key, millis: $0.value.millis, count: $0.value.count) }
            .sorted { $0.millis > $1.millis }
    }

    /// Wall-clock span in which *any* recorded command buffer was on the GPU,
    /// and the span from the first start to the last end.
    ///
    /// Per-role sums double-count: the decode path deliberately runs the routed
    /// MoE buffer concurrently with the next layer's attention, so adding the
    /// roles together can exceed the time that actually elapsed. Merging the
    /// intervals answers the question the sums cannot -- whether the GPU is
    /// saturated (busy ~= span, so the only gain left is cheaper kernels) or
    /// idle in the gaps (busy << span, so there is overlap still to win).
    public func kernelGPUOccupancy() -> (busyMillis: Double, spanMillis: Double) {
        guard !kernelGPUTimings.isEmpty else { return (0, 0) }
        let sorted = kernelGPUTimings.sorted { $0.start < $1.start }
        var busy: TimeInterval = 0
        var mergedStart = sorted[0].start
        var mergedEnd = sorted[0].end
        for t in sorted.dropFirst() {
            if t.start > mergedEnd {
                busy += mergedEnd - mergedStart
                mergedStart = t.start
                mergedEnd = t.end
            } else if t.end > mergedEnd {
                mergedEnd = t.end
            }
        }
        busy += mergedEnd - mergedStart
        let span = sorted.map(\.end).max()! - sorted[0].start
        return (busy * 1000, span * 1000)
    }

    /// Where the GPU's idle time actually sits, attributed to the transition
    /// it falls in.
    ///
    /// `kernelGPUOccupancy` says how much idle there is; this says where. Each
    /// gap between one buffer finishing and the next starting is charged to the
    /// pair of roles it separates, so "attn_tail_router -> moe_phase1_2_routed"
    /// accumulates the wait for the router readback and expert fetch, while
    /// "moe_phase1_2_routed -> attn_norm_qkv" accumulates the per-layer
    /// turnaround. Without this the only way to pick a target is to divide
    /// total idle by a buffer count and assume the quotient means something,
    /// which is exactly the reasoning that produced a failed optimisation.
    ///
    /// The gap is also broken down, but not as a strict partition: `host` is
    /// host-late time only, clamped to zero (the previous buffer's GPU end to
    /// this one's `kernelStartTime`, when submission came later), while
    /// `driver` (`kernelStartTime` to `kernelEndTime`) and `queue`
    /// (`kernelEndTime` to the GPU start) are both measured from
    /// `kernelStartTime` regardless. When the host submits early (the depth ≥
    /// 2 regime this chapter ships by default), `host` clamps to zero and the
    /// three no longer sum to the gap.
    public func kernelGPUGaps() -> [KernelGPUGap] {
        guard kernelGPUTimings.count > 1 else { return [] }
        let sorted = kernelGPUTimings.sorted { $0.start < $1.start }
        var acc: [String: KernelGPUGap] = [:]
        var previous = sorted[0]
        for current in sorted.dropFirst() {
            // Overlapping buffers contribute no gap; advance the frontier to
            // whichever end is later so a long buffer does not manufacture one.
            let gap = current.start - previous.end
            if gap > 0 {
                let key = "\(previous.role)->\(current.role)"
                var entry = acc[key] ?? KernelGPUGap(transition: key)
                entry.millis += gap * 1000
                entry.count += 1
                entry.hostMillis += max(0, current.kernelStart - previous.end) * 1000
                entry.driverMillis += max(0, current.kernelEnd - current.kernelStart) * 1000
                entry.queueMillis += max(0, current.start - current.kernelEnd) * 1000
                acc[key] = entry
            }
            if current.end > previous.end { previous = current }
        }
        return acc.values.sorted { $0.millis > $1.millis }
    }

    // MARK: - Routing trace (SHRIKE_ROUTE_TRACE)

    /// Appends `position layer e0 e1 ... e7` for one decode layer.
    ///
    /// Which experts a token actually routes to is the input to every question
    /// about how expert weights should reach the GPU -- how large the working
    /// set really is, how much reuse there is between consecutive tokens, and
    /// therefore whether a residency scheme that is not the slot cache could
    /// hold it. Synthetic access patterns answer none of that: a full sweep of
    /// the expert file measures thrash that decode never causes, and a random
    /// pattern measures the opposite. This dumps the real thing so a replay can
    /// be driven by it.
    ///
    /// Off unless `SHRIKE_ROUTE_TRACE` names a file. Diagnostic only.
    private func recordRouteTrace(layer: Int, position: Int, tile: Int? = nil, experts: [Int],
                                  rowCounts: [Int]? = nil, lastRows: [Int]? = nil) {
        guard routeTraceFD >= 0 else { return }
        writeRouteTraceLine(Self.formatRouteTraceLine(position: position, layer: layer,
                                                       tile: tile, experts: experts,
                                                       rowCounts: rowCounts, lastRows: lastRows))
    }

    /// Marks a request's start in `SHRIKE_ROUTE_TRACE`; called only from
    /// `runRawCompletion`, never from the settle rewrite's `prefillChunked` call.
    func recordRouteTraceRequestStart(cachedTokens: Int, promptTokens: Int) {
        guard routeTraceFD >= 0 else { return }
        writeRouteTraceLine(Self.formatRouteTraceLine(cachedTokens: cachedTokens,
                                                       promptTokens: promptTokens))
    }

    private func writeRouteTraceLine(_ line: String) {
        let bytes = Array(line.utf8)
        var written = 0
        while written < bytes.count {
            let n = bytes.withUnsafeBytes { raw -> Int in
                write(routeTraceFD, raw.baseAddress!.advanced(by: written),
                      bytes.count - written)
            }
            if n <= 0 { break }
            written += n
        }
    }

    /// Row counts and last-row indices for `SHRIKE_ROUTE_TRACE`'s prefill
    /// suffix, one pair per expert in the tile's own order; `sortedPairs` is
    /// sorted ascending by token within each expert's range, so the range's
    /// last element already holds the highest row.
    private func routeTraceRowCounts(forTile tile: PrefillMoETile,
                                     routes: PrefillMoEGroupedRoutes) throws
        -> (counts: [Int], lastRows: [Int]) {
        let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
        let counts = ranges.map(\.pairCount)
        let lastRows = ranges.map { Int(routes.sortedPairs[$0.pairStart + $0.pairCount - 1].token) }
        return (counts, lastRows)
    }

    /// One `SHRIKE_ROUTE_TRACE` line: bare `position layer e0 e1 ...` for a
    /// decode layer, `p position layer tile e0 e1 ... | n0 n1 ...` (or
    /// `n0:l0 n1:l1 ...` once `lastRows` is given, `lK` the highest row index
    /// `eK` was routed from in this tile's chunk) for one prefill tile, or
    /// `r cachedTokens promptTokens` for a request's start, so a replay can
    /// split on the first field; the suffix is omitted when `rowCounts` is
    /// nil and stays bare counts when `lastRows` is nil, so an older parser
    /// still reads the line.
    static func formatRouteTraceLine(position: Int, layer: Int, tile: Int? = nil,
                                     experts: [Int], rowCounts: [Int]? = nil,
                                     lastRows: [Int]? = nil) -> String {
        var line = tile.map { "p \(position) \(layer) \($0)" } ?? "\(position) \(layer)"
        for expert in experts { line += " \(expert)" }
        if let rowCounts, !rowCounts.isEmpty {
            line += " |"
            for (index, count) in rowCounts.enumerated() {
                if let lastRows, index < lastRows.count {
                    line += " \(count):\(lastRows[index])"
                } else {
                    line += " \(count)"
                }
            }
        }
        line += "\n"
        return line
    }

    static func formatRouteTraceLine(cachedTokens: Int, promptTokens: Int) -> String {
        "r \(cachedTokens) \(promptTokens)\n"
    }

    /// Appends one exact pre-plan routing/cache observation. `resident` is
    /// captured before cache planning so a later miss reservation cannot make
    /// the trace falsely report an expert as absent.
    private func recordPrefetchTrace(layer: Int,
                                     position: Int,
                                     experts: [Int],
                                     misses: [Int],
                                     resident: [Int],
                                     nextLayerPrediction: [Int]) {
        guard prefetchTraceFD >= 0 else { return }
        let line = "{\"position\":\(position),\"layer\":\(layer),\"probe_distance\":1,\"experts\":\(experts),\"misses\":\(misses),\"resident\":\(resident),\"next_layer_prediction\":\(nextLayerPrediction)}\n"
        let bytes = Array(line.utf8)
        var written = 0
        while written < bytes.count {
            let count = bytes.withUnsafeBytes { raw -> Int in
                write(prefetchTraceFD, raw.baseAddress!.advanced(by: written),
                      bytes.count - written)
            }
            if count <= 0 { break }
            written += count
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        do {
            for (spanIndex, span) in spans.enumerated() {
                try Task.checkCancellation()
                let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
                let upper = tokens.index(lower, offsetBy: span.tokenCount)
                try await executePrefillChunk(
                    tokens: tokens[lower..<upper],
                    startPosition: span.startPosition,
                    outputMode: outputMode,
                    logits: logits,
                    scratch: scratch,
                    config: config,
                    writeFinalHead: spanIndex == spans.count - 1)
                try Task.checkCancellation()
                onProgress(span.completedCount)
            }
        } catch {
            // Any failure — cancellation, a GPU command-buffer error, an I/O
            // error mid-routed-fetch — may have written partial KV rows and
            // left the chunk state dirty. Reset so the next request does not
            // trip `chunkedRunnerDirty` on a stale in-flight chunk.
            reset()
            throw error
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg, runtime: config)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    /// lint:allow-long the orchestrator for one prefill chunk: scratch setup,
    /// the per-layer dispatch, and the head. Each stage it calls is its own
    /// method; what remains is the sequence, and inlining less of it would
    /// only hide the order the stages must run in.
    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     participatesInCarry: Bool = true) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires a KV cache")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        // KV grows on demand rather than reserving maxContext, so make room for
        // this chunk before anything writes into it.
        try kv?.reserve(tokens: startPosition + tokens.count)
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }


        let layerViews = try makeLayerPrefillViews()

        // Reused UInt32 token-ID buffer, sized to the largest chunk seen so
        // far and grown on demand (R23); the prefill hot path never allocates
        // a Metal buffer per chunk.
        let tokenBytes = tokens.count * MemoryLayout<UInt32>.stride
        let tokenBuffer: MTLBuffer
        if let existing = prefillTokenBuffer, existing.length >= tokenBytes {
            tokenBuffer = existing
        } else {
            guard let made = ctx.device.makeBuffer(length: tokenBytes,
                                                   options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            made.label = "prefill.tokenIDs"
            prefillTokenBuffer = made
            tokenBuffer = made
        }
        let tokenPtr = tokenBuffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, token) in tokens.enumerated() {
            tokenPtr[i] = UInt32(bitPattern: token)
        }
        let D = cfg.hiddenSize
        let eps: Float = cfg.rmsNormEps
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = try model.embedding()


        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        try prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: embedOutScale,
                            vocab: UInt32(cfg.vocabSize))

        // Track A: whether this chunk's full-attention layers run on the ANE.
        // Non-4096 chunk configs and prompts beyond the sidecar's history
        // variants stay on the GPU; continuity is enforced inside
        // eligibleChunk so a fallback mid-prompt sticks for the rest of the
        // request.
        let aneChunk: ANEPrefillAttention? = {
            guard let ane = anePrefill,
                  ane.eligibleChunk(startPosition: startPosition,
                                    tokenCount: tokens.count,
                                    configChunkTokens: config.chunkTokens)
            else { return nil }
            return ane
        }()

        let prefillProfile = ProcessInfo.processInfo.environment["SHRIKE_PHASES"] != nil
        var prefillRouteNanos: UInt64 = 0
        var prefillTileNanos: UInt64 = 0
        var prefillTailNanos: UInt64 = 0
        var prefillActiveExperts: UInt64 = 0
        let prefillDescendingSweep = Self.prefillChunkSweepIsDescending(
            mode: prefillSweepMode,
            carried: prefillLastChunkDescending,
            startPosition: startPosition,
            chunkTokens: config.chunkTokens,
            participatesInCarry: participatesInCarry)
        if participatesInCarry, !prefillSweepMode.usesComputedOrder {
            prefillLastChunkDescending = prefillDescendingSweep
        }

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
            let prefillLayerStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            model.beginOpeningRoutedExpertStreamer(layer: L)
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            try prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if isLinear {
                try encodeLinearAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D)
            } else if cfg.layerIsMLA(L) {
                try encodeMLAAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    startPosition: startPosition)
            } else if let ane = aneChunk, ane.coveredLayers.contains(L) {
                try await runANEFullAttentionPrefill(
                    ane: ane, cb: &cb, layer: L, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    startPosition: startPosition, kvDim: kvDim)
            } else {
                try encodeFullAttentionPrefill(
                    cb: cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D, startPosition: startPosition,
                    isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                    qDim: qDim, kvDim: kvDim, rmsEps: eps)
            }
            // Plain pre-norm residual block: hidden += attention branch,
            // then one post-attention norm feeds router, shared expert,
            // and routed phase 1 (routedX doubles as moeX).
            try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                           hidden: scratch.hidden,
                                           delta: scratch.h1,
                                           count: t * D)
            try prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.postAttention.buffer,
                                   weightOffset: Int(views.postAttention.offset),
                                   out: scratch.routedX,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if L < cfg.numLeadingDenseLayers {
                // Leading dense-MLP layer (Kimi layer 0): no router, no
                // routed experts — one SwiGLU block at the layer's own
                // intermediate, folded like the shared branch.
                let dense = sharedExpertProjections[L]
                try prefillSharedExpert.encodeBlock(commandBuffer: cb,
                                                x: scratch.routedX,
                                                y: scratch.h1,
                                                gate: dense.gate,
                                                up: dense.up,
                                                down: dense.down,
                                                scratchGate: scratch.sharedGateScratch,
                                                scratchUp: scratch.sharedUpScratch,
                                                scratchAct: scratch.sharedActScratch,
                                                queryCount: t,
                                                d: D,
                                                intermediate: cfg.denseIntermediateSize,
                                                xStrideElements: D,
                                                yStrideElements: D)
                try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
            } else {
                try await encodeRoutedMoEPrefill(
                    cb: &cb, layer: L, views: views, scratch: scratch,
                    tokenCount: t, hiddenSize: D,
                    startPosition: startPosition,
                    descendingSweep: prefillDescendingSweep,
                    participatesInCarry: participatesInCarry,
                    layerStart: prefillLayerStart,
                    routeNanos: &prefillRouteNanos,
                    tileNanos: &prefillTileNanos,
                    tailNanos: &prefillTailNanos,
                    activeExperts: &prefillActiveExperts)
            }
        }

        if prefillProfile {
            let prefillTotal = prefillRouteNanos + prefillTileNanos + prefillTailNanos
            print("[prefill phases over \(t) tokens, \(prefillTotal / 1_000_000) ms total]")
            print("  route readback + GPU: \(String(format: "%.1f", Double(prefillRouteNanos) / 1e6)) ms")
            print("  expert fetch + tiles: \(String(format: "%.1f", Double(prefillTileNanos) / 1e6)) ms")
            print("  tail + residual:      \(String(format: "%.1f", Double(prefillTailNanos) / 1e6)) ms")
            let perLayer = Double(prefillActiveExperts) / Double(max(1, cfg.numLayers))
            print("  active experts/layer: \(String(format: "%.2f", perLayer))"
                + " (topK=\(cfg.topKExperts), max possible \(t * cfg.topKExperts))")
        }

        if writeFinalHead {
            try encodeFinalHead(logits: logits, scratch: scratch,
                                tokenCount: t, hiddenSize: D, rmsEps: eps,
                                outputMode: outputMode)
        }

        aneChunk?.finishChunk(startPosition: startPosition,
                              tokenCount: tokens.count)
        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    /// One full-attention layer's prefill attention on the Neural Engine
    /// (Track A). The layer's input norm is already encoded on `cb`; this
    /// stages it out, waits, predicts, and rebinds `cb` so the rest of the
    /// layer (KV quantization, residual, MoE) proceeds exactly as on the GPU
    /// path — `stagingK`/`stagingV` stand in for `kStage`/`vStage` and the
    /// attention output is blitted into `h1` for the generic residual tail.
    private func runANEFullAttentionPrefill(
        ane: ANEPrefillAttention,
        cb: inout MTLCommandBuffer,
        layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int,
        kvDim: Int
    ) async throws {
        let halfBytes = MemoryLayout<Float16>.stride
        guard let stage = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        stage.copy(from: scratch.normed, sourceOffset: 0,
                   to: ane.stagingNormed, destinationOffset: 0,
                   size: t * D * halfBytes)
        stage.endEncoding()
        cb.commit()
        try waitForCompletion(cb)
        recordKernelGPU(role: "prefill_ane_stage", cb)

        try await ane.predict(layer: L, history: startPosition, tokenCount: t)
        ane.appendShadow(layer: L, startPosition: startPosition, tokenCount: t)
        // Start the next covered layer's model load now: it overlaps the MoE
        // stage the caller is about to encode and run on the GPU, which is
        // roughly an order of magnitude longer than the ~0.5 s load.
        if let next = ((L + 1)..<cfg.numLayers).first(where: {
            ane.coveredLayers.contains($0)
        }) {
            ane.preload(layer: next, history: startPosition)
        }

        guard let next = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb = next
        if let kv {
            try copyPrefillKVToCache(commandBuffer: cb,
                                     kv: kv,
                                     layer: L,
                                     startPosition: startPosition,
                                     tokenCount: t,
                                     keySource: ane.stagingK,
                                     valueSource: ane.stagingV,
                                     bytesPerToken: kvDim * halfBytes)
        }
        guard let out = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        out.copy(from: ane.stagingOut, sourceOffset: 0,
                 to: scratch.h1, destinationOffset: 0,
                 size: t * D * halfBytes)
        out.endEncoding()
    }

    /// lint:allow-long the orchestrator for one decode step, in the same
    /// shape as executePrefillChunk: embed, the per-layer dispatch, the head.
    /// One routed decode layer's command buffers, encoded but not committed.
    /// The next layer is encoded while the GPU runs the current one, so the
    /// post-readback critical path is commits only.
    private struct HeldLayerCommands {
        let layer: Int
        let attnCB: MTLCommandBuffer
        let softmaxCB: MTLCommandBuffer?
        /// nil when the tail stage is folded into `attnCB` (one CB per layer;
        /// only the gpt-oss and plain paths keep the split, their o_proj must
        /// run after the separately committed softmax CB).
        let tailCB: MTLCommandBuffer?
        /// The speculative routed command; the shared-expert chain rides at
        /// its head.
        let specCB: MTLCommandBuffer
        let overlapCompletionClock: CommandCompletionClock?
        /// The tag the classifier stamps on this layer's host readback; zero
        /// when no classifier ran (the host then reads the raw buffers).
        let readbackTag: UInt32

        /// The CB whose completion publishes the router output.
        var routerCB: MTLCommandBuffer { tailCB ?? attnCB }
    }

    private func commitHeldLayerCommands(_ cmds: HeldLayerCommands) {
        cmds.attnCB.commit()
        cmds.softmaxCB?.commit()
        cmds.tailCB?.commit()
        // Queued before the tailCB wait, not after: the GPU runs the shared
        // MLP and the whole routed layer while the CPU blocks on tailCB for
        // the routing.
        cmds.specCB.commit()
    }

    private func encodeLayerCommands(layer L: Int, position: Int)
        throws -> HeldLayerCommands {
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = cfg.rmsNormEps
        let isLinear = cfg.layerIsLinear(L)
        let inNorm = try model.inputNorm(layer: L)
        let postAttn = try model.postAttnNorm(layer: L)
        let routerW = try model.router(layer: L)
        let nextRouterW: TensorView?
        if L + 1 < cfg.numLayers, L + 1 >= cfg.numLeadingDenseLayers {
            nextRouterW = try model.router(layer: L + 1)
        } else {
            nextRouterW = nil
        }
        let residencyResources = try model.routedExpertResidency(layer: L)
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (onesPerExpertScale!, 0)
        guard let attnCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        // GDN, MLA, and gated attention encode everything through o_proj on
        // attnCB, so the tail stage folds into the same CB — one submission
        // and one boundary per layer. gpt-oss and the plain path keep a
        // separate tail CB: their o_proj must run after the softmax CB,
        // which commits between the two.
        var tailCB: MTLCommandBuffer?
        if !(isLinear || cfg.layerIsMLA(L) || cfg.attnOutputGate) {
            guard let split = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            tailCB = split
        }
        // GDN and gated layers run input norm → attention → tail on one
        // serial encoder: on the M1 an encoder boundary costs more span than
        // the small dispatches around it. KDA and MLA keep their CB-internal
        // encoders, so their span cannot share one.
        var layerEncoder: MTLComputeCommandEncoder?
        if (isLinear && !cfg.linearAttentionPerChannelDecay)
            || (!isLinear && !cfg.layerIsMLA(L) && cfg.attnOutputGate) {
            guard let enc = attnCB.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            layerEncoder = enc
        }
        if let layerEncoder {
            rms.encodeBF16W(encoder: layerEncoder,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
        } else {
            try rms.encodeBF16W(commandBuffer: attnCB,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
        }
        var softmaxCB: MTLCommandBuffer?
        try encodeDecodeAttention(attnCB: attnCB, tailCB: tailCB ?? attnCB,
                                  softmaxCB: &softmaxCB,
                                  layerEncoder: layerEncoder,
                                  layer: L, position: position,
                                  isLinear: isLinear, rmsEps: eps)
        routerReadbackTag = RouterHostReadback.nextTag(after: routerReadbackTag)
        let readbackTag = routerReadbackTag
        try encodeDecodeTailStage(
            tailCB: tailCB ?? attnCB, layerEncoder: layerEncoder,
            layer: L, routerW: routerW,
            nextRouterW: nextRouterW, postAttn: postAttn,
            perExpertScale: perExpertScale,
            residency: (table: residencyResources.table, readbackTag: readbackTag),
            speculative: specDispatchArguments,
            d: D, eps: eps)
        layerEncoder?.endEncoding()
        let overlapCompletionClock = runnerStatsEnabled ? CommandCompletionClock() : nil
        let specCB = try encodeSpeculativeRouted(
            layer: L,
            residency: residencyResources,
            arguments: specDispatchArguments,
            completionClock: overlapCompletionClock)
        return HeldLayerCommands(
            layer: L, attnCB: attnCB, softmaxCB: softmaxCB, tailCB: tailCB,
            specCB: specCB,
            overlapCompletionClock: overlapCompletionClock,
            readbackTag: readbackTag)
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        // Decode must not share RAM with an idle ANE context (Track A):
        // prompts that end exactly on a chunk boundary reach here with the
        // last model still resident. No-op when ANE prefill is off or empty.
        anePrefill?.releaseModels()
        updateGPUCaptureWindow()
        try kv?.reserve(tokens: position + 1)
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let eps: Float = cfg.rmsNormEps
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        var pendingRoutedCommand: PendingRoutedCommand?

        /// Drain a routed layer's command buffers, surfacing any `.error`
        /// (R1/R2): the routed-CB failure must fail the generation rather than
        /// print-and-continue into silently corrupt output. The per-layer call
        /// (waitIfNeeded: false) runs right after the next layer's tailCB
        /// wait, so the routed CBs have completed on the GPU and their spans
        /// are valid — recording them here (not only in the waitIfNeeded
        /// drain) makes SHRIKE_KERNEL_STATS cover every layer instead of just
        /// the final layer of each token.

        // Embed lookup + sqrt(H) fused.
        let emb = try model.embedding()
        let embedCB = try runSync { cb in
            if let affineEmbed {
                try affineEmbed.encode(commandBuffer: cb,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: UInt32(bitPattern: token),
                             d: D, outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            } else {
                try embedInt4.encode(commandBuffer: cb,
                             table:  emb.buffer, tableOffset:  Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden,
                             tokenId: UInt32(bitPattern: token),
                             d: D,
                             outScale: embedOutScale,
                             vocab: UInt32(cfg.vocabSize))
            }
        }
        guard embedCB != nil else {
            throw ModelError.residentBufferWrapFailed
        }
        if let embedCB { recordKernelGPU(role: "embed", embedCB) }

        // Records a previous token left behind when it threw belong to that token.
        deferredGPURecords.removeAll()
        var heldNext: HeldLayerCommands?
        for L in 0..<cfg.numLayers {
            let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let isLinear = cfg.layerIsLinear(L)
            let isDense = L < cfg.numLeadingDenseLayers

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // GDN/MLA/gated layers run the whole stage — input norm through
            // router — in one CB (role "attn_layer"). gpt-oss and plain
            // layers keep the attn/softmax/tail CB split, whose commit order
            // sequences o_proj after the softmax. Same queue either way, one
            // wait on the last CB; only the router readback forces the
            // barrier.
            if isDense {
                let inNorm = try model.inputNorm(layer: L)
                let postAttn = try model.postAttnNorm(layer: L)
                guard let attnCB = ctx.queue.makeCommandBuffer(),
                      let tailCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try rms.encodeBF16W(commandBuffer: attnCB,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
                var softmaxCB: MTLCommandBuffer?
                try encodeDecodeAttention(attnCB: attnCB, tailCB: tailCB,
                                          softmaxCB: &softmaxCB,
                                          layerEncoder: nil,
                                          layer: L, position: position,
                                          isLinear: isLinear, rmsEps: eps)
                try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                               hidden: hidden,
                                               delta: oOut,
                                               count: cfg.hiddenSize)
                try rms.encodeBF16W(commandBuffer: tailCB,
                                x: hidden,
                                weight: postAttn.buffer,
                                weightOffset: Int(postAttn.offset),
                                out: routedX,
                                d: D, eps: eps)
                // Leading dense-MLP layer (Kimi layer 0): no router, no
                // routed experts — the shared-expert kernels run the layer's
                // own SwiGLU and the residual folds here.
                let dense = sharedExpertProjections[L]
                try shared.encode(commandBuffer: tailCB,
                                  x: routedX,
                                  gate: dense.gate,
                                  up: dense.up,
                                  down: dense.down,
                                  y: h1Buf,
                                  scratchGate: denseScratchGate,
                                  scratchUp: denseScratchUp,
                                  scratchAct: denseScratchAct)
                try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                               hidden: hidden,
                                               delta: h1Buf,
                                               count: cfg.hiddenSize)
                attnCB.commit()
                if let attentionCB = softmaxCB {
                    attentionCB.commit()
                }
                tailCB.commit()
                try waitForRouterCompletion(tailCB)
                recordKernelGPU(role: "attn_norm_qkv", attnCB)
                if let attentionCB = softmaxCB {
                    recordKernelGPU(role: "attn_softmax", attentionCB)
                }
                recordKernelGPU(role: "attn_tail_router", tailCB)
                totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start
                totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
                continue
            }
            let cmds: HeldLayerCommands
            if let held = heldNext, held.layer == L {
                cmds = held
                heldNext = nil
            } else {
                heldNext = nil
                cmds = try encodeLayerCommands(layer: L, position: position)
            }
            commitHeldLayerCommands(cmds)
            if L + 1 < cfg.numLayers,
               L + 1 >= cfg.numLeadingDenseLayers {
                heldNext = try encodeLayerCommands(layer: L + 1, position: position)
            }
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let wordWake = cmds.readbackTag != 0
            if wordWake {
                try waitForRouterReadback(cmds)
            } else {
                try waitForRouterCompletion(cmds.routerCB)
            }
            let woke = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            recordRouterWake(cmds.routerCB, wokeAt: woke, deferred: wordWake)
            for (role, cb) in layerKernelRecords(cmds, layer: L) {
                recordKernelGPU(role: role, cb, deferred: wordWake)
            }
            let waitNanos = woke - tWait
            totalWaitNanos &+= waitNanos
            var prevRoutedUs: Double = 0
            if let pending = pendingRoutedCommand {
                if pending.cb.gpuEndTime > 0 {
                    prevRoutedUs = (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1_000_000
                }
                try finishPendingRoutedCommand(pending, waitIfNeeded: false,
                                               deferTimings: wordWake)
                pendingRoutedCommand = nil
            }
            if wordWake { try drainDeferredGPURecords(waitIfNeeded: false) }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos
            let hostReadback = try decodeRouterHostReadback(for: cmds)
            let predictedNextLayer: [Int]
            if L + 1 < cfg.numLayers {
                if let hostReadback {
                    predictedNextLayer = hostReadback.predictedIDs.map {
                        min(Int($0), cfg.numExperts - 1)
                    }
                } else {
                    let ptr = prefetchPredictionIndices.contents().bindMemory(
                        to: UInt32.self, capacity: cfg.topKExperts)
                    predictedNextLayer = (0..<cfg.topKExperts).map {
                        min(Int(ptr[$0]), cfg.numExperts - 1)
                    }
                }
            } else {
                predictedNextLayer = []
            }

            // CPU readback to fetch routed-expert blobs from disk. The expert
            // id list is reused host scratch (R16); the runner is single-flight
            // per generation, so it never aliases concurrent decode work.
            try await encodeDecodeRoutedMoE(
                layer: L, position: position,
                attnCB: cmds.attnCB, tailCB: cmds.routerCB,
                specCB: cmds.specCB,
                overlapCompletionClock: cmds.overlapCompletionClock,
                pending: &pendingRoutedCommand,
                bodyStart: tBodyStart, cb1Start: tCb1Start,
                waitMark: tWait, waitNanos: waitNanos,
                previousRoutedMicros: prevRoutedUs,
                hostReadback: hostReadback,
                predictedNextLayer: predictedNextLayer)
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }
        try drainDeferredGPURecords(waitIfNeeded: true)

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = try model.finalNorm()
        let lm    = try model.lmHead()
        let gFinalNorm: (MTLCommandBuffer) throws -> Void = { cb in
            try self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.encodePrimaryGEMV(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) throws -> Void = { cb in
            try self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                if let headCB = try runSync(gFusionHead) {
                    recordKernelGPU(role: "head_fused", headCB)
                }
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                guard let headCB = try runSync({ cb in
                    try gFinalNorm(cb)
                    try gLmHead(cb)
                }) else {
                    throw ModelError.residentBufferWrapFailed
                }
                recordKernelGPU(role: "head_logits", headCB)
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    private func encodeLinearAttentionDecode(encoder: MTLComputeCommandEncoder,
                                             layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // Safe to share one encoder: a serial compute encoder guarantees each
        // dispatch sees the previous dispatch's writes. The encoder is the
        // caller's layer encoder — Stage B runs the whole layer span on it.
        if model.attentionWeightBits == 4 {
            gdn.encodeInputProjections(encoder: encoder,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)
        } else {
            encodePrimaryGEMV(encoder: encoder, projection: qkvW,
                              x: normed, y: gdnQKVRaw,
                              m: UInt32(la.qkvDim), n: D)
            encodePrimaryGEMV(encoder: encoder, projection: zW,
                              x: normed, y: gdnZ,
                              m: UInt32(la.valueDim), n: D)
            encodePrimaryGEMV(encoder: encoder, projection: aW,
                              x: normed, y: gdnA,
                              m: UInt32(la.numVHeads), n: D)
            encodePrimaryGEMV(encoder: encoder, projection: bW,
                              x: normed, y: gdnB,
                              m: UInt32(la.numVHeads), n: D)
        }

        gdn.encodeConvDecode(encoder: encoder,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(encoder: encoder, convOut: gdnConvOut)
        gdn.encodeDeltaStepDecode(encoder: encoder,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        gdn.encodeGatedNorm(encoder: encoder,
                            y: gdnY,
                            z: gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: gdnOut)
        encodePrimaryGEMV(encoder: encoder,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Kimi MLA, one decode step: q_proj GEMV, per-head absorbed embed
    /// ([W_UKᵀ·q_nope | q_pe] via `mla_embed_q`), kv_a projection straight
    /// into the fused cache row, latent RMSNorm in place (k_pe untouched),
    /// split-KV MQA attention over the rows with V as the latent prefix,
    /// per-head unembed (W_UV), o_proj into `oOut`.
    private func encodeMLAAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int,
                                          position: Int, seqLen: UInt32) throws {
        guard let mla, let kv, let mlaQRaw, let mlaQ, let mlaAttnOut,
              let mlaUnembedOut, let mlaCfg = cfg.mla else {
            throw ModelError.internalInconsistency(
                detail: "MLA layer \(L) without MLA kernels (arch mask misconfiguration)")
        }
        let D = UInt32(cfg.hiddenSize)
        let H = cfg.numHeads
        let qkDim = mlaCfg.latentDim + mlaCfg.qkRopeDim
        let qProjW = try model.qProj(layer: L)
        let kvAW = try model.kimiKVAProj(layer: L)
        let kvANorm = try model.kimiKVALayernorm(layer: L)
        let embedQ = try model.kimiEmbedQ(layer: L)
        let unembedOut = try model.kimiUnembedOut(layer: L)
        let outW = try model.oProj(layer: L)

        try encodePrimaryGEMV(commandBuffer: cb, projection: qProjW,
                          x: normed, y: mlaQRaw,
                          m: UInt32(H * (mlaCfg.qkNopeDim + mlaCfg.qkRopeDim)),
                          n: D)
        let slot = kv.kSlot(layer: L, position: position)
        try encodePrimaryGEMV(commandBuffer: cb, projection: kvAW,
                          x: normed, y: slot.buffer, yOffset: slot.offset,
                          m: UInt32(qkDim), n: D)
        try rms.encodeBF16WRows(commandBuffer: cb,
                            x: slot.buffer, xOffset: slot.offset,
                            weight: kvANorm.buffer,
                            weightOffset: Int(kvANorm.offset),
                            out: slot.buffer, outOffset: slot.offset,
                            d: UInt32(mlaCfg.latentDim), rows: 1,
                            rowStrideElements: UInt32(qkDim),
                            eps: cfg.rmsNormEps)
        try mla.encodeEmbedQ(commandBuffer: cb, embedQ: embedQ,
                         qRaw: mlaQRaw, y: mlaQ, tokens: 1)
        let kvView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
        try attention.encodeMLA(commandBuffer: cb,
                            q: mlaQ,
                            kv: kvView.buffer, kvOffset: kvView.offset,
                            out: mlaAttnOut,
                            qkDim: UInt32(qkDim),
                            vDim: UInt32(mlaCfg.latentDim),
                            numQHeads: UInt32(H),
                            seqLen: seqLen,
                            scale: Float(cfg.attentionScale))
        try mla.encodeUnembed(commandBuffer: cb, unembedOut: unembedOut,
                          attn: mlaAttnOut, y: mlaUnembedOut, tokens: 1)
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: mlaUnembedOut, y: oOut,
                    m: D, n: UInt32(H * mlaCfg.valueHeadDim))
    }

    /// Kimi KDA, one decode step: fused in_proj_qkv and in_proj_b GEMVs, the
    /// low-rank decay chain (f_a → f_b into the per-channel `a` buffer) and
    /// output-gate chain (g_a → g_b into the z slot, staged through the same
    /// low-rank scratch — separate encoders, so hazard tracking serializes the
    /// reuse), then conv → qk norm → per-channel delta → sigmoid-gated norm →
    /// o_proj.
    private func encodeKDADecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut,
              let gdnLowRank else {
            throw ModelError.internalInconsistency(
                detail: "KDA layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let low = UInt32(la.keyHeadDim)
        let qkvW = try model.linearInProjQKV(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let fA = try model.kimiFAProj(layer: L)
        let fB = try model.kimiFBProj(layer: L)
        let gA = try model.kimiGAProj(layer: L)
        let gB = try model.kimiGBProj(layer: L)
        let outW = try model.oProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.kimiALog(layer: L)
        let dtBias = try model.kimiDtBias(layer: L)
        let oNormW = try model.kimiONorm(layer: L)

        try encodePrimaryGEMV(commandBuffer: cb, projection: qkvW,
                          x: normed, y: gdnQKVRaw,
                          m: UInt32(la.qkvDim), n: D)
        try encodePrimaryGEMV(commandBuffer: cb, projection: bW,
                          x: normed, y: gdnB,
                          m: UInt32(la.numVHeads), n: D)
        try encodePrimaryGEMV(commandBuffer: cb, projection: fA,
                          x: normed, y: gdnLowRank, m: low, n: D)
        try encodePrimaryGEMV(commandBuffer: cb, projection: fB,
                          x: gdnLowRank, y: gdnA,
                          m: UInt32(la.numVHeads * la.keyHeadDim), n: low)
        try encodePrimaryGEMV(commandBuffer: cb, projection: gA,
                          x: normed, y: gdnLowRank, m: low, n: D)
        try encodePrimaryGEMV(commandBuffer: cb, projection: gB,
                          x: gdnLowRank, y: gdnZ,
                          m: UInt32(la.valueDim), n: low)

        try gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        try gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        try gdn.encodeGatedNorm(commandBuffer: cb,
                            y: gdnY,
                            z: gdnZ,
                            weight: oNormW.buffer,
                            weightOffset: Int(oNormW.offset),
                            out: gdnOut)
        try encodePrimaryGEMV(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    /// Tail stage of one routed decode layer: residual fold, post-attention
    /// norm, router (+ next-layer probe), and residency classification, in
    /// one serial encoder.
    private func encodeDecodeTailStage(
        tailCB: MTLCommandBuffer,
        layerEncoder: MTLComputeCommandEncoder? = nil,
        layer L: Int,
        routerW: TensorView,
        nextRouterW: TensorView?,
        postAttn: TensorView,
        perExpertScale: (buffer: any MTLBuffer, offset: Int),
        residency: (table: any MTLBuffer, readbackTag: UInt32),
        speculative: MoE.SpeculativeDispatchArguments,
        d D: UInt32,
        eps: Float
    ) throws {
        let ownsEncoder = layerEncoder == nil
        guard let tailEncoder = layerEncoder ?? tailCB.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        rms.encodeResidualAddBF16W(encoder: tailEncoder,
                                   hidden: hidden,
                                   delta: oOut,
                                   weight: postAttn.buffer,
                                   weightOffset: Int(postAttn.offset),
                                   out: routedX,
                                   d: D, eps: eps)
        if let nextRouterW {
            let probeLayer = L + 1
            moe.encodeRouterPair(
                encoder: tailEncoder,
                first: MoE.RouterOperands(
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    effectiveScale: effectiveScaleBuffers[L],
                    logitBias: routerLogitBias[L].buffer, logitBiasOffset: routerLogitBias[L].offset,
                    outIndices: outIndices, outWeights: outWeights),
                second: MoE.RouterOperands(
                    weights: nextRouterW.buffer, weightsOffset: Int(nextRouterW.offset),
                    scales: nextRouterW.buffer, scalesOffset: Int(nextRouterW.scaleOffset),
                    biases: nextRouterW.buffer, biasesOffset: Int(nextRouterW.biasOffset),
                    effectiveScale: effectiveScaleBuffers[probeLayer],
                    logitBias: routerLogitBias[probeLayer].buffer,
                    logitBiasOffset: routerLogitBias[probeLayer].offset,
                    outIndices: prefetchPredictionIndices, outWeights: prefetchPredictionWeights),
                hidden: routedX,
                perExpertScale: perExpertScale.buffer, perExpertScaleOffset: perExpertScale.offset,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
        } else {
            moe.encodeRouter(encoder: tailEncoder,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                logitBias: routerLogitBias[L].buffer,
                logitBiasOffset: routerLogitBias[L].offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
        }
        moe.encodeResidencyClassification(
            encoder: tailEncoder,
            topKIndices: outIndices,
            residencyTable: residency.table,
            hitCount: residencyHitCount,
            hitPositions: residencyHitPositions,
            missCount: residencyMissCount,
            missPositions: residencyMissPositions,
            missExperts: residencyMissExperts,
            resolvedSlots: residencyResolvedSlots,
            resolvedGenerations: residencyResolvedGenerations,
            topK: UInt32(cfg.topKExperts),
            numExperts: UInt32(cfg.numExperts),
            speculative: speculative,
            hostReadback: MoE.RouterHostReadbackArguments(
                buffer: routerHostReadback, tag: residency.readbackTag,
                topKWeights: outWeights,
                predictedIndices: prefetchPredictionIndices))
        if ownsEncoder { tailEncoder.endEncoding() }
    }

    private func encodeGatedFullQKVProjection(
        encoder: MTLComputeCommandEncoder,
        layer: Int,
        qOutput: MTLBuffer,
        kOutput: (buffer: MTLBuffer, offset: Int),
        vOutput: (buffer: MTLBuffer, offset: Int),
        qDimension: UInt32,
        kvDimension: UInt32
    ) throws {
        let q = try model.qProj(layer: layer)
        let k = try model.kProj(layer: layer)
        let v = try model.vProj(layer: layer)
        let hiddenDimension = UInt32(cfg.hiddenSize)
        if model.attentionWeightBits == 4 {
            fusedQKVGEMV.encode(encoder: encoder,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qOutput,
                            kOut: kOutput.buffer, kOutOffset: kOutput.offset,
                            vOut: vOutput.buffer, vOutOffset: vOutput.offset,
                            qRows: 2 * qDimension,
                            kvRows: kvDimension,
                            n: hiddenDimension)
        } else {
            encodePrimaryGEMV(encoder: encoder, projection: q,
                              x: normed, y: qOutput,
                              m: 2 * qDimension, n: hiddenDimension)
            encodePrimaryGEMV(encoder: encoder, projection: k,
                              x: normed, y: kOutput.buffer,
                              yOffset: kOutput.offset,
                              m: kvDimension, n: hiddenDimension)
            encodePrimaryGEMV(encoder: encoder, projection: v,
                              x: normed, y: vOutput.buffer,
                              yOffset: vOutput.offset,
                              m: kvDimension, n: hiddenDimension)
        }
    }

    private func encodeGatedFullAttentionDecode(encoder: MTLComputeCommandEncoder,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, rope != nil, let qPackedScratch, let attnGateScratch else {
            throw ModelError.internalInconsistency(
                detail: "attn_output_gate layer \(L) without gate kernels (arch mask misconfiguration)")
        }
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "full attention requires a KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = cfg.rmsNormEps
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let quantizedKV = kv.precision.isQuantized
        let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
        let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        // Safe to share one encoder: a serial compute encoder guarantees each
        // dispatch sees the previous dispatch's writes. The encoder is the
        // caller's layer encoder — Stage B runs the whole layer span on it.
        try encodeGatedFullQKVProjection(
            encoder: encoder, layer: L, qOutput: qPackedScratch,
            kOutput: kWrite, vOutput: vWrite,
            qDimension: qDim, kvDimension: kvDim)
        elementwise.encodeSplitQGate(encoder: encoder,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        try fusedQKVEpilogue.encode(encoder: encoder,
                                    q: qScratch,
                                    k: kWrite.buffer,
                                    kOffset: kWrite.offset,
                                    v: vWrite.buffer,
                                    vOffset: vWrite.offset,
                                    qWeight: qNormW.buffer,
                                    qWeightOffset: Int(qNormW.offset),
                                    kWeight: kNormW.buffer,
                                    kWeightOffset: Int(kNormW.offset),
                                    headDim: UInt32(headDim),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKV),
                                    position: UInt32(position),
                                    theta: Float(cfg.fullRopeTheta),
                                    rotatedPairs: rotaryDim / 2,
                                    eps: eps,
                                    subdimRope: true,
                                    normalizeV: false)
        if quantizedKV {
            try encodeQuantizedKV(encoder: encoder, kv: kv, layer: L,
                                  position: position, keySource: kStage,
                                  valueSource: vStage, elementCount: Int(kvDim))
        }
        let keyView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
        let valueView = kv.valueView(layer: L, validTokenCount: Int(seqLen))
        try attention.encodeFull(encoder: encoder,
                             q: qScratch,
                             k: keyView.buffer, kOffset: keyView.offset,
                             v: valueView.buffer, vOffset: valueView.offset,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale),
                             kvFormat: keyView)
        elementwise.encodeSigmoidGateMul(encoder: encoder,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        encodePrimaryGEMV(encoder: encoder,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    /// gpt-oss q/k/v projections with their additive BF16 biases and the
    /// arch-YaRN NeoX rotation on q and k — no QK norms, V unrotated.
    private func encodeGptOssQKVProjection(
        _ cb: MTLCommandBuffer,
        layer L: Int,
        position: Int,
        kWrite: (buffer: MTLBuffer, offset: Int),
        vWrite: (buffer: MTLBuffer, offset: Int),
        qDimension qDim: UInt32,
        kvDimension kvDim: UInt32,
        elementwise: Elementwise,
        rope: RoPE
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim
        let qBias = try model.qProjBias(layer: L)
        let kBias = try model.kProjBias(layer: L)
        let vBias = try model.vProjBias(layer: L)
        try encodePrimaryGEMV(commandBuffer: cb,
                              projection: try model.qProj(layer: L),
                              x: normed, y: qScratch, m: qDim, n: D)
        try encodePrimaryGEMV(commandBuffer: cb,
                              projection: try model.kProj(layer: L),
                              x: normed,
                              y: kWrite.buffer, yOffset: kWrite.offset,
                              m: kvDim, n: D)
        try encodePrimaryGEMV(commandBuffer: cb,
                              projection: try model.vProj(layer: L),
                              x: normed,
                              y: vWrite.buffer, yOffset: vWrite.offset,
                              m: kvDim, n: D)
        try elementwise.encodeBiasAdd(commandBuffer: cb,
                                      x: qScratch,
                                      bias: qBias.buffer,
                                      biasOffset: Int(qBias.offset),
                                      rowElems: Int(qDim))
        try elementwise.encodeBiasAdd(commandBuffer: cb,
                                      x: kWrite.buffer, xOffset: kWrite.offset,
                                      bias: kBias.buffer,
                                      biasOffset: Int(kBias.offset),
                                      rowElems: Int(kvDim))
        try elementwise.encodeBiasAdd(commandBuffer: cb,
                                      x: vWrite.buffer, xOffset: vWrite.offset,
                                      bias: vBias.buffer,
                                      biasOffset: Int(vBias.offset),
                                      rowElems: Int(kvDim))
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                                  data: qScratch,
                                  position: UInt32(position),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(cfg.numHeads),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
        try rope.encodeNeoxSubdim(commandBuffer: cb,
                                  data: kWrite.buffer,
                                  dataOffset: kWrite.offset,
                                  position: UInt32(position),
                                  headDim: UInt32(headDim),
                                  numHeads: UInt32(cfg.numFullKVHeads),
                                  rotaryDim: rotaryDim,
                                  theta: Float(cfg.fullRopeTheta))
    }

    /// gpt-oss decode attention: biased QKV + YaRN via
    /// `encodeGptOssQKVProjection`, sinks in the softmax, full or
    /// sliding-window dispatch by the layer mask, biased o_proj. Mirrors the
    /// non-gated branch's command-buffer split: QKV + RoPE on `attnCB`, the
    /// softmax on its own buffer via `softmaxCB`, o_proj on `tailCB`.
    private func encodeGptOssAttentionDecode(attnCB: MTLCommandBuffer,
                                             tailCB: MTLCommandBuffer,
                                             softmaxCB: inout MTLCommandBuffer?,
                                             layer L: Int,
                                             position: Int,
                                             seqLen: UInt32) throws {
        guard let elementwise, let rope else {
            throw ModelError.internalInconsistency(
                detail: "gpt-oss attention layer \(L) without bias/rope kernels (arch misconfiguration)")
        }
        guard let kv else {
            throw ModelError.internalInconsistency(
                detail: "attention requires a KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let quantizedKV = kv.precision.isQuantized
        let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
        let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
        let oBias = try model.oProjBias(layer: L)
        let sinks = try model.attentionSinks(layer: L)

        try encodeGptOssQKVProjection(attnCB, layer: L, position: position,
                                      kWrite: kWrite, vWrite: vWrite,
                                      qDimension: qDim, kvDimension: kvDim,
                                      elementwise: elementwise, rope: rope)
        if quantizedKV {
            try encodeQuantizedKV(commandBuffer: attnCB, kv: kv, layer: L,
                                  position: position, keySource: kStage,
                                  valueSource: vStage, elementCount: Int(kvDim))
        }
        let keyView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
        let valueView = kv.valueView(layer: L, validTokenCount: Int(seqLen))
        guard let attentionCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        softmaxCB = attentionCB
        if cfg.layerIsFull(L) {
            try attention.encodeFull(commandBuffer: attentionCB,
                                     q: qScratch,
                                     k: keyView.buffer, kOffset: keyView.offset,
                                     v: valueView.buffer, vOffset: valueView.offset,
                                     out: attnOut,
                                     headDim: UInt32(headDim),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(numKV),
                                     seqLen: seqLen,
                                     scale: Float(cfg.attentionScale),
                                     sinks: sinks.buffer,
                                     sinksOffset: Int(sinks.offset),
                                     kvFormat: keyView)
        } else {
            let ringCapacity = kv.ringCapacity(layer: L)
            let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                ? UInt32(ringCapacity)
                : 0
            try attention.encodeSWA(commandBuffer: attentionCB,
                                    q: qScratch,
                                    k: kSlot.buffer, kOffset: 0,
                                    v: vSlot.buffer, vOffset: 0,
                                    out: attnOut,
                                    headDim: UInt32(headDim),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKV),
                                    seqLen: seqLen,
                                    window: UInt32(cfg.slidingWindow),
                                    scale: Float(cfg.attentionScale),
                                    ringCapacity: activeRingCapacity,
                                    sinks: sinks.buffer,
                                    sinksOffset: Int(sinks.offset),
                                    kvFormat: keyView)
        }
        try encodePrimaryGEMV(commandBuffer: tailCB,
                              projection: try model.oProj(layer: L),
                              x: attnOut, y: oOut, m: D, n: qDim)
        try elementwise.encodeBiasAdd(commandBuffer: tailCB,
                                      x: oOut,
                                      bias: oBias.buffer,
                                      biasOffset: Int(oBias.offset),
                                      rowElems: Int(D))
    }

    private func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   projection p: TensorView,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        try encodePrimaryGEMV(commandBuffer: cb,
                          weights: p.buffer, weightsOffset: Int(p.offset),
                          scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                          biases: p.buffer, biasesOffset: Int(p.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
    }

    private func encodePrimaryGEMV(encoder: MTLComputeCommandEncoder,
                                   projection p: TensorView,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) {
        encodePrimaryGEMV(encoder: encoder,
                          weights: p.buffer, weightsOffset: Int(p.offset),
                          scales: p.buffer, scalesOffset: Int(p.scaleOffset),
                          biases: p.buffer, biasesOffset: Int(p.biasOffset),
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
    }

    private func encodePrimaryGEMV(commandBuffer cb: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int,
                                   scales: MTLBuffer, scalesOffset: Int,
                                   biases: MTLBuffer, biasesOffset: Int,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) throws {
        if let affine {
            try affine.encode(commandBuffer: cb,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
        } else {
            try int4.encode(commandBuffer: cb,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        }
    }

    private func encodePrimaryGEMV(encoder: MTLComputeCommandEncoder,
                                   weights: MTLBuffer, weightsOffset: Int,
                                   scales: MTLBuffer, scalesOffset: Int,
                                   biases: MTLBuffer, biasesOffset: Int,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   y: MTLBuffer, yOffset: Int = 0,
                                   m: UInt32, n: UInt32) {
        if let affine {
            affine.encode(encoder: encoder,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
        } else {
            int4.encode(encoder: encoder,
                        weights: weights, weightsOffset: weightsOffset,
                        scales: scales, scalesOffset: scalesOffset,
                        biases: biases, biasesOffset: biasesOffset,
                        x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                        m: m, n: n)
        }
    }

    private func runSync(_ body: (MTLCommandBuffer) throws -> Void) throws -> MTLCommandBuffer? {
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        try body(cb)
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
        return cb
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
    }

    /// Builds, gates, and commits the miss-fixup routed CB: the I/O
    /// event wait, phase 1
    /// (full or hit-split subset), the phase-2 reduce, the residual tail,
    /// and the S3b layer-done signal wiring.
    private func buildAndCommitMissFixupCommand(
        eventLoad: RoutedExpertLoadOperation?,
        phase1HitCB: MTLCommandBuffer?,
        phase1HitSplitArgBuf: MTLBuffer?,
        phase1MissSlots: [UInt32],
        routedBufs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        topK: UInt32,
        d D: UInt32,
        f FmoE: UInt32
    ) throws -> (cb: MTLCommandBuffer, commitNanos: UInt64) {
        // The phase-2 reduce already folded the shared branch (h1Buf
        // as its residual); the tail is a plain residual add.
        let gTail: (MTLCommandBuffer) throws -> Void = { [self] cb in
            try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                           hidden: hidden,
                                           delta: h2Buf,
                                           count: cfg.hiddenSize)
        }
        guard let routedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let ioToken = eventLoad?.storage.completionToken
        let ioStatus = ioToken.map { ($0.status, $0.statusOffset) }
        // A host-spin late commit (spin on the I/O timeline, commit the CB
        // wait-free) measured null on the M1 — the parked-CB wake is as fast
        // as a fresh-commit schedule (v10 T3, 2026-08-31); keep the encoded
        // wait.
        if let token = ioToken {
            routedCB.encodeWaitForEvent(token.event, value: token.value)
        }
        let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
            ? phase1HitSplitArgBuf
            : nil
        let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
            routedBlobs: routedBufs,
            topK: topK,
            routedBufferOffsets: decodeRoutedOffsetsScratch)
        let missesOnly = phase1HitCB != nil && !phase1MissSlots.isEmpty
        if missesOnly {
            totalHitFixupLayers &+= 1
            writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
            try moe.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count),
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        } else {
            try moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus?.0,
                ioStatusOffset: ioStatus?.1 ?? 0)
        }
        try moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                               routedArgBuffer: argBuf,
                                               routedBlobs: routedBufs,
                                               routedOffsets: routedOffsets,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: h1Buf,
                                               y: h2Buf,
                                               d: D,
                                               f: FmoE,
                                               topK: topK,
                                               ioStatus: ioStatus?.0,
                                               ioStatusOffset: ioStatus?.1 ?? 0)
        try gTail(routedCB)
        let commitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        routedCB.commit()
        return (routedCB, commitNanos)
    }

    /// Polls instead of parking the thread, trading a busy core for the
    /// scheduler-wake latency on the per-layer router wait (v10 T5's spin).
    /// Falls back to blocking after ~1s so a stalled CB cannot wedge a core.
    private nonisolated func waitForRouterCompletion(_ cb: MTLCommandBuffer) throws {
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 1_000_000_000
        while clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline {
            let status = cb.status
            if status == .completed { return }
            if status == .error {
                throw ModelError.commandBufferFailed(
                    detail: String(describing: cb.error))
            }
        }
        try waitForCompletion(cb)
    }

    /// The word wake (v14 lever B): poll the classifier's tagged copy, which lands
    /// before the driver marks the command complete; after ~1s fall back to
    /// the status wait so a stalled or failed command surfaces there.
    private func waitForRouterReadback(_ cmds: HeldLayerCommands) throws {
        let words = routerHostReadback.contents().bindMemory(
            to: UInt32.self,
            capacity: RouterHostReadback.wordCount(topK: cfg.topKExperts))
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 1_000_000_000
        var spins = 0
        while clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline {
            if RouterHostReadback.isComplete(words: words, topK: cfg.topKExperts,
                                             tag: cmds.readbackTag) {
                return
            }
            spins &+= 1
            if spins % 256 == 0, cmds.routerCB.status == .error {
                throw ModelError.commandBufferFailed(
                    detail: String(describing: cmds.routerCB.error))
            }
        }
        totalRouterWakeFallbacks &+= 1
        try waitForCompletion(cmds.routerCB)
    }

    private func decodeRouterHostReadback(for cmds: HeldLayerCommands) throws
        -> RouterHostReadback? {
        guard cmds.readbackTag != 0 else { return nil }
        let words = routerHostReadback.contents().bindMemory(
            to: UInt32.self,
            capacity: RouterHostReadback.wordCount(topK: cfg.topKExperts))
        guard let readback = RouterHostReadback.decode(
            words: words, topK: cfg.topKExperts, tag: cmds.readbackTag)
        else {
            throw ModelError.internalInconsistency(
                detail: "router host readback carries a stale tag after the router's wake")
        }
        return readback
    }

    private func layerKernelRecords(_ cmds: HeldLayerCommands, layer L: Int)
        -> [(role: String, cb: MTLCommandBuffer)] {
        guard let tailCB = cmds.tailCB else {
            return [(cfg.layerIsLinear(L) ? "attn_layer_linear" : "attn_layer_kv", cmds.attnCB)]
        }
        var records = [("attn_norm_qkv", cmds.attnCB)]
        if let attentionCB = cmds.softmaxCB { records.append(("attn_softmax", attentionCB)) }
        records.append(("attn_tail_router", tailCB))
        return records
    }

    private func recordKernelGPU(role: String, _ cb: MTLCommandBuffer, deferred: Bool) {
        guard deferred else { return recordKernelGPU(role: role, cb) }
        guard kernelGPUTimingsEnabled else { return }
        deferredGPURecords.append(.kernel(role: role, cb: cb))
    }

    private func recordRouterWake(_ cb: MTLCommandBuffer, wokeAt woke: UInt64, deferred: Bool) {
        if deferred {
            deferredGPURecords.append(.routerWake(cb: cb, wokeAt: woke))
            return
        }
        guard cb.gpuEndTime > 0 else { return }
        let gpuEnd = UInt64(cb.gpuEndTime * 1_000_000_000)
        totalRouterWakeNanos &+= woke > gpuEnd ? woke - gpuEnd : 0
    }

    private enum DeferredGPURecord {
        case kernel(role: String, cb: MTLCommandBuffer)
        case routerWake(cb: MTLCommandBuffer, wokeAt: UInt64)
        case routed(PendingRoutedCommand, ioCompletedNanos: UInt64)
        case prefetchRace(completions: [Int: UInt64], landed: [Int], tailCB: MTLCommandBuffer)

        var commandBuffers: [(label: String, cb: MTLCommandBuffer)] {
            switch self {
            case .kernel(let role, let cb):
                return [(role, cb)]
            case .routerWake(let cb, _):
                return [("router command buffer", cb)]
            case .prefetchRace(_, _, let tailCB):
                return [("attn_tail_router", tailCB)]
            case .routed(let pending, _):
                var buffers = [("routed layer command buffer", pending.cb)]
                if let specCB = pending.specCB {
                    buffers.append(("speculative routed command buffer", specCB))
                }
                if let phase1HitCB = pending.phase1HitCB {
                    buffers.append(("routed phase-1 hit command buffer", phase1HitCB))
                }
                return buffers
            }
        }
    }

    /// Applies the records whose commands the driver has marked complete; a
    /// failed command throws here with its name, since under the word wake the
    /// immediate error checks ran before the mark existed.
    private func drainDeferredGPURecords(waitIfNeeded: Bool) throws {
        guard !deferredGPURecords.isEmpty else { return }
        var kept: [DeferredGPURecord] = []
        defer { deferredGPURecords = kept }
        for record in deferredGPURecords {
            for (label, cb) in record.commandBuffers where cb.status == .error {
                throw ModelError.commandBufferFailed(
                    detail: "\(label): \(String(describing: cb.error))")
            }
            let ready = record.commandBuffers.allSatisfy { $0.cb.status == .completed }
            guard ready || waitIfNeeded else {
                kept.append(record)
                continue
            }
            if !ready {
                for (_, cb) in record.commandBuffers { try waitForCompletion(cb) }
            }
            switch record {
            case .kernel(let role, let cb):
                recordKernelGPU(role: role, cb)
            case .routerWake(let cb, let woke):
                recordRouterWake(cb, wokeAt: woke, deferred: false)
            case .routed(let pending, let ioCompletedNanos):
                try recordRoutedCommandTimings(pending, ioCompletedNanos: ioCompletedNanos)
            case .prefetchRace(let completions, let landed, let tailCB):
                countPrefetchRace(completions: completions, landed: landed, tailCB: tailCB)
            }
        }
        deferredGPURecords = kept
    }


    // MARK: - Chunked prefill helpers

    /// Per-layer tensor views resolved once before the chunk loop.
    private struct LayerPrefillQKVViews {
        let inputNorm: TensorView
        let postAttention: TensorView
        // nil on leading dense-MLP layers (no routed experts).
        let router: TensorView?
        // Softmax-attention layers only (nil on linear-attention layers).
        let q: TensorView?
        let k: TensorView?
        let v: TensorView?
        let o: TensorView?
        let qNorm: TensorView?
        let kNorm: TensorView?
        // Gated-DeltaNet linear-attention layers only. On per-channel-decay
        // architectures (Kimi KDA) linZ/linA stay nil, the low-rank chains
        // fill linFA…linGB, and linOut/linALog/linDtBias/linNorm carry the
        // KDA-named tensors.
        let linQKV: TensorView?
        let linZ: TensorView?
        let linA: TensorView?
        let linB: TensorView?
        let linOut: TensorView?
        let linConv: TensorView?
        let linALog: TensorView?
        let linDtBias: TensorView?
        let linNorm: TensorView?
        let linFA: TensorView?
        let linFB: TensorView?
        let linGA: TensorView?
        let linGB: TensorView?
        // Kimi MLA layers only (q and o ride the standard slots).
        let mlaKVA: TensorView?
        let mlaKVALayernorm: TensorView?
        let mlaEmbedQ: TensorView?
        let mlaUnembedOut: TensorView?
    }

    private func encodeAffineProjection(commandBuffer: MTLCommandBuffer,
                              family: PrefillProjectionFamily,
                              weights: TensorView,
                              x: MTLBuffer,
                              y: MTLBuffer,
                              rows: Int,
                              columns: Int,
                              tokenCount: Int,
                              xStrideElements: Int,
                              yStrideElements: Int) throws {
        if tokenCount >= prefillMatrixMinRows,
           family == .q || family == .kv || family == .o,
           let candidate = prefillMPPAffineInt4 {
            let path = try candidate.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                y: y,
                m: tokenCount,
                n: rows,
                k: columns)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        if PrefillProjectionDispatchPolicy.selectedDispatch(
                for: family,
                chunkTokens: tokenCount,
                minimumRows: prefillMatrixMinRows) == .qmm {
            try prefillQMM.encode(commandBuffer: commandBuffer,
                              weights: weights.buffer,
                              weightsOffset: Int(weights.offset),
                              scales: weights.buffer,
                              scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer,
                              biasesOffset: Int(weights.biasOffset),
                              x: x,
                              y: y,
                              t: tokenCount,
                              n: rows,
                              k: columns)
            return
        }
        for row in 0..<tokenCount {
            try encodePrimaryGEMV(
                commandBuffer: commandBuffer,
                projection: weights,
                x: x,
                xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                y: y,
                yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                m: UInt32(rows),
                n: UInt32(columns))
        }
    }

    private func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                       source: MTLBuffer,
                       destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                       sourceTokenOffset: Int,
                       tokenCount: Int,
                       bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: source,
                  sourceOffset: sourceTokenOffset * bytesPerToken,
                  to: destination.buffer,
                  destinationOffset: destination.offset,
                  size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    private func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                              kv: KVCacheManager,
                              layer: Int,
                              startPosition: Int,
                              tokenCount: Int,
                              keySource: MTLBuffer,
                              valueSource: MTLBuffer,
                              bytesPerToken: Int) throws {
        if kv.precision.isQuantized {
            guard let kvQuantizer else {
                throw ModelError.internalInconsistency(
                    detail: "quantized KV cache has no quantizer")
            }
            let elements = bytesPerToken / MemoryLayout<Float16>.stride
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(layer: layer, start: startPosition,
                                             count: firstSpan),
                tokenCount: firstSpan,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(layer: layer, start: startPosition,
                                               count: firstSpan),
                tokenCount: firstSpan,
                elementCount: elements)
            guard firstSpan < tokenCount else { return }
            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let sourceOffset = firstSpan * bytesPerToken
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: keySource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.keyRangeView(layer: layer, start: secondStart,
                                             count: secondCount),
                tokenCount: secondCount,
                elementCount: elements)
            try kvQuantizer.encode(
                commandBuffer: commandBuffer,
                source: valueSource,
                sourceOffset: sourceOffset,
                sourceTokenStrideElements: elements,
                destination: kv.valueRangeView(layer: layer, start: secondStart,
                                               count: secondCount),
                tokenCount: secondCount,
                elementCount: elements)
            return
        }
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
        let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keyFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
        let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keySecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueSecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
    }

    private func encodeQuantizedKV(commandBuffer: MTLCommandBuffer,
                                   kv: KVCacheManager,
                                   layer: Int,
                                   position: Int,
                                   keySource: MTLBuffer,
                                   valueSource: MTLBuffer,
                                   elementCount: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { encoder.endEncoding() }
        try encodeQuantizedKV(encoder: encoder, kv: kv, layer: layer,
                              position: position, keySource: keySource,
                              valueSource: valueSource, elementCount: elementCount)
    }

    private func encodeQuantizedKV(encoder: MTLComputeCommandEncoder,
                                   kv: KVCacheManager,
                                   layer: Int,
                                   position: Int,
                                   keySource: MTLBuffer,
                                   valueSource: MTLBuffer,
                                   elementCount: Int) throws {
        guard let kvQuantizer else {
            throw ModelError.internalInconsistency(
                detail: "quantized KV cache has no quantizer")
        }
        kvQuantizer.encode(
            encoder: encoder,
            source: keySource,
            sourceTokenStrideElements: elementCount,
            destination: kv.keyRangeView(layer: layer, start: position, count: 1),
            tokenCount: 1,
            elementCount: elementCount)
        kvQuantizer.encode(
            encoder: encoder,
            source: valueSource,
            sourceTokenStrideElements: elementCount,
            destination: kv.valueRangeView(layer: layer, start: position, count: 1),
            tokenCount: 1,
            elementCount: elementCount)
    }

    /// Gated-DeltaNet (linear attention) branch of one chunked-prefill layer.
    ///
    /// The chunked scan (v12 P4) when the runner, the kernels and the scratch
    /// all allow it; the serial kernel for a chunk below `GDN.chunkTokens`
    /// rows, an uncompiled shape, or `SHRIKE_GDN_PREFILL_SCAN=serial`.
    private func encodeGDNDeltaStep(
        cb: MTLCommandBuffer, gdn: GDN, gdnState: GDNStateManager, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        aLog: MTLBuffer, aLogOffset: Int, dtBias: MTLBuffer, dtBiasOffset: Int,
        rows t: Int
    ) throws {
        if gdnPrefillScanChunked, gdn.chunkedScanAvailable,
           let factors = scratch.gdnChunkFactors, t >= GDN.chunkTokens {
            try gdn.encodeDeltaStepPrefillChunked(commandBuffer: cb,
                                                  convOut: scratch.gdnConvOut,
                                                  aProj: scratch.gdnA,
                                                  bProj: scratch.gdnB,
                                                  aLog: aLog, aLogOffset: aLogOffset,
                                                  dtBias: dtBias, dtBiasOffset: dtBiasOffset,
                                                  state: gdnState.stateBuffer(layer: L),
                                                  y: scratch.gdnY,
                                                  rows: t, factors: factors)
        } else {
            try gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                           convOut: scratch.gdnConvOut,
                                           aProj: scratch.gdnA,
                                           bProj: scratch.gdnB,
                                           aLog: aLog, aLogOffset: aLogOffset,
                                           dtBias: dtBias, dtBiasOffset: dtBiasOffset,
                                           state: gdnState.stateBuffer(layer: L),
                                           y: scratch.gdnY,
                                           rows: t)
        }
    }

    /// lint:allow-long one layer's linear-attention pipeline is a single
    /// ordered sequence -- in-projection, causal conv, QK norm, delta step,
    /// gated norm, out-projection -- sharing scratch buffers at every step.
    /// Splitting it would thread a dozen buffers through sub-functions to make
    /// a line count smaller while making the data flow harder to follow.
    private func encodeLinearAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int
    ) throws {
        // Gated-DeltaNet linear attention over the chunk: batched
        // projections, causal conv (+ tail carry), delta-rule
        // recurrence, gated norm, out_proj. No KV writes, no
        // attention, no blit.
        guard let gdn, let gdnState else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) without GDN kernels (arch mask misconfiguration)")
        }
        // `LayerPrefillQKVViews` fills the linear_attn slots exactly
        // when `layerIsLinear(L)`, so these are provably non-nil; the
        // guard turns a future arch/view regression into a thrown
        // error instead of a force-unwrap trap.
        guard let linQKV = views.linQKV,
              let linB = views.linB,
              let linConv = views.linConv,
              let linALog = views.linALog,
              let linDtBias = views.linDtBias,
              let linNorm = views.linNorm,
              let linOut = views.linOut else {
            throw ModelError.internalInconsistency(
                detail: "linear-attention layer \(L) is missing a required linear_attn tensor view")
        }
        let la = cfg.linearAttention
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: linQKV,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: la.qkvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.qkvDim)
        if cfg.linearAttentionPerChannelDecay {
            try encodeKDAPrefillChains(cb: cb, layer: L, views: views,
                                       scratch: scratch, tokenCount: t,
                                       hiddenSize: D)
        } else {
            guard let linZ = views.linZ, let linA = views.linA else {
                throw ModelError.internalInconsistency(
                    detail: "linear-attention layer \(L) is missing a required linear_attn tensor view")
            }
            try encodeAffineProjection(commandBuffer: cb,
                                 family: .kv,
                                 weights: linZ,
                                 x: scratch.normed,
                                 y: scratch.gdnZ,
                                 rows: la.valueDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: la.valueDim)
            try encodeAffineProjection(commandBuffer: cb,
                                 family: .kv,
                                 weights: linA,
                                 x: scratch.normed,
                                 y: scratch.gdnA,
                                 rows: la.numVHeads,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: la.numVHeads)
        }
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linB,
                             x: scratch.normed,
                             y: scratch.gdnB,
                             rows: la.numVHeads,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: la.numVHeads)
        let convW = linConv
        let tail = gdnState.convTailBuffer(layer: L)
        try gdn.encodeConvPrefill(commandBuffer: cb,
                              tail: tail,
                              qkvRows: scratch.q,
                              convWeight: convW.buffer,
                              convWeightOffset: Int(convW.offset),
                              out: scratch.gdnConvOut,
                              rows: t)
        try gdn.encodeConvTailUpdate(commandBuffer: cb,
                                 tail: tail,
                                 qkvRows: scratch.q,
                                 rows: t)
        try gdn.encodeQKNorm(commandBuffer: cb,
                         convOut: scratch.gdnConvOut,
                         rows: t)
        try encodeGDNDeltaStep(cb: cb, gdn: gdn, gdnState: gdnState, layer: L,
                               scratch: scratch,
                               aLog: linALog.buffer, aLogOffset: Int(linALog.offset),
                               dtBias: linDtBias.buffer, dtBiasOffset: Int(linDtBias.offset),
                               rows: t)
        let gatedNormW = linNorm
        try gdn.encodeGatedNorm(commandBuffer: cb,
                            y: scratch.gdnY,
                            z: scratch.gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: scratch.attentionOutput,
                            rows: t)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .o,
                             weights: linOut,
                             x: scratch.attentionOutput,
                             y: scratch.h1,
                             rows: D,
                             columns: la.valueDim,
                             tokenCount: t,
                             xStrideElements: la.valueDim,
                             yStrideElements: D)
    }

    /// Kimi MLA branch of one chunked-prefill layer: batched q_proj, kv_a
    /// projection into `kStage` + latent RMSNorm in place, blit of the fused
    /// rows into the cache, absorbed per-head embed, causal MQA attention
    /// with V as the latent prefix, per-head unembed, o_proj into `h1`.
    private func encodeMLAAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, startPosition: Int
    ) throws {
        guard let mla, let kv, let mlaCfg = cfg.mla,
              let qProjW = views.q, let outW = views.o,
              let kvAW = views.mlaKVA, let kvANorm = views.mlaKVALayernorm,
              let embedQ = views.mlaEmbedQ, let unembedW = views.mlaUnembedOut else {
            throw ModelError.internalInconsistency(
                detail: "MLA layer \(L) is missing a required tensor view")
        }
        let H = cfg.numHeads
        let qkDim = mlaCfg.latentDim + mlaCfg.qkRopeDim
        let qRawDim = H * (mlaCfg.qkNopeDim + mlaCfg.qkRopeDim)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: qProjW,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: qRawDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: qRawDim)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: kvAW,
                             x: scratch.normed,
                             y: scratch.kStage,
                             rows: qkDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: qkDim)
        try rms.encodeBF16WRows(commandBuffer: cb,
                            x: scratch.kStage,
                            weight: kvANorm.buffer,
                            weightOffset: Int(kvANorm.offset),
                            out: scratch.kStage,
                            d: UInt32(mlaCfg.latentDim), rows: t,
                            rowStrideElements: UInt32(qkDim),
                            eps: cfg.rmsNormEps)
        try copyPrefillKV(commandBuffer: cb,
                      source: scratch.kStage,
                      destination: kv.kRange(layer: L, start: startPosition, count: t),
                      sourceTokenOffset: 0,
                      tokenCount: t,
                      bytesPerToken: qkDim * MemoryLayout<Float16>.stride)
        try mla.encodeEmbedQ(commandBuffer: cb, embedQ: embedQ,
                         qRaw: scratch.q, y: scratch.mlaQ, tokens: t)
        let kvView = kv.keyView(layer: L, validTokenCount: startPosition + t)
        let params = PrefillAttentionParams(
            startPosition: UInt32(startPosition),
            queryCount: UInt32(t),
            headDim: UInt32(qkDim),
            numQHeads: UInt32(H),
            numKVHeads: 1,
            kvValidCount: UInt32(startPosition + t),
            slidingWindow: 0,
            kvTokenStrideElements: UInt32(qkDim),
            qTokenStrideElements: UInt32(H * qkDim),
            oTokenStrideElements: UInt32(H * mlaCfg.latentDim),
            scale: Float(cfg.attentionScale))
        try prefillAttention.encodeMLACausal(commandBuffer: cb,
                                         q: scratch.mlaQ,
                                         kv: kvView.buffer, kvOffset: kvView.offset,
                                         out: scratch.attentionOutput,
                                         params: params,
                                         vDim: UInt32(mlaCfg.latentDim))
        try mla.encodeUnembed(commandBuffer: cb, unembedOut: unembedW,
                          attn: scratch.attentionOutput,
                          y: scratch.mlaUnembed, tokens: t)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .o,
                             weights: outW,
                             x: scratch.mlaUnembed,
                             y: scratch.h1,
                             rows: D,
                             columns: H * mlaCfg.valueHeadDim,
                             tokenCount: t,
                             xStrideElements: H * mlaCfg.valueHeadDim,
                             yStrideElements: D)
    }

    /// Kimi KDA prefill chains: f_a → f_b fills the per-channel `a` buffer
    /// and g_a → g_b fills the z slot, both staged through `gdnLowRank`
    /// (separate encoders, so hazard tracking serializes the reuse).
    private func encodeKDAPrefillChains(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int
    ) throws {
        guard let linFA = views.linFA, let linFB = views.linFB,
              let linGA = views.linGA, let linGB = views.linGB else {
            throw ModelError.internalInconsistency(
                detail: "KDA layer \(L) is missing a low-rank chain tensor view")
        }
        let la = cfg.linearAttention
        let low = la.keyHeadDim
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linFA,
                             x: scratch.normed,
                             y: scratch.gdnLowRank,
                             rows: low,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: low)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linFB,
                             x: scratch.gdnLowRank,
                             y: scratch.gdnA,
                             rows: la.numVHeads * la.keyHeadDim,
                             columns: low,
                             tokenCount: t,
                             xStrideElements: low,
                             yStrideElements: la.numVHeads * la.keyHeadDim)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linGA,
                             x: scratch.normed,
                             y: scratch.gdnLowRank,
                             rows: low,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: low)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: linGB,
                             x: scratch.gdnLowRank,
                             y: scratch.gdnZ,
                             rows: la.valueDim,
                             columns: low,
                             tokenCount: t,
                             xStrideElements: low,
                             yStrideElements: la.valueDim)
    }

    /// Softmax-attention branch of one chunked-prefill layer.
    ///
    /// lint:allow-long same shape as the linear branch: QKV projection, RoPE,
    /// KV-cache write and attention are one ordered pipeline over shared
    /// scratch, and the intermediate buffers have no meaning outside it.
    private func encodeFullAttentionPrefill(
        cb: MTLCommandBuffer, layer L: Int,
        views: LayerPrefillQKVViews, scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int, hiddenSize D: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int, rmsEps eps: Float
    ) throws {
        let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
        try encodeAffineProjection(commandBuffer: cb,
                             family: .q,
                             weights: views.q!,
                             x: scratch.normed,
                             y: scratch.q,
                             rows: qProjRows,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: qProjRows)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.k!,
                             x: scratch.normed,
                             y: scratch.kStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim)
        try encodeAffineProjection(commandBuffer: cb,
                             family: .kv,
                             weights: views.v!,
                             x: scratch.normed,
                             y: scratch.vStage,
                             rows: kvDim,
                             columns: D,
                             tokenCount: t,
                             xStrideElements: D,
                             yStrideElements: kvDim)

        // The attention input Q: the packed q_proj output is split
        // into per-head query/gate halves for gated architectures.
        let attnQ: MTLBuffer
        if cfg.attnOutputGate {
            try elementwise!.encodeSplitQGate(commandBuffer: cb,
                                          packed: scratch.q,
                                          q: scratch.attnQ,
                                          gate: scratch.attnGate,
                                          heads: cfg.numHeads,
                                          dim: headDim,
                                          rows: t)
            attnQ = scratch.attnQ
        } else {
            attnQ = scratch.q
        }

        if cfg.hasAttentionBiases {
            let qBias = try model.qProjBias(layer: L)
            let kBias = try model.kProjBias(layer: L)
            let vBias = try model.vProjBias(layer: L)
            try elementwise!.encodeBiasAdd(commandBuffer: cb,
                                       x: attnQ,
                                       bias: qBias.buffer,
                                       biasOffset: Int(qBias.offset),
                                       rowElems: qDim, rows: t)
            try elementwise!.encodeBiasAdd(commandBuffer: cb,
                                       x: scratch.kStage,
                                       bias: kBias.buffer,
                                       biasOffset: Int(kBias.offset),
                                       rowElems: kvDim, rows: t)
            try elementwise!.encodeBiasAdd(commandBuffer: cb,
                                       x: scratch.vStage,
                                       bias: vBias.buffer,
                                       biasOffset: Int(vBias.offset),
                                       rowElems: kvDim, rows: t)
        }

        if cfg.ropeNeoxSubdim && !cfg.hasQKNorms {
            let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
            try prefillQKVEpilogue.encodeNeoxSubdimNoNorm(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: Float(cfg.fullRopeTheta),
                rotaryDim: rotaryDim)
        } else if cfg.ropeNeoxSubdim {
            let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
            try prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                commandBuffer: cb,
                q: attnQ,
                k: scratch.kStage,
                qWeight: views.qNorm!.buffer,
                qWeightOffset: Int(views.qNorm!.offset),
                kWeight: views.kNorm!.buffer,
                kWeightOffset: Int(views.kNorm!.offset),
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                qTokenStrideElements: UInt32(qDim),
                kvTokenStrideElements: UInt32(kvDim),
                theta: Float(cfg.fullRopeTheta),
                rotaryDim: rotaryDim,
                eps: eps)
        } else {
            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            try prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: attnQ,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm!.buffer,
                                       qWeightOffset: Int(views.qNorm!.offset),
                                       kWeight: views.kNorm!.buffer,
                                       kWeightOffset: Int(views.kNorm!.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)
        }

        if let kv {
            let bytes = t * kvDim * MemoryLayout<Float16>.stride
            try copyPrefillKVToCache(commandBuffer: cb,
                                     kv: kv,
                                     layer: L,
                                     startPosition: startPosition,
                                     tokenCount: t,
                                     keySource: scratch.kStage,
                                     valueSource: scratch.vStage,
                                     bytesPerToken: bytes / t)
        }
        let kvView = kv?.keyView(layer: L, validTokenCount: startPosition + t)
        let params = PrefillAttentionParams(
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(cfg.numHeads),
                numKVHeads: UInt32(numKVHeads),
                kvValidCount: UInt32(startPosition + t),
                slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                kvTokenStrideElements: UInt32(kvDim),
                qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim),
                scale: Float(cfg.attentionScale),
                kvBits: UInt32(kvView?.precision.rawValue ?? 16),
                kvTokenStrideBytes: UInt32(kvView?.stride ?? (kvDim * 2)),
                kvValueBytes: UInt32(kvView?.valueBytes ?? (kvDim * 2)),
                kvGroupSize: UInt32(kvView?.groupSize
                    ?? KVCacheManager.quantizationGroupSize))
        if let kv {
                let keyView = kv.keyView(layer: L, validTokenCount: startPosition + t)
                let valueView = kv.valueView(layer: L, validTokenCount: startPosition + t)
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                let sinks = cfg.hasAttentionSinks
                    ? try model.attentionSinks(layer: L)
                    : nil
                try prefillAttention.encodeCausal(commandBuffer: cb,
                                              q: attnQ,
                                              k: keyView.buffer,
                                              v: valueView.buffer,
                                              out: scratch.attentionOutput,
                                              params: params,
                                              kvRingCapacity: activeRingCapacity,
                                              sinks: sinks?.buffer,
                                              sinksOffset: sinks.map { Int($0.offset) } ?? 0,
                                              path: prefillAttentionPath,
                                              minimumQueries: UInt32(prefillMatrixMinRows))
        } else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill attention requires a KV cache")
        }
        if cfg.attnOutputGate {
            try elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                              out: scratch.attentionOutput,
                                              gate: scratch.attnGate,
                                              count: t * qDim)
        }
        try encodeAffineProjection(commandBuffer: cb,
                                 family: .o,
                                 weights: views.o!,
                                 x: scratch.attentionOutput,
                                 y: scratch.h1,
                                 rows: D,
                                 columns: qDim,
                                 tokenCount: t,
                                 xStrideElements: qDim,
                                 yStrideElements: D)
        if cfg.hasAttentionBiases {
            let oBias = try model.oProjBias(layer: L)
            try elementwise!.encodeBiasAdd(commandBuffer: cb,
                                       x: scratch.h1,
                                       bias: oBias.buffer,
                                       biasOffset: Int(oBias.offset),
                                       rowElems: D, rows: t)
        }
    }

    /// Resolve every layer's tensor views once, before the chunk loop.
    private func makeLayerPrefillViews() throws -> [LayerPrefillQKVViews] {
        try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            let isKDA = isLinear && cfg.linearAttentionPerChannelDecay
            let isMLAL = cfg.layerIsMLA(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: L < cfg.numLeadingDenseLayers
                    ? nil : try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: (isLinear || isMLAL) ? nil : try model.kProj(layer: L),
                v: (isLinear || isMLAL) ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: (isLinear || isMLAL || !cfg.hasQKNorms)
                    ? nil : try model.qNorm(layer: L),
                kNorm: (isLinear || isMLAL || !cfg.hasQKNorms)
                    ? nil : try model.kNorm(layer: L),
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: (isLinear && !isKDA) ? try model.linearInProjZ(layer: L) : nil,
                linA: (isLinear && !isKDA) ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear
                    ? (isKDA ? try model.oProj(layer: L)
                             : try model.linearOutProj(layer: L)) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear
                    ? (isKDA ? try model.kimiALog(layer: L)
                             : try model.linearALog(layer: L)) : nil,
                linDtBias: isLinear
                    ? (isKDA ? try model.kimiDtBias(layer: L)
                             : try model.linearDtBias(layer: L)) : nil,
                linNorm: isLinear
                    ? (isKDA ? try model.kimiONorm(layer: L)
                             : try model.linearNorm(layer: L)) : nil,
                linFA: isKDA ? try model.kimiFAProj(layer: L) : nil,
                linFB: isKDA ? try model.kimiFBProj(layer: L) : nil,
                linGA: isKDA ? try model.kimiGAProj(layer: L) : nil,
                linGB: isKDA ? try model.kimiGBProj(layer: L) : nil,
                mlaKVA: isMLAL ? try model.kimiKVAProj(layer: L) : nil,
                mlaKVALayernorm: isMLAL ? try model.kimiKVALayernorm(layer: L) : nil,
                mlaEmbedQ: isMLAL ? try model.kimiEmbedQ(layer: L) : nil,
                mlaUnembedOut: isMLAL ? try model.kimiUnembedOut(layer: L) : nil)
        }
    }

    /// Final norm and lm_head for the last chunk, writing logits or a fused
    /// greedy token depending on the output mode.
    private func encodeFinalHead(
        logits: MTLBuffer,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        rmsEps eps: Float,
        outputMode: PrefillOutputMode
    ) throws {
        let finalNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let finalCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: finalCB,
                hidden: scratch.hidden,
                hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                normWeight: finalNorm.buffer,
                normOffset: Int(finalNorm.offset),
                weights: lm.buffer,
                weightsOffset: Int(lm.offset),
                scales: lm.buffer,
                scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer,
                biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: UInt32(D),
                vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
        } else {
            try prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                             hiddenBlock: scratch.hidden,
                                             row: t - 1,
                                             rowStrideElements: D,
                                             normWeight: finalNorm.buffer,
                                             normWeightOffset: Int(finalNorm.offset),
                                             weights: lm.buffer,
                                             weightsOffset: Int(lm.offset),
                                             scales: lm.buffer,
                                             scalesOffset: Int(lm.scaleOffset),
                                             biases: lm.buffer,
                                             biasesOffset: Int(lm.biasOffset),
                                             logits: logits,
                                             d: UInt32(D),
                                             vocab: UInt32(cfg.vocabSize),
                                             rmsEps: eps)
        }
        finalCB.commit()
        try waitForCompletion(finalCB)
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
        }
    }

    private struct PrefillRouting {
        let routes: PrefillMoEGroupedRoutes
        let schedulerConfig: PrefillRoutedTileSchedulerConfig
    }

    /// The router readback, pair building and grouping of one prefill chunk:
    /// host work that runs while the shared expert's command buffer is on the GPU.
    private func buildPrefillRoutes(layer L: Int,
                                    tokenCount t: Int,
                                    scratch: PrefillChunkScratchBuffers,
                                    descendingSweep: Bool,
                                    participatesInCarry: Bool) throws -> PrefillRouting {
        let routeCount = t * cfg.topKExperts
        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        // Reused per-chunk host scratch (R38): cleared in place so
        // the routed-tile planner never allocates per chunk.
        routeIDScratch.removeAll(keepingCapacity: true)
        routeWeightScratch.removeAll(keepingCapacity: true)
        routeIDScratch.reserveCapacity(routeCount)
        routeWeightScratch.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDScratch.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeightScratch.append(weightPtr[i])
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDScratch,
                                                       weights: routeWeightScratch,
                                                       queryCount: t,
                                                       topK: cfg.topKExperts)
        let schedulerConfig: PrefillRoutedTileSchedulerConfig
        if let slotCount = model.routedExpertCacheSlotCount() {
            guard let fitted = prefillRoutedTileSchedulerConfig.fitting(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
            }
            schedulerConfig = fitted
        } else {
            schedulerConfig = prefillRoutedTileSchedulerConfig
        }
        var expertTileCounts: [Int]?
        var sortKeys = model.routedExpertPhysicalOffsets(layer: L)
        var sortDescending = descendingSweep
        if Self.prefillChunkUsesComputedSweepOrder(mode: prefillSweepMode,
                                                   participatesInCarry: participatesInCarry) {
            var lastRowByExpert: [UInt32: Int] = [:]
            var rowsByExpert: [UInt32: Int] = [:]
            lastRowByExpert.reserveCapacity(min(pairs.count, cfg.numExperts))
            rowsByExpert.reserveCapacity(min(pairs.count, cfg.numExperts))
            for pair in pairs {
                let row = Int(pair.token)
                lastRowByExpert[pair.expert] = max(lastRowByExpert[pair.expert] ?? row, row)
                rowsByExpert[pair.expert, default: 0] += 1
            }
            if prefillSweepMode == .recency {
                let balanced = PrefillSweepOrder.recencyBalanced(
                    rowsByExpert: rowsByExpert,
                    lastRowByExpert: lastRowByExpert,
                    tail: prefillSweepTail,
                    tileWidth: schedulerConfig.tileExperts)
                // The array path `groupTokenExpertPairs` already takes for
                // `alternate`/`fixed`/`carry`: one sort key per expert id, no
                // per-comparison dictionary lookup.
                sortKeys = PrefillSweepOrder.expertSortKeys(forOrder: balanced.order, numExperts: cfg.numExperts)
                expertTileCounts = balanced.tileExpertCounts
            } else {
                let order = PrefillSweepOrder.residentFirstBalanced(
                    rowsByExpert: rowsByExpert,
                    lastRowByExpert: lastRowByExpert,
                    resident: try residentExpertMask(layer: L),
                    slots: model.routedExpertCacheSlotCount() ?? 0,
                    tileWidth: schedulerConfig.tileExperts)
                sortKeys = PrefillSweepOrder.expertSortKeys(forOrder: order, numExperts: cfg.numExperts)
            }
            sortDescending = false
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: schedulerConfig.tileExperts,
            expertSortKeys: sortKeys,
            descending: sortDescending,
            expertTileCounts: expertTileCounts)
        return PrefillRouting(routes: routes, schedulerConfig: schedulerConfig)
    }

    /// `.resident`'s per-chunk snapshot of layer `L`'s pool-resident
    /// experts: `prefillResidentExpertScratch` is allocated once to
    /// `cfg.numExperts` and cleared in place on every later call, so the
    /// sweep never allocates per chunk; empty when the model streams with
    /// no expert cache to query.
    private func residentExpertMask(layer L: Int) throws -> [Bool] {
        if prefillResidentExpertScratch.count != cfg.numExperts {
            prefillResidentExpertScratch = [Bool](repeating: false, count: cfg.numExperts)
        } else {
            for i in 0..<prefillResidentExpertScratch.count {
                prefillResidentExpertScratch[i] = false
            }
        }
        guard model.routedExpertCacheSlotCount() != nil else {
            return prefillResidentExpertScratch
        }
        for expert in try model.routedExpertResidentIDs(layer: L)
            where expert >= 0 && expert < prefillResidentExpertScratch.count {
            prefillResidentExpertScratch[expert] = true
        }
        return prefillResidentExpertScratch
    }

    /// The shared (dense) expert branch of one prefill chunk and its scalar
    /// gate, on whichever of the matrix and per-token paths this chunk earns.
    private func encodeSharedExpertBlock(commandBuffer sharedCB: MTLCommandBuffer,
                                         layer L: Int,
                                         scratch: PrefillChunkScratchBuffers,
                                         tokenCount t: Int,
                                         hiddenSize D: Int) throws {
        let matrixMPP = cfg.hasSharedExpert
            ? prefillSharedExpert.matrixPath(for: prefillMPPAffineInt4,
                                             queryCount: t,
                                             d: D,
                                             intermediate: cfg.intermediateSize,
                                             minimumRows: prefillMatrixMinRows)
            : nil
        if cfg.hasSharedExpert {
            let sharedProj = sharedExpertProjections[L]
            if let mpp = matrixMPP {
                try prefillSharedExpert.encodeChunk(
                    commandBuffer: sharedCB,
                    mpp: mpp,
                    x: scratch.routedX,
                    y: scratch.h1,
                    gate: sharedProj.gate,
                    up: sharedProj.up,
                    down: sharedProj.down,
                    scratchGate: scratch.sharedGateScratch,
                    scratchUp: scratch.sharedUpScratch,
                    queryCount: t,
                    d: D,
                    intermediate: cfg.intermediateSize,
                    minimumRows: prefillMatrixMinRows)
            } else {
                try prefillSharedExpert.encodeBlock(
                    commandBuffer: sharedCB,
                    x: scratch.routedX,
                    y: scratch.h1,
                    gate: sharedProj.gate,
                    up: sharedProj.up,
                    down: sharedProj.down,
                    scratchGate: scratch.sharedGateScratch,
                    scratchUp: scratch.sharedUpScratch,
                    scratchAct: scratch.sharedActScratch,
                    queryCount: t,
                    d: D,
                    intermediate: cfg.intermediateSize,
                    xStrideElements: D,
                    yStrideElements: D)
            }
        } else {
            // No shared expert (gpt-oss): scratch.h1 still holds the
            // attention branch; zero it so the reduce folds nothing.
            guard let blit = sharedCB.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.fill(buffer: scratch.h1,
                      range: 0..<(t * D * MemoryLayout<Float16>.stride),
                      value: 0)
            blit.endEncoding()
        }
        guard cfg.sharedExpertGated else { return }
        // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
        // per chunk row.
        let gateView = sharedExpertProjections[L].scalarGate!
        let halfBytes = MemoryLayout<Float16>.stride
        if matrixMPP != nil {
            try elementwise!.encodeScalarGateRows(
                commandBuffer: sharedCB,
                weights: gateView,
                x: scratch.routedX,
                gate: scratch.sharedScalarGate,
                rows: t, d: D)
            try elementwise!.encodeSigmoidScalarMulRows(
                commandBuffer: sharedCB,
                y: scratch.h1,
                gate: scratch.sharedScalarGate,
                rows: t, d: D)
            return
        }
        for row in 0..<t {
            try int8ScalarGate!.encode(
                commandBuffer: sharedCB,
                weights: gateView.buffer,
                weightsOffset: Int(gateView.offset),
                scales: gateView.buffer,
                scalesOffset: Int(gateView.scaleOffset),
                biases: gateView.buffer,
                biasesOffset: Int(gateView.biasOffset),
                x: scratch.routedX,
                xOffset: row * D * halfBytes,
                y: scratch.sharedScalarGate,
                yOffset: row * halfBytes,
                m: 1, n: UInt32(D))
        }
        for row in 0..<t {
            try elementwise!.encodeSigmoidScalarMul(
                commandBuffer: sharedCB,
                y: scratch.h1,
                yOffset: row * D * halfBytes,
                gate: scratch.sharedScalarGate,
                gateOffset: row * halfBytes,
                count: D)
        }
    }

    private struct PendingPrefillBatch {
        let tileIndices: [Int]
        let commandBuffer: MTLCommandBuffer
        let fetches: [PrefillStreamedTileFetchResult]
        let argumentBuffers: [PrefillStreamedTileArgumentBuffer]

        var assignedSlots: [Int] { fetches.flatMap(\.plannedAssignedSlots) }
    }

    private final class OpenPrefillBatch {
        let commandBuffer: MTLCommandBuffer
        private(set) var tileIndices: [Int] = []
        private(set) var fetches: [PrefillStreamedTileFetchResult] = []
        private(set) var argumentBuffers: [PrefillStreamedTileArgumentBuffer] = []

        init(commandBuffer: MTLCommandBuffer) {
            self.commandBuffer = commandBuffer
        }

        var assignedSlots: [Int] { fetches.flatMap(\.plannedAssignedSlots) }

        func append(tileIndex: Int,
                    fetch: PrefillStreamedTileFetchResult,
                    argumentBuffer: PrefillStreamedTileArgumentBuffer) {
            tileIndices.append(tileIndex)
            fetches.append(fetch)
            argumentBuffers.append(argumentBuffer)
        }

        func sealed() -> PendingPrefillBatch {
            PendingPrefillBatch(tileIndices: tileIndices,
                                commandBuffer: commandBuffer,
                                fetches: fetches,
                                argumentBuffers: argumentBuffers)
        }
    }

    private func drainOldestPendingBatch(
        _ pendingBatches: inout [PendingPrefillBatch],
        lifetime tileLifetime: inout PrefillStreamedTileSlotLifetime) throws {
        guard !pendingBatches.isEmpty else { return }
        let batch = pendingBatches.removeFirst()
        // `withExtendedLifetime` takes a non-throwing closure, so the wait
        // error is captured here and rethrown after the blobs are released.
        var drainError: Error?
        withExtendedLifetime((batch.fetches, batch.argumentBuffers)) {
            do {
                try waitForCompletion(batch.commandBuffer)
                recordKernelGPU(role: "prefill_routed_tile", batch.commandBuffer)
            } catch {
                drainError = error
            }
        }
        if let drainError {
            throw drainError
        }
        for (tileIndex, fetch) in zip(batch.tileIndices, batch.fetches)
        where !fetch.plannedMissSlots.isEmpty {
            try tileLifetime.complete(tileIndex: tileIndex)
        }
    }

    /// Router, routed-expert fetch and the MoE tail for one prefill layer.
    ///
    /// lint:allow-long one layer's MoE stage is a single ordered pipeline:
    /// route readback, expert streaming, tiled phase-1/phase-2, then the
    /// residual tail. It rebinds the command buffer partway through (the
    /// resident buffer wraps between layers), so the stages share mutable
    /// encoding state and cannot be separated without threading it back out.
    private func encodeRoutedMoEPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int,
        descendingSweep: Bool,
        participatesInCarry: Bool,
        layerStart prefillLayerStart: UInt64,
        routeNanos prefillRouteNanos: inout UInt64,
        tileNanos prefillTileNanos: inout UInt64,
        tailNanos prefillTailNanos: inout UInt64,
        activeExperts prefillActiveExperts: inout UInt64
    ) async throws {
        var prefillRouteEnd = prefillLayerStart
        var prefillTileEnd = prefillLayerStart
        let perExpertScale: (buffer: any MTLBuffer, offset: Int) =
            (onesPerExpertScale!, 0)
        guard let router = views.router else {
            throw ModelError.internalInconsistency(
                detail: "routed-MoE prefill on layer \(L) without a router view")
        }
        if let poolResidency {
            poolResidency.include(try model.routedExpertResidency(layer: L).expertPool)
        }
        try prefillRouter.encodeBlock(
                    commandBuffer: cb,
                    weights: router.buffer,
                    weightsOffset: Int(router.offset),
                    scales: router.buffer,
                    scalesOffset: Int(router.scaleOffset),
                    biases: router.buffer,
                    biasesOffset: Int(router.biasOffset),
                    hidden: scratch.routedX,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: perExpertScale.offset,
                    logitBias: routerLogitBias[L].buffer,
                    logitBiasOffset: routerLogitBias[L].offset,
                    outIndices: scratch.routeIDs,
                    outWeights: scratch.routeWeights,
                    queryCount: UInt32(t),
                    numExperts: UInt32(cfg.numExperts),
                    d: UInt32(D),
                    topK: UInt32(cfg.topKExperts),
                    hiddenStrideElements: UInt32(D))

                cb.commit()
                guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeSharedExpertBlock(commandBuffer: sharedCB,
                                            layer: L,
                                            scratch: scratch,
                                            tokenCount: t,
                                            hiddenSize: D)
                // One queue runs buffers in commit order, so sharedCB's read of
                // routedX is ordered after cb; committing it before the wait lets
                // its GPU time cover the host routing below.
                if prefillRouteOverlap { sharedCB.commit() }
                try waitForCompletion(cb)
                // Prefill had no occupancy instrumentation at all: these buffers
                // never reached recordKernelGPU, so SHRIKE_KERNEL_STATS reported
                // only the decode tokens of a request and prefill looked idle.
                // Split by layer kind: the Track A go/no-go needs to know how
                // the attention-block time divides between full-attention
                // layers (whole block is ANE-expressible) and Gated-DeltaNet
                // layers (only the dense projections are; the recurrent scan
                // is not representable in a static Core ML graph).
                recordKernelGPU(role: cfg.layerIsLinear(L) ? "prefill_gdn_router"
                                    : "prefill_attn_router", cb)

                let routing = try buildPrefillRoutes(layer: L, tokenCount: t, scratch: scratch,
                                                     descendingSweep: descendingSweep,
                                                     participatesInCarry: participatesInCarry)
                let routes = routing.routes
                let schedulerConfig = routing.schedulerConfig
                prefillRouteEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                prefillRouteNanos &+= prefillRouteEnd - prefillLayerStart
                // One group per *distinct* expert this chunk touches. For a
                // 1-token chunk this is topK; for a speculative 2-token verify
                // it is the union of the two tokens' routes, which is what
                // decides whether the extra row rides along on weights the
                // first row already pulled in or pays for its own.
                prefillActiveExperts &+= UInt64(routes.groups.count)

                if !prefillRouteOverlap { sharedCB.commit() }
                try waitForCompletion(sharedCB)
                recordKernelGPU(role: "prefill_shared_expert", sharedCB)

                let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                    device: ctx.device,
                    routes: routes)
                let routedOffsets = try model.routedExpertOffsets(layer: L)
                var tailError: Error?
                let tileDriver = ExpertStreamedTileDriver(
                    runner: self,
                    layer: L,
                    routes: routes,
                    scratch: scratch,
                    metadata: metadata,
                    routedOffsets: routedOffsets,
                    hiddenSize: D,
                    startPosition: startPosition,
                    protection: chunkExpertProtection(routes: routes))
                try await PrefillRoutedTileSequencer(
                    scheduler: PrefillRoutedTileScheduler(config: schedulerConfig))
                    .run(tileCount: routes.tiles.count, driver: tileDriver)
                prefillTileEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                prefillTileNanos &+= prefillTileEnd - prefillRouteEnd
                guard let tailCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                  routePartials: scratch.routePartials,
                                                  routeWeights: scratch.routeWeights,
                                                  h2: scratch.h2,
                                                  queryCount: UInt32(t),
                                                  topK: UInt32(cfg.topKExperts),
                                                  d: UInt32(D))
                // Plain pre-norm tail: hidden += gated shared branch
                // + routed branch.
                try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
                try elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                               hidden: scratch.hidden,
                                               delta: scratch.h2,
                                               count: t * D)
                tailCB.commit()
                withExtendedLifetime(metadata) {
                    do {
                        try waitForCompletion(tailCB)
                        recordKernelGPU(role: "prefill_moe_reduce", tailCB)
                    } catch {
                        // Rethrown after `metadata` is released.
                        tailError = error
                    }
                }
                if let error = tailError {
                    tailError = nil
                    throw error
                }
                if L + 1 < cfg.numLayers {
                    guard let nextCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    cb = nextCB
                }
                prefillTailNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prefillTileEnd
    }

    /// A fresh `PrefillChunkExpertProtection` seeded from the whole chunk's
    /// routed experts.
    private func chunkExpertProtection(routes: PrefillMoEGroupedRoutes) -> PrefillChunkExpertProtection {
        PrefillChunkExpertProtection(routedGroups: routes.groups, expertsPerLayer: cfg.numExperts)
    }

    /// One prefill layer's routed tiles as `PrefillRoutedTileSequencer`
    /// drives them: the pool's kept plans and begun fetches by tile, the open
    /// and pending command buffers, the slot lifetimes and the chunk's expert
    /// protection.
    private final class ExpertStreamedTileDriver: PrefillRoutedTileDriver {
        /// unchecked-invariant: built inside `encodeRoutedMoEPrefill`, handed to
        /// the sequencer's `run` and released before that call returns, never
        /// stored or captured, so the runner always outlives it.
        private unowned let runner: RealForwardRunner
        private let layer: Int
        private let routes: PrefillMoEGroupedRoutes
        private let scratch: PrefillChunkScratchBuffers
        private let metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers
        private let routedOffsets: MoEExpertOffsets
        private let hiddenSize: Int
        private let startPosition: Int
        private var protection: PrefillChunkExpertProtection
        private var pendingBatches: [PendingPrefillBatch] = []
        private var openBatch: OpenPrefillBatch?
        private var tileLifetime = PrefillStreamedTileSlotLifetime()
        private var keptPlans: [Int: RoutedExpertFetchPlan] = [:]
        private var begunFetches: [Int: PrefillStreamedTileFetchBegin] = [:]

        init(runner: RealForwardRunner,
             layer: Int,
             routes: PrefillMoEGroupedRoutes,
             scratch: PrefillChunkScratchBuffers,
             metadata: PrefillGroupedRoutedMoEStreamedMetadataBuffers,
             routedOffsets: MoEExpertOffsets,
             hiddenSize: Int,
             startPosition: Int,
             protection: PrefillChunkExpertProtection) {
            self.runner = runner
            self.layer = layer
            self.routes = routes
            self.scratch = scratch
            self.metadata = metadata
            self.routedOffsets = routedOffsets
            self.hiddenSize = hiddenSize
            self.startPosition = startPosition
            self.protection = protection
        }

        var openBatchTiles: Int { openBatch?.tileIndices.count ?? 0 }
        var openBatchSlots: [Int] { openBatch?.assignedSlots ?? [] }
        var pendingDepth: Int { pendingBatches.count }
        var pendingAssignedSlots: [Int] { pendingBatches.flatMap(\.assignedSlots) }
        private var heldSlots: Set<Int> { Set(openBatchSlots + pendingAssignedSlots) }

        func plan(tile: Int, avoidingInFlight inFlight: Int?) throws -> Bool {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tile, routes: routes)
            protection.planning(expertIDs)
            var avoiding = heldSlots
            if let inFlight {
                guard let fetch = begunFetches[inFlight] else {
                    throw ModelError.indexCorrupt(
                        detail: "routed tile \(inFlight) planned as in flight before its fetch was begun")
                }
                avoiding.formUnion(fetch.plan.assignedSlots)
            }
            let plan = try runner.model.planRoutedExpertsIfPossible(
                layer: layer,
                experts: expertIDs,
                avoidingSlots: avoiding,
                protectedExperts: protection.remaining)
            keptPlans[tile] = plan
            return plan != nil
        }

        func begin(tile: Int) throws {
            let fetch: PrefillStreamedTileFetchBegin
            if let plan = keptPlans.removeValue(forKey: tile) {
                do {
                    fetch = try PrefillStreamedTileBinding.beginFetchForTile(
                        model: runner.model, layer: layer, tileIndex: tile, routes: routes,
                        plannedFetch: plan)
                } catch {
                    // The streamer's begin throws only before it executes the
                    // plan, whose miss slots stay reserved until abandoned.
                    try? runner.model.abandonRoutedExpertPlan(plan)
                    throw error
                }
            } else {
                fetch = try PrefillStreamedTileBinding.beginFetchForTile(
                    model: runner.model, layer: layer, tileIndex: tile, routes: routes,
                    avoidingSlots: heldSlots, protectedExperts: protection.remaining)
            }
            // Kept before the lifetime check so a throw below is still waited out.
            begunFetches[tile] = fetch
            if runner.routeTraceFD >= 0 {
                let counted = try runner.routeTraceRowCounts(forTile: routes.tiles[tile], routes: routes)
                runner.recordRouteTrace(layer: layer, position: startPosition, tile: tile,
                                        experts: fetch.expertIDs, rowCounts: counted.counts,
                                        lastRows: counted.lastRows)
            }
            let missSlots = fetch.plan.misses.map { fetch.plan.assignedSlots[$0] }
            if !missSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tile, plannedSlots: missSlots)
            }
        }

        func abandonPlan(tile: Int) throws {
            guard let plan = keptPlans.removeValue(forKey: tile) else { return }
            try runner.model.abandonRoutedExpertPlan(plan)
        }

        func encode(tile: Int) async throws {
            guard let begun = begunFetches[tile] else {
                throw ModelError.indexCorrupt(
                    detail: "routed tile \(tile) encoded before its fetch was begun")
            }
            let views = try await begun.operation.completion()
            begunFetches.removeValue(forKey: tile)
            let fetch = try PrefillStreamedTileBinding.bindingForCompletedFetch(begin: begun, views: views)
            let tileRange = routes.tiles[tile]
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tileRange.pairStart),
                                                  pairCount: Int(tileRange.pairCount))
            let argumentBuffer = try runner.prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: runner.ctx.device, binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tileRange.pairStart,
                pairCount: tileRange.pairCount,
                d: UInt32(hiddenSize),
                routedIntermediate: UInt32(runner.cfg.moeIntermediateSize),
                topK: UInt32(runner.cfg.topKExperts),
                hiddenStrideElements: UInt32(hiddenSize),
                binding: fetch.binding,
                offsets: routedOffsets)
            let batch = try currentOpenBatch()
            try runner.encodeRoutedTileExperts(commandBuffer: batch.commandBuffer,
                                               scratch: scratch,
                                               sortedPairs: metadata.sortedPairs,
                                               routes: routes,
                                               tile: tileRange,
                                               binding: fetch.binding,
                                               argumentBuffer: argumentBuffer,
                                               params: streamedParams)
            batch.append(tileIndex: tile, fetch: fetch, argumentBuffer: argumentBuffer)
        }

        func commitOpenBatch() {
            guard let batch = openBatch else { return }
            openBatch = nil
            batch.commandBuffer.commit()
            pendingBatches.append(batch.sealed())
        }

        func drainOldestBatch() throws {
            try runner.drainOldestPendingBatch(&pendingBatches, lifetime: &tileLifetime)
        }

        func abandonBegunFetches() {
            for fetch in begunFetches.values {
                _ = try? fetch.operation.wait()
            }
            begunFetches.removeAll()
            for plan in keptPlans.values {
                try? runner.model.abandonRoutedExpertPlan(plan)
            }
            keptPlans.removeAll()
        }

        private func currentOpenBatch() throws -> OpenPrefillBatch {
            if let openBatch { return openBatch }
            guard let batchCB = runner.ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            let batch = OpenPrefillBatch(commandBuffer: batchCB)
            openBatch = batch
            return batch
        }
    }

    /// One routed tile's experts: the grouped GEMMs over every expert, else
    /// the per-expert GEMMs with the scalar microbatch path for the experts
    /// below the tile threshold, else the scalar path whole.
    private func encodeRoutedTileExperts(
        commandBuffer tileCB: MTLCommandBuffer,
        scratch: PrefillChunkScratchBuffers,
        sortedPairs: MTLBuffer,
        routes: PrefillMoEGroupedRoutes,
        tile: PrefillMoETile,
        binding: PrefillStreamedTileBinding,
        argumentBuffer: PrefillStreamedTileArgumentBuffer,
        params: PrefillGroupedRoutedMoEStreamedParams
    ) throws {
        func encodeScalar(pairStart: UInt32, pairCount: UInt32) throws {
            var scalarParams = params
            scalarParams.pairStart = pairStart
            scalarParams.pairCount = pairCount
            _ = try prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.routedX,
                sortedPairs: sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: binding,
                params: scalarParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
        }

        let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
        // `usesRoutedExpertMatrixPath` is the same predicate the scratch layout
        // sizes the staging with, so the branch can never outrun its buffers.
        guard scratch.layout.usesRoutedExpertMatrixPath,
              let mpp = prefillGroupedMoE.matrixPath(
                for: prefillMPPAffineInt4,
                d: Int(params.d),
                intermediate: Int(params.routedIntermediate)) else {
            try encodeScalar(pairStart: tile.pairStart, pairCount: tile.pairCount)
            return
        }
        if prefillRoutedGEMMGrouped, prefillGroupedMoE.groupedPathAvailable(for: mpp) {
            let tailTile = tailTileInForce(for: mpp)
            let waves = try PrefillGroupedRoutedMoE.planExpertWaves(
                ranges: ranges,
                binding: binding,
                stagingRows: scratch.routedExpertStaging.rowBlock,
                tailTile: tailTile)
            try prefillGroupedMoE.encodeGroupedExpertGEMMs(
                commandBuffer: tileCB,
                mpp: mpp,
                hidden: scratch.routedX,
                sortedPairs: sortedPairs,
                routePartials: scratch.routePartials,
                binding: binding,
                argumentBuffer: argumentBuffer,
                waves: waves,
                staging: scratch.routedExpertStaging,
                params: params,
                tailTile: tailTile)
            return
        }
        let leftovers = try prefillGroupedMoE.encodeExpertGEMMs(
            commandBuffer: tileCB,
            mpp: mpp,
            hidden: scratch.routedX,
            sortedPairs: sortedPairs,
            routePartials: scratch.routePartials,
            binding: binding,
            ranges: ranges,
            staging: scratch.routedExpertStaging,
            params: params)
        guard leftovers.count < ranges.count else {
            try encodeScalar(pairStart: tile.pairStart, pairCount: tile.pairCount)
            return
        }
        for leftover in leftovers {
            try encodeScalar(pairStart: UInt32(leftover.pairStart),
                             pairCount: UInt32(leftover.pairCount))
        }
    }

    /// Attention stage of one decode layer: the gated-DeltaNet branch or the
    /// softmax branch, both writing into `oOut` for the residual add.
    ///
    /// lint:allow-long the two branches are alternatives over the same set of
    /// scratch buffers; splitting them apart again would only re-create the
    /// dispatch this method exists to hold.
    private func encodeDecodeAttention(
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        softmaxCB: inout MTLCommandBuffer?,
        layerEncoder: MTLComputeCommandEncoder?,
        layer L: Int,
        position: Int,
        isLinear: Bool,
        rmsEps eps: Float
    ) throws {
        let seqLen = UInt32(position + 1)
        if isLinear, cfg.linearAttentionPerChannelDecay {
            // Kimi KDA keeps its own CB-internal encoders (the low-rank
            // scratch is reused across its projection chains).
            try encodeKDADecode(attnCB, layer: L)
        } else if isLinear {
            // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
            // fixed-size recurrent state updated in place.
            guard let layerEncoder else {
                throw ModelError.internalInconsistency(
                    detail: "GDN decode layer \(L) without a layer encoder")
            }
            try encodeLinearAttentionDecode(encoder: layerEncoder, layer: L)
        } else if cfg.layerIsMLA(L) {
            // Kimi MLA: absorbed MQA over one fused [latent | k_pe] FP16
            // row per token, NoPE.
            try encodeMLAAttentionDecode(attnCB, layer: L,
                                         position: position, seqLen: seqLen)
        } else if cfg.attnOutputGate {
            // Qwen full attention: packed [query ; gate] q_proj, real
            // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
            guard let layerEncoder else {
                throw ModelError.internalInconsistency(
                    detail: "gated attention layer \(L) without a layer encoder")
            }
            try encodeGatedFullAttentionDecode(encoder: layerEncoder, layer: L,
                                               position: position,
                                               seqLen: seqLen)
        } else if cfg.hasAttentionBiases {
            // gpt-oss attention: biased q/k/v/o, no QK norms, no output
            // gate, arch YaRN RoPE, sinks, full/SWA by the layer mask.
            try encodeGptOssAttentionDecode(attnCB: attnCB, tailCB: tailCB,
                                            softmaxCB: &softmaxCB, layer: L,
                                            position: position, seqLen: seqLen)
        } else {
            try encodePlainAttentionDecode(attnCB: attnCB, tailCB: tailCB,
                                           softmaxCB: &softmaxCB, layer: L,
                                           position: position, seqLen: seqLen,
                                           rmsEps: eps)
        }

        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
    }

    /// Plain (non-gated, unbiased) full/SWA attention, one decode step:
    /// fused QKV + rope/norm epilogue on `attnCB`, the softmax pass on its
    /// own CB, o_proj on `tailCB`.
    private func encodePlainAttentionDecode(
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        softmaxCB: inout MTLCommandBuffer?,
        layer L: Int,
        position: Int,
        seqLen: UInt32,
        rmsEps eps: Float
    ) throws {
        let D = UInt32(cfg.hiddenSize)
        let isFull = cfg.fullAttentionLayerMask[L] == 1
        let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
        let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
        let qDim     = UInt32(cfg.numHeads * headDimL)
        let kvDim    = UInt32(numKVL * headDimL)
        do {
            let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let quantizedKV = kv?.precision.isQuantized == true
            let kWrite = quantizedKV ? (buffer: kStage, offset: 0) : kSlot
            let vWrite = quantizedKV ? (buffer: vStage, offset: 0) : vSlot
            let q     = try model.qProj(layer: L)
            let k     = try model.kProj(layer: L)
            // Under the K=V quirk full layers reuse k_proj; otherwise
            // v_proj is a real tensor.
            let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
            let o     = try model.oProj(layer: L)
            let qNorm = try model.qNorm(layer: L)
            let kNorm = try model.kNorm(layer: L)

            try fusedQKVGEMV.encode(commandBuffer: attnCB,
                                qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                x: normed,
                                qOut: qScratch,
                                kOut: kWrite.buffer, kOutOffset: kWrite.offset,
                                vOut: vWrite.buffer, vOutOffset: vWrite.offset,
                                qRows: qDim,
                                kvRows: kvDim,
                                n: D)

            let rotated = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDimL / 2)
            try fusedQKVEpilogue.encode(commandBuffer: attnCB,
                                    q: qScratch,
                                    k: kWrite.buffer,
                                    kOffset: kWrite.offset,
                                    v: vWrite.buffer,
                                    vOffset: vWrite.offset,
                                    qWeight: qNorm.buffer,
                                    qWeightOffset: Int(qNorm.offset),
                                    kWeight: kNorm.buffer,
                                    kWeightOffset: Int(kNorm.offset),
                                    headDim: UInt32(headDimL),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKVL),
                                    position: UInt32(position),
                                    theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                    rotatedPairs: rotated,
                                    eps: eps)

            guard let kv else {
                throw ModelError.internalInconsistency(
                    detail: "attention requires a KV cache")
            }
            if quantizedKV {
                try encodeQuantizedKV(commandBuffer: attnCB, kv: kv, layer: L,
                                      position: position, keySource: kStage,
                                      valueSource: vStage, elementCount: Int(kvDim))
            }
            let keyView = kv.keyView(layer: L, validTokenCount: Int(seqLen))
            let valueView = kv.valueView(layer: L, validTokenCount: Int(seqLen))
            guard let attentionCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            softmaxCB = attentionCB
            if isFull {
                try attention.encodeFull(commandBuffer: attentionCB,
                                     q: qScratch,
                                     k: keyView.buffer, kOffset: keyView.offset,
                                     v: valueView.buffer, vOffset: valueView.offset,
                                     out: attnOut,
                                     headDim: UInt32(headDimL),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(numKVL),
                                     seqLen: seqLen,
                                     scale: Float(cfg.attentionScale),
                                     kvFormat: keyView)
            } else {
                let ringCapacity = kv.ringCapacity(layer: L)
                let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                    ? UInt32(ringCapacity)
                    : 0
                try attention.encodeSWA(commandBuffer: attentionCB,
                                    q: qScratch,
                                    k: kSlot.buffer, kOffset: 0,
                                    v: vSlot.buffer, vOffset: 0,
                                    out: attnOut,
                                    headDim: UInt32(headDimL),
                                    numQHeads: UInt32(cfg.numHeads),
                                    numKVHeads: UInt32(numKVL),
                                    seqLen: seqLen,
                                    window: UInt32(cfg.slidingWindow),
                                    scale: Float(cfg.attentionScale),
                                    ringCapacity: activeRingCapacity,
                                    kvFormat: keyView)
            }
            try int4.encode(commandBuffer: tailCB,
                        weights: o.buffer, weightsOffset: Int(o.offset),
                        scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                        biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                        x: attnOut, y: oOut, m: D, n: qDim)
        }
    }

    // MARK: - Decode routed-expert helpers

    /// A routed-expert command whose completion is deferred to the next layer.
    private struct PendingRoutedCommand {
        let cb: MTLCommandBuffer
        let phase1HitCB: MTLCommandBuffer?
        let specCB: MTLCommandBuffer?
        let expertLease: RoutedExpertLease?
        let storageOperation: RoutedExpertLoadOperation?
        let overlapCompletionClock: CommandCompletionClock?
        let expectedOverlapCompletions: Int
        let hitCommitNanos: UInt64
        let routedCommitNanos: UInt64
        let kernelRole: String
        let encodeAndCommitNanos: UInt64
    }

    /// Diagnostic-only completion clock used to measure the I/O tail left
    /// after already-runnable GPU work. It is allocated only with
    /// SHRIKE_RUNNER_STATS, never in the production hot path.
    /// unchecked-invariant: completion timestamps are mutated and read only
    /// while holding `lock`.
    private final class CommandCompletionClock: @unchecked Sendable {
        private let lock = NSLock()
        private var completionCount = 0
        private var latestCompletion: UInt64 = 0

        func track(_ commandBuffer: MTLCommandBuffer) {
            commandBuffer.addCompletedHandler { [self] _ in
                lock.lock()
                completionCount += 1
                latestCompletion = max(
                    latestCompletion,
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
                lock.unlock()
            }
        }

        func latest(expected: Int) -> UInt64? {
            lock.lock()
            defer { lock.unlock() }
            guard completionCount == expected else { return nil }
            return latestCompletion
        }
    }

    private func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                    waitIfNeeded: Bool,
                                    deferTimings: Bool = false) throws {
        defer { pending.expertLease?.release() }
        if waitIfNeeded {
            if let phase1HitCB = pending.phase1HitCB {
                try waitForCompletion(phase1HitCB)
            }
            try waitForCompletion(pending.cb)
        } else if let err = pending.cb.error {
            throw ModelError.commandBufferFailed(
                detail: "routed layer command buffer: \(err)")
        }
        var ioCompletedNanos: UInt64 = 0
        if let operation = pending.storageOperation {
            // Event-gated commands cannot complete before this operation is
            // terminal, so this is an error check, not a successful-I/O host
            // wait. A failed read is surfaced after safe no-op kernels have
            // prevented incomplete slot bytes from being dereferenced.
            try operation.storage.wait()
            totalIOQueueNanos &+= operation.storage.submissionToStartNanos
            totalIoNanos &+= operation.storage.loadNanos
            totalMissIoNanos &+= operation.storage.loadNanos
            ioCompletedNanos = operation.storage.completedNanos
        }
        if let phase1HitCB = pending.phase1HitCB, let err = phase1HitCB.error {
            throw ModelError.commandBufferFailed(
                detail: "routed phase-1 hit command buffer: \(err)")
        }
        totalCb2Nanos &+= pending.encodeAndCommitNanos
        if deferTimings, !waitIfNeeded {
            deferredGPURecords.append(.routed(pending, ioCompletedNanos: ioCompletedNanos))
        } else {
            try recordRoutedCommandTimings(pending, ioCompletedNanos: ioCompletedNanos)
        }
    }

    /// The terms that read a command's GPU stamps, so the word wake can defer
    /// them until the driver has marked the commands complete.
    private func recordRoutedCommandTimings(_ pending: PendingRoutedCommand,
                                            ioCompletedNanos: UInt64) throws {
        if let specCB = pending.specCB {
            try waitForCompletion(specCB)
            recordKernelGPU(role: "moe_spec_routed", specCB)
        }
        if pending.storageOperation != nil {
            let gpuStartNanos = UInt64(max(0, pending.cb.gpuStartTime) * 1_000_000_000)
            if gpuStartNanos > ioCompletedNanos, ioCompletedNanos > 0 {
                totalFixupWakeNanos &+= gpuStartNanos - ioCompletedNanos
            }
            if let latest = pending.overlapCompletionClock?.latest(
                expected: pending.expectedOverlapCompletions),
               ioCompletedNanos > latest {
                totalExposedIoNanos &+= ioCompletedNanos - latest
            }
        }
        if let phase1HitCB = pending.phase1HitCB {
            recordKernelGPU(role: "moe_phase1_hit", phase1HitCB)
            if pending.hitCommitNanos > 0, phase1HitCB.kernelStartTime > 0 {
                let kernelStart = UInt64(phase1HitCB.kernelStartTime * 1_000_000_000)
                let gpuStart = UInt64(max(0, phase1HitCB.gpuStartTime) * 1_000_000_000)
                totalHitCommitToKernelNanos &+= kernelStart > pending.hitCommitNanos
                    ? kernelStart - pending.hitCommitNanos : 0
                totalHitKernelToGPUNanos &+= gpuStart > kernelStart ? gpuStart - kernelStart : 0
            }
        }
        recordKernelGPU(role: pending.kernelRole, pending.cb)
        if pending.routedCommitNanos > 0, pending.cb.kernelStartTime > 0 {
            let kernelStart = UInt64(pending.cb.kernelStartTime * 1_000_000_000)
            totalFixupCommitToKernelNanos &+= kernelStart > pending.routedCommitNanos
                ? kernelStart - pending.routedCommitNanos : 0
        }
    }


    private func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<slots.count { ptr[i] = slots[i] }
    }

    /// The shared-expert chain (or the h1Buf zero-fill when the arch has
    /// none), encoded at the head of the spec CB. It depends only on
    /// `routedX`, which the tail produces, so the command is committed before
    /// the router readback and the GPU runs it while the CPU waits for the
    /// routing; encoding it after the readback left a measured 7.88 ms/token
    /// of GPU idle in the `attn_tail_router -> shared_expert` transition.
    private func encodeSharedExpertWork(into cb: MTLCommandBuffer,
                                        layer L: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        guard cfg.hasSharedExpert else {
            // No shared expert (gpt-oss): the phase-2 reduce still seeds from
            // h1Buf, so pin it to zero in place of the dense MLP output.
            guard let blit = cb.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.fill(buffer: h1Buf,
                      range: 0..<(cfg.hiddenSize * MemoryLayout<Float16>.stride),
                      value: 0)
            blit.endEncoding()
            return
        }
        let sharedProj = sharedExpertProjections[L]
        // Serial encoder on purpose: a .concurrent encoder here — however
        // barriered — segfaults the AGX driver when a later encoder on this
        // CB encodes an indirect dispatch (macOS 26 / M4 HAL200,
        // insertIndirectTGOptKernel null deref).
        if let fused = shared.int4FusedDecode {
            guard let sharedEncoder = cb.makeComputeCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            defer { sharedEncoder.endEncoding() }
            try fused.encodeGateUp(encoder: sharedEncoder,
                                   x: routedX,
                                   gate: sharedProj.gate,
                                   up: sharedProj.up,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp)
            if cfg.sharedExpertGated {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                int8ScalarGate!.encode(encoder: sharedEncoder,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: routedX,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
            }
            try fused.encodeFusedDown(encoder: sharedEncoder,
                                      down: sharedProj.down,
                                      gateIn: denseScratchGate,
                                      upIn: denseScratchUp,
                                      y: h1Buf,
                                      scalarGate: cfg.sharedExpertGated
                                          ? sharedScalarGateBuf! : nil)
            return
        }
        // B1c stage 1: one encoder for the whole shared-expert chain — the
        // per-kernel encoders cost more span than the GEMVs they wrapped.
        guard let sharedEncoder = cb.makeComputeCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        defer { sharedEncoder.endEncoding() }
        try shared.encode(encoder: sharedEncoder,
                          x: routedX,
                          gate: sharedProj.gate,
                          up: sharedProj.up,
                          down: sharedProj.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate,
                          scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        if cfg.sharedExpertGated {
            // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
            let gateView = sharedProj.scalarGate!
            int8ScalarGate!.encode(encoder: sharedEncoder,
                                   weights: gateView.buffer,
                                   weightsOffset: Int(gateView.offset),
                                   scales: gateView.buffer,
                                   scalesOffset: Int(gateView.scaleOffset),
                                   biases: gateView.buffer,
                                   biasesOffset: Int(gateView.biasOffset),
                                   x: routedX,
                                   y: sharedScalarGateBuf!,
                                   m: 1, n: D)
            elementwise!.encodeSigmoidScalarMul(encoder: sharedEncoder,
                                                y: h1Buf,
                                                gate: sharedScalarGateBuf!,
                                                count: cfg.hiddenSize)
        }
    }

    private static func makeSpeculativeDispatchArguments(
        cfg: ArchConfig,
        device: MTLDevice
    ) throws -> MoE.SpeculativeDispatchArguments {
        guard let args = device.makeBuffer(length: MoE.specDispatchArgsLength,
                                           options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        args.label = "decode.specArgs"
        return MoE.SpeculativeDispatchArguments(
            arguments: args,
            phase1Threadgroups: MoE.specPhase1FullGrid(
                f: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts)),
            phase2Threadgroups: MoE.specPhase2FullGrid(
                d: UInt32(cfg.hiddenSize)),
            tailThreadgroups: MoE.specTailFullGrid(
                d: UInt32(cfg.hiddenSize),
                threadgroupWidth: Elementwise.residualAddThreadgroupWidth))
    }

    /// The speculative pool-addressed phase-1/phase-2 (v9), committed before
    /// the tail wait so it sizes itself from the classifier's indirect
    /// arguments: full grids on an all-hit layer, zero grids otherwise.
    private func encodeSpeculativeRouted(
        layer L: Int,
        residency: ExpertResidencyResources,
        arguments: MoE.SpeculativeDispatchArguments,
        completionClock: CommandCompletionClock?
    ) throws -> MTLCommandBuffer {
        let pool = residency.expertPool
        guard let cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        // Stage C: the shared-expert chain rides at the head of the spec CB —
        // same main-queue commit position as the old separate sharedCB, so
        // h1Buf ordering is unchanged, and the spec CB commits on every
        // layer, so miss layers still produce h1Buf for the fixup reduce.
        try encodeSharedExpertWork(into: cb, layer: L)
        completionClock?.track(cb)
        let offsets = try model.routedExpertOffsets(layer: L)
        try moe.encodeSpecPhase1U16Load(
            commandBuffer: cb,
            expertPool: pool,
            poolSlotStride: residency.poolSlotStride,
            resolvedSlots: residencyResolvedSlots,
            routedOffsets: offsets,
            x: routedX,
            acts: moeActs,
            d: UInt32(cfg.hiddenSize),
            f: UInt32(cfg.moeIntermediateSize),
            topK: UInt32(cfg.topKExperts),
            indirectArguments: arguments.arguments)
        try moe.encodeSpecPhase2Reduce(
            commandBuffer: cb,
            expertPool: pool,
            poolSlotStride: residency.poolSlotStride,
            resolvedSlots: residencyResolvedSlots,
            routedOffsets: offsets,
            acts: moeActs,
            routingWeights: outWeights,
            residual: h1Buf,
            y: h2Buf,
            d: UInt32(cfg.hiddenSize),
            f: UInt32(cfg.moeIntermediateSize),
            topK: UInt32(cfg.topKExperts),
            indirectArguments: arguments.arguments)
        try elementwise!.encodeResidualAddIndirect(
            commandBuffer: cb,
            hidden: hidden,
            delta: h2Buf,
            count: cfg.hiddenSize,
            indirectArguments: arguments.arguments,
            indirectOffset: MoE.specTailArgsOffset)
        return cb
    }

    /// Routed-expert stage of one decode layer: top-k readback, expert fetch,
    /// phase-1/phase-2 encode, and the deferred completion hand-off.
    ///
    /// lint:allow-long one pipeline whose phases share the fetch plan, the
    /// argument buffer and the slot scratch; the layer trace at the end reports
    /// timings from every phase, so splitting it would mean threading those
    /// back out purely to shorten a function.
    private func encodeDecodeRoutedMoE(
        layer L: Int,
        position: Int,
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
        specCB: MTLCommandBuffer,
        overlapCompletionClock: CommandCompletionClock?,
        pending pendingRoutedCommand: inout PendingRoutedCommand?,
        bodyStart tBodyStart: UInt64,
        cb1Start tCb1Start: UInt64,
        waitMark tWait: UInt64,
        waitNanos: UInt64,
        previousRoutedMicros prevRoutedUs: Double,
        hostReadback: RouterHostReadback?,
        predictedNextLayer: [Int]
    ) async throws {
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let readbackStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        decodeExpertsScratch.removeAll(keepingCapacity: true)
        decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
        if let hostReadback {
            for id in hostReadback.expertIDs {
                decodeExpertsScratch.append(min(Int(id), cfg.numExperts - 1))
            }
        } else {
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                decodeExpertsScratch.append(min(Int(idxPtr[i]), cfg.numExperts - 1))
            }
        }
        totalRouterReadbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readbackStarted
        if runnerStatsEnabled {
            if totalRankWeightMass.count != cfg.topKExperts {
                totalRankWeightMass = [Double](repeating: 0, count: cfg.topKExperts)
            }
            if let hostReadback {
                for i in 0..<cfg.topKExperts {
                    totalRankWeightMass[i] += Double(Float16(bitPattern: hostReadback.weightBits[i]))
                }
            } else {
                let wPtr = outWeights.contents().bindMemory(to: Float16.self,
                                                            capacity: cfg.topKExperts)
                for i in 0..<cfg.topKExperts {
                    totalRankWeightMass[i] += Double(wPtr[i])
                }
            }
            totalRankWeightLayers &+= 1
        }
        let experts = decodeExpertsScratch
        recordRouteTrace(layer: L, position: position, experts: experts)

        let routedOffsets = try model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let canUsePlannedFetch = cfg.topKExperts <= MoE.maxStreamedExperts
        let residentBeforePlan = prefetchTraceFD >= 0
            ? try model.routedExpertResidentIDs(layer: L) : []
        let landedPrefetches = predictivePrefetch.readyCells(
            layer: L, experts: experts, joinNanos: Self.prefetchJoinNanos)
        // The classifier's view, taken before this plan: a landing it missed
        // is swapped in all the same and reported adopted for the fixup.
        let gpuMissedExperts: Set<Int>? = hostReadback.map { readback in
            Set(readback.missPositions.map { Int(readback.expertIDs[Int($0)]) })
        }
        let cachePlanStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedFetch: RoutedExpertFetchPlan?
        do {
            plannedFetch = canUsePlannedFetch
                ? try model.planRoutedExperts(layer: L, experts: experts,
                                              gpuMissedExperts: gpuMissedExperts,
                                              leasedLandings: Set(landedPrefetches.keys))
                : nil
        } catch {
            predictivePrefetch.unlease(layer: L, experts: Set(landedPrefetches.keys))
            throw error
        }
        totalCachePlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cachePlanStarted
        let landed = Array(landedPrefetches.keys)
        recordPrefetchRace(landed: landed, layer: L, tailCB: tailCB)
        if !landed.isEmpty {
            let swapped = plannedFetch?.freedCells.count ?? 0
            let adoptedCount = plannedFetch?.adopted.count ?? 0
            // A landed hit is one the classifier counted; without one none is.
            if gpuMissedExperts != nil {
                totalPrefetchLandedHits &+= UInt64(max(0, swapped - adoptedCount))
            }
            predictivePrefetch.consume(layer: L, experts: Set(landed),
                                       freedCells: plannedFetch?.freedCells ?? [:])
        }
        let missesForTrace = plannedFetch.map { plan in
            plan.misses.map { experts[$0] }
        } ?? experts
        recordPrefetchTrace(layer: L, position: position, experts: experts,
                            misses: missesForTrace, resident: residentBeforePlan,
                            nextLayerPrediction: predictedNextLayer)
        let pinStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let expertLease = try plannedFetch.map { try model.pinRoutedExperts(for: $0) }
        totalRoutedPinNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - pinStarted
        // v4.2 Phase B: once slots and generations are reserved and pinned,
        // submit real storage immediately. Hit partitioning, argument binding,
        // and command encoding below now overlap the reader queue.
        let submitStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedLoad = try plannedFetch.map {
            try model.beginFetchRoutedExperts(plan: $0, eventDriven: !$0.misses.isEmpty)
        }
        totalRoutedSubmitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - submitStarted
        if let plannedLoad, !plannedLoad.plan.misses.isEmpty {
            predictivePrefetch.noteDemandSubmission()
        }
        var transferredExpertLease = false
        var phase1HitCB: MTLCommandBuffer?
        defer {
            if !transferredExpertLease {
                // A thrown fetch/encode must not make a hit slot evictable
                // while its already-committed phase-1 command is still reading.
                if let phase1HitCB, let expertLease {
                    try? waitForCompletion(phase1HitCB)
                    expertLease.release()
                } else {
                    expertLease?.release()
                }
            }
        }
        var phase1HitSplitArgBuf: MTLBuffer?
        decodeHitSplitRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeHitSplitRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        decodeHitSlotsScratch.removeAll(keepingCapacity: true)
        decodeMissSlotsScratch.removeAll(keepingCapacity: true)

        if let plan = plannedFetch {
            DecodeExpertPartition.populate(
                topK: cfg.topKExperts,
                missIndices: plan.misses,
                adoptedIndices: plan.adopted,
                hits: &decodeHitSlotsScratch,
                misses: &decodeMissSlotsScratch)
            // The spec command computed the GPU's partition, not the plan's;
            // the fixup must finish exactly what it skipped. CPU planning is
            // still the eviction authority: a prefetch the planner adopted is
            // a miss there and a resident hit here, and any other mismatch
            // means metadata publication raced or became stale. Fail closed
            // rather than executing a different partition.
            guard let hostReadback else {
                throw ModelError.internalInconsistency(
                    detail: "GPU residency classification ran without its host readback")
            }
            guard hostReadback.missPositions.elementsEqual(decodeMissSlotsScratch) else {
                throw ModelError.internalInconsistency(
                    detail: "GPU residency classification disagrees with cache plan")
            }
        }
        // Capture the populated arrays. Capturing them before `populate` made
        // empty value-semantic snapshots and silently disabled hit/fixup.
        let phase1HitSlots = decodeHitSlotsScratch
        let phase1MissSlots = decodeMissSlotsScratch
        func encodeRoutedPhase1Subset(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            activeSlots: MTLBuffer,
            activeSlotIndices: [UInt32],
            activeCount: UInt32,
            ioStatus: MTLBuffer? = nil,
            ioStatusOffset: Int = 0
        ) throws {
            try moe.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: cb,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                activeSlots: activeSlots,
                activeSlotIndices: activeSlotIndices,
                activeCount: activeCount,
                d: D,
                f: FmoE,
                topK: topK,
                ioStatus: ioStatus,
                ioStatusOffset: ioStatusOffset)
        }

        let fixupHasMisses = !phase1MissSlots.isEmpty
            || plannedFetch.map { !$0.misses.isEmpty } == true
        var argBufStartedForEncode: UInt64 = 0
        if let plan = plannedFetch,
           plan.hits > 0,
           !phase1HitSlots.isEmpty,
           fixupHasMisses {
            let argBufStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let plannedBlobs = try model.routedExpertBuffers(for: plan)
            for blob in plannedBlobs {
                decodeHitSplitRoutedBufsScratch.append(blob.buffer)
                decodeHitSplitRoutedOffsetsScratch.append(Int(blob.offset))
            }
            phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: decodeHitSplitRoutedBufsScratch,
                topK: topK,
                routedBufferOffsets: decodeHitSplitRoutedOffsetsScratch)
            argBufStartedForEncode = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            totalHitSplitArgBufNanos &+= argBufStartedForEncode - argBufStarted
            if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, fixupHasMisses {
                writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                guard let cb = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                try encodeRoutedPhase1Subset(
                    cb,
                    argBuf: argBuf,
                    routedBufs: decodeHitSplitRoutedBufsScratch,
                    activeSlots: moeHitActiveSlots,
                    activeSlotIndices: phase1HitSlots,
                    activeCount: UInt32(phase1HitSlots.count))
                phase1HitCB = cb
            }
        }

        var hitCommitNanos: UInt64 = 0
        if let cb = phase1HitCB {
            overlapCompletionClock?.track(cb)
            hitCommitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            cb.commit()
            totalHitSplitEncodeNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - argBufStartedForEncode
        }
        let missCount = plannedFetch?.misses.count ?? experts.count
        let fixupMissCount = plannedFetch == nil
            ? experts.count : max(missCount, phase1MissSlots.count)
        let completionClock = missCount > 0 ? overlapCompletionClock : nil
        let expectedOverlapCompletions = phase1HitCB == nil ? 1 : 2

        // Routed-expert pread — overlaps the shared MLP GPU work above.
        let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        let eventLoad = plannedLoad.flatMap { operation -> RoutedExpertLoadOperation? in
            operation.storage.completionToken == nil ? nil : operation
        }
        if let eventLoad {
            // Slot resources and offsets are known from the reservation. Their
            // bytes are consumed only after the shared-event wait encoded
            // below, so no successful completion has to resume this task.
            blobs = try model.routedExpertBuffers(for: eventLoad.plan)
            totalExpertIOHostWaitsAvoided &+= 1
        } else if let plannedFetch, plannedFetch.misses.isEmpty {
            // An all-hit layer has already pinned its current slot generations.
            // Do not manufacture a completed storage operation and an async
            // continuation only to retrieve the same cache views.
            blobs = try model.routedExpertBuffers(for: plannedFetch)
            totalExpertIOHostWaitsAvoided &+= 1
        } else {
            guard plannedFetch == nil else {
                throw ModelError.internalInconsistency(
                    detail: "a demand batch with misses carried no completion token")
            }
            blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        let layerIo = eventLoad == nil
            ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart : 0
        if eventLoad == nil { totalIoNanos &+= layerIo }
        if missCount > 0 && eventLoad == nil {
            totalMissIoNanos &+= layerIo
            if let latest = completionClock?.latest(expected: expectedOverlapCompletions) {
                let overlapEnd = max(tIoStart, latest)
                if overlapEnd < tIoStart + layerIo {
                    totalExposedIoNanos &+= tIoStart + layerIo - overlapEnd
                }
            }
        }
        schedulePredictivePrefetch(layer: L, predicted: predictedNextLayer,
                                   demand: plannedLoad?.storage)
        decodeRoutedBufsScratch.removeAll(keepingCapacity: true)
        decodeRoutedOffsetsScratch.removeAll(keepingCapacity: true)
        for blob in blobs {
            decodeRoutedBufsScratch.append(blob.buffer)
            decodeRoutedOffsetsScratch.append(Int(blob.offset))
        }
        let routedBufs = decodeRoutedBufsScratch
        let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // v9 S3a: on an all-hit layer in speculative mode, the already-running
        // spec CB wrote moeActs/h2Buf/hidden itself — it IS the layer's routed
        // command; nothing classic gets encoded.
        if plannedFetch != nil,
           fixupMissCount == 0 {
            guard pendingRoutedCommand == nil else {
                throw ModelError.internalInconsistency(
                    detail: "routed command-buffer pipeline not drained before queuing the next layer")
            }
            pendingRoutedCommand = PendingRoutedCommand(
                cb: specCB,
                phase1HitCB: nil,
                specCB: nil,
                expertLease: expertLease,
                storageOperation: eventLoad,
                overlapCompletionClock: eventLoad == nil ? nil : overlapCompletionClock,
                expectedOverlapCompletions: expectedOverlapCompletions,
                hitCommitNanos: 0,
                routedCommitNanos: 0,
                kernelRole: "moe_spec_routed",
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            transferredExpertLease = true
            totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
            return
        }
        let hitSplitFixup = phase1HitCB != nil && !phase1MissSlots.isEmpty
        guard pendingRoutedCommand == nil else {
            // The pipeline drains the previous layer's routed CB before
            // queuing the next, so this is a logic error, not a user
            // condition — but it must fail the generation, not trap.
            throw ModelError.internalInconsistency(
                detail: "routed command-buffer pipeline not drained before queuing the next layer")
        }
        let fixupBuildStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let (routedCB, routedCommitNanos) = try buildAndCommitMissFixupCommand(
            eventLoad: eventLoad,
            phase1HitCB: phase1HitCB,
            phase1HitSplitArgBuf: phase1HitSplitArgBuf,
            phase1MissSlots: phase1MissSlots,
            routedBufs: routedBufs,
            routedOffsets: routedOffsets,
            topK: topK, d: D, f: FmoE)
        totalFixupBuildNanos &+= routedCommitNanos - fixupBuildStarted
        pendingRoutedCommand = PendingRoutedCommand(
            cb: routedCB,
            phase1HitCB: phase1HitCB,
            specCB: specCB,
            expertLease: expertLease,
            storageOperation: eventLoad,
            overlapCompletionClock: eventLoad == nil ? nil : overlapCompletionClock,
            expectedOverlapCompletions: expectedOverlapCompletions,
            hitCommitNanos: hitCommitNanos,
            routedCommitNanos: routedCommitNanos,
            kernelRole: !hitSplitFixup
                ? "moe_phase1_2_routed"
                : missCount == 0
                    ? "moe_phase1_miss_fixup_phase2_adopted"
                    : "moe_phase1_miss_fixup_phase2",
            encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
        transferredExpertLease = true
        totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tBodyStart
        if layerTraceEnabled,
           position < 3 || position % 16 == 0 {
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Under the word wake the stamps may not exist yet.
            func gpuMicros(_ cb: MTLCommandBuffer) -> String {
                cb.gpuEndTime > 0 ? String(Int((cb.gpuEndTime - cb.gpuStartTime) * 1_000_000)) : "pending"
            }
            print("Shrike layer pos=\(position) L=\(L) "
                + "body_us=\((now - tBodyStart) / 1000) "
                + "wait_us=\(waitNanos / 1000) io_us=\(layerIo / 1000) "
                + "cb1_us=\((tWait - tCb1Start) / 1000) "
                + "cb2_us=\((now - tCb2Start) / 1000) "
                + "gpu_attn_us=\(gpuMicros(attnCB)) gpu_tail_us=\(gpuMicros(tailCB)) "
                + "gpu_routed_us=\(prevRoutedUs > 0 ? String(Int(prevRoutedUs)) : "pending")")
        }
    }
}
