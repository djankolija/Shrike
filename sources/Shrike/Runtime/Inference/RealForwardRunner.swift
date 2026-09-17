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
public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, BoundaryLogitProducer, @unchecked Sendable {
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

    public var prefillDescription: String {
        Self.prefillDescription(routerBits: prefillRouter.weightBits,
                                poolResidencyUnavailableReason: poolResidencyUnavailableReason,
                                prefetchTrace: prefetchTraceFD >= 0,
                                expertCache: Self.expertCacheDescription(model.streamingMode))
    }

    static func expertCacheDescription(_ mode: ExpertStreamingMode) -> String {
        switch mode {
        case .pread(let slotCount, let perLayer, let policy):
            let routed = perLayer?.filter { $0 > 0 }
            let slots = routed.flatMap { counts -> String? in
                guard let low = counts.min(), let high = counts.max() else { return nil }
                return "\(low)..\(high)"
            } ?? "uniform:\(slotCount)"
            return "expert_slots=\(slots) policy=\(policy.label)"
        }
    }

    static func prefillDescription(routerBits: Int,
                                   poolResidencyUnavailableReason: String?,
                                   prefetchTrace: Bool,
                                   expertCache: String? = nil) -> String {
        var description = "prefill_router_bits=\(routerBits)"
        if let poolResidencyUnavailableReason {
            description += " prefill_pool_residency=unavailable reason=\(poolResidencyUnavailableReason)"
        }
        if prefetchTrace {
            description += " prefetch_trace=on"
        }
        if let expertCache {
            description += " \(expertCache)"
        }
        return description
    }

    /// The tail tile in force for a GEMM instance: 32 unless the kernel has
    /// no 32-row instantiation for this model's K's.
    private func tailTileInForce(for mpp: MPPPrefillInt4QMM) -> Int {
        guard mpp.groupedRowTile32Available(forK: cfg.hiddenSize),
              mpp.groupedRowTile32Available(forK: cfg.moeIntermediateSize) else { return 0 }
        return Self.prefillTailTile
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
    private var agreedCells: MTLBuffer { residencyReadback.agreedCells } // [numLayers * topK] UInt32
    private var residencyHitCount: MTLBuffer { residencyReadback.hitCount }
    private var residencyHitPositions: MTLBuffer { residencyReadback.hitPositions }
    private var residencyMissCount: MTLBuffer { residencyReadback.missCount }
    private var residencyMissPositions: MTLBuffer { residencyReadback.missPositions }
    private var residencyMissExperts: MTLBuffer { residencyReadback.missExperts }
    private var residencyResolvedSlots: MTLBuffer { residencyReadback.resolvedSlots }
    private var routerHostReadback: MTLBuffer { residencyReadback.hostReadback }
    private var greedyTokenBuf: MTLBuffer { decodeScratch.greedyTokenBuf } // 4 B UInt32 fused-head output
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private var qPackedScratch: MTLBuffer? { decodeScratch.qPackedScratch } // [2 * N_HEADS * head_dim] packed [q ; gate]
    private var attnGateScratch: MTLBuffer? { decodeScratch.attnGateScratch } // [N_HEADS * head_dim]
    /// The committed token's command, recorded at the next produce.
    private var runningToken: TokenCommand?
    /// The next token's command, its layers encoded a layer per word during
    /// the running one, committed on the boundary word after the stop check.
    private var heldToken: TokenCommand?
    private var boundaryTokenWord: MTLBuffer?
    private var wordClock: DecodeWordClock
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
    /// The values reserved for encoded layers and not yet handed to a batch
    /// or published: what the drain publishes on the way out of a pass.
    private var armedAgreedTokens: [ExpertIOCompletionToken] = []

    /// Two routed tiles pending, one per command buffer, the next tile's
    /// fetch begun before the current one is awaited (v12 P16, v13 T3).
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig(
        maxPendingDepth: 2,
        tilesPerCommandBuffer: 1,
        fetchLookahead: 1)
    /// The expert pools held in a queue residency set (v12 P12), nil with
    /// the reason when the device refused one.
    private let poolResidency: ExpertPoolResidency?
    private let poolResidencyUnavailableReason: String?
    /// Each expert block's remainder of ≤ 32 rows packed into a 32-row tile
    /// instead of a padded 64-row one.
    private static let prefillTailTile = 32
    /// The row-count floor below which the attention, projection and
    /// shared-expert matrix kernels fall back to their scalar paths (v13 T3:
    /// the 21-row follow-up turn runs on the matrix kernels).
    private static let prefillMatrixMinRows = 16

    private static func makePoolResidency(context: MetalContext)
        -> (holder: ExpertPoolResidency?, unavailableReason: String?) {
        do {
            return (try ExpertPoolResidency(device: context.device, queue: context.queue), nil)
        } catch {
            return (nil, "\(error)")
        }
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
    private var routerReadbackTag: UInt32 = 0
    /// Bookkeeping that needs a command's GPU stamps, which the word wake reads before they exist.

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
        self.wordClock = DecodeWordClock(layers: cfg.numLayers)
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
        if cfg.family == .qwen36, !runtimeConfiguration.attentionFallbackAllowed {
            guard Attention.streamServesShape(headDim: cfg.fullHeadDim, numQHeads: cfg.numHeads,
                                              numKVHeads: cfg.numFullKVHeads) else {
                throw ModelError.unsupportedArchitecture(detail:
                    "the streaming attention scan (v19) serves head dim 256 and eight query "
                    + "heads per KV head; this model has head dim \(cfg.fullHeadDim) and "
                    + "\(cfg.numHeads) query heads over \(cfg.numFullKVHeads) KV heads, and "
                    + "the fallback to the shared kernel is disabled for served models")
            }
            if runtimeConfiguration.kvCachePrecision != .int8 {
                print("Shrike attention: the streaming scan serves int8 KV rows; "
                      + "\(runtimeConfiguration.kvCachePrecision.label) rows run the v11 shared kernel")
            }
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
                                     partialLoopVariant: .stream),
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
        /// One row of top-k cells per layer, host-written at the word: the
        /// miss at position p reads into `agreedCells[p]`, a hit's the sentinel.
        let agreedCells: MTLBuffer
        let hitCount: MTLBuffer
        let hitPositions: MTLBuffer
        let missCount: MTLBuffer
        let missPositions: MTLBuffer
        let missExperts: MTLBuffer
        let resolvedSlots: MTLBuffer
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
        let agreedCells = try buf(cfg.numLayers * topK, u32, label: "decode.agreedCells")
        memset(agreedCells.contents(), 0xff, agreedCells.length)
        return ResidencyReadbackBuffers(
            agreedCells: agreedCells,
            hitCount: try buf(1, u32, label: "decode.residencyHitCount"),
            hitPositions: try buf(topK, u32, label: "decode.residencyHitPositions"),
            missCount: try buf(1, u32, label: "decode.residencyMissCount"),
            missPositions: try buf(topK, u32, label: "decode.residencyMissPositions"),
            missExperts: try buf(topK, u32, label: "decode.residencyMissExperts"),
            resolvedSlots: try buf(topK, u32, label: "decode.residencyResolvedSlots"),
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
        discardBoundaryState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        discardBoundaryState()
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
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    // Overlap-analysis counters (SHRIKE_RUNNER_STATS): the per-layer wall spent
    // waiting on the attention+router command buffer (covers the previous
    // layer's routed CB plus this layer's cb1 on the GPU) and the per-layer
    // loop-body wall. body = cb1 + wait + readback/plan + io + cb2.
    public private(set) var totalWaitNanos: UInt64 = 0
    public private(set) var totalBodyNanos: UInt64 = 0
    public private(set) var totalMissIoNanos: UInt64 = 0
    /// Routed layers whose route missed an expert, so the fixup computed.
    public private(set) var totalHitFixupLayers: UInt64 = 0
    /// Misses given a pool victim on the path because the ring had no free
    /// cell for them (v20 T3.1).
    public private(set) var totalAgreedOverflow: UInt64 = 0
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
    public private(set) var totalRoutedSubmitNanos: UInt64 = 0
    public private(set) var totalRouterWakeFallbacks: UInt64 = 0
    public var prefetchStatistics: ExpertPrefetchStatistics { predictivePrefetch.statistics }
    public private(set) var totalBoundaryWakeFallbacks: UInt64 = 0
    public private(set) var totalIOQueueNanos: UInt64 = 0
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
    private var pendingProbeDumpPosition: Int?

    static let prefetchJoinNanos: UInt64 = 400_000

    public func resetKernelGPUTimings() {
        kernelGPUTimings.removeAll(keepingCapacity: true)
        wordClock = DecodeWordClock(layers: cfg.numLayers)
    }

    /// The per-layer clock of the token under one command (v20 T3.2), for the
    /// kernel stats; nil before a token ran.
    public func wordClockLine() -> String? {
        wordClock.tokens > 0 ? wordClock.line() : nil
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
        dumpPendingProbeRankings()
        guard routeTraceFD >= 0 else { return }
        writeRouteTraceLine(Self.formatRouteTraceLine(cachedTokens: cachedTokens,
                                                       promptTokens: promptTokens))
    }

    /// The loop reports a decode position's input id, since on the boundary
    /// path the runner never sees it on the host.
    func recordRouteTraceToken(position: Int, id: Int32) {
        guard routeTraceFD >= 0 else { return }
        writeRouteTraceLine(Self.formatRouteTraceTokenLine(position: position, id: id))
    }

    private func recordRouteTracePrefillRows(layer: Int, startPosition: Int,
                                             rowCount: Int, ids: [UInt32]) {
        guard routeTraceFD >= 0 else { return }
        var lines = ""
        for row in 0..<rowCount {
            let start = row * cfg.topKExperts
            let experts = ids[start..<start + cfg.topKExperts].map { Int($0) }
            lines += Self.formatRouteTracePrefillRowLine(position: startPosition + row,
                                                         layer: layer, experts: experts)
        }
        writeRouteTraceLine(lines)
    }

    static func formatRouteTraceTokenLine(position: Int, id: Int32) -> String {
        "t \(position) \(id)\n"
    }

    static func formatRouteTracePrefillRowLine(position: Int, layer: Int, experts: [Int]) -> String {
        var line = "q \(position) \(layer)"
        for expert in experts { line += " \(expert)" }
        line += "\n"
        return line
    }

    /// The select kernel's order: the logit, or its sigmoid, plus the bias,
    /// descending, the lower index first on a tie.
    static func probeRanking(logits: UnsafePointer<Float>, bias: UnsafePointer<UInt16>?,
                             sigmoid: Bool, count: Int, width: Int) -> [Int] {
        var scored: [(score: Float, index: Int)] = []
        scored.reserveCapacity(count)
        for expert in 0..<count {
            let raw = logits[expert]
            let base = sigmoid ? 1 / (1 + exp(-raw)) : raw
            let shift = bias.map { Float(bitPattern: UInt32($0[expert]) << 16) } ?? 0
            scored.append((base + shift, expert))
        }
        scored.sort { $0.score > $1.score || ($0.score == $1.score && $0.index < $1.index) }
        return scored.prefix(width).map(\.index)
    }

    static let probeRankingWidth = 32
    static let probeDistances = 3

    /// Distance one is the pair's own slot; the capture's further distances
    /// take the banks after it.
    static func probeSlot(layer: Int, distance: Int, bank: Int, numLayers: Int) -> Int {
        (distance - 1) * 2 * numLayers + bank * numLayers + layer
    }

    /// The routers two and three layers ahead on this layer's state, scores
    /// only, into their probe slots; the capture reads them with the pair's.
    private func encodeProbeDistanceScores(encoder: MTLComputeCommandEncoder, layer L: Int,
                                           hidden: MTLBuffer, probeBank: Int, d D: UInt32) throws {
        for distance in 2...Self.probeDistances where L + distance < cfg.numLayers {
            let slot = Self.probeSlot(layer: L, distance: distance, bank: probeBank,
                                      numLayers: cfg.numLayers)
            guard slot < MoE.probeLogitsSlots else { return }
            let target = L + distance
            let router = try model.router(layer: target)
            moe.encodeRouterScores(
                encoder: encoder,
                weights: router.buffer, weightsOffset: Int(router.offset),
                scales: router.buffer, scalesOffset: Int(router.scaleOffset),
                biases: router.buffer, biasesOffset: Int(router.biasOffset),
                hidden: hidden,
                effectiveScale: effectiveScaleBuffers[target],
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts),
                probeSlot: slot)
        }
    }

    /// Runs at the next pass's entry or the next request's start, after the
    /// pass's commands have completed, so the slots are coherent.
    private func dumpPendingProbeRankings() {
        guard prefetchTraceFD >= 0, let position = pendingProbeDumpPosition else { return }
        pendingProbeDumpPosition = nil
        let stride = MoE.probeLogitsStride
        let base = moe.probeLogitsBuffer.contents().bindMemory(
            to: Float.self, capacity: stride * MoE.probeLogitsSlots)
        var lines = ""
        for layer in 0..<(cfg.numLayers - 1) {
            var fields = "\"position\":\(position),\"layer\":\(layer)"
            for distance in 1...Self.probeDistances where layer + distance < cfg.numLayers {
                let slot = Self.probeSlot(layer: layer, distance: distance, bank: position & 1,
                                          numLayers: cfg.numLayers)
                guard slot < MoE.probeLogitsSlots else { continue }
                let biasEntry = routerLogitBias[layer + distance]
                var biasPointer: UnsafePointer<UInt16>?
                if biasEntry.buffer.storageMode == .shared {
                    biasPointer = UnsafePointer(biasEntry.buffer.contents()
                        .advanced(by: biasEntry.offset)
                        .bindMemory(to: UInt16.self, capacity: cfg.numExperts))
                }
                let ranking = Self.probeRanking(logits: base.advanced(by: slot * stride),
                                                bias: biasPointer, sigmoid: moe.selectsOnSigmoid,
                                                count: cfg.numExperts, width: Self.probeRankingWidth)
                let key = distance == 1 ? "probe_ranking" : "probe_ranking_d\(distance)"
                fields += ",\"\(key)\":\(ranking)"
            }
            lines += "{\(fields)}\n"
        }
        writeTraceLine(lines, to: prefetchTraceFD)
    }

    private func writeTraceLine(_ line: String, to fd: Int32) {
        let bytes = Array(line.utf8)
        var written = 0
        while written < bytes.count {
            let count = bytes.withUnsafeBytes { raw -> Int in
                write(fd, raw.baseAddress!.advanced(by: written), bytes.count - written)
            }
            if count <= 0 { return }
            written += count
        }
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
        discardBoundaryState()
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func produce(token: Int32?, position: Int, into logits: MTLBuffer,
                        tokenWord: MTLBuffer,
                        sample: @escaping (MTLComputeCommandEncoder) throws -> Void) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .logits,
                               boundaryWord: tokenWord,
                               sample: sample)
    }

    /// The token word's wake: the sampler writes it before the command's
    /// embed and completion; after a second the token's command is waited on
    /// with the deadline, so a wait the drain missed ends loudly.
    public func awaitBoundaryToken() throws -> Int32 {
        guard let token = runningToken, let word = boundaryTokenWord else {
            throw ModelError.internalInconsistency(
                detail: "no boundary command is pending a token")
        }
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 1_000_000_000
        var spins = 0
        while clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline {
            let value = word.contents().load(as: UInt32.self)
            if value != Self.boundaryTokenSentinel {
                wordClock.boundary(at: clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
                return Int32(bitPattern: value)
            }
            spins &+= 1
            if spins % 256 == 0, token.cb.status == .error {
                throw ModelError.commandBufferFailed(
                    detail: "the boundary: \(Self.describeCommandBufferError(token.cb.error))")
            }
        }
        totalBoundaryWakeFallbacks &+= 1
        try Self.awaitCompletion(of: token.cb, deadlineNanos: Self.commandDeadlineNanos,
                                 naming: "the boundary word")
        let value = word.contents().load(as: UInt32.self)
        guard value != Self.boundaryTokenSentinel else {
            throw ModelError.internalInconsistency(
                detail: "the token's command completed without writing its token word")
        }
        wordClock.boundary(at: clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
        return Int32(bitPattern: value)
    }

    static let boundaryTokenSentinel: UInt32 = 0xFFFF_FFFF
    static let commandDeadlineNanos: UInt64 = 10_000_000_000

    private func discardBoundaryState() {
        heldToken = nil
        boundaryTokenWord = nil
        drainArmedAgreedTokens()
        if let running = runningToken {
            runningToken = nil
            try? Self.awaitCompletion(of: running.cb, deadlineNanos: Self.commandDeadlineNanos,
                                      naming: "the previous token")
        }
    }

    private func takeHeldToken(position: Int) throws -> TokenCommand {
        guard let held = heldToken, held.position == position else {
            throw ModelError.internalInconsistency(
                detail: "a continued pass at \(position) needs the command the previous boundary encoded ahead")
        }
        heldToken = nil
        return held
    }

    /// The previous token's command, complete once its boundary word landed
    /// and the next token is committed behind it: its error surfaced, its GPU
    /// span recorded under the `token` role.
    private func finishToken(_ token: TokenCommand?) throws {
        guard let token else { return }
        try Self.awaitCompletion(of: token.cb, deadlineNanos: Self.commandDeadlineNanos,
                                 naming: "token \(token.position)")
        recordKernelGPU(role: "token", token.cb)
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

    /// The orchestrator for one prefill chunk: scratch setup, the per-layer dispatch, and the head.
    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        try validatePrefillChunk(tokens: tokens, startPosition: startPosition,
                                 scratch: scratch, config: config)


        let layerViews = try makeLayerPrefillViews()

        let tokenBuffer = try makePrefillTokenBuffer(tokens: tokens)
        let D = cfg.hiddenSize
        let eps: Float = cfg.rmsNormEps
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = try model.embedding()


        var cb = try encodePrefillChunkEmbed(startPosition: startPosition,
                                             embedding: emb, tokenBuffer: tokenBuffer,
                                             scratch: scratch, tokenCount: t,
                                             hiddenSize: D, outScale: embedOutScale)

        let aneChunk = probeANEPrefillChunk(startPosition: startPosition,
                                            tokenCount: tokens.count, config: config)

        for L in 0..<cfg.numLayers {
            try Task.checkCancellation()
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
                    startPosition: startPosition)
            }
        }

        try finishPrefillChunk(writeFinalHead: writeFinalHead, logits: logits,
                               scratch: scratch, tokenCount: t, hiddenSize: D,
                               rmsEps: eps, outputMode: outputMode,
                               aneChunk: aneChunk, startPosition: startPosition)
    }

    private func validatePrefillChunk(tokens: ArraySlice<Int32>, startPosition: Int,
                                      scratch: PrefillChunkScratchBuffers,
                                      config: PrefillRuntimeConfig) throws {
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
    }

    private func makePrefillTokenBuffer(tokens: ArraySlice<Int32>) throws -> MTLBuffer {
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
        return tokenBuffer
    }

    private func encodePrefillChunkEmbed(startPosition: Int, embedding emb: TensorView,
                                         tokenBuffer: MTLBuffer,
                                         scratch: PrefillChunkScratchBuffers,
                                         tokenCount t: Int, hiddenSize D: Int,
                                         outScale embedOutScale: Float) throws -> MTLCommandBuffer {
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: t)

        guard let cb = ctx.queue.makeCommandBuffer() else {
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
        return cb
    }

    private func probeANEPrefillChunk(startPosition: Int, tokenCount: Int,
                                      config: PrefillRuntimeConfig) -> ANEPrefillAttention? {
        // Track A: whether this chunk's full-attention layers run on the ANE.
        // Non-4096 chunk configs and prompts beyond the sidecar's history
        // variants stay on the GPU; continuity is enforced inside
        // eligibleChunk so a fallback mid-prompt sticks for the rest of the
        // request.
        guard let ane = anePrefill,
              ane.eligibleChunk(startPosition: startPosition,
                                tokenCount: tokenCount,
                                configChunkTokens: config.chunkTokens)
        else { return nil }
        return ane
    }

    private func finishPrefillChunk(writeFinalHead: Bool, logits: MTLBuffer,
                                    scratch: PrefillChunkScratchBuffers,
                                    tokenCount t: Int, hiddenSize D: Int, rmsEps eps: Float,
                                    outputMode: PrefillOutputMode,
                                    aneChunk: ANEPrefillAttention?, startPosition: Int) throws {
        if writeFinalHead {
            try encodeFinalHead(logits: logits, scratch: scratch,
                                tokenCount: t, hiddenSize: D, rmsEps: eps,
                                outputMode: outputMode)
        }

        aneChunk?.finishChunk(startPosition: startPosition,
                              tokenCount: t)
        kv?.advance(by: t)
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

    /// One routed decode layer's command buffers, encoded but not committed.
    /// The next layer is encoded while the GPU runs the current one, so the
    /// post-readback critical path is commits only.
    /// A routed layer inside the token's command: the tag its classifier
    /// stamps on the host readback and the value its fixup waits on.
    private struct TokenLayer {
        let layer: Int
        let readbackTag: UInt32
        let agreedToken: ExpertIOCompletionToken
    }

    /// One token's command (v20 T3.2): every layer and the boundary as
    /// encoders of one command buffer, the routed layers listed in order for
    /// the host's word loop.
    private final class TokenCommand {
        let position: Int
        let cb: MTLCommandBuffer
        var layers: [TokenLayer] = []
        var encodedLayers = 0

        init(position: Int, cb: MTLCommandBuffer) {
            self.position = position
            self.cb = cb
        }
    }

    /// The token's command asks Metal for its encoders' execution status, so
    /// a fault names the encoder; the option measured free on the mini's four
    /// shapes (v20 T3.2's arms), so it is always on.
    private func makeTokenCommand(position: Int) throws -> TokenCommand {
        let descriptor = MTLCommandBufferDescriptor()
        descriptor.errorOptions = .encoderExecutionStatus
        guard let cb = ctx.queue.makeCommandBuffer(descriptor: descriptor) else {
            throw ModelError.residentBufferWrapFailed
        }
        cb.label = "token \(position)"
        return TokenCommand(position: position, cb: cb)
    }

    /// Encodes the layers not yet encoded, up to `last` inclusive.
    private func encodeLayers(into token: TokenCommand, upTo last: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = cfg.rmsNormEps
        while token.encodedLayers <= last {
            let L = token.encodedLayers
            if L < cfg.numLeadingDenseLayers {
                try encodeDenseLayer(into: token.cb, layer: L, position: token.position,
                                     isLinear: cfg.layerIsLinear(L), d: D, rmsEps: eps)
            } else {
                token.layers.append(
                    try encodeRoutedLayer(into: token.cb, layer: L, position: token.position))
            }
            token.encodedLayers += 1
        }
    }

    /// One routed layer as encoders of the token's command: the input norm,
    /// the attention and the tail with the classifier (one serial encoder on
    /// GDN and gated layers), the speculative routed work, the wait on the
    /// layer's value and the agreed fixup.
    private func encodeRoutedLayer(into cb: MTLCommandBuffer, layer L: Int,
                                   position: Int) throws -> TokenLayer {
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
        // GDN and gated layers run input norm, attention and the tail on one
        // serial encoder: on the M1 an encoder boundary costs more span than
        // the small dispatches around it. KDA and MLA keep their own encoders.
        var layerEncoder: MTLComputeCommandEncoder?
        if (isLinear && !cfg.linearAttentionPerChannelDecay)
            || (!isLinear && !cfg.layerIsMLA(L) && cfg.attnOutputGate) {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc.label = "layer \(L) attention"
            layerEncoder = enc
        }
        if let layerEncoder {
            rms.encodeBF16W(encoder: layerEncoder,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
        } else {
            try rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
        }
        try encodeDecodeAttention(cb: cb, layerEncoder: layerEncoder,
                                  layer: L, position: position,
                                  isLinear: isLinear, rmsEps: eps)
        routerReadbackTag = RouterHostReadback.nextTag(after: routerReadbackTag)
        let readbackTag = routerReadbackTag
        try encodeDecodeTailStage(
            tailCB: cb, layerEncoder: layerEncoder,
            layer: L, routerW: routerW,
            nextRouterW: nextRouterW, postAttn: postAttn,
            perExpertScale: perExpertScale,
            residency: (table: residencyResources.table, readbackTag: readbackTag),
            speculative: specDispatchArguments,
            d: D, eps: eps, probeBank: position & 1)
        layerEncoder?.endEncoding()
        try encodeSpeculativeRouted(
            into: cb,
            layer: L,
            residency: residencyResources,
            arguments: specDispatchArguments)
        let agreedToken = try model.reserveExpertIOCompletionToken()
        armedAgreedTokens.append(agreedToken)
        try encodeAgreedFixup(into: cb, layer: L,
                              residency: residencyResources,
                              arguments: specDispatchArguments, token: agreedToken)
        return TokenLayer(layer: L, readbackTag: readbackTag, agreedToken: agreedToken)
    }

    /// One decode step (v20 T3.2): the token's command, every layer and the
    /// boundary as its encoders, committed once; the host then feeds each
    /// routed layer's reads at its word and encodes the next token a layer per
    /// word, to be committed on the boundary word after the stop check.
    private func produceToken(token: Int32?,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode,
                              boundaryWord: MTLBuffer? = nil,
                              sample: ((MTLComputeCommandEncoder) throws -> Void)? = nil) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        // Decode must not share RAM with an idle ANE context (Track A):
        // prompts that end exactly on a chunk boundary reach here with the
        // last model still resident. No-op when ANE prefill is off or empty.
        anePrefill?.releaseModels()
        dumpPendingProbeRankings()
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
        let current: TokenCommand
        if let token {
            discardBoundaryState()
            current = try makeTokenCommand(position: position)
            try encodeEmbed(into: current.cb, token: token, d: D, outScale: embedOutScale)
        } else {
            current = try takeHeldToken(position: position)
        }
        try encodeLayers(into: current, upTo: cfg.numLayers - 1)
        let boundaryPath = boundaryWord != nil && sample != nil
        var fusedHead = false
        if let boundaryWord, let sample {
            try encodeBoundary(into: current.cb, logits: logits, d: D, rmsEps: eps,
                               outScale: embedOutScale, tokenWord: boundaryWord, sample: sample)
        } else {
            fusedHead = try encodeHead(into: current.cb, emitHead, logits: logits,
                                       outputMode: outputMode, d: D, rmsEps: eps)
        }
        let previous = runningToken
        if let boundaryWord {
            boundaryWord.contents().storeBytes(of: Self.boundaryTokenSentinel, as: UInt32.self)
            boundaryTokenWord = boundaryWord
        }
        let committed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        current.cb.commit()
        runningToken = current
        wordClock.beginToken(at: committed)
        var pending: PendingAgreedLayer?
        var next: TokenCommand?
        do {
            try finishToken(previous)
            if boundaryPath, position + 1 < maxContext, cfg.numLayers > 0 {
                next = try makeTokenCommand(position: position + 1)
                try kv?.reserve(tokens: position + 2)
            }
            for tokenLayer in current.layers {
                let L = tokenLayer.layer
                let tBodyStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                if let next { try encodeLayers(into: next, upTo: L) }
                let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                totalCb1Nanos &+= tWait - tBodyStart
                try waitForWord(tokenLayer, of: current)
                let woke = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                totalWaitNanos &+= woke - tWait
                wordClock.word(layer: L, at: woke)
                if let previous = pending {
                    try finishPendingAgreedLayer(previous)
                    pending = nil
                }
                let readback = try decodeRouterHostReadback(tag: tokenLayer.readbackTag)
                let predictedNextLayer: [Int] = L + 1 < cfg.numLayers
                    ? readback.predictedIDs.map { min(Int($0), cfg.numExperts - 1) }
                    : []
                pending = try serviceAgreedLayer(
                    layer: L, position: position, cb: current.cb, token: tokenLayer.agreedToken,
                    readback: readback, predictedNextLayer: predictedNextLayer,
                    bodyStart: tBodyStart)
            }
            if let last = pending {
                try finishPendingAgreedLayer(last)
                pending = nil
            }
            if let next { try encodeLayers(into: next, upTo: cfg.numLayers - 1) }
        } catch {
            // The fold's invariant: a committed command waits on nothing the
            // host will not publish, so the pass drains before it unwinds.
            if let pending { abandonPendingAgreedLayer(pending) }
            drainArmedAgreedTokens()
            try? Self.awaitCompletion(of: current.cb, deadlineNanos: Self.commandDeadlineNanos,
                                      naming: "the drained token")
            runningToken = nil
            heldToken = nil
            throw error
        }
        if prefetchTraceFD >= 0 { pendingProbeDumpPosition = position }
        kv?.advance()
        if boundaryPath {
            heldToken = next
        } else {
            try Self.awaitCompletion(of: current.cb, deadlineNanos: Self.commandDeadlineNanos,
                                     naming: "token \(position)")
            recordKernelGPU(role: "token", current.cb)
            runningToken = nil
            wordClock.endToken()
            if fusedHead { lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self) }
        }
    }

    private enum EmbedTokenSource {
        case constant(UInt32)
        case word(MTLBuffer)
    }

    private func encodeEmbed(into cb: MTLCommandBuffer, token: Int32,
                             d D: UInt32, outScale: Float) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.label = "embed"
        try encodeEmbedLookup(encoder, token: .constant(UInt32(bitPattern: token)),
                              d: D, outScale: outScale)
        encoder.endEncoding()
    }

    private func encodeEmbedLookup(_ encoder: MTLComputeCommandEncoder, token: EmbedTokenSource,
                                   d D: UInt32, outScale: Float) throws {
        let emb = try model.embedding()
        let vocab = UInt32(cfg.vocabSize)
        switch (affineEmbed, token) {
        case (let affine?, .constant(let id)):
            affine.encode(encoder: encoder,
                          table: emb.buffer, tableOffset: Int(emb.offset),
                          scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                          biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                          out: hidden, tokenId: id, d: D, outScale: outScale, vocab: vocab)
        case (let affine?, .word(let word)):
            affine.encode(encoder: encoder,
                          table: emb.buffer, tableOffset: Int(emb.offset),
                          scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                          biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                          out: hidden, tokenBuffer: word, d: D, outScale: outScale, vocab: vocab)
        case (nil, .constant(let id)):
            embedInt4.encode(encoder: encoder,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenId: id, d: D, outScale: outScale, vocab: vocab)
        case (nil, .word(let word)):
            embedInt4.encode(encoder: encoder,
                             table: emb.buffer, tableOffset: Int(emb.offset),
                             scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                             biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                             out: hidden, tokenBuffer: word, d: D, outScale: outScale, vocab: vocab)
        }
    }

    /// The sentinel goes into the word before the commit so the host can tell the
    /// sampler's write from the previous token's.
    /// The boundary as the token's last encoders: the final norm, the head,
    /// the caller's sampler writing the token word, the next embed from it.
    private func encodeBoundary(into cb: MTLCommandBuffer, logits: MTLBuffer,
                                d D: UInt32, rmsEps eps: Float, outScale: Float,
                                tokenWord: MTLBuffer,
                                sample: (MTLComputeCommandEncoder) throws -> Void) throws {
        let fNorm = try model.finalNorm()
        let lm = try model.lmHead()
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.label = "boundary"
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        encodeFinalNorm(encoder, weights: fNorm, d: D, rmsEps: eps)
        encodeLMHead(encoder, weights: lm, into: logits, d: D)
        try sample(encoder)
        try encodeEmbedLookup(encoder, token: .word(tokenWord), d: D, outScale: outScale)
        encoder.endEncoding()
        totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
    }

    private func encodeFinalNorm(_ encoder: MTLComputeCommandEncoder, weights fNorm: TensorView,
                                 d D: UInt32, rmsEps eps: Float) {
        rms.encodeBF16W(encoder: encoder, x: hidden,
                        weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                        out: normed, d: D, eps: eps)
    }

    private func encodeLMHead(_ encoder: MTLComputeCommandEncoder, weights lm: TensorView,
                              into logits: MTLBuffer, d D: UInt32) {
        encodePrimaryGEMV(encoder: encoder,
                          weights: lm.buffer, weightsOffset: Int(lm.offset),
                          scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                          biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                          x: normed, y: logits, m: UInt32(cfg.vocabSize), n: D)
    }

    private func encodeDenseLayer(into cb: MTLCommandBuffer, layer L: Int, position: Int,
                                  isLinear: Bool, d D: UInt32, rmsEps eps: Float) throws {
        let inNorm = try model.inputNorm(layer: L)
        let postAttn = try model.postAttnNorm(layer: L)
        try rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                        out: normed,
                        d: D, eps: eps)
        try encodeDecodeAttention(cb: cb, layerEncoder: nil,
                                  layer: L, position: position,
                                  isLinear: isLinear, rmsEps: eps)
        try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                       hidden: hidden,
                                       delta: oOut,
                                       count: cfg.hiddenSize)
        try rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: postAttn.buffer,
                        weightOffset: Int(postAttn.offset),
                        out: routedX,
                        d: D, eps: eps)
        // Leading dense-MLP layer (Kimi layer 0): no router, no
        // routed experts — the shared-expert kernels run the layer's
        // own SwiGLU and the residual folds here.
        let dense = sharedExpertProjections[L]
        try shared.encode(commandBuffer: cb,
                          x: routedX,
                          gate: dense.gate,
                          up: dense.up,
                          down: dense.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate,
                          scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        try elementwise!.encodeResidualAdd(commandBuffer: cb,
                                       hidden: hidden,
                                       delta: h1Buf,
                                       count: cfg.hiddenSize)
    }

    /// The synchronous head as the token's last encoders: the fused greedy
    /// head into `greedyTokenBuf`, or the final norm and lm_head into the
    /// logits; the caller waits for the command. Returns whether the fused
    /// head was encoded.
    private func encodeHead(into cb: MTLCommandBuffer, _ wanted: Bool, logits: MTLBuffer,
                            outputMode: PrefillOutputMode,
                            d D: UInt32, rmsEps eps: Float) throws -> Bool {
        guard wanted else { return false }
        let fNorm = try model.finalNorm()
        let lm    = try model.lmHead()
        let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if useFusedGreedyHead && outputMode == .greedyIfAvailable {
            try fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: greedyTokenBuf,
                d: D, vocab: UInt32(cfg.vocabSize),
                rmsEps: eps)
            totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            return true
        }
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.label = "head"
        encodeFinalNorm(encoder, weights: fNorm, d: D, rmsEps: eps)
        encodeLMHead(encoder, weights: lm, into: logits, d: D)
        encoder.endEncoding()
        totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
        return false
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
        eps: Float,
        probeBank: Int = 0
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
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts),
                probeSlot: Self.probeSlot(layer: L, distance: 1, bank: probeBank,
                                          numLayers: cfg.numLayers))
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
        if prefetchTraceFD >= 0 {
            try encodeProbeDistanceScores(encoder: tailEncoder, layer: L, hidden: routedX,
                                          probeBank: probeBank, d: D)
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
    /// non-gated branch as encoders of the token's command: QKV + RoPE, the
    /// softmax, o_proj, in that order.
    private func encodeGptOssAttentionDecode(attnCB: MTLCommandBuffer,
                                             tailCB: MTLCommandBuffer,
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
        let attentionCB = attnCB
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

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let err = cb.error {
            throw ModelError.commandBufferFailed(detail: String(describing: err))
        }
    }

    /// The word wake (v14 lever B): the classifier's tagged copy lands before
    /// the driver marks anything. The token's command cannot complete before
    /// the host has serviced every later layer, so past the first second the
    /// wait keeps polling the word, gently, up to the deadline; a wait
    /// nothing will publish then ends loudly, naming the layer.
    private func waitForWord(_ layer: TokenLayer, of token: TokenCommand) throws {
        let words = routerHostReadback.contents().bindMemory(
            to: UInt32.self,
            capacity: RouterHostReadback.wordCount(topK: cfg.topKExperts))
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let spinUntil = started + 1_000_000_000
        let deadline = started + Self.commandDeadlineNanos
        var spins = 0
        var fallenBack = false
        while true {
            if RouterHostReadback.isComplete(words: words, topK: cfg.topKExperts,
                                             tag: layer.readbackTag) {
                return
            }
            spins &+= 1
            if spins % 256 == 0 {
                switch token.cb.status {
                case .error:
                    throw ModelError.commandBufferFailed(
                        detail: "layer \(layer.layer): \(Self.describeCommandBufferError(token.cb.error))")
                case .completed:
                    throw ModelError.internalInconsistency(
                        detail: "layer \(layer.layer)'s command completed without its word")
                default:
                    break
                }
            }
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if now >= spinUntil {
                if !fallenBack {
                    fallenBack = true
                    totalRouterWakeFallbacks &+= 1
                }
                guard now < deadline else {
                    throw ModelError.commandBufferFailed(
                        detail: "layer \(layer.layer)'s word: not written within "
                            + "\(Self.commandDeadlineNanos / 1_000_000) ms")
                }
                usleep(100)
            }
        }
    }

    /// A bounded completion wait: the status polled until the command
    /// completes or errs, or `deadlineNanos` pass, which throws naming what
    /// was waited on rather than hanging on a wait nothing will publish.
    static func awaitCompletion(of cb: MTLCommandBuffer, deadlineNanos: UInt64,
                                naming what: String) throws {
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let deadline = started + deadlineNanos
        while true {
            switch cb.status {
            case .completed:
                return
            case .error:
                throw ModelError.commandBufferFailed(
                    detail: "\(what): \(describeCommandBufferError(cb.error))")
            default:
                break
            }
            let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            guard now < deadline else {
                throw ModelError.commandBufferFailed(
                    detail: "\(what): the command did not complete within "
                        + "\(deadlineNanos / 1_000_000) ms")
            }
            if now - started > 2_000_000 { usleep(100) }
        }
    }

    /// A command's error with the encoder that faulted named, when the
    /// command was made with `encoderExecutionStatus` (v20 T3.2).
    static func describeCommandBufferError(_ error: (any Error)?) -> String {
        guard let error else { return "no error recorded" }
        let nsError = error as NSError
        guard let infos = nsError.userInfo[MTLCommandBufferEncoderInfoErrorKey]
                as? [any MTLCommandBufferEncoderInfo], !infos.isEmpty else {
            return String(describing: error)
        }
        let faulted = infos.enumerated()
            .filter { $0.element.errorState == .faulted }
            .map { $0.element.label.isEmpty ? "encoder #\($0.offset)" : $0.element.label }
        let affected = infos.filter { $0.errorState == .affected }.count
        return "\(String(describing: error)); faulted: "
            + "\(faulted.isEmpty ? "none reported" : faulted.joined(separator: ", ")); "
            + "affected encoders: \(affected)"
    }

    private func decodeRouterHostReadback(tag: UInt32) throws -> RouterHostReadback {
        let words = routerHostReadback.contents().bindMemory(
            to: UInt32.self,
            capacity: RouterHostReadback.wordCount(topK: cfg.topKExperts))
        guard let readback = RouterHostReadback.decode(
            words: words, topK: cfg.topKExperts, tag: tag)
        else {
            throw ModelError.internalInconsistency(
                detail: "router host readback carries a stale tag after the router's wake")
        }
        return readback
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
        if tokenCount >= Self.prefillMatrixMinRows,
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
                minimumRows: Self.prefillMatrixMinRows) == .qmm {
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
    /// The chunked scan (v12 P4) when the kernels and the scratch allow it;
    /// the serial kernel for a chunk below `GDN.chunkTokens` rows or an
    /// uncompiled shape.
    private func encodeGDNDeltaStep(
        cb: MTLCommandBuffer, gdn: GDN, gdnState: GDNStateManager, layer L: Int,
        scratch: PrefillChunkScratchBuffers,
        aLog: MTLBuffer, aLogOffset: Int, dtBias: MTLBuffer, dtBiasOffset: Int,
        rows t: Int
    ) throws {
        if gdn.chunkedScanAvailable,
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

    /// One layer's linear-attention pipeline: in-projection, causal conv, QK norm, delta step, gated norm, out-projection.
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
        try encodePrefillRoPEEpilogue(cb: cb, views: views, scratch: scratch, attnQ: attnQ,
                                      tokenCount: t, startPosition: startPosition,
                                      isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                                      qDim: qDim, kvDim: kvDim, rmsEps: eps)
        try encodePrefillCausalAttention(cb: cb, layer: L, scratch: scratch, attnQ: attnQ,
                                         tokenCount: t, startPosition: startPosition,
                                         isFull: isFull, headDim: headDim, numKVHeads: numKVHeads,
                                         qDim: qDim, kvDim: kvDim)
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

    private func encodePrefillRoPEEpilogue(
        cb: MTLCommandBuffer, views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers, attnQ: MTLBuffer,
        tokenCount t: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int, rmsEps eps: Float
    ) throws {
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
    }

    private func encodePrefillCausalAttention(
        cb: MTLCommandBuffer, layer L: Int,
        scratch: PrefillChunkScratchBuffers, attnQ: MTLBuffer,
        tokenCount t: Int, startPosition: Int,
        isFull: Bool, headDim: Int, numKVHeads: Int,
        qDim: Int, kvDim: Int
    ) throws {
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
                                              minimumQueries: UInt32(Self.prefillMatrixMinRows))
        } else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill attention requires a KV cache")
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
                                    scratch: PrefillChunkScratchBuffers) throws -> PrefillRouting {
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
        if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
            guard let fitted = Self.prefillRoutedTileSchedulerConfig.fitting(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tiles cannot fit the \(slotCount)-slot expert cache")
            }
            schedulerConfig = fitted
        } else {
            schedulerConfig = Self.prefillRoutedTileSchedulerConfig
        }
        var lastRowByExpert: [UInt32: Int] = [:]
        var rowsByExpert: [UInt32: Int] = [:]
        lastRowByExpert.reserveCapacity(min(pairs.count, cfg.numExperts))
        rowsByExpert.reserveCapacity(min(pairs.count, cfg.numExperts))
        for pair in pairs {
            let row = Int(pair.token)
            lastRowByExpert[pair.expert] = max(lastRowByExpert[pair.expert] ?? row, row)
            rowsByExpert[pair.expert, default: 0] += 1
        }
        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert,
            lastRowByExpert: lastRowByExpert,
            resident: try residentExpertMask(layer: L),
            slots: model.routedExpertCacheSlotCount(layer: L) ?? 0,
            tileWidth: schedulerConfig.tileExperts)
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: schedulerConfig.tileExperts,
            expertSortKeys: PrefillSweepOrder.expertSortKeys(forOrder: order, numExperts: cfg.numExperts))
        return PrefillRouting(routes: routes, schedulerConfig: schedulerConfig)
    }

    /// The resident-first sweep's per-chunk snapshot of layer `L`'s pool-resident
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
                                             minimumRows: Self.prefillMatrixMinRows)
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
                    minimumRows: Self.prefillMatrixMinRows)
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
    private func encodeRoutedMoEPrefill(
        cb: inout MTLCommandBuffer,
        layer L: Int,
        views: LayerPrefillQKVViews,
        scratch: PrefillChunkScratchBuffers,
        tokenCount t: Int,
        hiddenSize D: Int,
        startPosition: Int
    ) async throws {
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
                sharedCB.commit()
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

                let routing = try buildPrefillRoutes(layer: L, tokenCount: t, scratch: scratch)
                recordRouteTracePrefillRows(layer: L, startPosition: startPosition, rowCount: t,
                                            ids: routeIDScratch)
                let routes = routing.routes
                let schedulerConfig = routing.schedulerConfig

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

    /// One routed tile's experts: the grouped GEMMs over every expert when
    /// the matrix path is in force, else the scalar microbatch path whole.
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
                intermediate: Int(params.routedIntermediate)),
              prefillGroupedMoE.groupedPathAvailable(for: mpp) else {
            try encodeScalar(pairStart: tile.pairStart, pairCount: tile.pairCount)
            return
        }
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
    }

    /// Attention stage of one decode layer: the gated-DeltaNet branch or the
    /// softmax branch, both writing into `oOut` for the residual add.
    private func encodeDecodeAttention(
        cb: MTLCommandBuffer,
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
            try encodeKDADecode(cb, layer: L)
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
            try encodeMLAAttentionDecode(cb, layer: L,
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
            try encodeGptOssAttentionDecode(attnCB: cb, tailCB: cb, layer: L,
                                            position: position, seqLen: seqLen)
        } else {
            try encodePlainAttentionDecode(attnCB: cb, tailCB: cb, layer: L,
                                           position: position, seqLen: seqLen,
                                           rmsEps: eps)
        }

        // Plain pre-norm residual block: hidden += attention branch,
        // then one post-attention norm feeds router, shared expert,
        // and routed phase 1 (routedX doubles as moeX).
    }

    /// Plain (non-gated, unbiased) full/SWA attention, one decode step, as
    /// encoders of the token's command: fused QKV + rope/norm epilogue, the
    /// softmax pass, o_proj.
    private func encodePlainAttentionDecode(
        attnCB: MTLCommandBuffer,
        tailCB: MTLCommandBuffer,
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
            let attentionCB = attnCB
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

    // MARK: - The agreed cells (v20 T3.1)

    static let agreedCellSentinel: UInt32 = 0xffff_ffff

    private func agreedCellsOffset(layer L: Int) -> Int {
        L * cfg.topKExperts * MemoryLayout<UInt32>.stride
    }

    /// The layer's fixup, encoded with the layer before its router has run:
    /// the wait on the layer's value, then phase 1 over the host's cell row
    /// (the sentinel at the hits) and phase 2 over the eight resolving a
    /// sentinel through that row, the speculative kernels behind the batch's
    /// status word, sized by the classifier's fixup grids.
    private func encodeAgreedFixup(into cb: MTLCommandBuffer, layer L: Int,
                                   residency: ExpertResidencyResources,
                                   arguments: MoE.SpeculativeDispatchArguments,
                                   token: ExpertIOCompletionToken) throws {
        cb.encodeWaitForEvent(token.event, value: token.value)
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        encoder.label = "layer \(L) fixup"
        defer { encoder.endEncoding() }
        let offsets = try model.routedExpertOffsets(layer: L)
        let cellsOffset = agreedCellsOffset(layer: L)
        let d = UInt32(cfg.hiddenSize)
        let f = UInt32(cfg.moeIntermediateSize)
        let topK = UInt32(cfg.topKExperts)
        moe.encodeSpecPhase1U16Load(
            encoder: encoder,
            expertPool: residency.expertPool,
            poolSlotStride: residency.poolSlotStride,
            resolvedSlots: agreedCells, resolvedSlotsOffset: cellsOffset,
            routedOffsets: offsets,
            x: routedX, acts: moeActs,
            d: d, f: f, topK: topK,
            indirectArguments: arguments.arguments,
            indirectOffset: MoE.fixupPhase1ArgsOffset,
            ioStatus: token.status, ioStatusOffset: token.statusOffset)
        moe.encodeSpecPhase2Reduce(
            encoder: encoder,
            expertPool: residency.expertPool,
            poolSlotStride: residency.poolSlotStride,
            resolvedSlots: residencyResolvedSlots,
            fallbackCells: agreedCells, fallbackCellsOffset: cellsOffset,
            routedOffsets: offsets,
            acts: moeActs, routingWeights: outWeights,
            residual: h1Buf, y: h2Buf, hidden: hidden,
            d: d, f: f, topK: topK,
            indirectArguments: arguments.arguments,
            indirectOffset: MoE.fixupPhase2ArgsOffset,
            ioStatus: token.status, ioStatusOffset: token.statusOffset)
    }

    private func disarmAgreedToken(_ token: ExpertIOCompletionToken) {
        armedAgreedTokens.removeAll { $0.value == token.value }
    }

    /// The fold's invariant in its T3.1 form: every value a committed or held
    /// command waits on is published, by the batch, by the host at the word,
    /// or here on the way out of a pass, so no GPU wait is left unsatisfied.
    private func drainArmedAgreedTokens() {
        for token in armedAgreedTokens {
            model.publishExpertIOCompletionToken(token, succeeded: false)
        }
        armedAgreedTokens.removeAll()
    }

    /// One routed layer at its word: the route, the landed predictions
    /// leased, every miss given its cell, the reads issued with the layer's
    /// value, the next layer's prediction issued.
    private struct AgreedLayerContext {
        let layer: Int
        let position: Int
        let cb: MTLCommandBuffer
        let bodyStart: UInt64
        let readback: RouterHostReadback
        let predictedNextLayer: [Int]
        let experts: [Int]
        let token: ExpertIOCompletionToken
        var residentBeforePlan: [Int] = []
        var leasedPredictions: [Int: Int] = [:]
        var claimedCells: [Int: Int] = [:]
        var readExperts: [Int] = []
        var readCells: [Int] = []
        var overflowExperts: [Int: Int] = [:]
        var operation: ExpertLoadOperation?
    }

    /// A serviced layer carried to the next wake, where its batch is checked,
    /// its plan run off the path and its cells consumed.
    private struct PendingAgreedLayer {
        let layer: Int
        let position: Int
        let cb: MTLCommandBuffer
        let experts: [Int]
        let missExperts: Set<Int>
        let leasedPredictions: [Int]
        let claimedExperts: Set<Int>
        let readExperts: [Int]
        let operation: ExpertLoadOperation
        let residentBeforePlan: [Int]
        let predictedNextLayer: [Int]
    }

    private func serviceAgreedLayer(layer L: Int, position: Int, cb: MTLCommandBuffer,
                                    token: ExpertIOCompletionToken,
                                    readback: RouterHostReadback, predictedNextLayer: [Int],
                                    bodyStart: UInt64) throws -> PendingAgreedLayer {
        let experts = readDecodeRouterReadback(layer: L, position: position,
                                               hostReadback: readback)
        var context = AgreedLayerContext(
            layer: L, position: position, cb: cb, bodyStart: bodyStart,
            readback: readback, predictedNextLayer: predictedNextLayer,
            experts: experts, token: token)
        do {
            try joinAgreedLandings(&context)
            try agreeCells(&context)
            let pending = try submitAgreedReads(&context)
            schedulePredictivePrefetch(layer: L, predicted: predictedNextLayer,
                                       demand: pending.operation)
            totalBodyNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - bodyStart
            return pending
        } catch {
            abandonAgreedContext(context)
            throw error
        }
    }

    private func readDecodeRouterReadback(
        layer L: Int, position: Int, hostReadback: RouterHostReadback
    ) -> [Int] {
        let readbackStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        decodeExpertsScratch.removeAll(keepingCapacity: true)
        decodeExpertsScratch.reserveCapacity(cfg.topKExperts)
        for id in hostReadback.expertIDs {
            decodeExpertsScratch.append(min(Int(id), cfg.numExperts - 1))
        }
        totalRouterReadbackNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - readbackStarted
        if runnerStatsEnabled {
            if totalRankWeightMass.count != cfg.topKExperts {
                totalRankWeightMass = [Double](repeating: 0, count: cfg.topKExperts)
            }
            for i in 0..<cfg.topKExperts {
                totalRankWeightMass[i] += Double(Float16(bitPattern: hostReadback.weightBits[i]))
            }
            totalRankWeightLayers &+= 1
        }
        let experts = decodeExpertsScratch
        recordRouteTrace(layer: L, position: position, experts: experts)
        return experts
    }

    private func joinAgreedLandings(_ context: inout AgreedLayerContext) throws {
        context.residentBeforePlan = prefetchTraceFD >= 0
            ? try model.routedExpertResidentIDs(layer: context.layer) : []
        context.leasedPredictions = predictivePrefetch.readyCells(
            layer: context.layer, experts: context.experts, joinNanos: Self.prefetchJoinNanos)
    }

    /// Every miss of the classifier's list gets its cell: a landing the ring
    /// leased (no read), a free ring cell claimed as a landing, or, when the
    /// ring has none, the pool's victim chosen here for this miss alone; the
    /// row written before the batch publishes the value the fixup waits on.
    private func agreeCells(_ context: inout AgreedLayerContext) throws {
        let L = context.layer
        let missPositions = context.readback.missPositions.map { Int($0) }
        let unleased = missPositions.map { context.experts[$0] }
            .filter { context.leasedPredictions[$0] == nil }
        context.claimedCells = predictivePrefetch.claimDemand(layer: L, experts: unleased)
        let row = agreedCells.contents().advanced(by: agreedCellsOffset(layer: L))
            .assumingMemoryBound(to: UInt32.self)
        for position in 0..<cfg.topKExperts { row[position] = Self.agreedCellSentinel }
        for position in missPositions {
            let expert = context.experts[position]
            row[position] = UInt32(try agreeCell(for: expert, &context))
        }
        totalAgreedOverflow &+= UInt64(context.overflowExperts.count)
    }

    private func agreeCell(for expert: Int, _ context: inout AgreedLayerContext) throws -> Int {
        let L = context.layer
        if let landed = context.leasedPredictions[expert] { return landed }
        if let claimed = context.claimedCells[expert] {
            if try model.claimRoutedExpertLanding(layer: L, expert: expert, cell: claimed) {
                context.readExperts.append(expert)
                context.readCells.append(claimed)
                return claimed
            }
            predictivePrefetch.consume(layer: L, experts: [expert], freedCells: [:])
            context.claimedCells[expert] = nil
        }
        // A prediction issued between the join and the claim: joined now.
        if let late = predictivePrefetch.readyCells(
            layer: L, experts: [expert], joinNanos: Self.prefetchJoinNanos)[expert] {
            context.leasedPredictions[expert] = late
            return late
        }
        guard let victim = try model.reserveRoutedExpertOverflowSlot(
            layer: L, expert: expert, protecting: context.experts) else {
            throw ModelError.expertCacheUnplaceable(
                detail: "layer \(L) has no cell for expert \(expert)")
        }
        context.overflowExperts[expert] = victim
        context.readExperts.append(expert)
        context.readCells.append(victim)
        return victim
    }

    private func submitAgreedReads(_ context: inout AgreedLayerContext) throws -> PendingAgreedLayer {
        let L = context.layer
        let submitStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let operation = try model.beginAgreedRoutedReads(
            layer: L, experts: context.readExperts, cells: context.readCells,
            token: context.token)
        context.operation = operation
        disarmAgreedToken(context.token)
        predictivePrefetch.attachDemand(layer: L, experts: Set(context.claimedCells.keys),
                                        operation: operation)
        totalRoutedSubmitNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - submitStarted
        if !context.readExperts.isEmpty { predictivePrefetch.noteDemandSubmission() }
        let missExperts = Set(context.readback.missPositions.map { context.experts[Int($0)] })
        if !missExperts.isEmpty { totalHitFixupLayers &+= 1 }
        return PendingAgreedLayer(
            layer: L, position: context.position, cb: context.cb, experts: context.experts,
            missExperts: missExperts,
            leasedPredictions: Array(context.leasedPredictions.keys),
            claimedExperts: Set(context.claimedCells.keys),
            readExperts: context.readExperts, operation: operation,
            residentBeforePlan: context.residentBeforePlan,
            predictedNextLayer: context.predictedNextLayer)
    }

    private func abandonAgreedContext(_ context: AgreedLayerContext) {
        if let operation = context.operation { _ = try? operation.wait() }
        let held = Set(context.leasedPredictions.keys).union(context.claimedCells.keys)
        predictivePrefetch.consume(layer: context.layer, experts: held, freedCells: [:])
        for (expert, cell) in context.overflowExperts {
            try? model.abandonRoutedExpertOverflowSlot(layer: context.layer, expert: expert, cell: cell)
        }
    }

    private func abandonPendingAgreedLayer(_ pending: PendingAgreedLayer) {
        _ = try? pending.operation.wait()
        let leased = Set(pending.leasedPredictions).union(pending.claimedExperts)
        predictivePrefetch.consume(layer: pending.layer, experts: leased, freedCells: [:])
    }

    /// The previous layer at this wake: its command's error, its batch's
    /// (a failed read names the layer), the io rows, then the plan.
    private func finishPendingAgreedLayer(_ pending: PendingAgreedLayer) throws {
        if let err = pending.cb.error {
            throw ModelError.commandBufferFailed(
                detail: "layer \(pending.layer): \(Self.describeCommandBufferError(err))")
        }
        do {
            try pending.operation.wait()
        } catch {
            abandonPendingAgreedLayer(pending)
            throw ModelError.expertReadFailed(layer: pending.layer,
                                              detail: String(describing: error))
        }
        if !pending.readExperts.isEmpty {
            totalIOQueueNanos &+= pending.operation.submissionToStartNanos
            totalIoNanos &+= pending.operation.loadNanos
            totalMissIoNanos &+= pending.operation.loadNanos
        }
        try planAgreedLayer(pending)
    }

    /// The plan, off the path: the hits' use counts and promotions, every
    /// leased cell swapped into the pool by index against a victim chosen
    /// now, the freed cells back to the ring, the trace rows.
    private func planAgreedLayer(_ pending: PendingAgreedLayer) throws {
        let leased = Set(pending.leasedPredictions).union(pending.claimedExperts)
        let planStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plan: RoutedExpertFetchPlan
        do {
            guard let planned = try model.planRoutedExperts(
                layer: pending.layer, experts: pending.experts,
                gpuMissedExperts: pending.missExperts, leasedLandings: leased,
                missesCounted: pending.readExperts.count) else {
                throw ModelError.expertCacheUnplaceable(
                    detail: "layer \(pending.layer): the route does not fit its pool")
            }
            plan = planned
        } catch {
            predictivePrefetch.unlease(layer: pending.layer, experts: leased)
            throw error
        }
        totalCachePlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - planStarted
        guard plan.misses.isEmpty else {
            try? model.abandonRoutedExpertPlan(plan)
            predictivePrefetch.consume(layer: pending.layer, experts: leased,
                                       freedCells: plan.freedCells)
            throw ModelError.internalInconsistency(
                detail: "layer \(pending.layer): an agreed read did not land for experts "
                    + "\(plan.misses.map { plan.experts[$0] })")
        }
        totalPrefetchLandedHits &+= UInt64(
            pending.leasedPredictions.filter { !pending.missExperts.contains($0) }.count)
        predictivePrefetch.consume(layer: pending.layer, experts: leased,
                                   freedCells: plan.freedCells)
        recordPrefetchTrace(layer: pending.layer, position: pending.position,
                            experts: pending.experts, misses: pending.readExperts,
                            resident: pending.residentBeforePlan,
                            nextLayerPrediction: pending.predictedNextLayer)
    }

    /// The shared-expert chain (or the h1Buf zero-fill when the arch has
    /// none), encoded at the head of the spec CB. It depends only on
    /// `routedX`, which the tail produces, so the command is committed before
    /// the router readback and the GPU runs it while the CPU waits for the
    /// routing; encoding it after the readback left a measured 7.88 ms/token
    /// of GPU idle in the `attn_tail_router -> shared_expert` transition.
    private func encodeSharedExpertZeroFill(into cb: MTLCommandBuffer) throws {
        // No shared expert (gpt-oss): the phase-2 reduce still seeds from
        // h1Buf, so pin it to zero in place of the dense MLP output.
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.fill(buffer: h1Buf,
                  range: 0..<(cfg.hiddenSize * MemoryLayout<Float16>.stride),
                  value: 0)
        blit.endEncoding()
    }

    private func encodeSharedExpertWork(on sharedEncoder: MTLComputeCommandEncoder,
                                        layer L: Int) throws {
        let D = UInt32(cfg.hiddenSize)
        let sharedProj = sharedExpertProjections[L]
        if let fused = shared.int4FusedDecode {
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
                d: UInt32(cfg.hiddenSize)))
    }

    /// The speculative pool-addressed phase-1/phase-2 (v9), committed before
    /// the tail wait so it sizes itself from the classifier's indirect
    /// arguments: full grids on an all-hit layer, zero grids otherwise.
    private func encodeSpeculativeRouted(
        into cb: MTLCommandBuffer,
        layer L: Int,
        residency: ExpertResidencyResources,
        arguments: MoE.SpeculativeDispatchArguments
    ) throws {
        let pool = residency.expertPool
        if !cfg.hasSharedExpert {
            try encodeSharedExpertZeroFill(into: cb)
        }
        let offsets = try model.routedExpertOffsets(layer: L)
        // One serial encoder for the shared chain and the routed work; a
        // .concurrent encoder segfaults the AGX driver when an indirect
        // dispatch follows it on the command (macOS 26 / M4, v9's trap).
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        encoder.label = "layer \(L) routed"
        defer { encoder.endEncoding() }
        if cfg.hasSharedExpert {
            try encodeSharedExpertWork(on: encoder, layer: L)
        }
        moe.encodeSpecPhase1U16Load(
            encoder: encoder,
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
        moe.encodeSpecPhase2Reduce(
            encoder: encoder,
            expertPool: pool,
            poolSlotStride: residency.poolSlotStride,
            resolvedSlots: residencyResolvedSlots,
            routedOffsets: offsets,
            acts: moeActs,
            routingWeights: outWeights,
            residual: h1Buf,
            y: h2Buf,
            hidden: hidden,
            d: UInt32(cfg.hiddenSize),
            f: UInt32(cfg.moeIntermediateSize),
            topK: UInt32(cfg.topKExperts),
            indirectArguments: arguments.arguments)
    }

}
