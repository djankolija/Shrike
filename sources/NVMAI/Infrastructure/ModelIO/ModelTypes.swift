import Foundation
import Metal

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape, and family-specific kernel behavior. Stored in
/// `manifest.json -> arch.family`; absent means the compatible Qwen3.5-MoE
/// 35B-A3B baseline used by Qwen 3.6 and Ornith 1.5.
public enum ModelFamily: String, Sendable, Equatable, Codable, CaseIterable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    case gptOss20b = "gpt_oss_20b"
    case kimiLinear48b = "kimi_linear_48b"
}

/// Multi-head latent attention projection split for layers with mask value 3.
/// The runtime's MQA form derives from these: one cache row per token is
/// [latent | rope] (headDim = latentDim + qkRopeDim), and the per-head
/// embed/unembed GEMVs use qkNopeDim / valueHeadDim.
public struct MLAConfig: Sendable, Equatable {
    public let latentDim: Int
    public let qkNopeDim: Int
    public let qkRopeDim: Int
    public let valueHeadDim: Int

    public init(latentDim: Int, qkNopeDim: Int, qkRopeDim: Int,
                valueHeadDim: Int) {
        self.latentDim = latentDim
        self.qkNopeDim = qkNopeDim
        self.qkRopeDim = qkRopeDim
        self.valueHeadDim = valueHeadDim
    }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int

    public init(numKHeads: Int, numVHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convKernelSize: Int) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention,
/// 3 = multi-head latent attention (MLA).
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE baseline so earlier manifests validate unchanged.
    public let family: ModelFamily
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention (Qwen 3.6 uses 0.0625 = 256^-0.5).
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize). False for Qwen 3.6.
    public let embeddingScaledBySqrtHidden: Bool
    /// Router carries `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors. False (Qwen 3.6): plain quantized linear router with
    /// renormalized top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Dual-branch FFN sandwich: pre/post feedforward norms plus a per-layer
    /// residual scalar. False (Qwen 3.6) = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. True (Qwen/NeoX sub-dim): rotation confined to
    /// the first `rotaryDim` elements, pairing (i, rotaryDim/2 + i), frequency
    /// divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig
    /// Leading layers whose MLP is dense rather than routed; they have no
    /// packed-expert layer file.
    public let numLeadingDenseLayers: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        family: ModelFamily = .qwen36,
        attnOutputGate: Bool = true,
        attentionScale: Double = 0.0625,
        embeddingScaledBySqrtHidden: Bool = false,
        routerScaled: Bool = false,
        ffnSandwichNorms: Bool = false,
        sharedExpertGated: Bool = true,
        ropeNeoxSubdim: Bool = true,
        linearAttention: LinearAttentionConfig = .none,
        numLeadingDenseLayers: Int = 0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.family = family
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.linearAttention = linearAttention
        self.numLeadingDenseLayers = numLeadingDenseLayers
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
    public static let qwen36_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: .qwen36,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    /// Native one-layer Qwen3.5-MoE MTP draft for compatible Qwen/Ornith
    /// targets. Its 65,536-token KV cache is at most
    /// 128 MiB in FP16 and smaller with compressed storage; truncating draft
    /// context can only lower acceptance because every emitted token is still
    /// verified by the full target.
    public static let qwen36MTP = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 65_536,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 1,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: [1],
        hiddenActivation: "silu",
        family: .qwen36MTP,
        attnOutputGate: true,
        attentionScale: 0.0625,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true)

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// gpt-oss-20b: 24 alternating sliding-window(128)/full-attention layers
    /// starting sliding, 32 routed experts (top-4) with additive expert and
    /// router biases, attention sinks, q/k/v/o biases, no QK-norm, no output
    /// gate, no shared expert, YaRN RoPE (factor 32 over original 4096) over
    /// the full head dim, clamped SwiGLU (limit 7.0, alpha 1.702).
    public static let gptOss20b = ArchConfig(
        hiddenSize: 2880,
        intermediateSize: 0,
        moeIntermediateSize: 2880,
        numHeads: 64,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 201_088,
        slidingWindow: 128,
        finalLogitSoftcap: 0.0,
        ropeTheta: 150_000.0,
        fullRopeTheta: 150_000.0,
        partialRotaryFactor: 1.0,
        numLayers: 24,
        numExperts: 32,
        topKExperts: 4,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: (0..<24).map { UInt8($0 % 2 == 0 ? 0 : 1) },
        hiddenActivation: "silu",
        family: .gptOss20b,
        attnOutputGate: false,
        attentionScale: 0.125,    // 64^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        linearAttention: .none)

    /// Kimi-Linear-48B-A3B: 27 layers — KDA (per-channel-decay gated DeltaNet)
    /// on 20, NoPE MLA on 7 (every 4th and the last) run as MQA over one
    /// [latent 512 | rope 64] row (headDim 576; score scale stays 192^-0.5,
    /// the original per-head q dim). Layer 0's MLP is dense (9216); the other
    /// 26 carry 256 routed experts (top-8, sigmoid router with correction
    /// bias, renormalized, scaled 2.446) plus one ungated shared expert.
    public static let kimiLinear48bA3b = ArchConfig(
        hiddenSize: 2304,
        intermediateSize: 1024,
        moeIntermediateSize: 1024,
        numHeads: 32,
        numKVHeads: 1,
        numFullKVHeads: 1,
        headDim: 576,
        fullHeadDim: 576,
        vocabSize: 163_840,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 10_000.0,
        partialRotaryFactor: 0.0,
        numLayers: 27,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.kimiLinearLayerMask(),
        hiddenActivation: "silu",
        family: .kimiLinear48b,
        attnOutputGate: false,
        attentionScale: 0.07216878364870323,   // 192^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: false,
        linearAttention: LinearAttentionConfig(
            numKHeads: 32, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4),
        numLeadingDenseLayers: 1)

    private static func kimiLinearLayerMask() -> [UInt8] {
        // MLA (3) on 1-indexed layers {4, 8, 12, 16, 20, 24, 27}; KDA (2)
        // everywhere else.
        var mask = [UInt8](repeating: 2, count: 27)
        for oneIndexed in [4, 8, 12, 16, 20, 24, 27] { mask[oneIndexed - 1] = 3 }
        return mask
    }

    /// Registry keyed by `manifest.arch.family` for auto-detection at load.
    public static let knownArchitectures: [ModelFamily: ArchConfig] = [
        .qwen36: .qwen36_35B_A3B,
        .qwen36MTP: .qwen36MTP,
        .gptOss20b: .gptOss20b,
        .kimiLinear48b: .kimiLinear48bA3b,
    ]

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
        shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        if intermediateSize > 0 {
            shapes.append((m: intermediateSize, n: hiddenSize))
            shapes.append((m: hiddenSize, n: intermediateSize))
        }
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = [(m: numExperts, n: hiddenSize)]
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// gpt-oss expert FFNs carry additive biases and use the clamped SwiGLU
    /// (limit 7.0, alpha 1.702) with a `+1` on the linear half.
    public var expertsHaveAdditiveBiases: Bool { family == .gptOss20b }
    public var usesClampedSwiGLU: Bool { family == .gptOss20b }
    /// gpt-oss attention carries additive q/k/v/o projection biases and a
    /// per-Q-head sink logit, all resident BF16; no QK norms, no output gate.
    public var hasAttentionBiases: Bool { family == .gptOss20b }
    public var hasAttentionSinks: Bool { family == .gptOss20b }
    /// Learned per-head Q/K RMS norms before RoPE (Qwen); gpt-oss has none.
    public var hasQKNorms: Bool { family != .gptOss20b }
    /// gpt-oss has no shared-expert FFN (its `intermediateSize` is 0); the
    /// MoE reduce then folds a zeroed branch instead of a dense MLP output.
    public var hasSharedExpert: Bool { intermediateSize > 0 }
    /// Arch-mandated YaRN rope scaling (gpt-oss: factor 32 over original
    /// 4096, theta 150000, full-head-dim rotation), applied unconditionally —
    /// independent of the user context-extension mode, which stays qwen-only.
    var archYaRN: YaRNRoPEParameters? {
        guard family == .gptOss20b else { return nil }
        return YaRNRoPEParameters(headDim: fullHeadDim,
                                  partialRotaryFactor: partialRotaryFactor,
                                  theta: fullRopeTheta,
                                  targetContextTokens: 32 * 4_096,
                                  originalContextTokens: 4_096)
    }

    /// MLA projection split for layers with mask value 3; nil for
    /// architectures without them. Kimi-Linear's [128 nope | 64 rope] query
    /// halves against a 512-latent cache row.
    public var mla: MLAConfig? {
        guard family == .kimiLinear48b else { return nil }
        return MLAConfig(latentDim: 512, qkNopeDim: 128, qkRopeDim: 64,
                         valueHeadDim: 128)
    }
    /// KDA's decay is per channel (`g[head, dk]`); Qwen's GDN decay is one
    /// scalar per head.
    public var linearAttentionPerChannelDecay: Bool { family == .kimiLinear48b }
    /// Kimi router: scores are sigmoid(logits); top-k selects by score plus
    /// `e_score_correction_bias`, weights are the original scores of the
    /// selected renormalized (÷ sum + 1e-20) and scaled by 2.446. Other
    /// families keep their softmax-over-selected routers.
    public var routerUsesSigmoidScores: Bool { family == .kimiLinear48b }
    public var routerHasCorrectionBias: Bool { family == .kimiLinear48b }
    public var routedScalingFactor: Double { family == .kimiLinear48b ? 2.446 : 1.0 }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsSWA(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 0 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public func layerIsMLA(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 3 }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
    public var hasMLALayers: Bool { fullAttentionLayerMask.contains(3) }
    public var hasSlidingWindowLayers: Bool { fullAttentionLayerMask.contains(0) }
}

/// Failure modes for the validation gates in `Model.load`.
public enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case unsupportedArchitecture(detail: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case expertCacheUnplaceable(detail: String)
    /// A Metal command buffer reported `.error`; the GPU work it carried
    /// (decode layer, head, or routed-expert pass) did not complete.
    case commandBufferFailed(detail: String)
    /// A runtime invariant the code believes is impossible was violated
    /// (arch/kernel mismatch, pipeline state corruption). Thrown instead of
    /// trapping so generation fails loudly without crashing the process.
    case internalInconsistency(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .unsupportedArchitecture(let detail):
            return "unsupported architecture: \(detail)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .expertCacheUnplaceable(let detail):
            return "expert cache cannot place requested experts: \(detail)"
        case .commandBufferFailed(let detail):
            return "Metal command buffer failed: \(detail)"
        case .internalInconsistency(let detail):
            return "internal inconsistency: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
/// unchecked-invariant: every stored property is a `let`; only MTLBuffer's
/// lack of Sendable forces @unchecked. The view describes where a tensor sits
/// in the resident buffer and never mutates it.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
