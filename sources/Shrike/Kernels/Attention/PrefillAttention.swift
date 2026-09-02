import Foundation
import Metal

struct PrefillAttentionParams: Sendable, Equatable {
    var startPosition: UInt32
    var queryCount: UInt32
    var headDim: UInt32
    var numQHeads: UInt32
    var numKVHeads: UInt32
    var kvValidCount: UInt32
    var slidingWindow: UInt32
    var kvTokenStrideElements: UInt32
    var qTokenStrideElements: UInt32
    var oTokenStrideElements: UInt32
    var scale: Float
    var kvBits: UInt32
    var kvTokenStrideBytes: UInt32
    var kvValueBytes: UInt32
    var kvGroupSize: UInt32

    init(startPosition: UInt32,
                queryCount: UInt32,
                headDim: UInt32,
                numQHeads: UInt32,
                numKVHeads: UInt32,
                kvValidCount: UInt32,
                slidingWindow: UInt32,
                kvTokenStrideElements: UInt32,
                qTokenStrideElements: UInt32,
                oTokenStrideElements: UInt32,
                scale: Float,
                kvBits: UInt32 = 16,
                kvTokenStrideBytes: UInt32 = 0,
                kvValueBytes: UInt32 = 0,
                kvGroupSize: UInt32 = UInt32(KVCacheManager.quantizationGroupSize)) {
        self.startPosition = startPosition
        self.queryCount = queryCount
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.kvValidCount = kvValidCount
        self.slidingWindow = slidingWindow
        self.kvTokenStrideElements = kvTokenStrideElements
        self.qTokenStrideElements = qTokenStrideElements
        self.oTokenStrideElements = oTokenStrideElements
        self.scale = scale
        self.kvBits = kvBits
        self.kvTokenStrideBytes = kvTokenStrideBytes
        self.kvValueBytes = kvValueBytes
        self.kvGroupSize = kvGroupSize
    }
}


enum PrefillAttentionError: Error, CustomStringConvertible {
    case tensorOpsUnavailable(reason: String)
    case commandEncoderFailed
    case shadowAllocationFailed(bytes: Int)

    public var description: String {
        switch self {
        case .tensorOpsUnavailable(let reason):
            return "TensorOps 2D prefill attention requested but unavailable: \(reason)"
        case .commandEncoderFailed:
            return "Failed to create Metal compute command encoder"
        case .shadowAllocationFailed(let bytes):
            return "Could not allocate \(bytes)-byte KV shadow buffers for matrix prefill attention"
        }
    }
}


final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoMLACausal: MTLComputePipelineState?
    private let psoFullTensorOps2DValidityV2: MTLComputePipelineState?
    /// K7: recorded once at init so an explicit TensorOps path request can
    /// throw the real reason instead of a bare `preconditionFailure`.
    private let tensorOpsUnavailableReason: String
    private let psoKVDequant: MTLComputePipelineState?
    private let psoCausalMatrix: MTLComputePipelineState?
    let matrixUnavailableReason: String
    /// Tile geometry: rows of queries per threadgroup and simdgroups per
    /// threadgroup. `SHRIKE_ATTN_MATRIX_TILE=r64s8` selects the wider variant
    /// for A/B measurement; the default is the measured production choice.
    static let matrixTile: (rows: Int, simdgroups: Int) =
        ProcessInfo.processInfo.environment["SHRIKE_ATTN_MATRIX_TILE"] == "r64s8"
            ? (64, 8) : (32, 4)
    static let matrixHeadDim: UInt32 = 256
    private var shadowK: MTLBuffer?
    private var shadowV: MTLBuffer?

    var matrixPathAvailable: Bool { psoCausalMatrix != nil && psoKVDequant != nil }

    init(context: MetalContext, supportsMLA: Bool = false) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoMLACausal = supportsMLA
            ? try context.pipeline("attention_prefill_mla_causal")
            : nil
        var kvDequant: MTLComputePipelineState?
        var causalMatrix: MTLComputePipelineState?
        var matrixReason = ""
        do {
            kvDequant = try context.pipeline("attention_prefill_kv_dequant")
            causalMatrix = try context.pipeline(
                "attention_prefill_causal_matrix_r\(Self.matrixTile.rows)s\(Self.matrixTile.simdgroups)")
        } catch {
            kvDequant = nil
            causalMatrix = nil
            matrixReason = "\(error)"
        }
        self.psoKVDequant = kvDequant
        self.psoCausalMatrix = causalMatrix
        self.matrixUnavailableReason = matrixReason
        if context.device.supportsFamily(.apple10) {
            do {
                self.psoFullTensorOps2DValidityV2 = try context.pipeline(
                    "attention_prefill_full_tensorops_2d_validity_v2")
                self.tensorOpsUnavailableReason = ""
            } catch {
                self.psoFullTensorOps2DValidityV2 = nil
                self.tensorOpsUnavailableReason = "\(error)"
            }
        } else {
            self.psoFullTensorOps2DValidityV2 = nil
            self.tensorOpsUnavailableReason =
                "device does not support Apple10 MPP tensor operations"
        }
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                             path: RuntimePrefillAttentionPath = .causalTiled) throws {
        validate(params)

        if path == .causalMatrix, matrixPathAvailable,
           Self.matrixPathAccepts(params, kvRingCapacity: kvRingCapacity, hasSinks: sinks != nil) {
            try encodeMatrix(commandBuffer: commandBuffer,
                             q: q, qOffset: qOffset,
                             k: k, kOffset: kOffset,
                             v: v, vOffset: vOffset,
                             out: out, outOffset: outOffset,
                             params: params)
            return
        }

        let requestsTensorOps = path == .fullTensorOps2DPreferred
            || path == .fullTensorOps2DValidityV2
            || path == .causalMatrix
        // The pinned model uses 512/16/2 only for full attention; its
        // sliding-window layers use 256/16/8. A future model that reuses this
        // shape for sliding attention must add a full-visibility check here.
        let tensorOpsShape = requestsTensorOps
            && params.kvBits == 16
            && kvRingCapacity == 0
            && sinks == nil
            && params.headDim == 512
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && params.scale == 1.0
        let tensorOpsPipeline = tensorOpsShape ? psoFullTensorOps2DValidityV2 : nil
        let useTensorOps = tensorOpsPipeline != nil
        let pipeline: MTLComputePipelineState
        if let tensorOpsPipeline {
            pipeline = tensorOpsPipeline
        } else if tensorOpsShape && path == .fullTensorOps2DValidityV2 {
            // K7: the caller explicitly requested the TensorOps path — fail
            // loudly with the recorded reason instead of crashing or silently
            // running a different kernel. Only auto-selected paths fall back.
            throw PrefillAttentionError.tensorOpsUnavailable(
                reason: tensorOpsUnavailableReason.isEmpty
                    ? "TensorOps pipeline failed to compile"
                    : tensorOpsUnavailableReason)
        } else {
            // Explicit mode also falls back for incompatible shapes. Benchmark
            // fixtures must use 512/16/2 to prove that TensorOps ran.
            pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity,
                                           hasSinks: sinks != nil)
        }
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = useTensorOps
            ? 128
            : roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "tiled prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        if let sinks { enc.setBuffer(sinks, offset: sinksOffset, index: 5) }
        let groups = useTensorOps
            ? MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads) / 8,
                      depth: 1)
            : MTLSize(width: Int(params.queryCount),
                      height: Int(params.numQHeads),
                      depth: 1)
        enc.dispatchThreadgroups(
            groups,
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }


    /// MLA (Kimi-Linear) causal prefill over fused FP16 cache rows.
    /// `params.headDim` is the qk width (576); output rows are `vDim` per
    /// head, so `oTokenStrideElements = numQHeads * vDim` — smaller than the
    /// generic validator allows, hence the dedicated checks here.
    func encodeMLACausal(commandBuffer: MTLCommandBuffer,
                         q: MTLBuffer, qOffset: Int = 0,
                         kv: MTLBuffer, kvOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         params: PrefillAttentionParams,
                         vDim: UInt32) throws {
        guard let pipeline = psoMLACausal else {
            throw PrefillAttentionError.tensorOpsUnavailable(
                reason: "encodeMLACausal on a PrefillAttention built without supportsMLA")
        }
        precondition(params.headDim > 0 && params.headDim % 32 == 0
                     && params.headDim <= 576,
                     "MLA qk width must be a positive multiple of 32 up to 576")
        precondition(vDim > 0 && vDim <= params.headDim,
                     "MLA vDim must fit the KV row prefix")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numKVHeads == 1, "MLA runs as MQA")
        precondition(params.kvBits == 16, "MLA KV rows are always FP16")
        precondition(params.slidingWindow == 0, "MLA layers are full attention")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * vDim,
                     "output token stride is too small")
        precondition(params.kvTokenStrideElements >= params.headDim,
                     "KV token stride is too small")
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(kv, offset: kvOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 3)
        var vd = vDim
        enc.setBytes(&vd, length: MemoryLayout<UInt32>.size, index: 4)
        let threadCount = Int(params.headDim)
        precondition(threadCount <= pipeline.maxTotalThreadsPerThreadgroup,
                     "MLA prefill attention requires headDim <= maxTotalThreadsPerThreadgroup")
        enc.dispatchThreadgroups(
            MTLSize(width: Int(params.queryCount),
                    height: Int(params.numQHeads),
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadCount, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The matrix kernel is written for the 256-wide, 16/2-head, fully visible
    /// causal shape; everything else keeps the scalar kernel.
    static let matrixPathMinimumQueries: UInt32 = 32
    /// The KV shadow (see `ensureShadow`) grows with `kvValidCount` and is
    /// never released; this ceiling keeps a very long context off the matrix
    /// path instead of letting the shadow grow without bound.
    static let matrixPathMaxContext: UInt32 = 65_536

    static func matrixPathAccepts(_ params: PrefillAttentionParams,
                                  kvRingCapacity: UInt32,
                                  hasSinks: Bool) -> Bool {
        params.headDim == matrixHeadDim
            && params.numQHeads == 16
            && params.numKVHeads == 2
            && kvRingCapacity == 0
            && !hasSinks
            && (params.slidingWindow == 0 || params.slidingWindow >= params.kvValidCount)
            && params.queryCount >= matrixPathMinimumQueries
            && params.kvValidCount > 0
            && params.kvValidCount <= matrixPathMaxContext
    }

    private func encodeMatrix(commandBuffer: MTLCommandBuffer,
                              q: MTLBuffer, qOffset: Int,
                              k: MTLBuffer, kOffset: Int,
                              v: MTLBuffer, vOffset: Int,
                              out: MTLBuffer, outOffset: Int,
                              params: PrefillAttentionParams) throws {
        guard let psoKVDequant, let psoCausalMatrix else {
            throw PrefillAttentionError.tensorOpsUnavailable(reason: matrixUnavailableReason)
        }
        let elements = Int(params.numKVHeads * params.headDim)
        let shadowBytes = Int(params.kvValidCount) * elements * MemoryLayout<Float16>.stride
        let (shadowK, shadowV) = try ensureShadow(bytes: shadowBytes)

        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw PrefillAttentionError.commandEncoderFailed
        }
        var p = params
        enc.setComputePipelineState(psoKVDequant)
        enc.setBuffer(k, offset: kOffset, index: 0)
        enc.setBuffer(v, offset: vOffset, index: 1)
        enc.setBuffer(shadowK, offset: 0, index: 2)
        enc.setBuffer(shadowV, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        enc.dispatchThreads(
            MTLSize(width: elements, height: Int(params.kvValidCount), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(psoCausalMatrix)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(shadowK, offset: 0, index: 1)
        enc.setBuffer(shadowV, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        let rows = Self.matrixTile.rows
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(params.queryCount) + rows - 1) / rows,
                    height: Int(params.numQHeads),
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.matrixTile.simdgroups,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    private static let shadowQuantumBytes = 8 << 20

    private func ensureShadow(bytes: Int) throws -> (MTLBuffer, MTLBuffer) {
        if let shadowK, let shadowV, shadowK.length >= bytes, shadowV.length >= bytes {
            return (shadowK, shadowV)
        }
        let quantum = Self.shadowQuantumBytes
        let rounded = (bytes + quantum - 1) / quantum * quantum
        guard let newK = context.device.makeBuffer(length: rounded, options: .storageModePrivate),
              let newV = context.device.makeBuffer(length: rounded, options: .storageModePrivate) else {
            throw PrefillAttentionError.shadowAllocationFailed(bytes: rounded)
        }
        shadowK = newK
        shadowV = newV
        return (newK, newV)
    }

    private func validate(_ params: PrefillAttentionParams) {
        precondition(params.headDim > 0, "headDim must be positive")
        precondition(params.queryCount > 0, "queryCount must be positive")
        precondition(params.numQHeads > 0, "numQHeads must be positive")
        precondition(params.numKVHeads > 0, "numKVHeads must be positive")
        precondition(params.numQHeads % params.numKVHeads == 0,
                     "numQHeads must be divisible by numKVHeads")
        precondition(params.qTokenStrideElements >= params.numQHeads * params.headDim,
                     "q token stride is too small")
        precondition(params.oTokenStrideElements >= params.numQHeads * params.headDim,
                     "output token stride is too small")
        if params.kvBits == 16 {
            precondition(params.kvTokenStrideElements >= params.numKVHeads * params.headDim,
                         "KV token stride is too small")
        } else {
            precondition(params.kvBits == 4 || params.kvBits == 8,
                         "KV bits must be 4, 8, or 16")
            precondition(params.kvTokenStrideBytes > 0,
                         "quantized KV token stride must be positive")
        }
        precondition(params.startPosition + params.queryCount <= params.kvValidCount,
                     "kvValidCount must include all in-flight query rows")
    }


    private func roundUp(_ value: Int, toMultipleOf multiple: Int) -> Int {
        ((value + multiple - 1) / multiple) * multiple
    }

    private func causalTiledPipeline(kvRingCapacity: UInt32,
                                     hasSinks: Bool) -> MTLComputePipelineState {
        guard kvRingCapacity > 0 || hasSinks else { return psoCausalTiled }
        var constants: [MetalFunctionConstant] = []
        if kvRingCapacity > 0 {
            constants.append(MetalFunctionConstant(index: 76, value: .uint32(kvRingCapacity)))
        }
        if hasSinks {
            constants.append(MetalFunctionConstant(index: 122, value: .bool(true)))
        }
        do {
            return try context.pipeline("attention_prefill_causal_tiled",
                                        constants: constants)
        } catch {
            preconditionFailure("failed to build prefill attention pipeline: \(error)")
        }
    }
}
