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
    case matrixPathUnavailable(reason: String)
    case commandEncoderFailed
    case shadowAllocationFailed(bytes: Int)
    case qGroupAllocationFailed(bytes: Int)

    public var description: String {
        switch self {
        case .matrixPathUnavailable(let reason):
            return "the matrix prefill attention path is unavailable: \(reason)"
        case .commandEncoderFailed:
            return "Failed to create Metal compute command encoder"
        case .shadowAllocationFailed(let bytes):
            return "Could not allocate \(bytes)-byte KV shadow buffers for matrix prefill attention"
        case .qGroupAllocationFailed(let bytes):
            return "Could not allocate a \(bytes)-byte group-major Q buffer for matrix prefill attention"
        }
    }
}


final class PrefillAttention {
    private let context: MetalContext
    private let psoCausalTiled: MTLComputePipelineState
    private let psoMLACausal: MTLComputePipelineState?
    private let psoKVDequant: MTLComputePipelineState?
    private let psoCausalMatrix: MTLComputePipelineState?
    private let psoQGroupPack: MTLComputePipelineState?
    let matrixUnavailableReason: String
    static let matrixHeadDim: UInt32 = 256
    static let matrixGroupHeads: UInt32 = 8
    /// `g2k256d` (v12 P7): a 128-thread threadgroup owns two query positions ×
    /// the eight query heads of one KV head against 256-key tiles.
    static let matrixKernelName = "attention_prefill_causal_matrix_g2k256d"
    static let matrixQueryRows = 2
    static let matrixThreadsPerThreadgroup = 128
    private var shadowK: MTLBuffer?
    private var shadowV: MTLBuffer?
    private var qGroup: MTLBuffer?

    var matrixPathAvailable: Bool {
        psoCausalMatrix != nil && psoKVDequant != nil && psoQGroupPack != nil
    }

    init(context: MetalContext, supportsMLA: Bool = false) throws {
        self.context = context
        self.psoCausalTiled = try context.pipeline("attention_prefill_causal_tiled")
        self.psoMLACausal = supportsMLA
            ? try context.pipeline("attention_prefill_mla_causal")
            : nil
        var kvDequant: MTLComputePipelineState?
        var causalMatrix: MTLComputePipelineState?
        var qGroupPack: MTLComputePipelineState?
        var matrixReason = ""
        do {
            kvDequant = try context.pipeline("attention_prefill_kv_dequant")
            causalMatrix = try context.pipeline(Self.matrixKernelName)
            qGroupPack = try context.pipeline("attention_prefill_q_group_pack")
        } catch {
            kvDequant = nil
            causalMatrix = nil
            qGroupPack = nil
            matrixReason = "\(error)"
        }
        self.psoKVDequant = kvDequant
        self.psoCausalMatrix = causalMatrix
        self.psoQGroupPack = qGroupPack
        self.matrixUnavailableReason = matrixReason
    }

    func encodeCausal(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int = 0,
                             k: MTLBuffer, kOffset: Int = 0,
                             v: MTLBuffer, vOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             params: PrefillAttentionParams,
                             kvRingCapacity: UInt32 = 0,
                             sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                             minimumQueries: UInt32 = matrixPathMinimumQueries) throws {
        if matrixPathAvailable,
           Self.matrixPathAccepts(params, kvRingCapacity: kvRingCapacity, hasSinks: sinks != nil,
                                  minimumQueries: minimumQueries) {
            validate(params)
            try encodeMatrix(commandBuffer: commandBuffer,
                             q: q, qOffset: qOffset,
                             k: k, kOffset: kOffset,
                             v: v, vOffset: vOffset,
                             out: out, outOffset: outOffset,
                             params: params)
            return
        }
        try encodeTiled(commandBuffer: commandBuffer,
                        q: q, qOffset: qOffset,
                        k: k, kOffset: kOffset,
                        v: v, vOffset: vOffset,
                        out: out, outOffset: outOffset,
                        params: params,
                        kvRingCapacity: kvRingCapacity,
                        sinks: sinks, sinksOffset: sinksOffset)
    }

    func encodeTiled(commandBuffer: MTLCommandBuffer,
                     q: MTLBuffer, qOffset: Int = 0,
                     k: MTLBuffer, kOffset: Int = 0,
                     v: MTLBuffer, vOffset: Int = 0,
                     out: MTLBuffer, outOffset: Int = 0,
                     params: PrefillAttentionParams,
                     kvRingCapacity: UInt32 = 0,
                     sinks: MTLBuffer? = nil, sinksOffset: Int = 0) throws {
        validate(params)
        let pipeline = causalTiledPipeline(kvRingCapacity: kvRingCapacity,
                                           hasSinks: sinks != nil)
        let headDim = Int(params.headDim)
        let threadWidth = max(1, pipeline.threadExecutionWidth)
        let threadCount = roundUp(max(threadWidth, headDim), toMultipleOf: threadWidth)
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
        enc.dispatchThreadgroups(
            MTLSize(width: Int(params.queryCount),
                    height: Int(params.numQHeads),
                    depth: 1),
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
            throw PrefillAttentionError.matrixPathUnavailable(
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
    /// causal shape; everything else keeps the scalar kernel. The runner
    /// passes its own minimum to `encodeCausal`.
    static let matrixPathMinimumQueries: UInt32 = 32
    /// The KV shadow (see `ensureShadow`) grows with `kvValidCount` and is
    /// never released; this ceiling keeps a very long context off the matrix
    /// path instead of letting the shadow grow without bound.
    static let matrixPathMaxContext: UInt32 = 65_536

    static func matrixPathAccepts(_ params: PrefillAttentionParams,
                                  kvRingCapacity: UInt32,
                                  hasSinks: Bool,
                                  minimumQueries: UInt32 = matrixPathMinimumQueries) -> Bool {
        params.headDim == matrixHeadDim
            && params.numKVHeads == 2
            && params.numQHeads == params.numKVHeads * matrixGroupHeads
            && kvRingCapacity == 0
            && !hasSinks
            && (params.slidingWindow == 0 || params.slidingWindow >= params.kvValidCount)
            && params.queryCount >= minimumQueries
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
            throw PrefillAttentionError.matrixPathUnavailable(reason: matrixUnavailableReason)
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

        let qGroup = try encodeQGroupPack(encoder: enc, q: q, qOffset: qOffset, params: params)
        enc.setComputePipelineState(psoCausalMatrix)
        enc.setBuffer(qGroup, offset: 0, index: 0)
        enc.setBuffer(shadowK, offset: 0, index: 1)
        enc.setBuffer(shadowV, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 4)
        let rows = Self.matrixQueryRows
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(params.queryCount) + rows - 1) / rows,
                    height: Int(params.numKVHeads),
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.matrixThreadsPerThreadgroup,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Packs the strided Q rows group-major, `[numKVHeads][queryCount * 8][headDim]`
    /// with row stride `headDim`, and returns the buffer holding them.
    func encodeQGroupPack(encoder enc: MTLComputeCommandEncoder,
                          q: MTLBuffer, qOffset: Int,
                          params: PrefillAttentionParams) throws -> MTLBuffer {
        guard let psoQGroupPack else {
            throw PrefillAttentionError.matrixPathUnavailable(reason: matrixUnavailableReason)
        }
        let bytes = Int(params.queryCount) * Int(params.numQHeads) * Int(params.headDim)
            * MemoryLayout<Float16>.stride
        let qGroup = try ensureQGroup(bytes: bytes)
        var p = params
        enc.setComputePipelineState(psoQGroupPack)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(qGroup, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<PrefillAttentionParams>.stride, index: 2)
        enc.dispatchThreads(
            MTLSize(width: Int(params.headDim), height: Int(params.numQHeads),
                    depth: Int(params.queryCount)),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        return qGroup
    }

    private static let shadowQuantumBytes = 8 << 20

    /// Chunk-sized, not context-sized: grown in the shadow's quanta and never
    /// released, like the shadow.
    private func ensureQGroup(bytes: Int) throws -> MTLBuffer {
        if let qGroup, qGroup.length >= bytes { return qGroup }
        let quantum = Self.shadowQuantumBytes
        let rounded = (bytes + quantum - 1) / quantum * quantum
        guard let buffer = context.device.makeBuffer(length: rounded, options: .storageModePrivate) else {
            throw PrefillAttentionError.qGroupAllocationFailed(bytes: rounded)
        }
        qGroup = buffer
        return buffer
    }

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
