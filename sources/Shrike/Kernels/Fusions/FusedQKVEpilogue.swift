import Foundation
import Metal

/// Single-kernel Q/K/V epilogue.
///
/// Equivalent to:
///   Q = rmsnorm_bf16w_perhead(Q, q_norm); rope(Q)
///   K = rmsnorm_bf16w_perhead(K, k_norm); rope(K)
///   V = rmsnorm_no_scale_perhead(V)            (skipped when normalizeV: false)
///
/// The rope is proportional NeoX by default; `subdimRope: true` selects the
/// Qwen-style partial rotation (pairs (i, RP+i) inside the first 2·RP
/// elements, frequencies over 2·RP) matching `rope_neox_subdim`.
final class FusedQKVEpilogue {
    private struct Shape: Hashable {
        var headDim: UInt32
        var numQHeads: UInt32
        var numKVHeads: UInt32
        var rotatedPairs: UInt32
        var subdimRope: Bool
    }

    private let context: MetalContext
    private let pso: MTLComputePipelineState
    private var specializedPSOs: [Shape: MTLComputePipelineState]
    private static let realDecodeShapes: [Shape] = [
        Shape(headDim: 256, numQHeads: 16, numKVHeads: 8, rotatedPairs: 128,
              subdimRope: false),
        Shape(headDim: 512, numQHeads: 16, numKVHeads: 2, rotatedPairs: 64,
              subdimRope: false),
        Shape(headDim: 256, numQHeads: 16, numKVHeads: 2, rotatedPairs: 32,
              subdimRope: true),
    ]

    init(context: MetalContext) throws {
        self.context = context
        self.pso = try context.pipeline("fused_qkv_epilogue")
        var variants: [Shape: MTLComputePipelineState] = [:]
        for shape in Self.realDecodeShapes {
            variants[shape] = try Self.makePipeline(context: context, shape: shape)
        }
        self.specializedPSOs = variants
    }

    private static func makePipeline(context: MetalContext,
                                     shape: Shape) throws -> MTLComputePipelineState {
        try context.pipeline(
            "fused_qkv_epilogue",
            constants: [
                MetalFunctionConstant(index: 82, value: .uint32(shape.headDim)),
                MetalFunctionConstant(index: 83, value: .uint32(shape.numQHeads)),
                MetalFunctionConstant(index: 84, value: .uint32(shape.numKVHeads)),
                MetalFunctionConstant(index: 85, value: .uint32(shape.rotatedPairs)),
                MetalFunctionConstant(index: 86, value: .bool(true)),
                MetalFunctionConstant(index: 87, value: .bool(shape.subdimRope)),
            ])
    }

    private func pipeline(for shape: Shape) throws -> MTLComputePipelineState {
        if let cached = specializedPSOs[shape] { return cached }
        // The subdim pairing exists only behind its function constant, so an
        // unspecialized fallback would silently run the proportional math.
        guard shape.subdimRope else { return pso }
        let made = try Self.makePipeline(context: context, shape: shape)
        specializedPSOs[shape] = made
        return made
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                       q: MTLBuffer,
                       qOffset: Int = 0,
                       k: MTLBuffer,
                       kOffset: Int = 0,
                       v: MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: MTLBuffer,
                       kWeightOffset: Int = 0,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       position: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float,
                       subdimRope: Bool = false,
                       normalizeV: Bool = true) throws {
        guard let enc = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { enc.endEncoding() }
        try encode(encoder: enc,
                   q: q, qOffset: qOffset,
                   k: k, kOffset: kOffset,
                   v: v, vOffset: vOffset,
                   qWeight: qWeight, qWeightOffset: qWeightOffset,
                   kWeight: kWeight, kWeightOffset: kWeightOffset,
                   headDim: headDim,
                   numQHeads: numQHeads,
                   numKVHeads: numKVHeads,
                   position: position,
                   theta: theta,
                   rotatedPairs: rotatedPairs,
                   eps: eps,
                   subdimRope: subdimRope,
                   normalizeV: normalizeV)
    }

    func encode(encoder enc: MTLComputeCommandEncoder,
                       q: MTLBuffer,
                       qOffset: Int = 0,
                       k: MTLBuffer,
                       kOffset: Int = 0,
                       v: MTLBuffer,
                       vOffset: Int = 0,
                       qWeight: MTLBuffer,
                       qWeightOffset: Int = 0,
                       kWeight: MTLBuffer,
                       kWeightOffset: Int = 0,
                       headDim: UInt32,
                       numQHeads: UInt32,
                       numKVHeads: UInt32,
                       position: UInt32,
                       theta: Float,
                       rotatedPairs: UInt32,
                       eps: Float,
                       subdimRope: Bool = false,
                       normalizeV: Bool = true) throws {
        precondition(headDim <= 512,
                     "headDim > 512 exceeds the fused QKV epilogue scratch")
        precondition(rotatedPairs * 2 <= headDim,
                     "rotatedPairs must fit inside one NeoX head")
        enc.setComputePipelineState(
            try pipeline(for: Shape(headDim: headDim,
                                    numQHeads: numQHeads,
                                    numKVHeads: numKVHeads,
                                    rotatedPairs: rotatedPairs,
                                    subdimRope: subdimRope)))
        enc.setBuffer(q,       offset: qOffset,       index: 0)
        enc.setBuffer(k,       offset: kOffset,       index: 1)
        enc.setBuffer(v,       offset: vOffset,       index: 2)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 3)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 4)
        var headDimVar = headDim
        var numQVar = numQHeads
        var numKVVar = numKVHeads
        var posVar = position
        var thetaVar = theta
        var rotatedVar = rotatedPairs
        var epsVar = eps
        enc.setBytes(&headDimVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&numQVar,    length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&numKVVar,   length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&posVar,     length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&thetaVar,   length: MemoryLayout<Float>.size,  index: 9)
        enc.setBytes(&rotatedVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&epsVar,     length: MemoryLayout<Float>.size,  index: 11)

        let threads = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let groups = Int(numQHeads + (normalizeV ? 2 : 1) * numKVHeads)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
