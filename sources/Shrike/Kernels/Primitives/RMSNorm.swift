import Foundation
import Metal

/// Swift wrapper for the RMSNorm kernel family.
///
///   inv  = 1 / sqrt(mean(x[i]^2) + eps)
///   y[i] = x[i] * inv * weight[i]            // BF16 weight
///   y[i] = x[i] * inv                        // .noScale
///
/// One row per dispatch — callers needing per-head dispatch (q/k_norm) loop
/// across heads and offset both `x` and `out`. Single-row is the shape we hit
/// at decode: a token at a time, 61 norm dispatches per token before fusion.
final class RMSNorm {

    private let psoBF16: MTLComputePipelineState
    private let psoResidualBF16: MTLComputePipelineState
    private let psoNoScale: MTLComputePipelineState
    private let psoBF16PerHead: MTLComputePipelineState
    private let psoBF16Rows: MTLComputePipelineState
    private let psoNoScalePerHead: MTLComputePipelineState
    private let psoBF16D2816: MTLComputePipelineState
    private let psoNoScaleD2816: MTLComputePipelineState
    private let psoBF16PerHead256: MTLComputePipelineState
    private let psoBF16PerHead512: MTLComputePipelineState
    private let psoNoScalePerHead256: MTLComputePipelineState
    private let psoNoScalePerHead512: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoBF16     = try context.pipeline("rmsnorm_bf16w")
        self.psoResidualBF16 = try context.pipeline("residual_add_rmsnorm_bf16w")
        self.psoNoScale  = try context.pipeline("rmsnorm_no_scale")
        self.psoBF16PerHead    = try context.pipeline("rmsnorm_bf16w_perhead")
        self.psoBF16Rows       = try context.pipeline("rmsnorm_bf16w_rows")
        self.psoNoScalePerHead = try context.pipeline("rmsnorm_no_scale_perhead")
        self.psoBF16D2816 = try Self.specializedPipeline(context,
                                                         "rmsnorm_bf16w",
                                                         d: 2816)
        self.psoNoScaleD2816 = try Self.specializedPipeline(context,
                                                            "rmsnorm_no_scale",
                                                            d: 2816)
        self.psoBF16PerHead256 = try Self.specializedPipeline(context,
                                                              "rmsnorm_bf16w_perhead",
                                                              d: 256)
        self.psoBF16PerHead512 = try Self.specializedPipeline(context,
                                                              "rmsnorm_bf16w_perhead",
                                                              d: 512)
        self.psoNoScalePerHead256 = try Self.specializedPipeline(context,
                                                                 "rmsnorm_no_scale_perhead",
                                                                 d: 256)
        self.psoNoScalePerHead512 = try Self.specializedPipeline(context,
                                                                 "rmsnorm_no_scale_perhead",
                                                                 d: 512)
    }

    /// Encode the BF16-weight variant (weighted norms).
    func encodeBF16W(commandBuffer: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            d: UInt32,
                            eps: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeBF16W(encoder: enc, x: x, xOffset: xOffset,
                    weight: weight, weightOffset: weightOffset,
                    out: out, outOffset: outOffset, d: d, eps: eps)
        enc.endEncoding()
    }

    func encodeBF16W(encoder: MTLComputeCommandEncoder,
                            x: MTLBuffer, xOffset: Int = 0,
                            weight: MTLBuffer, weightOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            d: UInt32,
                            eps: Float) {
        encodeWeighted(encoder: encoder,
                       pso: d == 2816 ? psoBF16D2816 : psoBF16,
                       x: x, xOffset: xOffset,
                       weight: weight, weightOffset: weightOffset,
                       out: out, outOffset: outOffset,
                       d: d, eps: eps)
    }

    /// Fused in-place residual add + weighted norm, bitwise identical to
    /// encodeResidualAdd followed by encodeBF16W.
    func encodeResidualAddBF16W(encoder enc: MTLComputeCommandEncoder,
                                hidden: MTLBuffer, hiddenOffset: Int = 0,
                                delta: MTLBuffer, deltaOffset: Int = 0,
                                weight: MTLBuffer, weightOffset: Int = 0,
                                out: MTLBuffer, outOffset: Int = 0,
                                d: UInt32,
                                eps: Float) {
        enc.setComputePipelineState(psoResidualBF16)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(delta,  offset: deltaOffset,  index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var dVar   = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 5)
        dispatchOneRow(enc: enc, pso: psoResidualBF16)
    }

    /// Encode the no-scale variant (v_norm, router internal norm).
    func encodeNoScale(commandBuffer: MTLCommandBuffer,
                              x: MTLBuffer, xOffset: Int = 0,
                              out: MTLBuffer, outOffset: Int = 0,
                              d: UInt32,
                              eps: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let pso = d == 2816 ? psoNoScaleD2816 : psoNoScale
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,   offset: xOffset,   index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var dVar   = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 3)
        dispatchOneRow(enc: enc, pso: pso)
        enc.endEncoding()
    }

    /// BF16-weight RMSNorm applied to `numHeads` contiguous heads of width
    /// `headDim` in one dispatch. `weight` is the shared
    /// [headDim] per-head gain. In-place safe (each head's threadgroup touches
    /// only its own region).
    func encodeBF16WPerHead(commandBuffer: MTLCommandBuffer,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   weight: MTLBuffer, weightOffset: Int = 0,
                                   out: MTLBuffer, outOffset: Int = 0,
                                   headDim: UInt32, numHeads: Int,
                                   eps: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeBF16WPerHead(encoder: enc, x: x, xOffset: xOffset,
                           weight: weight, weightOffset: weightOffset,
                           out: out, outOffset: outOffset,
                           headDim: headDim, numHeads: numHeads, eps: eps)
        enc.endEncoding()
    }

    func encodeBF16WPerHead(encoder enc: MTLComputeCommandEncoder,
                                   x: MTLBuffer, xOffset: Int = 0,
                                   weight: MTLBuffer, weightOffset: Int = 0,
                                   out: MTLBuffer, outOffset: Int = 0,
                                   headDim: UInt32, numHeads: Int,
                                   eps: Float) {
        let pso = perHeadPipeline(headDim: headDim,
                                  base: psoBF16PerHead,
                                  p256: psoBF16PerHead256,
                                  p512: psoBF16PerHead512)
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var hd = headDim
        var epsVar = eps
        enc.setBytes(&hd,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 4)
        let w = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    /// No-scale RMSNorm over `numHeads` contiguous heads (v_norm), one dispatch.
    func encodeNoScalePerHead(commandBuffer: MTLCommandBuffer,
                                     x: MTLBuffer, xOffset: Int = 0,
                                     out: MTLBuffer, outOffset: Int = 0,
                                     headDim: UInt32, numHeads: Int,
                                     eps: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        let pso = perHeadPipeline(headDim: headDim,
                                  base: psoNoScalePerHead,
                                  p256: psoNoScalePerHead256,
                                  p512: psoNoScalePerHead512)
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,   offset: xOffset,   index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var hd = headDim
        var epsVar = eps
        enc.setBytes(&hd,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 3)
        let w = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: numHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// BF16-weight RMSNorm over the leading `d` elements of `rows` rows of
    /// `rowStride` elements each, one dispatch (MLA kv_a latent norm — the
    /// trailing k_pe elements of each row stay untouched). In-place safe.
    func encodeBF16WRows(commandBuffer: MTLCommandBuffer,
                         x: MTLBuffer, xOffset: Int = 0,
                         weight: MTLBuffer, weightOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0,
                         d: UInt32, rows: Int, rowStrideElements: UInt32,
                         eps: Float) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(psoBF16Rows)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var dVar = d
        var epsVar = eps
        var strideVar = rowStrideElements
        enc.setBytes(&dVar,      length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar,    length: MemoryLayout<Float>.size,  index: 4)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 5)
        let w = min(Int(psoBF16Rows.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeWeighted(encoder enc: MTLComputeCommandEncoder,
                                pso: MTLComputePipelineState,
                                x: MTLBuffer, xOffset: Int,
                                weight: MTLBuffer, weightOffset: Int,
                                out: MTLBuffer, outOffset: Int,
                                d: UInt32,
                                eps: Float) {
        enc.setComputePipelineState(pso)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var dVar   = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 4)
        dispatchOneRow(enc: enc, pso: pso)
    }

    private func dispatchOneRow(enc: MTLComputeCommandEncoder,
                                pso: MTLComputePipelineState) {
        // One threadgroup, 256 threads. 256 is a multiple of the 32-lane SIMD
        // width on Apple silicon — no trailing partial SIMD-group to special-case.
        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            d: UInt32) throws -> MTLComputePipelineState {
        try context.pipeline(
            name,
            constants: [
                MetalFunctionConstant(index: 30, value: .uint32(d)),
                MetalFunctionConstant(index: 31, value: .bool(true)),
            ])
    }

    private func perHeadPipeline(headDim: UInt32,
                                 base: MTLComputePipelineState,
                                 p256: MTLComputePipelineState,
                                 p512: MTLComputePipelineState) -> MTLComputePipelineState {
        switch headDim {
        case 256: return p256
        case 512: return p512
        default:  return base
        }
    }
}
