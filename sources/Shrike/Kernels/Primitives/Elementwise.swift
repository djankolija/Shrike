import Foundation
import Metal

/// Small elementwise kernels used by the Qwen 3.6 layer graph: the
/// full-attention output gate, the shared-expert scalar gate, and the plain
/// pre-norm residual add (architectures without a fused sandwich tail).
final class Elementwise {
    private let sigmoidGateMulPSO: MTLComputePipelineState
    private let sigmoidScalarMulPSO: MTLComputePipelineState
    private let residualAddPSO: MTLComputePipelineState
    private let splitQGatePSO: MTLComputePipelineState
    private let concatRowsPSO: MTLComputePipelineState
    private let biasAddPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.sigmoidGateMulPSO = try context.pipeline("sigmoid_gate_mul_fp16")
        self.sigmoidScalarMulPSO = try context.pipeline("sigmoid_scalar_mul_fp16")
        self.residualAddPSO = try context.pipeline("residual_add_fp16")
        self.splitQGatePSO = try context.pipeline("split_q_gate_fp16")
        self.concatRowsPSO = try context.pipeline("concat_rows_fp16")
        self.biasAddPSO = try context.pipeline("bias_add_fp16")
    }

    /// packed [H, 2D] per-head [query ; gate] → q [H, D], gate [H, D].
    /// `rows` > 1 processes consecutive token rows (packed stride 2*H*D,
    /// output strides H*D).
    ///
    /// K5: all rows are dispatched from ONE encoder (a per-row dispatch loop
    /// with per-row buffer offsets) instead of creating one encoder per row.
    func encodeSplitQGate(commandBuffer: MTLCommandBuffer,
                          packed: MTLBuffer, packedOffset: Int = 0,
                          q: MTLBuffer, qOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          heads: Int, dim: Int, rows: Int = 1) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeSplitQGate(encoder: encoder, packed: packed, packedOffset: packedOffset,
                         q: q, qOffset: qOffset, gate: gate, gateOffset: gateOffset,
                         heads: heads, dim: dim, rows: rows)
        encoder.endEncoding()
    }

    func encodeSplitQGate(encoder: MTLComputeCommandEncoder,
                          packed: MTLBuffer, packedOffset: Int = 0,
                          q: MTLBuffer, qOffset: Int = 0,
                          gate: MTLBuffer, gateOffset: Int = 0,
                          heads: Int, dim: Int, rows: Int = 1) {
        let rowElems = heads * dim
        encoder.setComputePipelineState(splitQGatePSO)
        var headCount = UInt32(heads)
        var headDim = UInt32(dim)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&headDim, length: MemoryLayout<UInt32>.size, index: 4)
        for row in 0..<rows {
            encoder.setBuffer(packed, offset: packedOffset + row * 2 * rowElems * 2, index: 0)
            encoder.setBuffer(q, offset: qOffset + row * rowElems * 2, index: 1)
            encoder.setBuffer(gate, offset: gateOffset + row * rowElems * 2, index: 2)
            dispatch(encoder, pipeline: splitQGatePSO, threads: rowElems)
        }
    }

    /// out[i] *= sigmoid(gate[i])
    func encodeSigmoidGateMul(commandBuffer: MTLCommandBuffer,
                              out: MTLBuffer, outOffset: Int = 0,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeSigmoidGateMul(encoder: encoder, out: out, outOffset: outOffset,
                             gate: gate, gateOffset: gateOffset, count: count)
        encoder.endEncoding()
    }

    func encodeSigmoidGateMul(encoder: MTLComputeCommandEncoder,
                              out: MTLBuffer, outOffset: Int = 0,
                              gate: MTLBuffer, gateOffset: Int = 0,
                              count: Int) {
        encoder.setComputePipelineState(sigmoidGateMulPSO)
        encoder.setBuffer(out, offset: outOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidGateMulPSO, threads: count)
    }

    /// y[i] *= sigmoid(gate[0])
    func encodeSigmoidScalarMul(commandBuffer: MTLCommandBuffer,
                                y: MTLBuffer, yOffset: Int = 0,
                                gate: MTLBuffer, gateOffset: Int = 0,
                                count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { encoder.endEncoding() }
        encodeSigmoidScalarMul(encoder: encoder, y: y, yOffset: yOffset,
                               gate: gate, gateOffset: gateOffset, count: count)
    }

    func encodeSigmoidScalarMul(encoder: MTLComputeCommandEncoder,
                                y: MTLBuffer, yOffset: Int = 0,
                                gate: MTLBuffer, gateOffset: Int = 0,
                                count: Int) {
        encoder.setComputePipelineState(sigmoidScalarMulPSO)
        encoder.setBuffer(y, offset: yOffset, index: 0)
        encoder.setBuffer(gate, offset: gateOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: sigmoidScalarMulPSO, threads: count)
    }

    /// x[i] += bias[i % rowElems] — a resident BF16 bias row broadcast over
    /// `rows` consecutive FP16 token rows (rows == 1 for decode).
    func encodeBiasAdd(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       bias: MTLBuffer, biasOffset: Int = 0,
                       rowElems: Int, rows: Int = 1) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(biasAddPSO)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(bias, offset: biasOffset, index: 1)
        var elementCount = UInt32(rows * rowElems)
        var rowElementCount = UInt32(rowElems)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&rowElementCount, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(encoder, pipeline: biasAddPSO, threads: rows * rowElems)
        encoder.endEncoding()
    }

    /// hidden[i] += delta[i]
    func encodeResidualAdd(commandBuffer: MTLCommandBuffer,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           delta: MTLBuffer, deltaOffset: Int = 0,
                           count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeResidualAdd(encoder: encoder, hidden: hidden, hiddenOffset: hiddenOffset,
                          delta: delta, deltaOffset: deltaOffset, count: count)
        encoder.endEncoding()
    }

    func encodeResidualAdd(encoder: MTLComputeCommandEncoder,
                           hidden: MTLBuffer, hiddenOffset: Int = 0,
                           delta: MTLBuffer, deltaOffset: Int = 0,
                           count: Int) {
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: hiddenOffset, index: 0)
        encoder.setBuffer(delta, offset: deltaOffset, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        dispatch(encoder, pipeline: residualAddPSO, threads: count)
    }

    /// The kernel is bounds-checked, so an indirect ceil-grid dispatch is
    /// bit-identical to the exact-grid `encodeResidualAdd`.
    func encodeResidualAddIndirect(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   delta: MTLBuffer,
                                   count: Int,
                                   indirectArguments: MTLBuffer,
                                   indirectOffset: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(residualAddPSO)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(delta, offset: 0, index: 1)
        var elementCount = UInt32(count)
        encoder.setBytes(&elementCount, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.dispatchThreadgroups(
            indirectBuffer: indirectArguments,
            indirectBufferOffset: indirectOffset,
            threadsPerThreadgroup: MTLSize(width: residualAddThreadgroupWidth,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }

    static let residualAddThreadgroupWidth = 256

    var residualAddThreadgroupWidth: Int {
        min(residualAddPSO.maxTotalThreadsPerThreadgroup,
            Self.residualAddThreadgroupWidth)
    }

    func encodeConcatRows(commandBuffer: MTLCommandBuffer,
                          lhs: MTLBuffer,
                          rhs: MTLBuffer,
                          out: MTLBuffer,
                          rows: Int,
                          dim: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(concatRowsPSO)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var rowCount = UInt32(rows)
        var dimension = UInt32(dim)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(encoder, pipeline: concatRowsPSO, threads: rows * dim)
        encoder.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
