import Metal

/// The shared expert's gate and up INT4 GEMVs as one grid over the same input.
final class FusedGateUpGEMV {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private static let rowsPerThreadgroup = 8

    private let pipeline: MTLComputePipelineState
    private let specializedPipelines: [Shape: MTLComputePipelineState]

    init(context: MetalContext,
         additionalShapes: [(m: Int, n: Int)] = []) throws {
        self.pipeline = try context.pipeline(
            "dequant_int4_shared_gate_up_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 512)
        var specializedPipelines: [Shape: MTLComputePipelineState] = [:]
        for raw in additionalShapes {
            let shape = Shape(m: UInt32(raw.m), n: UInt32(raw.n))
            specializedPipelines[shape] = try context.pipeline(
                "dequant_int4_shared_gate_up_gemv_simd",
                constants: [
                    MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                    MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                    MetalFunctionConstant(index: 22, value: .bool(true)),
                ],
                maxTotalThreadsPerThreadgroup: 512)
        }
        self.specializedPipelines = specializedPipelines
    }

    func encode(encoder: MTLComputeCommandEncoder,
                gate: SharedExpertProjection,
                up: SharedExpertProjection,
                x: MTLBuffer,
                xOffset: Int = 0,
                gateOut: MTLBuffer,
                gateOutOffset: Int = 0,
                upOut: MTLBuffer,
                upOutOffset: Int = 0) {
        precondition(gate.rows == up.rows && gate.cols == up.cols,
                     "FusedGateUpGEMV needs gate and up of one shape")
        precondition(gate.cols % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(gate.weightsOffset % 2 == 0 && up.weightsOffset % 2 == 0,
                     "dequant_int4_shared_gate_up_gemv_simd needs 2-aligned weights offsets")
        let shape = Shape(m: gate.rows, n: gate.cols)
        encoder.setComputePipelineState(specializedPipelines[shape] ?? pipeline)
        encoder.setBuffer(gate.weights, offset: gate.weightsOffset, index: 0)
        encoder.setBuffer(gate.scales, offset: gate.scalesOffset, index: 1)
        encoder.setBuffer(gate.biases, offset: gate.biasesOffset, index: 2)
        encoder.setBuffer(up.weights, offset: up.weightsOffset, index: 3)
        encoder.setBuffer(up.scales, offset: up.scalesOffset, index: 4)
        encoder.setBuffer(up.biases, offset: up.biasesOffset, index: 5)
        encoder.setBuffer(x, offset: xOffset, index: 6)
        encoder.setBuffer(gateOut, offset: gateOutOffset, index: 7)
        encoder.setBuffer(upOut, offset: upOutOffset, index: 8)
        var mValue = gate.rows
        var nValue = gate.cols
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 10)

        let totalRows = 2 * Int(gate.rows)
        encoder.dispatchThreadgroups(
            MTLSize(width: (totalRows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
    }
}
