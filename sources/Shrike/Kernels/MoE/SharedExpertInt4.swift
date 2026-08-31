import Foundation
import Metal

public enum SharedExpertError: Error, CustomStringConvertible {
    case unsupportedWeightBits(Int)
    case dimensionMismatch(String)
    case scratchTooSmall(String)

    public var description: String {
        switch self {
        case .unsupportedWeightBits(let bits):
            return "SharedExpert unsupported weight bits: \(bits)"
        case .dimensionMismatch(let detail):
            return "SharedExpert dimension mismatch: \(detail)"
        case .scratchTooSmall(let detail):
            return "SharedExpert scratch too small: \(detail)"
        }
    }
}

public final class SharedExpertInt4 {
    private struct Shape: Hashable {
        var m: UInt32
        var n: UInt32
    }

    private let int4: DequantInt4GEMV
    private let geluMulPSO: MTLComputePipelineState
    // The fused down-projection kernel bakes silu into its operand read, so
    // it exists only for silu architectures; gelu falls back to the split
    // chain. The gated flag (FC 27) changes semantics, not just loop bounds,
    // so every fused PSO carries it explicitly.
    private let fusedDownGated: MTLComputePipelineState?
    private let fusedDownUngated: MTLComputePipelineState?
    private let specializedFusedDownGated: [Shape: MTLComputePipelineState]
    private let specializedFusedDownUngated: [Shape: MTLComputePipelineState]

    /// `decodeShapes` compiles constant-folded GEMV variants for the shapes
    /// this runtime issues every decode token (same mechanism and measured
    /// payoff as `DequantInt4GEMV.additionalShapes`).
    public init(context: MetalContext, siluActivation: Bool = false,
                decodeShapes: [(m: Int, n: Int)] = []) throws {
        self.int4 = try DequantInt4GEMV(context: context,
                                        additionalShapes: decodeShapes)
        self.geluMulPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        if siluActivation {
            func fusedPipeline(gated: Bool, shape: Shape?) throws -> MTLComputePipelineState {
                var constants = [
                    MetalFunctionConstant(index: 27, value: .bool(gated)),
                ]
                if let shape {
                    constants += [
                        MetalFunctionConstant(index: 20, value: .uint32(shape.m)),
                        MetalFunctionConstant(index: 21, value: .uint32(shape.n)),
                        MetalFunctionConstant(index: 22, value: .bool(true)),
                    ]
                }
                return try context.pipeline(
                    "dequant_int4_shared_down_fused",
                    constants: constants,
                    maxTotalThreadsPerThreadgroup: 512)
            }
            self.fusedDownGated = try fusedPipeline(gated: true, shape: nil)
            self.fusedDownUngated = try fusedPipeline(gated: false, shape: nil)
            var gated: [Shape: MTLComputePipelineState] = [:]
            var ungated: [Shape: MTLComputePipelineState] = [:]
            for raw in decodeShapes {
                let shape = Shape(m: UInt32(raw.m), n: UInt32(raw.n))
                gated[shape] = try fusedPipeline(gated: true, shape: shape)
                ungated[shape] = try fusedPipeline(gated: false, shape: shape)
            }
            self.specializedFusedDownGated = gated
            self.specializedFusedDownUngated = ungated
        } else {
            self.fusedDownGated = nil
            self.fusedDownUngated = nil
            self.specializedFusedDownGated = [:]
            self.specializedFusedDownUngated = [:]
        }
    }

    public var supportsFusedDecodeChain: Bool { fusedDownGated != nil }

    /// The gate and up GEMVs of the fused decode chain. The caller must
    /// order their scratch writes before `encodeFusedDown` reads them (a
    /// serial encoder does this implicitly).
    public func encodeGateUp(encoder: MTLComputeCommandEncoder,
                             x: MTLBuffer, xOffset: Int = 0,
                             gate: SharedExpertProjection,
                             up: SharedExpertProjection,
                             scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                             scratchUp: MTLBuffer, scratchUpOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols))")
        }
        let required = Int(gate.rows) * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
              scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length else {
            throw SharedExpertError.scratchTooSmall("need \(required) bytes per intermediate buffer")
        }
        int4.encode(encoder: encoder,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchGate, yOffset: scratchGateOffset,
                    m: gate.rows, n: gate.cols)
        int4.encode(encoder: encoder,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, scalesOffset: up.scalesOffset,
                    biases: up.biases, biasesOffset: up.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchUp, yOffset: scratchUpOffset,
                    m: up.rows, n: up.cols)
    }

    /// The fused down projection: silu(gate)·up, the down GEMV, and (when
    /// `scalarGate` is non-nil) the sigmoid scalar gate, in one dispatch.
    public func encodeFusedDown(encoder: MTLComputeCommandEncoder,
                                down: SharedExpertProjection,
                                gateIn: MTLBuffer, gateInOffset: Int = 0,
                                upIn: MTLBuffer, upInOffset: Int = 0,
                                y: MTLBuffer, yOffset: Int = 0,
                                scalarGate: MTLBuffer? = nil,
                                scalarGateOffset: Int = 0) throws {
        guard let base = scalarGate == nil ? fusedDownUngated : fusedDownGated else {
            throw SharedExpertError.dimensionMismatch(
                "fused decode chain requires silu activation")
        }
        precondition(down.cols % UInt32(Quantization.groupSize) == 0,
                     "N must be a multiple of \(Quantization.groupSize)")
        precondition(down.weightsOffset % 2 == 0,
                     "dequant_int4_shared_down_fused needs a 2-aligned weightsOffset")
        let inputBytes = Int(down.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard gateInOffset >= 0, gateInOffset + inputBytes <= gateIn.length,
              upInOffset >= 0, upInOffset + inputBytes <= upIn.length else {
            throw SharedExpertError.scratchTooSmall("need \(inputBytes) bytes per intermediate buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }
        let shape = Shape(m: down.rows, n: down.cols)
        let pipeline = scalarGate == nil
            ? specializedFusedDownUngated[shape] ?? base
            : specializedFusedDownGated[shape] ?? base
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(down.weights, offset: down.weightsOffset, index: 0)
        encoder.setBuffer(down.scales, offset: down.scalesOffset, index: 1)
        encoder.setBuffer(down.biases, offset: down.biasesOffset, index: 2)
        encoder.setBuffer(gateIn, offset: gateInOffset, index: 3)
        encoder.setBuffer(upIn, offset: upInOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        var mValue = down.rows
        var nValue = down.cols
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 7)
        if let scalarGate {
            encoder.setBuffer(scalarGate, offset: scalarGateOffset, index: 8)
        }
        encoder.setThreadgroupMemoryLength(inputBytes, index: 0)
        let rowsPerThreadgroup = 8
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(down.rows) + rowsPerThreadgroup - 1) / rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * rowsPerThreadgroup,
                                           height: 1, depth: 1))
    }

    public func encode(commandBuffer cb: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw SharedExpertError.dimensionMismatch("encoder alloc failed")
        }
        defer { encoder.endEncoding() }
        try encode(encoder: encoder, x: x, xOffset: xOffset,
                   gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                   scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                   scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                   scratchAct: scratchAct, scratchActOffset: scratchActOffset)
    }

    public func encode(encoder: MTLComputeCommandEncoder,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0) throws {
        guard gate.rows == up.rows, gate.cols == up.cols,
              down.rows == gate.cols, down.cols == gate.rows else {
            throw SharedExpertError.dimensionMismatch(
                "gate=(\(gate.rows),\(gate.cols)) up=(\(up.rows),\(up.cols)) down=(\(down.rows),\(down.cols))")
        }
        let intermediate = Int(gate.rows)
        let required = intermediate * MemoryLayout<Float16>.stride
        guard scratchGateOffset >= 0, scratchGateOffset + required <= scratchGate.length,
              scratchUpOffset >= 0, scratchUpOffset + required <= scratchUp.length,
              scratchActOffset >= 0, scratchActOffset + required <= scratchAct.length else {
            throw SharedExpertError.scratchTooSmall("need \(required) bytes per intermediate buffer")
        }
        let inputBytes = Int(gate.cols) * MemoryLayout<Float16>.stride
        let outputBytes = Int(down.rows) * MemoryLayout<Float16>.stride
        guard xOffset >= 0, xOffset + inputBytes <= x.length else {
            throw SharedExpertError.scratchTooSmall("input range exceeds x buffer")
        }
        guard yOffset >= 0, yOffset + outputBytes <= y.length else {
            throw SharedExpertError.scratchTooSmall("output range exceeds y buffer")
        }

        int4.encode(encoder: encoder,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, scalesOffset: gate.scalesOffset,
                    biases: gate.biases, biasesOffset: gate.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchGate, yOffset: scratchGateOffset,
                    m: gate.rows, n: gate.cols)
        int4.encode(encoder: encoder,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, scalesOffset: up.scalesOffset,
                    biases: up.biases, biasesOffset: up.biasesOffset,
                    x: x, xOffset: xOffset,
                    y: scratchUp, yOffset: scratchUpOffset,
                    m: up.rows, n: up.cols)

        encoder.setComputePipelineState(geluMulPSO)
        encoder.setBuffer(scratchGate, offset: scratchGateOffset, index: 0)
        encoder.setBuffer(scratchUp, offset: scratchUpOffset, index: 1)
        encoder.setBuffer(scratchAct, offset: scratchActOffset, index: 2)
        var count = UInt32(intermediate)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(geluMulPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: intermediate, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))

        int4.encode(encoder: encoder,
                    weights: down.weights, weightsOffset: down.weightsOffset,
                    scales: down.scales, scalesOffset: down.scalesOffset,
                    biases: down.biases, biasesOffset: down.biasesOffset,
                    x: scratchAct, xOffset: scratchActOffset,
                    y: y, yOffset: yOffset,
                    m: down.rows, n: down.cols)
    }
}

public final class SharedExpertRuntime {
    private enum Implementation {
        case int4(SharedExpertInt4)
        case affine(SharedExpertAffineQuant)
        case int8(SharedExpertInt8)
    }

    private let implementation: Implementation
    public let weightBits: Int

    /// Non-nil when the int4 fused decode chain applies (int4 weights, silu).
    public var int4FusedDecode: SharedExpertInt4? {
        guard case .int4(let runtime) = implementation,
              runtime.supportsFusedDecodeChain else { return nil }
        return runtime
    }

    public init(context: MetalContext, weightBits: Int,
                siluActivation: Bool = false,
                decodeShapes: [(m: Int, n: Int)] = []) throws {
        self.weightBits = weightBits
        switch weightBits {
        case 4: self.implementation = .int4(try SharedExpertInt4(
            context: context, siluActivation: siluActivation,
            decodeShapes: decodeShapes))
        case 6: self.implementation = .affine(try SharedExpertAffineQuant(
            context: context, weightBits: weightBits,
            siluActivation: siluActivation))
        case 8: self.implementation = .int8(try SharedExpertInt8(
            context: context, siluActivation: siluActivation))
        default: throw SharedExpertError.unsupportedWeightBits(weightBits)
        }
    }

    public func encode(commandBuffer: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0) throws {
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .int8(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .affine(let runtime):
            try runtime.encode(commandBuffer: commandBuffer, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }

    public func encode(encoder: MTLComputeCommandEncoder,
                       x: MTLBuffer, xOffset: Int = 0,
                       gate: SharedExpertProjection,
                       up: SharedExpertProjection,
                       down: SharedExpertProjection,
                       y: MTLBuffer, yOffset: Int = 0,
                       scratchGate: MTLBuffer, scratchGateOffset: Int = 0,
                       scratchUp: MTLBuffer, scratchUpOffset: Int = 0,
                       scratchAct: MTLBuffer, scratchActOffset: Int = 0) throws {
        switch implementation {
        case .int4(let runtime):
            try runtime.encode(encoder: encoder, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .int8(let runtime):
            try runtime.encode(encoder: encoder, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        case .affine(let runtime):
            try runtime.encode(encoder: encoder, x: x, xOffset: xOffset,
                               gate: gate, up: up, down: down, y: y, yOffset: yOffset,
                               scratchGate: scratchGate, scratchGateOffset: scratchGateOffset,
                               scratchUp: scratchUp, scratchUpOffset: scratchUpOffset,
                               scratchAct: scratchAct, scratchActOffset: scratchActOffset)
        }
    }
}
