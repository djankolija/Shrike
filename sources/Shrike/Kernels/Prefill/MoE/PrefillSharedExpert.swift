import Foundation
import Metal

enum PrefillSharedExpertError: Error, Equatable {
    case chunkTooShort(Int)
    case weightBitsMismatch(expected: Int, got: Int)
}

final class PrefillSharedExpert {
    private let shared: SharedExpertRuntime
    private let activationPSO: MTLComputePipelineState

    /// Below this the per-token GEMV loop still wins: the matrix path pays a
    /// full 64-row tile whatever the chunk holds. The runner passes its own
    /// parsed minimum to `matrixPath`/`encodeChunk`; this is the anchor that
    /// parsed default derives from, not the shipped default itself.
    static let matrixPathMinimumRows = 32

    var weightBits: Int { shared.weightBits }

    /// The MPP instance `encodeChunk` may use for a chunk of `queryCount`
    /// rows, or nil when this chunk belongs on the per-token loop: no MPP
    /// object, a pipeline that failed to compile, a bit width that is not
    /// this runtime's own 4, too few rows to fill a tile, or a reduction
    /// length (`d`, `intermediate`) that is not a whole number of
    /// quantization groups, mirroring `PrefillGroupedRoutedMoE.matrixPath`.
    func matrixPath(for mpp: MPPPrefillInt4QMM?,
                    queryCount: Int,
                    d: Int,
                    intermediate: Int,
                    minimumRows: Int = matrixPathMinimumRows) -> MPPPrefillInt4QMM? {
        guard let mpp,
              mpp.isAvailable,
              weightBits == 4,
              mpp.weightBits == weightBits,
              queryCount >= minimumRows,
              d.isMultiple(of: MPPPrefillInt4QMM.tileK),
              intermediate.isMultiple(of: MPPPrefillInt4QMM.tileK) else { return nil }
        return mpp
    }

    init(context: MetalContext, weightBits: Int = 8, siluActivation: Bool = false) throws {
        self.shared = try SharedExpertRuntime(context: context,
                                              weightBits: weightBits,
                                              siluActivation: siluActivation)
        self.activationPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
    }

    func encodeBlock(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer,
                            xOffset: Int = 0,
                            y: MTLBuffer,
                            yOffset: Int = 0,
                            gate: SharedExpertInt8Proj,
                            up: SharedExpertInt8Proj,
                            down: SharedExpertInt8Proj,
                            scratchGate: MTLBuffer,
                            scratchGateOffset: Int = 0,
                            scratchUp: MTLBuffer,
                            scratchUpOffset: Int = 0,
                            scratchAct: MTLBuffer,
                            scratchActOffset: Int = 0,
                            queryCount: Int,
                            d: Int,
                            intermediate: Int,
                            xStrideElements: Int,
                            yStrideElements: Int) throws {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(d > 0, "d must be positive")
        precondition(intermediate > 0, "intermediate must be positive")
        precondition(xStrideElements >= d, "x stride is too small")
        precondition(yStrideElements >= d, "y stride is too small")
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }

        let halfBytes = MemoryLayout<Float16>.stride
        for row in 0..<queryCount {
            try shared.encode(commandBuffer: cb,
                              x: x,
                              xOffset: xOffset + row * xStrideElements * halfBytes,
                              gate: gate,
                              up: up,
                              down: down,
                              y: y,
                              yOffset: yOffset + row * yStrideElements * halfBytes,
                              scratchGate: scratchGate,
                              scratchGateOffset: scratchGateOffset,
                              scratchUp: scratchUp,
                              scratchUpOffset: scratchUpOffset,
                              scratchAct: scratchAct,
                              scratchActOffset: scratchActOffset)
        }
    }

    /// Whole-chunk form of `encodeBlock`: three GEMMs over the chunk's T rows
    /// and one activation, instead of T M=1 GEMV chains. `x` and `y` are
    /// contiguous T×d fp16; the scratches are contiguous T×intermediate.
    /// The activation folds into `scratchGate` — `silu_mul_fp16` reads and
    /// writes only its own element, so input and output may be the same buffer.
    func encodeChunk(commandBuffer cb: MTLCommandBuffer,
                     mpp: MPPPrefillInt4QMM,
                     x: MTLBuffer,
                     y: MTLBuffer,
                     gate: SharedExpertProjection,
                     up: SharedExpertProjection,
                     down: SharedExpertProjection,
                     scratchGate: MTLBuffer,
                     scratchUp: MTLBuffer,
                     queryCount: Int,
                     d: Int,
                     intermediate: Int,
                     minimumRows: Int = matrixPathMinimumRows) throws {
        guard queryCount >= minimumRows else {
            throw PrefillSharedExpertError.chunkTooShort(queryCount)
        }
        guard mpp.weightBits == weightBits else {
            throw PrefillSharedExpertError.weightBitsMismatch(expected: weightBits,
                                                              got: mpp.weightBits)
        }
        guard gate.rows == UInt32(intermediate), gate.cols == UInt32(d),
              up.rows == UInt32(intermediate), up.cols == UInt32(d),
              down.rows == UInt32(d), down.cols == UInt32(intermediate) else {
            throw SharedExpertInt8Error.dimensionMismatch(
                "expected gate/up=(\(intermediate),\(d)) down=(\(d),\(intermediate))")
        }
        let activationElements = queryCount * intermediate
        let activationBytes = activationElements * MemoryLayout<Float16>.stride
        guard scratchGate.length >= activationBytes,
              scratchUp.length >= activationBytes else {
            throw SharedExpertInt8Error.scratchTooSmall(
                "need \(activationBytes) bytes per intermediate buffer")
        }

        try project(gate, on: cb, mpp: mpp, x: x, y: scratchGate,
                    m: queryCount, n: intermediate, k: d)
        try project(up, on: cb, mpp: mpp, x: x, y: scratchUp,
                    m: queryCount, n: intermediate, k: d)

        guard let encoder = cb.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(activationPSO)
        encoder.setBuffer(scratchGate, offset: 0, index: 0)
        encoder.setBuffer(scratchUp, offset: 0, index: 1)
        encoder.setBuffer(scratchGate, offset: 0, index: 2)
        var count = UInt32(activationElements)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(activationPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: activationElements, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()

        try project(down, on: cb, mpp: mpp, x: scratchGate, y: y,
                    m: queryCount, n: d, k: intermediate)
    }

    private func project(_ projection: SharedExpertProjection,
                         on cb: MTLCommandBuffer,
                         mpp: MPPPrefillInt4QMM,
                         x: MTLBuffer,
                         y: MTLBuffer,
                         m: Int,
                         n: Int,
                         k: Int) throws {
        try mpp.encode(commandBuffer: cb,
                       weights: projection.weights,
                       weightsOffset: projection.weightsOffset,
                       scales: projection.scales,
                       scalesOffset: projection.scalesOffset,
                       biases: projection.biases,
                       biasesOffset: projection.biasesOffset,
                       x: x,
                       y: y,
                       m: m,
                       n: n,
                       k: k,
                       required: true)
    }
}
