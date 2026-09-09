import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct FusedGateUpGEMVTests {
    @Test func fusedGateUpGEMV_matchesTwoGEMVs_smallShape() throws {
        try Self.expectMatches(m: 64, n: 128, seed: 0x4755_0001)
    }

    @Test func fusedGateUpGEMV_matchesTwoGEMVs_remainderLoopAndStraddledThreadgroup() throws {
        try Self.expectMatches(m: 20, n: 192, seed: 0x4755_0002)
    }

    @Test func fusedGateUpGEMV_matchesTwoGEMVs_offsetWeights() throws {
        try Self.expectMatches(m: 40, n: 256, seed: 0x4755_0003, weightOffset: 2)
    }

    @Test func fusedGateUpGEMV_matchesTwoGEMVs_servedShapeSpecialized() throws {
        try Self.expectMatches(m: 512, n: 2048, seed: 0x4755_0004, specialized: true)
    }

    private static func expectMatches(m: Int,
                                      n: Int,
                                      seed: UInt64,
                                      weightOffset: Int = 0,
                                      specialized: Bool = false) throws {
        precondition(n % Quantization.groupSize == 0)
        var rng = SplitMix64(seed: seed)
        let ctx = try MetalContext()
        let gate = try Self.makeProjection(ctx, rows: m, n: n, rng: &rng, weightOffset: weightOffset)
        let up = try Self.makeProjection(ctx, rows: m, n: n, rng: &rng, weightOffset: weightOffset)
        let x = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let shapes: [(m: Int, n: Int)] = specialized ? [(m: m, n: n)] : []
        let gemv = try DequantInt4GEMV(context: ctx, additionalShapes: shapes)
        let fused = try FusedGateUpGEMV(context: ctx, additionalShapes: shapes)
        let device = ctx.device
        let xBuf = try #require(Fp16Buffer.make(device, halves: x))
        let gateSplit = try #require(Fp16Buffer.make(device, values: [Float](repeating: 111, count: m)))
        let upSplit = try #require(Fp16Buffer.make(device, values: [Float](repeating: 222, count: m)))
        let gateFused = try #require(Fp16Buffer.make(device, values: [Float](repeating: 333, count: m)))
        let upFused = try #require(Fp16Buffer.make(device, values: [Float](repeating: 444, count: m)))

        let cb = try #require(ctx.queue.makeCommandBuffer())
        let encoder = try #require(cb.makeComputeCommandEncoder())
        gemv.encode(encoder: encoder,
                    weights: gate.weights, weightsOffset: gate.weightsOffset,
                    scales: gate.scales, biases: gate.biases,
                    x: xBuf, y: gateSplit,
                    m: gate.rows, n: gate.cols)
        gemv.encode(encoder: encoder,
                    weights: up.weights, weightsOffset: up.weightsOffset,
                    scales: up.scales, biases: up.biases,
                    x: xBuf, y: upSplit,
                    m: up.rows, n: up.cols)
        fused.encode(encoder: encoder,
                     gate: gate, up: up,
                     x: xBuf,
                     gateOut: gateFused,
                     upOut: upFused)
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed)

        #expect(Self.halfWords(gateSplit, count: m) == Self.halfWords(gateFused, count: m),
                "gate m=\(m) n=\(n) offset=\(weightOffset) diverged from the plain GEMV")
        #expect(Self.halfWords(upSplit, count: m) == Self.halfWords(upFused, count: m),
                "up m=\(m) n=\(n) offset=\(weightOffset) diverged from the plain GEMV")
    }

    private static func makeProjection(_ ctx: MetalContext,
                                       rows: Int,
                                       n: Int,
                                       rng: inout SplitMix64,
                                       weightOffset: Int) throws -> SharedExpertProjection {
        let packedPerRow = n / 2
        let groups = n / Quantization.groupSize
        var weights = [UInt8](repeating: 0, count: rows * packedPerRow + weightOffset)
        var scales = [UInt16](repeating: 0, count: rows * groups)
        var biases = [UInt16](repeating: 0, count: rows * groups)
        for row in 0..<rows {
            let values = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
            let q = Quantization.quantizeInt4Affine(values)
            for i in 0..<packedPerRow { weights[weightOffset + row * packedPerRow + i] = q.packed[i] }
            for i in 0..<groups {
                scales[row * groups + i] = q.scales[i]
                biases[row * groups + i] = q.biases[i]
            }
        }
        let device = ctx.device
        return SharedExpertProjection(
            weights: try #require(device.makeBuffer(bytes: weights, length: weights.count,
                                                    options: .storageModeShared)),
            scales: try #require(device.makeBuffer(bytes: scales,
                                                   length: scales.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared)),
            biases: try #require(device.makeBuffer(bytes: biases,
                                                   length: biases.count * MemoryLayout<UInt16>.size,
                                                   options: .storageModeShared)),
            weightsOffset: weightOffset,
            rows: UInt32(rows), cols: UInt32(n))
    }

    private static func halfWords(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt16.self),
                                  count: count))
    }
}
