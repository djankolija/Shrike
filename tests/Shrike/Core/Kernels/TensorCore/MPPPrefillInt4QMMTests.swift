import Foundation
import Metal
import Testing
@testable import Shrike
import ShrikeValidationSupport

// MPP (matrix-product pipeline) TensorOps is a runtime/hardware capability:
// it is unavailable on some Apple Silicon configurations. These tests are
// gated with .enabled(if:) so the skip is *recorded* in the test output
// instead of silently passing or being dropped — an unavailable MPP path is
// an expected skip, never a green result. The un-gated
// unsupportedOrUnalignedInputsReportFallback test below still pins the
// fallback contract on every machine.
private let mppTensorOpsAvailable: Bool = {
    guard let context = try? MetalContext() else { return false }
    return MPPPrefillInt4QMM(context: context).isAvailable
}()

@Suite struct MPPPrefillInt4QMMTests {
    private struct Inputs {
        let bits: Int
        let packed: [UInt8]
        let scales: [UInt16]
        let biases: [UInt16]
        let x: [Float16]
    }

    /// `irregular` fills x, scales and biases with full-mantissa pseudo-random
    /// values: the default inputs (integers over 64, a few bf16 scales) have
    /// partial sums that are exact in fp32, so they cannot see a reduction
    /// order and compare bit-identical whatever the order.
    private static func makeInputs(m: Int,
                                   n: Int,
                                   k: Int,
                                   bits: Int = 4,
                                   adversarialAffine: Bool = false,
                                   irregular: Bool = false) -> Inputs {
        let groups = k / Quantization.groupSize
        var state: UInt32 = 0x9E37_79B9
        func next() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24)
        }
        var packed = [UInt8](repeating: 0, count: n * k * bits / 8)
        for index in packed.indices {
            packed[index] = UInt8(truncatingIfNeeded: index &* 37 &+ 0x29)
        }
        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        let adversarialScales: [UInt16] = [
            Quantization.bf16Bits(0.001),
            Quantization.bf16Bits(-0.0015),
            0x0001,
            0x8001,
        ]
        let adversarialBiases: [UInt16] = [
            Quantization.bf16Bits(-0.01),
            Quantization.bf16Bits(0.006),
            0x0001,
            0x8001,
        ]
        for row in 0..<n {
            for group in 0..<groups {
                let index = row * groups + group
                if irregular {
                    scales[index] = Quantization.bf16Bits(0.0005 + next() * 0.003)
                    biases[index] = Quantization.bf16Bits(-0.02 + next() * 0.04)
                } else if adversarialAffine {
                    scales[index] = adversarialScales[(row + group) % adversarialScales.count]
                    biases[index] = adversarialBiases[(row * 3 + group) % adversarialBiases.count]
                } else {
                    scales[index] = Quantization.bf16Bits(
                        0.001 + Float((row + group) % 5) * 0.00025)
                    biases[index] = Quantization.bf16Bits(
                        -0.01 + Float((row * 3 + group) % 7) * 0.002)
                }
            }
        }
        var x = [Float16](repeating: 0, count: m * k)
        for index in x.indices {
            x[index] = irregular
                ? Float16(next() - 0.5)
                : Float16(Float((index * 11) % 29 - 14) / 64.0)
        }
        return Inputs(bits: bits, packed: packed, scales: scales, biases: biases, x: x)
    }

    private static func makeBuffer<T>(device: MTLDevice,
                                      values: [T],
                                      prefixBytes: Int = 0) -> MTLBuffer? {
        let byteCount = values.count * MemoryLayout<T>.stride
        guard let buffer = device.makeBuffer(length: prefixBytes + byteCount,
                                             options: .storageModeShared) else {
            return nil
        }
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            buffer.contents().advanced(by: prefixBytes)
                .copyMemory(from: baseAddress, byteCount: byteCount)
        }
        return buffer
    }

    private static func cpuReference(_ inputs: Inputs,
                                     m: Int,
                                     n: Int,
                                     k: Int) -> [Float] {
        let groups = k / Quantization.groupSize
        let rowBytes = k * inputs.bits / 8
        let mask = UInt32((1 << inputs.bits) - 1)
        var output = [Float](repeating: 0, count: m * n)
        for token in 0..<m {
            for row in 0..<n {
                var accumulator: Float = 0
                for group in 0..<groups {
                    let scale = Quantization.bf16ToFloat(inputs.scales[row * groups + group])
                    let bias = Quantization.bf16ToFloat(inputs.biases[row * groups + group])
                    for localK in 0..<Quantization.groupSize {
                        let column = group * Quantization.groupSize + localK
                        let bitOffset = column * inputs.bits
                        let byteOffset = row * rowBytes + bitOffset / 8
                        let shift = bitOffset % 8
                        var word = UInt32(inputs.packed[byteOffset])
                        if shift + inputs.bits > 8 {
                            word |= UInt32(inputs.packed[byteOffset + 1]) << 8
                        }
                        let q = (word >> shift) & mask
                        let weight = Float(q) * scale + bias
                        accumulator.addProduct(weight, Float(inputs.x[token * k + column]))
                    }
                }
                output[token * n + row] = Float(Float16(accumulator))
            }
        }
        return output
    }

    @discardableResult
    private static func runShape(context: MetalContext,
                                 candidate: MPPPrefillInt4QMM,
                                 baseline: PrefillInt4QMM,
                                 m: Int,
                                 n: Int,
                                 k: Int,
                                 bits: Int = 4,
                                 adversarialAffine: Bool = false,
                                 weightOffset: Int = 0,
                                 scaleOffset: Int = 0,
                                 biasOffset: Int = 0,
                                 compareCPUReference: Bool = false,
                                 irregular: Bool = false) throws
        -> MPPPrefillInt4QMM.Path {
        let inputs = makeInputs(m: m, n: n, k: k, bits: bits,
                                adversarialAffine: adversarialAffine, irregular: irregular)
        guard let weights = makeBuffer(device: context.device,
                                       values: inputs.packed,
                                       prefixBytes: weightOffset),
              let scaleBuffer = makeBuffer(device: context.device,
                                           values: inputs.scales,
                                           prefixBytes: scaleOffset),
              let biasBuffer = makeBuffer(device: context.device,
                                          values: inputs.biases,
                                          prefixBytes: biasOffset),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let expectedBuffer = Fp16Buffer.make(context.device, count: m * n),
              let actualBuffer = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }

        try baseline.encode(commandBuffer: commandBuffer,
                        weights: weights,
                        weightsOffset: weightOffset,
                        scales: scaleBuffer,
                        scalesOffset: scaleOffset,
                        biases: biasBuffer,
                        biasesOffset: biasOffset,
                        x: input,
                        y: expectedBuffer,
                        t: m,
                        n: n,
                        k: k)
        let path = try candidate.encode(commandBuffer: commandBuffer,
                                    weights: weights,
                                    weightsOffset: weightOffset,
                                    scales: scaleBuffer,
                                    scalesOffset: scaleOffset,
                                    biases: biasBuffer,
                                    biasesOffset: biasOffset,
                                    x: input,
                                    y: actualBuffer,
                                    m: m,
                                    n: n,
                                    k: k)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(path == .affineThreadgroupF16)

        let baselineOutput = Fp16Buffer.read(expectedBuffer, count: m * n)
        let actual = Fp16Buffer.read(actualBuffer, count: m * n)
        let baselineMaxAbs = RelError.maxAbsDiff(actual, baselineOutput)
        let baselineRelative = RelError.compute(actual: actual, reference: baselineOutput)
        let byteExact = actual == baselineOutput
        #expect(baselineMaxAbs <= 0.03,
                "shape M=\(m) N=\(n) K=\(k) maxAbs=\(baselineMaxAbs) rel=\(baselineRelative) byteExact=\(byteExact)")
        #expect(baselineRelative <= 1e-3 || baselineMaxAbs <= 0.01,
                "shape M=\(m) N=\(n) K=\(k) maxAbs=\(baselineMaxAbs) rel=\(baselineRelative) byteExact=\(byteExact)")

        if compareCPUReference {
            let reference = cpuReference(inputs, m: m, n: n, k: k)
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let relative = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 0.03,
                    "CPU reference M=\(m) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(relative)")
            #expect(relative <= 1e-3,
                    "CPU reference M=\(m) N=\(n) K=\(k) maxAbs=\(maxAbs) rel=\(relative)")
        }
        return path
    }

    private static let variantShapes: [(m: Int, n: Int, k: Int)] = [
        (m: 64, n: 32, k: 128),
        (m: 33, n: 512, k: 2048),
        (m: 128, n: 2048, k: 512),
    ]

    private static func runPair(context: MetalContext,
                                first: MPPPrefillInt4QMM,
                                second: MPPPrefillInt4QMM,
                                m: Int,
                                n: Int,
                                k: Int,
                                bits: Int = 4,
                                weightOffset: Int = 0,
                                irregular: Bool = false) throws -> (first: [Float16], second: [Float16]) {
        let inputs = makeInputs(m: m, n: n, k: k, bits: bits, irregular: irregular)
        guard let weights = makeBuffer(device: context.device, values: inputs.packed,
                                       prefixBytes: weightOffset),
              let scales = makeBuffer(device: context.device, values: inputs.scales),
              let biases = makeBuffer(device: context.device, values: inputs.biases),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let firstOutput = Fp16Buffer.make(context.device, count: m * n),
              let secondOutput = Fp16Buffer.make(context.device, count: m * n),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }
        let firstPath = try first.encode(commandBuffer: commandBuffer,
                                         weights: weights, weightsOffset: weightOffset,
                                         scales: scales, biases: biases,
                                         x: input, y: firstOutput,
                                         m: m, n: n, k: k, required: true)
        let secondPath = try second.encode(commandBuffer: commandBuffer,
                                           weights: weights, weightsOffset: weightOffset,
                                           scales: scales, biases: biases,
                                           x: input, y: secondOutput,
                                           m: m, n: n, k: k, required: true)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(firstPath == .affineThreadgroupF16 && secondPath == .affineThreadgroupF16)
        return (Fp16Buffer.readHalf(firstOutput, count: m * n),
                Fp16Buffer.readHalf(secondOutput, count: m * n))
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func doubleBufferedDequantIsBitIdenticalToTheSingleBuffered() throws {
        let context = try MetalContext()
        let single = MPPPrefillInt4QMM(context: context, variant: .n32b1)
        let double = MPPPrefillInt4QMM(context: context, variant: .n32b2)
        #expect(double.isAvailable, "n32b2 pipeline unavailable")
        for shape in Self.variantShapes {
            let outputs = try Self.runPair(context: context, first: single, second: double,
                                           m: shape.m, n: shape.n, k: shape.k)
            let finite = outputs.second.allSatisfy(\.isFinite)
            #expect(finite, "shape \(shape) produced a non-finite output")
            let firstMismatch = zip(outputs.first, outputs.second).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset
            #expect(firstMismatch == nil,
                    "shape M=\(shape.m) N=\(shape.n) K=\(shape.k) first mismatch=\(firstMismatch ?? -1)")
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"),
          arguments: [MPPPrefillInt4QMM.TileVariant.n64b1, .n64b2])
    func wideNTileMatchesTheNarrowTile(variant: MPPPrefillInt4QMM.TileVariant) throws {
        let context = try MetalContext()
        let narrow = MPPPrefillInt4QMM(context: context, variant: .n32b1)
        let wide = MPPPrefillInt4QMM(context: context, variant: variant)
        let baseline = try PrefillInt4QMM(context: context)
        #expect(wide.isAvailable, "\(variant) pipeline unavailable")
        for shape in Self.variantShapes {
            try Self.runShape(context: context, candidate: wide, baseline: baseline,
                              m: shape.m, n: shape.n, k: shape.k, compareCPUReference: true)
            let outputs = try Self.runPair(context: context, first: narrow, second: wide,
                                           m: shape.m, n: shape.n, k: shape.k)
            let actual = outputs.second.map(Float.init)
            let reference = outputs.first.map(Float.init)
            let finite = actual.allSatisfy(\.isFinite)
            #expect(finite, "\(variant) shape \(shape) produced a non-finite output")
            let firstMismatch = zip(outputs.first, outputs.second).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset
            #expect(firstMismatch == nil,
                    "\(variant) M=\(shape.m) N=\(shape.n) K=\(shape.k) first mismatch=\(firstMismatch ?? -1)")
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 2e-2, "\(variant) M=\(shape.m) N=\(shape.n) K=\(shape.k) maxAbs=\(maxAbs) rel=\(rel)")
            #expect(rel <= 2e-2, "\(variant) M=\(shape.m) N=\(shape.n) K=\(shape.k) rel=\(rel) maxAbs=\(maxAbs)")
        }
    }

    private static func expectBitIdentical(_ outputs: (first: [Float16], second: [Float16]),
                                           _ label: String) {
        let finite = outputs.second.allSatisfy(\.isFinite)
        #expect(finite, "\(label) produced a non-finite output")
        let firstMismatch = zip(outputs.first, outputs.second).enumerated()
            .first { $0.element.0 != $0.element.1 }?.offset
        #expect(firstMismatch == nil, "\(label) first mismatch=\(firstMismatch ?? -1)")
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func vectorWeightLoadsAreBitIdenticalToByteLoads() throws {
        let context = try MetalContext()
        let byte = MPPPrefillInt4QMM(context: context, variant: .n32b1, weightLoads: .byte)
        let vector = MPPPrefillInt4QMM(context: context, variant: .n32b1, weightLoads: .vector)
        for shape in Self.variantShapes {
            for irregular in [false, true] {
                let outputs = try Self.runPair(context: context, first: byte, second: vector,
                                               m: shape.m, n: shape.n, k: shape.k, irregular: irregular)
                Self.expectBitIdentical(outputs, "vector M=\(shape.m) N=\(shape.n) K=\(shape.k) irregular=\(irregular)")
            }
        }
        let unaligned = try Self.runPair(context: context, first: byte, second: vector,
                                         m: 33, n: 512, k: 2048, weightOffset: 13)
        Self.expectBitIdentical(unaligned, "vector at weightOffset 13 (byte fallback)")
        let byte8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32b1, weightLoads: .byte)
        let vector8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32b1, weightLoads: .vector)
        let eightBit = try Self.runPair(context: context, first: byte8, second: vector8,
                                        m: 33, n: 35, k: 128, bits: 8)
        Self.expectBitIdentical(eightBit, "vector 8-bit M=33 N=35 K=128")
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func vectorWeightLoadsAreBitIdenticalToByteLoadsOnTheWideKTile() throws {
        let context = try MetalContext()
        let byte = MPPPrefillInt4QMM(context: context, variant: .n32k128b1, weightLoads: .byte)
        let vector = MPPPrefillInt4QMM(context: context, variant: .n32k128b1, weightLoads: .vector)
        for shape in Self.variantShapes {
            let outputs = try Self.runPair(context: context, first: byte, second: vector,
                                           m: shape.m, n: shape.n, k: shape.k, irregular: true)
            Self.expectBitIdentical(outputs, "vector K128 M=\(shape.m) N=\(shape.n) K=\(shape.k)")
        }
        let byte8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32k128b1, weightLoads: .byte)
        let vector8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32k128b1, weightLoads: .vector)
        let eightBit = try Self.runPair(context: context, first: byte8, second: vector8,
                                        m: 33, n: 35, k: 256, bits: 8, irregular: true)
        Self.expectBitIdentical(eightBit, "vector K128 8-bit M=33 N=35 K=256 (two-uint4 chunks)")
    }

    private static let wideK256Shapes: [(m: Int, n: Int, k: Int)] = [
        (m: 64, n: 32, k: 256),
        (m: 33, n: 512, k: 2048),
        (m: 128, n: 2048, k: 512),
    ]

    private static func expectWideKTileMatchesTheNarrowTile(variant: MPPPrefillInt4QMM.TileVariant,
                                                            shapes: [(m: Int, n: Int, k: Int)]) throws {
        let context = try MetalContext()
        let narrow = MPPPrefillInt4QMM(context: context, variant: .n32b1)
        let wide = MPPPrefillInt4QMM(context: context, variant: variant)
        let baseline = try PrefillInt4QMM(context: context)
        #expect(wide.isAvailable, "\(variant) pipeline unavailable")
        for shape in shapes {
            try runShape(context: context, candidate: wide, baseline: baseline,
                         m: shape.m, n: shape.n, k: shape.k, compareCPUReference: true,
                         irregular: true)
            let outputs = try runPair(context: context, first: narrow, second: wide,
                                      m: shape.m, n: shape.n, k: shape.k, irregular: true)
            let actual = outputs.second.map(Float.init)
            let reference = outputs.first.map(Float.init)
            let finite = actual.allSatisfy(\.isFinite)
            #expect(finite, "\(variant) shape \(shape) produced a non-finite output")
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 2e-2, "\(variant) M=\(shape.m) N=\(shape.n) K=\(shape.k) maxAbs=\(maxAbs) rel=\(rel)")
            #expect(rel <= 2e-2, "\(variant) M=\(shape.m) N=\(shape.n) K=\(shape.k) rel=\(rel) maxAbs=\(maxAbs)")
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func wideKTileMatchesTheNarrowTile() throws {
        try Self.expectWideKTileMatchesTheNarrowTile(variant: .n32k128b1, shapes: Self.variantShapes)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func wideK256TileMatchesTheNarrowTile() throws {
        try Self.expectWideKTileMatchesTheNarrowTile(variant: .n32k256b1, shapes: Self.wideK256Shapes)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func wideK256TileTakesTheWidestRungThatDividesK() throws {
        let context = try MetalContext()
        let widest = MPPPrefillInt4QMM(context: context, variant: .n32k256b1)
        let k128 = MPPPrefillInt4QMM(context: context, variant: .n32k128b1)
        let narrow = MPPPrefillInt4QMM(context: context, variant: .n32b1)
        let viaK128 = try Self.runPair(context: context, first: k128, second: widest,
                                       m: 64, n: 32, k: 384, irregular: true)
        Self.expectBitIdentical(viaK128, "n32k256b1 K 384 through the K128 rung")
        let viaK64 = try Self.runPair(context: context, first: narrow, second: widest,
                                      m: 33, n: 128, k: 2880, irregular: true)
        Self.expectBitIdentical(viaK64, "n32k256b1 K 2880 through the K64 rung")
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func vectorWeightLoadsAreBitIdenticalToByteLoadsOnTheK256Tile() throws {
        let context = try MetalContext()
        let byte = MPPPrefillInt4QMM(context: context, variant: .n32k256b1, weightLoads: .byte)
        let vector = MPPPrefillInt4QMM(context: context, variant: .n32k256b1, weightLoads: .vector)
        for shape in Self.wideK256Shapes {
            let outputs = try Self.runPair(context: context, first: byte, second: vector,
                                           m: shape.m, n: shape.n, k: shape.k, irregular: true)
            Self.expectBitIdentical(outputs, "vector K256 M=\(shape.m) N=\(shape.n) K=\(shape.k)")
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func wideK256TileMatchesTheNarrowTileAtEightBits() throws {
        let context = try MetalContext()
        let byte8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32k256b1, weightLoads: .byte)
        let vector8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32k256b1, weightLoads: .vector)
        let narrow8 = MPPPrefillInt4QMM(context: context, weightBits: 8, variant: .n32b1)
        for shape in [(m: 33, n: 35, k: 256), (m: 33, n: 512, k: 2048)] {
            let loads = try Self.runPair(context: context, first: byte8, second: vector8,
                                         m: shape.m, n: shape.n, k: shape.k, bits: 8, irregular: true)
            Self.expectBitIdentical(loads, "vector K256 8-bit M=\(shape.m) N=\(shape.n) K=\(shape.k) (four-uint4 chunks)")
            let widths = try Self.runPair(context: context, first: narrow8, second: vector8,
                                          m: shape.m, n: shape.n, k: shape.k, bits: 8, irregular: true)
            let actual = widths.second.map(Float.init)
            let reference = widths.first.map(Float.init)
            let maxAbs = RelError.maxAbsDiff(actual, reference)
            let rel = RelError.compute(actual: actual, reference: reference)
            #expect(maxAbs <= 2e-2, "n32k256b1 8-bit M=\(shape.m) N=\(shape.n) K=\(shape.k) maxAbs=\(maxAbs) rel=\(rel)")
            #expect(rel <= 2e-2, "n32k256b1 8-bit M=\(shape.m) N=\(shape.n) K=\(shape.k) rel=\(rel) maxAbs=\(maxAbs)")
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func wideKTileFallsBackOnARaggedK() throws {
        let context = try MetalContext()
        let narrow = MPPPrefillInt4QMM(context: context, variant: .n32b1)
        let wide = MPPPrefillInt4QMM(context: context, variant: .n32k128b1)
        for shape in [(m: 64, n: 32, k: 192), (m: 33, n: 128, k: 2880)] {
            let outputs = try Self.runPair(context: context, first: narrow, second: wide,
                                           m: shape.m, n: shape.n, k: shape.k)
            Self.expectBitIdentical(outputs, "n32k128b1 narrow fallback M=\(shape.m) N=\(shape.n) K=\(shape.k)")
        }
    }

    @Test func tileVariantsParseTheirEnvironmentNames() {
        typealias Tile = MPPPrefillInt4QMM.TileVariant
        #expect(Tile(tileN: "64", tileK: nil, buffers: "2", fallback: .n32b1) == .n64b2)
        #expect(Tile(tileN: nil, tileK: nil, buffers: "2", fallback: .n32b1) == .n32b2)
        #expect(Tile(tileN: "64", tileK: nil, buffers: nil, fallback: .n32b1) == .n64b1)
        #expect(Tile(tileN: "48", tileK: nil, buffers: "2", fallback: .n32b1) == nil)
        #expect(Tile(tileN: "32", tileK: nil, buffers: "3", fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: "128", buffers: nil, fallback: .n32b1) == .n32k128b1)
        #expect(Tile(tileN: "64", tileK: "128", buffers: nil, fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: "128", buffers: "2", fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: "96", buffers: nil, fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: "256", buffers: nil, fallback: .n32b1) == .n32k256b1)
        #expect(Tile(tileN: "64", tileK: "256", buffers: nil, fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: "256", buffers: "2", fallback: .n32b1) == nil)
        #expect(Tile(tileN: nil, tileK: nil, buffers: nil, fallback: .n32k256b1) == .n32k256b1)
        #expect(Tile(tileN: nil, tileK: "128", buffers: nil, fallback: .n32k256b1) == .n32k128b1)
        #expect(Tile.n32k256b1.tileK == 256 && Tile.n32k256b1.tileN == 32 && Tile.n32k256b1.dequantBuffers == 1)
        #expect(Tile.n32k256b1.kernelName == "mpp_prefill_affine_threadgroup_f16_n32k256b1")
        #expect(Tile.n32k256b1.groupedKernelName == "mpp_prefill_affine_grouped_f16_n32k256b1")
        #expect(Tile(tileN: nil, tileK: nil, buffers: nil, fallback: .n32k128b1) == .n32k128b1)
        #expect(Tile(tileN: nil, tileK: "64", buffers: nil, fallback: .n32k128b1) == .n32b1)
        #expect(Tile.n32k128b1.tileK == 128 && Tile.n64b2.tileK == 64)
        #expect(Tile.n32k128b1.kernelName == "mpp_prefill_affine_threadgroup_f16_n32k128b1")
        #expect(MPPPrefillInt4QMM.WeightLoads(rawValue: "vector") == .vector)
        #expect(MPPPrefillInt4QMM.TileVariant.n32b1.kernelName == "mpp_prefill_affine_threadgroup_f16")
        #expect(MPPPrefillInt4QMM.TileVariant.n32b1.groupedKernelName == "mpp_prefill_affine_grouped_f16")
        #expect(MPPPrefillInt4QMM.TileVariant.n64b2.kernelName == "mpp_prefill_affine_threadgroup_f16_n64b2")
        #expect(MPPPrefillInt4QMM.TileVariant.n64b2.groupedKernelName == "mpp_prefill_affine_grouped_f16_n64b2")
        #expect(MPPPrefillInt4QMM.TileVariant.n64b1.tileN == 64 && MPPPrefillInt4QMM.TileVariant.n64b1.dequantBuffers == 1)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func affineThreadgroupCandidateMatchesFP32AffineReference() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        let baseline = try PrefillInt4QMM(context: context)

        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 64, compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 64, n: 32, k: 128, adversarialAffine: true,
                          compareCPUReference: true)
        try Self.runShape(context: context, candidate: candidate, baseline: baseline,
                          m: 17, n: 35, k: 64, compareCPUReference: true)
        try Self.runShape(
            context: context, candidate: candidate, baseline: baseline,
            m: 17, n: 35, k: 128, adversarialAffine: true,
            weightOffset: 13, scaleOffset: 2, biasOffset: 6,
            compareCPUReference: true)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"),
          arguments: [8])
    func affineThreadgroupSupportsHigherBitWeights(bits: Int) throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context, weightBits: bits)
        let baseline = try PrefillInt4QMM(context: context, weightBits: bits)
        #expect(candidate.isAvailable)
        try Self.runShape(context: context,
                          candidate: candidate,
                          baseline: baseline,
                          m: 33,
                          n: 35,
                          k: 128,
                          bits: bits,
                          adversarialAffine: true,
                          compareCPUReference: true)
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func selectedProductionAttentionShapesMatchCurrentPolicy() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        let baseline = try PrefillInt4QMM(context: context)
        let shapes = [
            (name: "swa-q", n: 4096, k: 2816),
            (name: "swa-kv", n: 2048, k: 2816),
            (name: "swa-o", n: 2816, k: 4096),
            (name: "full-q", n: 8192, k: 2816),
            (name: "full-kv", n: 1024, k: 2816),
            (name: "full-o", n: 2816, k: 8192),
        ]
        for m in [32, 128] {
            for shape in shapes {
                let path = try Self.runShape(
                    context: context, candidate: candidate, baseline: baseline,
                    m: m, n: shape.n, k: shape.k)
                #expect(path == .affineThreadgroupF16,
                        "\(shape.name) M=\(m) unexpectedly fell back")
            }
        }
    }

    @Test(.enabled(if: mppTensorOpsAvailable,
                   "Requires runtime MPP TensorOps support"))
    func fullProductionShapeIsByteStableAcross32Dispatches() throws {
        let m = 32
        let n = 2816
        let k = 8192
        let outputElements = m * n
        let outputBytes = outputElements * MemoryLayout<Float16>.stride
        let inputs = Self.makeInputs(m: m, n: n, k: k)
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        guard let weights = Self.makeBuffer(device: context.device, values: inputs.packed),
              let scales = Self.makeBuffer(device: context.device, values: inputs.scales),
              let biases = Self.makeBuffer(device: context.device, values: inputs.biases),
              let input = Fp16Buffer.make(context.device, halves: inputs.x),
              let outputs = context.device.makeBuffer(length: outputBytes * 32,
                                                      options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        for run in 0..<32 {
            let path = try candidate.encode(commandBuffer: commandBuffer,
                                        weights: weights,
                                        scales: scales,
                                        biases: biases,
                                        x: input,
                                        y: outputs,
                                        yOffset: run * outputBytes,
                                        m: m,
                                        n: n,
                                        k: k)
            #expect(path == .affineThreadgroupF16)
        }
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let reference = outputs.contents().assumingMemoryBound(to: UInt16.self)
        for run in 1..<32 {
            let candidateOutput = outputs.contents()
                .advanced(by: run * outputBytes)
                .assumingMemoryBound(to: UInt16.self)
            var mismatch: Int?
            for index in 0..<outputElements where reference[index] != candidateOutput[index] {
                mismatch = index
                break
            }
            #expect(mismatch == nil, "dispatch \(run) first mismatch=\(mismatch ?? -1)")
        }
    }

    @Test func unsupportedOrUnalignedInputsReportFallback() throws {
        let context = try MetalContext()
        let candidate = MPPPrefillInt4QMM(context: context)
        guard let buffer = context.device.makeBuffer(length: 4096,
                                                     options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let unsupportedShape = try candidate.encode(commandBuffer: commandBuffer,
                                                weights: buffer,
                                                scales: buffer,
                                                biases: buffer,
                                                x: buffer,
                                                y: buffer,
                                                m: 1,
                                                n: 1,
                                                k: 65)
        let unalignedScale = try candidate.encode(commandBuffer: commandBuffer,
                                              weights: buffer,
                                              weightsOffset: 1,
                                              scales: buffer,
                                              scalesOffset: 1,
                                              biases: buffer,
                                              x: buffer,
                                              y: buffer,
                                              m: 1,
                                              n: 1,
                                              k: 64)
        #expect(unsupportedShape == .unavailable)
        #expect(unalignedScale == .unavailable)
    }
}
