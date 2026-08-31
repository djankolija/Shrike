import Foundation
import Metal
import Testing
@testable import Shrike
import ShrikeValidationSupport

@Suite struct SharedExpertInt4Tests {
    private static let d = 128
    private static let f = 64

    @Test func sharedExpertInt4MatchesAffineReference() throws {
        var rng = SeedTree(0x604).key("shared-expert-int4")
        let x = (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<Self.f).map { _ in (0..<Self.d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<Self.d).map { _ in (0..<Self.f).map { _ in rng.uniform(-0.4, 0.4) } }
        let gatePack = Self.pack(gate)
        let upPack = Self.pack(up)
        let downPack = Self.pack(down)
        let x16 = x.map { Float(Float16($0)) }
        let gateOut = DequantInt4GemvRef.apply(weightRows: gatePack.rows, x: x16, n: Self.d)
        let upOut = DequantInt4GemvRef.apply(weightRows: upPack.rows, x: x16, n: Self.d)
        let act = zip(gateOut, upOut).map { gateValue, upValue in
            let cube = gateValue * gateValue * gateValue
            let inner = 0.7978845608028654 * Double(gateValue + 0.044715 * cube)
            return Float(Float16(Float(0.5 * Double(gateValue) * (1 + tanh(inner))) * upValue))
        }
        let reference = DequantInt4GemvRef.apply(weightRows: downPack.rows, x: act, n: Self.f)

        let context = try MetalContext()
        let runtime = try SharedExpertInt4(context: context)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let yBuffer = try #require(Fp16Buffer.make(context.device, count: Self.d))
        let gateScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let upScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let actScratch = try #require(Fp16Buffer.make(context.device, count: Self.f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try runtime.encode(commandBuffer: commandBuffer,
                           x: xBuffer,
                           gate: Self.projection(context, gatePack, rows: Self.f, cols: Self.d),
                           up: Self.projection(context, upPack, rows: Self.f, cols: Self.d),
                           down: Self.projection(context, downPack, rows: Self.d, cols: Self.f),
                           y: yBuffer,
                           scratchGate: gateScratch,
                           scratchUp: upScratch,
                           scratchAct: actScratch)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let actual = Fp16Buffer.read(yBuffer, count: Self.d)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.quantInt4 * 4, "shared-expert int4 rel=\(error)")
    }

    /// Kimi's ungated shared expert and dense layer 0 run this kernel with
    /// SiLU at F = 1024 (and 9216); pin the SiLU path at the real MoE width.
    @Test func sharedExpertInt4SiLUMatchesReferenceAtKimiWidth() throws {
        let d = 128, f = 1024
        var rng = SeedTree(0x605).key("shared-expert-int4-silu")
        let x = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<d).map { _ in (0..<f).map { _ in rng.uniform(-0.4, 0.4) } }
        let gatePack = Self.pack(gate)
        let upPack = Self.pack(up)
        let downPack = Self.pack(down)
        let x16 = x.map { Float(Float16($0)) }
        let gateOut = DequantInt4GemvRef.apply(weightRows: gatePack.rows, x: x16, n: d)
        let upOut = DequantInt4GemvRef.apply(weightRows: upPack.rows, x: x16, n: d)
        let act = zip(gateOut, upOut).map { gateValue, upValue in
            Float(Float16(gateValue / (1 + exp(-gateValue)) * upValue))
        }
        let reference = DequantInt4GemvRef.apply(weightRows: downPack.rows, x: act, n: f)

        let context = try MetalContext()
        let runtime = try SharedExpertInt4(context: context, siluActivation: true)
        let xBuffer = try #require(Fp16Buffer.make(context.device, values: x))
        let yBuffer = try #require(Fp16Buffer.make(context.device, count: d))
        let gateScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let upScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let actScratch = try #require(Fp16Buffer.make(context.device, count: f))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try runtime.encode(commandBuffer: commandBuffer,
                           x: xBuffer,
                           gate: Self.projection(context, gatePack, rows: f, cols: d),
                           up: Self.projection(context, upPack, rows: f, cols: d),
                           down: Self.projection(context, downPack, rows: d, cols: f),
                           y: yBuffer,
                           scratchGate: gateScratch,
                           scratchUp: upScratch,
                           scratchAct: actScratch)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.status == .completed)
        let actual = Fp16Buffer.read(yBuffer, count: d)
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.quantInt4 * 4, "shared-expert silu@1024 rel=\(error)")
    }

    @Test func fusedDecodeChainGatedMatchesSplitChainBitwise() throws {
        try Self.runFusedChainBitwiseArm(d: 128, f: 256, gated: true, seed: 0x606)
    }

    /// f = 192 (three groups) drives the GEMV's scalar remainder loop, which
    /// the multiple-of-256 widths never touch.
    @Test func fusedDecodeChainRemainderLoopMatchesSplitChainBitwise() throws {
        try Self.runFusedChainBitwiseArm(d: 128, f: 192, gated: true, seed: 0x607)
    }

    @Test func fusedDecodeChainUngatedMatchesSplitChainBitwise() throws {
        try Self.runFusedChainBitwiseArm(d: 128, f: 256, gated: false, seed: 0x608)
    }

    private static func runFusedChainBitwiseArm(
        d: Int, f: Int, gated: Bool, seed: UInt64) throws {
        var rng = SeedTree(seed).key("shared-expert-int4-fused")
        let x = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let gate = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let up = (0..<f).map { _ in (0..<d).map { _ in rng.uniform(-0.4, 0.4) } }
        let down = (0..<d).map { _ in (0..<f).map { _ in rng.uniform(-0.4, 0.4) } }
        let scalarGateRow = (0..<d).map { _ in rng.uniform(-0.4, 0.4) }
        let gatePack = Self.pack(gate)
        let upPack = Self.pack(up)
        let downPack = Self.pack(down)
        let scalarQ = Quantization.quantizeInt8Affine(scalarGateRow)

        let context = try MetalContext()
        let runtime = try SharedExpertInt4(
            context: context, siluActivation: true,
            decodeShapes: [(m: f, n: d), (m: d, n: f)])
        let int8 = try DequantInt8GEMV(context: context)
        let elementwise = try Elementwise(context: context)
        let device = context.device
        let xBuffer = try #require(Fp16Buffer.make(device, values: x))
        let gateProj = Self.projection(context, gatePack, rows: f, cols: d)
        let upProj = Self.projection(context, upPack, rows: f, cols: d)
        let downProj = Self.projection(context, downPack, rows: d, cols: f)
        let scalarWeights = try #require(device.makeBuffer(
            bytes: scalarQ.packed, length: scalarQ.packed.count,
            options: .storageModeShared))
        let scalarScales = try #require(device.makeBuffer(
            bytes: scalarQ.scales, length: scalarQ.scales.count * 2,
            options: .storageModeShared))
        let scalarBiases = try #require(device.makeBuffer(
            bytes: scalarQ.biases, length: scalarQ.biases.count * 2,
            options: .storageModeShared))
        // Distinct sentinel fills so a dispatch that silently never ran
        // cannot pass as "equal".
        let ySplit = try #require(Fp16Buffer.make(device, values: [Float](repeating: 111, count: d)))
        let yFused = try #require(Fp16Buffer.make(device, values: [Float](repeating: 222, count: d)))
        let scalarSplit = try #require(Fp16Buffer.make(device, values: [333]))
        let scalarFused = try #require(Fp16Buffer.make(device, values: [444]))
        let gateScratch = try #require(Fp16Buffer.make(device, count: f))
        let upScratch = try #require(Fp16Buffer.make(device, count: f))
        let actScratch = try #require(Fp16Buffer.make(device, count: f))

        func encodeScalarGate(_ encoder: MTLComputeCommandEncoder, y: MTLBuffer) {
            int8.encode(encoder: encoder,
                        weights: scalarWeights,
                        scales: scalarScales,
                        biases: scalarBiases,
                        x: xBuffer,
                        y: y,
                        m: 1, n: UInt32(d))
        }

        let splitCB = try #require(context.queue.makeCommandBuffer())
        let splitEncoder = try #require(splitCB.makeComputeCommandEncoder())
        try runtime.encode(encoder: splitEncoder,
                           x: xBuffer,
                           gate: gateProj, up: upProj, down: downProj,
                           y: ySplit,
                           scratchGate: gateScratch,
                           scratchUp: upScratch,
                           scratchAct: actScratch)
        if gated {
            encodeScalarGate(splitEncoder, y: scalarSplit)
            elementwise.encodeSigmoidScalarMul(encoder: splitEncoder,
                                               y: ySplit,
                                               gate: scalarSplit,
                                               count: d)
        }
        splitEncoder.endEncoding()
        splitCB.commit()
        splitCB.waitUntilCompleted()
        #expect(splitCB.status == .completed)

        let fusedCB = try #require(context.queue.makeCommandBuffer())
        let fusedEncoder = try #require(fusedCB.makeComputeCommandEncoder())
        try runtime.encodeGateUp(encoder: fusedEncoder,
                                 x: xBuffer,
                                 gate: gateProj, up: upProj,
                                 scratchGate: gateScratch,
                                 scratchUp: upScratch)
        if gated {
            encodeScalarGate(fusedEncoder, y: scalarFused)
        }
        try runtime.encodeFusedDown(encoder: fusedEncoder,
                                    down: downProj,
                                    gateIn: gateScratch,
                                    upIn: upScratch,
                                    y: yFused,
                                    scalarGate: gated ? scalarFused : nil)
        fusedEncoder.endEncoding()
        fusedCB.commit()
        fusedCB.waitUntilCompleted()
        #expect(fusedCB.status == .completed)

        #expect(Self.halfWords(ySplit, count: d) == Self.halfWords(yFused, count: d),
                "fused shared chain d=\(d) f=\(f) gated=\(gated) diverged from the split chain")
        if gated {
            #expect(Self.halfWords(scalarSplit, count: 1) == Self.halfWords(scalarFused, count: 1))
        }
    }

    private static func halfWords(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt16.self)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    private static func pack(_ values: [[Float]]) ->
        (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        let rows = values.map(Quantization.quantizeInt4Affine)
        return (rows,
                rows.flatMap(\.packed),
                rows.flatMap(\.scales),
                rows.flatMap(\.biases))
    }

    private static func projection(
        _ context: MetalContext,
        _ packed: (rows: [Quantization.Int4AffineRow], packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertProjection {
        SharedExpertProjection(
            weights: context.device.makeBuffer(bytes: packed.packed,
                                                length: packed.packed.count,
                                                options: .storageModeShared)!,
            scales: context.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * 2,
                                               options: .storageModeShared)!,
            biases: context.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * 2,
                                               options: .storageModeShared)!,
            rows: UInt32(rows), cols: UInt32(cols))
    }
}
