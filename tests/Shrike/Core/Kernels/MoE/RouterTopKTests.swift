import Foundation
import Metal
import Testing
@testable import Shrike
import ShrikeValidationSupport

@Suite struct RouterTopKTests {
    private static let experts = 16
    private static let dimension = 128
    private static let topK = 8

    private struct Result {
        let indices: [UInt32]
        let weights: [Float]
    }

    @Test func productionRouterMatchesReference() throws {
        var rng = SplitMix64(seed: 0xA5B6_1234)
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert) * 0.01
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let invSqrtD = 1.0 / Float(Self.dimension).squareRoot()
        let effectiveScale = (0..<Self.dimension).map { _ in
            rng.uniform(0.5, 1.5) * invSqrtD
        }
        let expertScale = (0..<Self.experts).map { _ in rng.uniform(0.6, 1.4) }

        let expected = Self.reference(weights: weights,
                                      hidden: hidden,
                                      effectiveScale: effectiveScale,
                                      expertScale: expertScale)
        let actual = try Self.run(weights: weights,
                                  hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale)
        #expect(actual.indices == expected.indices)
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    @Test func topFourRouterMatchesReference() throws {
        var rng = SplitMix64(seed: 0xB44D_5678)
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert) * 0.01
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let effectiveScale = [Float](repeating: 1.0, count: Self.dimension)
        let expertScale = (0..<Self.experts).map { _ in rng.uniform(0.6, 1.4) }

        let expected = Self.reference(weights: weights,
                                      hidden: hidden,
                                      effectiveScale: effectiveScale,
                                      expertScale: expertScale,
                                      topK: 4)
        let actual = try Self.run(weights: weights,
                                  hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale,
                                  topK: 4)
        #expect(actual.indices == expected.indices)
        #expect(actual.indices.count == 4)
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    @Test func additiveLogitBiasShiftsSelectionAndWeights() throws {
        var rng = SplitMix64(seed: 0x0B1A_5EED)
        let weights = (0..<Self.experts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert) * 0.01
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(0.5, 1.5) }
        let effectiveScale = [Float](repeating: 1.0, count: Self.dimension)
        let expertScale = [Float](repeating: 1.0, count: Self.experts)
        // A large bias on expert 0 (otherwise the weakest) must put it first.
        var bias = [Float](repeating: 0, count: Self.experts)
        bias[0] = 100.0

        let unbiased = try Self.run(weights: weights, hidden: hidden,
                                    effectiveScale: effectiveScale,
                                    expertScale: expertScale, topK: 4)
        let biased = try Self.run(weights: weights, hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale, topK: 4,
                                  logitBias: bias)
        #expect(!unbiased.indices.contains(0))
        #expect(biased.indices.first == 0)
        #expect(biased.weights.first.map { $0 > 0.99 } == true)
    }

    @Test func productionRouterResolvesNearTieLikeReference() throws {
        var rng = SplitMix64(seed: 0x71E_0F4A)
        let pattern = (0..<Self.dimension).map { _ in rng.uniform(0.2, 1.0) }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(0.5, 1.5) }
        var gains = [Float](repeating: 0, count: Self.experts)
        for expert in 0..<7 { gains[expert] = 1.0 - Float(expert) * 0.05 }
        gains[7] = 0.5
        gains[8] = 0.5 * (1.0 + 1e-4)
        for expert in 9..<Self.experts {
            gains[expert] = 0.4 - Float(expert - 9) * 0.02
        }
        let weights = gains.map { gain in pattern.map { $0 * gain } }
        let effectiveScale = [Float](repeating: 1.0, count: Self.dimension)
        let expertScale = [Float](repeating: 1.0, count: Self.experts)

        let expected = Self.reference(weights: weights,
                                      hidden: hidden,
                                      effectiveScale: effectiveScale,
                                      expertScale: expertScale)
        let actual = try Self.run(weights: weights,
                                  hidden: hidden,
                                  effectiveScale: effectiveScale,
                                  expertScale: expertScale)
        #expect(actual.indices == expected.indices)
    }

    // MARK: - Sigmoid scoring (Kimi)

    private static let kimiExperts = 256
    private static let kimiTopK = 8
    private static let kimiScaling: Float = 2.446

    /// Selection sorts by `sigmoid(logit) + correction bias`; weights are the
    /// ORIGINAL sigmoid scores of the selected, ÷ (sum + 1e-20), × scaling.
    private static func sigmoidReference(weights: [[Float]],
                                         hidden: [Float],
                                         correctionBias: [Float],
                                         experts: Int,
                                         topK: Int,
                                         scaling: Float) -> Result {
        let rows = weights.map { Quantization.quantizeInt8Affine($0) }
        let logits = DequantInt8GemvRef.apply(weightRows: rows,
                                              x: hidden,
                                              n: Self.dimension)
        func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
        var paired: [(Float, UInt32)] = []
        paired.reserveCapacity(experts)
        for expert in 0..<experts {
            paired.append((sigmoid(logits[expert]) + correctionBias[expert],
                           UInt32(expert)))
        }
        paired.sort { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }
        let selected = Array(paired.prefix(topK))
        let original = selected.map { sigmoid(logits[Int($0.1)]) }
        let sum = original.reduce(0, +) + 1e-20
        let outputWeights = original.map { $0 / sum * scaling }
        return Result(indices: selected.map { $0.1 }, weights: outputWeights)
    }

    private static func makeSigmoidFixture(seed: UInt64)
        -> (weights: [[Float]], hidden: [Float], bias: [Float]) {
        var rng = SplitMix64(seed: seed)
        let weights = (0..<Self.kimiExperts).map { expert in
            (0..<Self.dimension).map { _ in
                rng.uniform(-0.05, 0.05) + Float(expert % 37) * 0.003
            }
        }
        let hidden = (0..<Self.dimension).map { _ in rng.uniform(-1.0, 1.0) }
        let bias = (0..<Self.kimiExperts).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(-0.1, 0.1)))
        }
        return (weights, hidden, bias)
    }

    private static func makeSigmoidBuffers(
        _ context: MetalContext,
        weights: [[Float]], hidden: [Float], bias: [Float]
    ) throws -> (weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
                 hidden: MTLBuffer, ones: MTLBuffer, ext: MTLBuffer,
                 correction: MTLBuffer) {
        let packedRows = weights.map { Quantization.quantizeInt8Affine($0) }
        let packed = packedRows.flatMap(\.packed)
        let scales = packedRows.flatMap(\.scales)
        let biases = packedRows.flatMap(\.biases)
        let onesE = [Float](repeating: 1.0, count: Self.kimiExperts)
            .map(Quantization.bf16Bits)
        let onesD = [Float](repeating: 1.0, count: Self.dimension)
            .map(Quantization.bf16Bits)
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales, length: scales.count * 2, options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases, length: biases.count * 2, options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let onesBuffer = context.device.makeBuffer(
                  bytes: onesE, length: onesE.count * 2, options: .storageModeShared),
              let extBuffer = context.device.makeBuffer(
                  bytes: onesD, length: onesD.count * 2, options: .storageModeShared),
              let correctionBuffer = context.device.makeBuffer(
                  bytes: bias.map(Quantization.bf16Bits),
                  length: bias.count * 2, options: .storageModeShared) else {
            throw CocoaError(.fileReadUnknown)
        }
        return (weightBuffer, scaleBuffer, biasBuffer, hiddenBuffer,
                onesBuffer, extBuffer, correctionBuffer)
    }

    @Test func sigmoidRouterMatchesReference_kimiShape() throws {
        let fixture = Self.makeSigmoidFixture(seed: 0x516_0001)
        let hidden16 = fixture.hidden.map { Float(Float16($0)) }
        let expected = Self.sigmoidReference(weights: fixture.weights,
                                             hidden: hidden16,
                                             correctionBias: fixture.bias,
                                             experts: Self.kimiExperts,
                                             topK: Self.kimiTopK,
                                             scaling: Self.kimiScaling)
        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             specializedNumExperts: UInt32(Self.kimiExperts),
                             specializedTopK: UInt32(Self.kimiTopK),
                             sigmoidRouterScores: true,
                             routedScalingFactor: Self.kimiScaling)
        let buffers = try Self.makeSigmoidBuffers(
            context, weights: fixture.weights, hidden: fixture.hidden,
            bias: fixture.bias)
        guard let indexBuffer = context.device.makeBuffer(
                  length: Self.kimiTopK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: Self.kimiTopK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        try kernel.encodeRouter(
            commandBuffer: commandBuffer,
            weights: buffers.weights, scales: buffers.scales,
            biases: buffers.biases, hidden: buffers.hidden,
            effectiveScale: buffers.ext,
            perExpertScale: buffers.ones,
            logitBias: buffers.correction,
            outIndices: indexBuffer, outWeights: weightBuffer,
            numExperts: UInt32(Self.kimiExperts),
            d: UInt32(Self.dimension),
            topK: UInt32(Self.kimiTopK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: Self.kimiTopK)
        let actual = Result(
            indices: (0..<Self.kimiTopK).map { indexPointer[$0] },
            weights: Fp16Buffer.read(weightBuffer, count: Self.kimiTopK))
        #expect(actual.indices == expected.indices)
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3, "sigmoid weights max err \(maxError)")
    }

    /// The correction bias moves an expert into the selection but must NOT
    /// inflate its weight — the weight comes from the original sigmoid score.
    @Test func sigmoidCorrectionBiasShiftsSelectionNotWeights() throws {
        let fixture = Self.makeSigmoidFixture(seed: 0x516_0002)
        let hidden16 = fixture.hidden.map { Float(Float16($0)) }
        let unbiased = Self.sigmoidReference(weights: fixture.weights,
                                             hidden: hidden16,
                                             correctionBias: [Float](repeating: 0,
                                                                     count: Self.kimiExperts),
                                             experts: Self.kimiExperts,
                                             topK: Self.kimiTopK,
                                             scaling: Self.kimiScaling)
        guard let weakest = (0..<Self.kimiExperts).first(where: { expert in
            !unbiased.indices.contains(UInt32(expert))
        }) else {
            Issue.record("no unselected expert to promote"); return
        }
        var bias = [Float](repeating: 0, count: Self.kimiExperts)
        bias[weakest] = 2.0

        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             specializedNumExperts: UInt32(Self.kimiExperts),
                             specializedTopK: UInt32(Self.kimiTopK),
                             sigmoidRouterScores: true,
                             routedScalingFactor: Self.kimiScaling)
        let buffers = try Self.makeSigmoidBuffers(
            context, weights: fixture.weights, hidden: fixture.hidden, bias: bias)
        guard let indexBuffer = context.device.makeBuffer(
                  length: Self.kimiTopK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device, count: Self.kimiTopK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        try kernel.encodeRouter(
            commandBuffer: commandBuffer,
            weights: buffers.weights, scales: buffers.scales,
            biases: buffers.biases, hidden: buffers.hidden,
            effectiveScale: buffers.ext,
            perExpertScale: buffers.ones,
            logitBias: buffers.correction,
            outIndices: indexBuffer, outWeights: weightBuffer,
            numExperts: UInt32(Self.kimiExperts),
            d: UInt32(Self.dimension),
            topK: UInt32(Self.kimiTopK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: Self.kimiTopK)
        let indices = (0..<Self.kimiTopK).map { indexPointer[$0] }
        let weightsOut = Fp16Buffer.read(weightBuffer, count: Self.kimiTopK)
        #expect(indices.first == UInt32(weakest),
                "bias +2.0 must put the promoted expert first in selection order")
        guard let promoted = indices.firstIndex(of: UInt32(weakest)) else { return }
        let others = weightsOut.enumerated()
            .filter { $0.offset != promoted }
            .map(\.element)
        #expect(weightsOut[promoted] < (others.max() ?? 0),
                "the promoted expert's weight must reflect its original sigmoid score, not the bias")
    }

    @Test(arguments: [PrefillRouter.Kind.block, .tiled])
    func sigmoidPrefillRouterMatchesReference(kind: PrefillRouter.Kind) throws {
        let fixtureA = Self.makeSigmoidFixture(seed: 0x516_0003)
        let fixtureB = Self.makeSigmoidFixture(seed: 0x516_0004)
        let rows = [fixtureA.hidden, fixtureB.hidden]
        let context = try MetalContext()
        let kernel = try PrefillRouter(context: context,
                                       weightBits: 8,
                                       sigmoidRouterScores: true,
                                       routedScalingFactor: Self.kimiScaling,
                                       kind: kind)
        let buffers = try Self.makeSigmoidBuffers(
            context, weights: fixtureA.weights,
            hidden: rows.flatMap { $0 }, bias: fixtureA.bias)
        guard let indexBuffer = context.device.makeBuffer(
                  length: 2 * Self.kimiTopK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let weightBuffer = Fp16Buffer.make(context.device,
                                                 count: 2 * Self.kimiTopK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        try kernel.encodeBlock(
            commandBuffer: commandBuffer,
            weights: buffers.weights, scales: buffers.scales,
            biases: buffers.biases, hidden: buffers.hidden,
            effectiveScale: buffers.ext,
            perExpertScale: buffers.ones,
            logitBias: buffers.correction,
            outIndices: indexBuffer, outWeights: weightBuffer,
            queryCount: 2,
            numExperts: UInt32(Self.kimiExperts),
            d: UInt32(Self.dimension),
            topK: UInt32(Self.kimiTopK),
            hiddenStrideElements: UInt32(Self.dimension))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: 2 * Self.kimiTopK)
        let weightsOut = Fp16Buffer.read(weightBuffer, count: 2 * Self.kimiTopK)
        for (row, hidden) in rows.enumerated() {
            let hidden16 = hidden.map { Float(Float16($0)) }
            let expected = Self.sigmoidReference(weights: fixtureA.weights,
                                                 hidden: hidden16,
                                                 correctionBias: fixtureA.bias,
                                                 experts: Self.kimiExperts,
                                                 topK: Self.kimiTopK,
                                                 scaling: Self.kimiScaling)
            let base = row * Self.kimiTopK
            let indices = (0..<Self.kimiTopK).map { indexPointer[base + $0] }
            #expect(indices == expected.indices, "row \(row) selection diverged")
            let maxError = zip((0..<Self.kimiTopK).map { weightsOut[base + $0] },
                               expected.weights)
                .map { abs($0 - $1) }
                .max() ?? 0
            #expect(maxError < 5e-3, "row \(row) sigmoid weights max err \(maxError)")
        }
    }

    private static func reference(weights: [[Float]],
                                  hidden: [Float],
                                  effectiveScale: [Float],
                                  expertScale: [Float],
                                  topK: Int = RouterTopKTests.topK) -> Result {
        let scaled = zip(hidden, effectiveScale).map { $0 * $1 }
        let rows = weights.map { Quantization.quantizeInt8Affine($0) }
        let logits = DequantInt8GemvRef.apply(weightRows: rows,
                                              x: scaled,
                                              n: Self.dimension)
        var paired: [(Float, UInt32)] = []
        paired.reserveCapacity(Self.experts)
        for expert in 0..<Self.experts {
            paired.append((logits[expert], UInt32(expert)))
        }
        paired.sort { lhs, rhs in
            lhs.0 == rhs.0 ? lhs.1 < rhs.1 : lhs.0 > rhs.0
        }
        let selected = Array(paired.prefix(topK))
        let maximum = selected.first?.0 ?? 0
        let exponents = selected.map { exp($0.0 - maximum) }
        let sum = exponents.reduce(0, +)
        let outputWeights = zip(selected, exponents).map { item, value in
            value / sum * expertScale[Int(item.1)]
        }
        return Result(indices: selected.map { $0.1 }, weights: outputWeights)
    }

    private static func run(weights: [[Float]],
                            hidden: [Float],
                            effectiveScale: [Float],
                            expertScale: [Float],
                            topK: Int = RouterTopKTests.topK,
                            logitBias: [Float]? = nil) throws -> Result {
        let packedRows = weights.map { Quantization.quantizeInt8Affine($0) }
        let groupsPerRow = Self.dimension / Quantization.groupSize
        let packed = packedRows.flatMap(\.packed)
        let scales = packedRows.flatMap(\.scales)
        let biases = packedRows.flatMap(\.biases)
        precondition(scales.count == Self.experts * groupsPerRow)

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        guard let weightBuffer = context.device.makeBuffer(
                  bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = context.device.makeBuffer(
                  bytes: scales,
                  length: scales.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                  bytes: biases,
                  length: biases.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(context.device, values: hidden),
              let effectiveBuffer = context.device.makeBuffer(
                  bytes: effectiveScale.map(Quantization.bf16Bits),
                  length: effectiveScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let expertScaleBuffer = context.device.makeBuffer(
                  bytes: expertScale.map(Quantization.bf16Bits),
                  length: expertScale.count * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let logitBiasBuffer = context.device.makeBuffer(
                  bytes: (logitBias ?? [Float](repeating: 0, count: Self.experts))
                      .map(Quantization.bf16Bits),
                  length: Self.experts * MemoryLayout<UInt16>.stride,
                  options: .storageModeShared),
              let indexBuffer = context.device.makeBuffer(
                  length: topK * MemoryLayout<UInt32>.stride,
                  options: .storageModeShared),
              let outputWeightBuffer = Fp16Buffer.make(context.device, count: topK),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw CocoaError(.fileReadUnknown)
        }
        try kernel.encodeRouter(
            commandBuffer: commandBuffer,
            weights: weightBuffer,
            scales: scaleBuffer,
            biases: biasBuffer,
            hidden: hiddenBuffer,
            effectiveScale: effectiveBuffer,
            perExpertScale: expertScaleBuffer,
            logitBias: logitBiasBuffer,
            outIndices: indexBuffer,
            outWeights: outputWeightBuffer,
            numExperts: UInt32(Self.experts),
            d: UInt32(Self.dimension),
            topK: UInt32(topK))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let indexPointer = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: topK)
        return Result(
            indices: (0..<topK).map { indexPointer[$0] },
            weights: Fp16Buffer.read(outputWeightBuffer, count: topK))
    }
}
