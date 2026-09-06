import Foundation
import Metal
import Testing
@testable import Shrike
import ShrikeValidationSupport

@Suite struct MoEFusedFFNTests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    @Test(arguments: [8])
    func affineRoutedPipelineSupportsQwenBits(bits: Int) throws {
        let blobs = (0..<Self.topK).map { _ in Self.makeConstantBlob(bits: bits) }
        let context = try MetalContext()
        let kernel = try MoE(context: context, siluActivation: true,
                             routedWeightBits: bits,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: 256)
        let routed = blobs.map {
            context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count)!
        }
        let x = Fp16Buffer.make(context.device,
                                halves: [Float16](repeating: 1, count: Self.dimension))!
        let acts = Fp16Buffer.make(context.device,
                                   count: Self.topK * Self.intermediate)!
        let weights = Fp16Buffer.make(context.device,
                                      halves: [Float16](repeating: 0.125, count: Self.topK))!
        let residual = Fp16Buffer.make(context.device, count: Self.dimension)!
        let output = Fp16Buffer.make(context.device, count: Self.dimension)!
        memset(residual.contents(), 0, residual.length)
        let args = kernel.makeRoutedArgumentBuffer(routedBlobs: routed, topK: 8)!
        let cb = context.queue.makeCommandBuffer()!
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb, routedArgBuffer: args, routedBlobs: routed,
            routedOffsets: blobs[0].offsets, x: x, acts: acts,
            d: 128, f: 64, topK: 8)
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: cb, routedArgBuffer: args, routedBlobs: routed,
            routedOffsets: blobs[0].offsets, acts: acts,
            routingWeights: weights, residual: residual, y: output,
            d: 128, f: 64, topK: 8)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let expected = Float(2) / (1 + exp(-Float(2)))
        let actual = Fp16Buffer.read(output, count: Self.dimension)
        #expect(actual.allSatisfy { abs($0 - expected) < 0.01 })
    }

    @Test func productionRoutedPipelineAndHitSplitMatchReference() throws {
        var rng = SeedTree(0x2D3).key("production-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<Self.topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let expected = MoeRef.applyStreamedRouted(
            x: x,
            residual: residual,
            routedGate: gates.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedUp: ups.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            routedDown: downs.map { rows in
                rows.map { Quantization.quantizeInt4Affine($0) }
            },
            indices: Array(0..<Self.topK),
            routingWeights: routingWeights,
            d: Self.dimension,
            f: Self.intermediate)
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let splitActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let splitOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let lowSlots = context.device.makeBuffer(
                bytes: [UInt32](0...3),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let highSlots = context.device.makeBuffer(
                bytes: [UInt32](4...7),
                length: 4 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        let splitCommand = context.queue.makeCommandBuffer()!
        for (slots, activeSlots) in [([UInt32](0...3), lowSlots),
                                     ([UInt32](4...7), highSlots)] {
            try kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: splitCommand,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: activeSlots,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
        }
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: splitCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        splitCommand.commit()
        splitCommand.waitUntilCompleted()
        #expect(splitCommand.error == nil)

        let full = Fp16Buffer.read(fullOutput, count: Self.dimension)
        let split = Fp16Buffer.read(splitOutput, count: Self.dimension)
        #expect(full == split)
        #expect(RelError.compute(actual: full, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    @Test func speculativePoolPipelineMatchesRoutedPipeline() throws {
        var rng = SeedTree(0x2D3).key("speculative-pool-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<Self.dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<Self.topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }
        let blobs = (0..<Self.topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        let poolSlotStride = ((blobs.map(\.bytes.count).max()! + 63) / 64) * 64
        let slotCount = 16
        let slotOfExpert: [UInt32] = [5, 2, 9, 0, 12, 3, 15, 7]
        guard routedBuffers.count == Self.topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let fullActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let specActs = Fp16Buffer.make(
                context.device, count: Self.topK * Self.intermediate),
              let fullOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let specOutput = Fp16Buffer.make(context.device, count: Self.dimension),
              let pool = context.device.makeBuffer(
                length: poolSlotStride * slotCount, options: .storageModeShared),
              let resolvedSlots = context.device.makeBuffer(
                bytes: slotOfExpert,
                length: slotOfExpert.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let indirectArgs = context.device.makeBuffer(
                length: MoE.specDispatchArgsLength, options: .storageModeShared),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(Self.topK)) else {
            Issue.record("buffer allocation failed")
            return
        }
        for (expert, blob) in blobs.enumerated() {
            blob.bytes.withUnsafeBytes { bytes in
                pool.contents()
                    .advanced(by: Int(slotOfExpert[expert]) * poolSlotStride)
                    .copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }

        let fullCommand = context.queue.makeCommandBuffer()!
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: fullActs,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: fullCommand,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: fullActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: fullOutput,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        fullCommand.commit()
        fullCommand.waitUntilCompleted()
        #expect(fullCommand.error == nil)

        func writeGrids(phase1: MTLSize, phase2: MTLSize, tail: MTLSize) {
            let grids: [UInt32] = [
                UInt32(phase1.width), UInt32(phase1.height), UInt32(phase1.depth),
                UInt32(phase2.width), UInt32(phase2.height), UInt32(phase2.depth),
                UInt32(tail.width), UInt32(tail.height), UInt32(tail.depth),
            ]
            grids.withUnsafeBytes {
                indirectArgs.contents().copyMemory(
                    from: $0.baseAddress!, byteCount: $0.count)
            }
        }
        func runSpec() throws {
            let command = context.queue.makeCommandBuffer()!
            try kernel.encodeSpecPhase1U16Load(
                commandBuffer: command,
                expertPool: pool,
                poolSlotStride: UInt64(poolSlotStride),
                resolvedSlots: resolvedSlots,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: specActs,
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK),
                indirectArguments: indirectArgs)
            try kernel.encodeSpecPhase2Reduce(
                commandBuffer: command,
                expertPool: pool,
                poolSlotStride: UInt64(poolSlotStride),
                resolvedSlots: resolvedSlots,
                routedOffsets: blobs[0].offsets,
                acts: specActs,
                routingWeights: routingBuffer,
                residual: residualBuffer,
                y: specOutput,
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK),
                indirectArguments: indirectArgs)
            command.commit()
            command.waitUntilCompleted()
            #expect(command.error == nil)
        }

        writeGrids(
            phase1: MoE.specPhase1FullGrid(f: UInt32(Self.intermediate),
                                           topK: UInt32(Self.topK)),
            phase2: MoE.specPhase2FullGrid(d: UInt32(Self.dimension)),
            tail: MoE.specTailFullGrid(
                d: UInt32(Self.dimension),
                threadgroupWidth: Elementwise.residualAddThreadgroupWidth))
        try runSpec()
        #expect(Fp16Buffer.read(specActs, count: Self.topK * Self.intermediate)
                == Fp16Buffer.read(fullActs, count: Self.topK * Self.intermediate))
        #expect(Fp16Buffer.read(specOutput, count: Self.dimension)
                == Fp16Buffer.read(fullOutput, count: Self.dimension))

        let elementwise = try Elementwise(context: context)
        guard let hiddenClassic = Fp16Buffer.make(context.device, values: residual),
              let hiddenSpec = Fp16Buffer.make(context.device, values: residual) else {
            Issue.record("hidden buffer allocation failed")
            return
        }
        let tailCommand = context.queue.makeCommandBuffer()!
        try elementwise.encodeResidualAdd(commandBuffer: tailCommand,
                                          hidden: hiddenClassic,
                                          delta: fullOutput,
                                          count: Self.dimension)
        try elementwise.encodeResidualAddIndirect(
            commandBuffer: tailCommand,
            hidden: hiddenSpec,
            delta: specOutput,
            count: Self.dimension,
            indirectArguments: indirectArgs,
            indirectOffset: MoE.specTailArgsOffset)
        tailCommand.commit()
        tailCommand.waitUntilCompleted()
        #expect(tailCommand.error == nil)
        #expect(Fp16Buffer.read(hiddenSpec, count: Self.dimension)
                == Fp16Buffer.read(hiddenClassic, count: Self.dimension))

        let sentinel: [Float] = (0..<Self.dimension).map { Float($0 % 7) - 3 }
        sentinel.enumerated().forEach { index, value in
            specOutput.contents()
                .bindMemory(to: Float16.self, capacity: Self.dimension)[index]
                = Float16(value)
        }
        writeGrids(phase1: MTLSize(width: 0, height: 1, depth: 1),
                   phase2: MTLSize(width: 0, height: 1, depth: 1),
                   tail: MTLSize(width: 0, height: 1, depth: 1))
        try runSpec()
        #expect(Fp16Buffer.read(specOutput, count: Self.dimension)
                == sentinel.map { Float(Float16($0)) })

        // Lever A: an absent slot makes the spec phase 1 skip that position, so
        // its activation row keeps whatever it held while the others match.
        let absentPosition = 3
        let rowSentinel: [Float] = (0..<Self.intermediate).map { Float($0 % 5) - 2 }
        let actsPointer = specActs.contents()
            .bindMemory(to: Float16.self, capacity: Self.topK * Self.intermediate)
        for position in 0..<Self.topK {
            for (index, value) in rowSentinel.enumerated() {
                actsPointer[position * Self.intermediate + index] = Float16(value)
            }
        }
        resolvedSlots.contents()
            .bindMemory(to: UInt32.self, capacity: Self.topK)[absentPosition] = 0xffff_ffff
        writeGrids(
            phase1: MoE.specPhase1FullGrid(f: UInt32(Self.intermediate),
                                           topK: UInt32(Self.topK)),
            phase2: MTLSize(width: 0, height: 1, depth: 1),
            tail: MTLSize(width: 0, height: 1, depth: 1))
        try runSpec()
        let partialActs = Fp16Buffer.read(specActs, count: Self.topK * Self.intermediate)
        let referenceActs = Fp16Buffer.read(fullActs, count: Self.topK * Self.intermediate)
        for position in 0..<Self.topK {
            let row = position * Self.intermediate..<(position + 1) * Self.intermediate
            if position == absentPosition {
                #expect(Array(partialActs[row]) == rowSentinel.map { Float(Float16($0)) })
            } else {
                #expect(Array(partialActs[row]) == Array(referenceActs[row]))
            }
        }
    }

    @Test func gptOssRoutedPipelineWithBiasesMatchesReference() throws {
        let topK = 4
        var rng = SeedTree(0x6F55).key("gptoss-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }
        func bf16RoundTrip(_ v: Float) -> Float {
            Float(bitPattern: UInt32(Quantization.bf16Bits(v)) << 16)
        }

        var gates = [[[Float]]](), ups = [[[Float]]](), downs = [[[Float]]]()
        var gateBiases = [[Float]](), upBiases = [[Float]](), downBiases = [[Float]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
            gateBiases.append((0..<Self.intermediate).map { _ in
                bf16RoundTrip(rng.uniform(-0.3, 0.3)) })
            upBiases.append((0..<Self.intermediate).map { _ in
                bf16RoundTrip(rng.uniform(-0.3, 0.3)) })
            downBiases.append((0..<Self.dimension).map { _ in
                bf16RoundTrip(rng.uniform(-0.3, 0.3)) })
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<topK).map { Float(Float16(0.1 + Float($0) * 0.05)) }

        var expected = residual
        for slot in 0..<topK {
            let out = MoeRef.runFFNGptOss(
                gateRows: gates[slot].map { Quantization.quantizeInt4Affine($0) },
                upRows: ups[slot].map { Quantization.quantizeInt4Affine($0) },
                downRows: downs[slot].map { Quantization.quantizeInt4Affine($0) },
                gateBias: gateBiases[slot],
                upBias: upBiases[slot],
                downBias: downBiases[slot],
                x: x, d: Self.dimension, f: Self.intermediate)
            for r in 0..<Self.dimension {
                expected[r] += routingWeights[slot] * out[r]
            }
        }

        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0],
                          gateBias: gateBiases[$0], upBias: upBiases[$0],
                          downBias: downBiases[$0])
        }
        let context = try MetalContext()
        let kernel = try MoE(context: context,
                             siluActivation: true,
                             specializedD: UInt32(Self.dimension),
                             specializedF: UInt32(Self.intermediate),
                             specializedNumExperts: 32,
                             specializedTopK: UInt32(topK),
                             expertAdditiveBiases: true,
                             clampedSwiGLU: true)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes, length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let acts = Fp16Buffer.make(context.device, count: topK * Self.intermediate),
              let output = Fp16Buffer.make(context.device, count: Self.dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers, topK: UInt32(topK)) else {
            Issue.record("buffer allocation failed")
            return
        }
        let cb = context.queue.makeCommandBuffer()!
        try kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: cb, routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers, routedOffsets: blobs[0].offsets,
            x: xBuffer, acts: acts,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        try kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: cb, routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers, routedOffsets: blobs[0].offsets,
            acts: acts, routingWeights: routingBuffer,
            residual: residualBuffer, y: output,
            d: UInt32(Self.dimension), f: UInt32(Self.intermediate),
            topK: UInt32(topK))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let actual = Fp16Buffer.read(output, count: Self.dimension)
        #expect(RelError.compute(actual: actual, reference: expected)
            < Tolerance.fp16ChainedReduction)
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]],
                                 gateBias: [Float]? = nil,
                                 upBias: [Float]? = nil,
                                 downBias: [Float]? = nil) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt4Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        var gateAB: UInt32 = 0, upAB: UInt32 = 0, downAB: UInt32 = 0
        if let gateBias, let upBias, let downBias {
            gateAB = UInt32(bytes.count)
            append(gateBias.map { Quantization.bf16Bits($0) })
            upAB = UInt32(bytes.count)
            append(upBias.map { Quantization.bf16Bits($0) })
            downAB = UInt32(bytes.count)
            append(downBias.map { Quantization.bf16Bits($0) })
        }
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB,
                gateABOff: gateAB, upABOff: upAB, downABOff: downAB))
    }

    private static func makeConstantBlob(bits: Int) -> RoutedBlob {
        func bf16(_ value: Float) -> UInt16 {
            UInt16(truncatingIfNeeded: value.bitPattern >> 16)
        }
        var bytes = [UInt8]()
        func appendZeros(rows: Int, cols: Int) -> UInt32 {
            let offset = UInt32(bytes.count)
            bytes += [UInt8](repeating: 0, count: rows * cols * bits / 8)
            return offset
        }
        func appendBF16(_ value: Float, count: Int) -> UInt32 {
            let offset = UInt32(bytes.count)
            let raw = bf16(value)
            for _ in 0..<count {
                bytes.append(UInt8(truncatingIfNeeded: raw))
                bytes.append(UInt8(truncatingIfNeeded: raw >> 8))
            }
            return offset
        }
        let gateW = appendZeros(rows: intermediate, cols: dimension)
        let gateS = appendBF16(0, count: intermediate * dimension / 64)
        let gateB = appendBF16(1.0 / 64.0, count: intermediate * dimension / 64)
        let upW = appendZeros(rows: intermediate, cols: dimension)
        let upS = appendBF16(0, count: intermediate * dimension / 64)
        let upB = appendBF16(0.5 / 64.0, count: intermediate * dimension / 64)
        let downW = appendZeros(rows: dimension, cols: intermediate)
        let downS = appendBF16(0, count: dimension * intermediate / 64)
        let downB = appendBF16(1.0 / 64.0, count: dimension * intermediate / 64)
        return RoutedBlob(bytes: bytes, offsets: MoEExpertOffsets(
            gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
            upWOff: upW, upSOff: upS, upBOff: upB,
            downWOff: downW, downSOff: downS, downBOff: downB))
    }
}
