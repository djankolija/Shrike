import Foundation
import Metal

/// Lives in the module because the encoders it drives are internal.
public enum PrefillRoutedGEMMBenchmark {
    public struct Result: Sendable {
        public let experts: Int
        public let rowsPerExpert: Int
        public let d: Int
        public let f: Int
        public let stagingRows: Int
        public let groupedWaves: Int
        public let perExpertMillisPerTile: Double
        public let groupedMillisPerTile: Double
        public let variant: String
        public let weightLoads: String
        public let rowTile: Int
        public let tailTile: Int
        public var gflopPerTile: Double {
            Double(experts) * 6.0 * Double(rowsPerExpert) * Double(d) * Double(f) / 1.0e9
        }
        public var perExpertTFLOPS: Double { gflopPerTile / perExpertMillisPerTile }
        public var groupedTFLOPS: Double { gflopPerTile / groupedMillisPerTile }
    }

    public static func run(context: MetalContext,
                           iterations: Int,
                           experts: Int = 8,
                           rowsPerExpert: Int = 128,
                           d: Int = 2048,
                           f: Int = 512,
                           stagingRows: Int = 1024,
                           rowTile rowTileRows: Int = 64,
                           tailTile: Int = 0) throws -> Result {
        guard let rowTile = MPPPrefillInt4QMM.GroupedRowTile(rawValue: rowTileRows) else {
            throw MPPPrefillInt4QMMError.invalidArguments("row tile \(rowTileRows) is not 64 or 32")
        }
        let device = context.device
        let mpp = MPPPrefillInt4QMM(context: context, weightBits: 4)
        guard mpp.isAvailable, mpp.groupedAvailable else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(
                reason: "MPP prefill pipelines unavailable on this device")
        }
        let grouped = try PrefillGroupedRoutedMoE(context: context, siluActivation: true, weightBits: 4)
        let tokens = experts * rowsPerExpert
        let pairs = (0..<tokens).map { token in
            PrefillTokenExpertPair(token: UInt32(token),
                                   expert: UInt32(token / rowsPerExpert),
                                   rank: 0,
                                   weight: 1)
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs, queryCount: tokens, topK: 1, numExperts: experts, tileExpertCount: 16)
        guard routes.tiles.count == 1 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expected one tile, got \(routes.tiles.count)")
        }
        let pool = SyntheticExpertPool(device: device, experts: experts, d: d, f: f)
        let binding = try PrefillStreamedTileBinding(expertIDs: Array(0..<experts), views: pool.views)
        let argumentBuffer = try grouped.makeStreamedArgumentBuffer(device: device, binding: binding)
        let tile = routes.tiles[0]
        let params = PrefillGroupedRoutedMoEStreamedParams(
            pairStart: tile.pairStart,
            pairCount: tile.pairCount,
            d: UInt32(d),
            routedIntermediate: UInt32(f),
            topK: 1,
            hiddenStrideElements: UInt32(d),
            binding: binding,
            offsets: pool.offsets)
        func shared(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed(label)
            }
            buffer.label = label
            return buffer
        }
        let hidden = try shared(tokens * d * MemoryLayout<Float16>.stride, "bench.hidden")
        let hiddenValues = hidden.contents().bindMemory(to: Float16.self, capacity: tokens * d)
        for i in 0..<(tokens * d) {
            hiddenValues[i] = Float16(Float((i % 17) - 8) * 0.125)
        }
        let sortedPairs = try shared(routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
                                     "bench.sortedPairs")
        routes.sortedPairs.withUnsafeBytes { bytes in
            sortedPairs.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        let routePartials = try shared(tokens * d * MemoryLayout<Float16>.stride, "bench.routePartials")
        let staging = try PrefillExpertStaging.allocate(device: device,
                                                        rowBlock: stagingRows,
                                                        hiddenSize: d,
                                                        intermediate: f)
        let ranges = try PrefillExpertPairRange.ranges(forTile: tile, routes: routes)
        let waves = try PrefillGroupedRoutedMoE.planExpertWaves(ranges: ranges,
                                                                binding: binding,
                                                                stagingRows: stagingRows,
                                                                rowTile: rowTile.rawValue,
                                                                tailTile: tailTile)

        func encodePerExpert(_ commandBuffer: MTLCommandBuffer) throws {
            let leftovers = try grouped.encodeExpertGEMMs(commandBuffer: commandBuffer,
                                                          mpp: mpp,
                                                          hidden: hidden,
                                                          sortedPairs: sortedPairs,
                                                          routePartials: routePartials,
                                                          binding: binding,
                                                          ranges: ranges,
                                                          staging: staging,
                                                          params: params)
            guard leftovers.isEmpty else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "per-expert path left \(leftovers.count) experts for the scalar path")
            }
        }
        func encodeGrouped(_ commandBuffer: MTLCommandBuffer) throws {
            try grouped.encodeGroupedExpertGEMMs(commandBuffer: commandBuffer,
                                                 mpp: mpp,
                                                 hidden: hidden,
                                                 sortedPairs: sortedPairs,
                                                 routePartials: routePartials,
                                                 binding: binding,
                                                 argumentBuffer: argumentBuffer,
                                                 waves: waves,
                                                 staging: staging,
                                                 params: params,
                                                 rowTile: rowTile,
                                                 tailTile: tailTile)
        }
        let perExpert = try timeEncodes(context: context, iterations: iterations, encodePerExpert)
        let groupedMillis = try timeEncodes(context: context, iterations: iterations, encodeGrouped)
        return Result(experts: experts,
                      rowsPerExpert: rowsPerExpert,
                      d: d,
                      f: f,
                      stagingRows: stagingRows,
                      groupedWaves: waves.count,
                      perExpertMillisPerTile: perExpert,
                      groupedMillisPerTile: groupedMillis,
                      variant: mpp.variant.rawValue,
                      weightLoads: mpp.weightLoads.rawValue,
                      rowTile: rowTile.rawValue,
                      tailTile: tailTile)
    }

    private static func timeEncodes(context: MetalContext,
                                    iterations: Int,
                                    _ encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
        guard let warm = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        try encode(warm)
        warm.commit()
        warm.waitUntilCompleted()
        if let error = warm.error { throw error }
        guard let timed = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        for _ in 0..<iterations {
            try encode(timed)
        }
        timed.commit()
        timed.waitUntilCompleted()
        if let error = timed.error { throw error }
        return (timed.gpuEndTime - timed.gpuStartTime) * 1000 / Double(iterations)
    }

    /// Gate, up and down per expert, each as packed int4 rows then bf16 group
    /// scales then bf16 group biases — the blob order the runtime's offsets assume.
    private struct SyntheticExpertPool {
        let views: [TensorView]
        let offsets: MoEExpertOffsets

        init(device: MTLDevice, experts: Int, d: Int, f: Int) {
            let group = Quantization.groupSize
            func projectionBytes(rows: Int, cols: Int) -> (packed: Int, scales: Int) {
                (rows * cols / 2, rows * (cols / group) * MemoryLayout<UInt16>.stride)
            }
            let gateUp = projectionBytes(rows: f, cols: d)
            let down = projectionBytes(rows: d, cols: f)
            var cursor = 0
            func take(_ bytes: Int) -> UInt32 {
                defer { cursor += bytes }
                return UInt32(cursor)
            }
            let offsets = MoEExpertOffsets(gateWOff: take(gateUp.packed),
                                           gateSOff: take(gateUp.scales),
                                           gateBOff: take(gateUp.scales),
                                           upWOff: take(gateUp.packed),
                                           upSOff: take(gateUp.scales),
                                           upBOff: take(gateUp.scales),
                                           downWOff: take(down.packed),
                                           downSOff: take(down.scales),
                                           downBOff: take(down.scales),
                                           gateABOff: 0,
                                           upABOff: 0,
                                           downABOff: 0)
            let stride = cursor
            let scaleBits = UInt16(truncatingIfNeeded: Float(0.02).bitPattern >> 16)
            var seed: UInt32 = 0x9E37_79B9
            self.offsets = offsets
            self.views = (0..<experts).map { expert in
                guard let buffer = device.makeBuffer(length: stride, options: .storageModeShared) else {
                    fatalError("could not allocate synthetic expert \(expert)")
                }
                let bytes = buffer.contents().bindMemory(to: UInt8.self, capacity: stride)
                for i in 0..<stride {
                    seed = seed &* 1_664_525 &+ 1_013_904_223
                    bytes[i] = UInt8(truncatingIfNeeded: seed >> 24)
                }
                let halves = buffer.contents().bindMemory(to: UInt16.self, capacity: stride / 2)
                for (scales, biases, count) in [
                    (offsets.gateSOff, offsets.gateBOff, gateUp.scales),
                    (offsets.upSOff, offsets.upBOff, gateUp.scales),
                    (offsets.downSOff, offsets.downBOff, down.scales),
                ] {
                    for i in 0..<(count / 2) {
                        halves[Int(scales) / 2 + i] = scaleBits
                        halves[Int(biases) / 2 + i] = 0
                    }
                }
                return TensorView(buffer: buffer,
                                  offset: 0,
                                  length: UInt64(stride),
                                  scaleOffset: 0,
                                  scaleLength: 0,
                                  biasOffset: 0,
                                  biasLength: 0,
                                  shape: (0, UInt32(expert), 0, 0),
                                  dtype: 0)
            }
        }
    }
}

extension PrefillRoutedGEMMBenchmark {
    public struct TileComparison: Sendable {
        public let m: Int
        public let n: Int
        public let k: Int
        public let bits: Int
        public let pair: String
        public let mismatches: Int
        public let maxAbsDiff: Float
        public let maxRelDiff: Float
        public let firstMismatch: Int
    }

    /// Runs two K-tile variants (by name, `n32b1` against `n32k128b1` by
    /// default) on one deterministic full-mantissa input and compares the fp16
    /// outputs element for element, so a box without the test suite (the mini)
    /// can bound how far one K reduction order is from the other.
    /// Inputs whose partial sums are exact in fp32 (integers over 64, a few
    /// bf16 scales) hide the reduction order — they compare bit-identical.
    public static func compareTileK(context: MetalContext,
                                    m: Int = 128,
                                    n: Int = 2048,
                                    k: Int = 2048,
                                    bits: Int = 4,
                                    narrow narrowName: String = "n32b1",
                                    wide wideName: String = "n32k128b1") throws -> TileComparison {
        let device = context.device
        guard let narrowVariant = MPPPrefillInt4QMM.TileVariant(rawValue: narrowName),
              let wideVariant = MPPPrefillInt4QMM.TileVariant(rawValue: wideName) else {
            throw MPPPrefillInt4QMMError.invalidArguments("unknown tile variant \(narrowName) / \(wideName)")
        }
        let narrow = MPPPrefillInt4QMM(context: context, weightBits: bits, variant: narrowVariant, weightLoads: .byte)
        let wide = MPPPrefillInt4QMM(context: context, weightBits: bits, variant: wideVariant, weightLoads: .byte)
        guard narrow.isAvailable, wide.isAvailable else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(reason: "MPP prefill pipelines unavailable on this device")
        }
        let groups = k / Quantization.groupSize
        var state: UInt32 = 0x9E37_79B9
        func next() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24)
        }
        var packed = [UInt8](repeating: 0, count: n * k * bits / 8)
        for index in packed.indices { packed[index] = UInt8(truncatingIfNeeded: index &* 37 &+ 0x29) }
        var scales = [UInt16](repeating: 0, count: n * groups)
        var biases = [UInt16](repeating: 0, count: n * groups)
        for index in scales.indices {
            scales[index] = Quantization.bf16Bits(0.0005 + next() * 0.003)
            biases[index] = Quantization.bf16Bits(-0.02 + next() * 0.04)
        }
        var x = [Float16](repeating: 0, count: m * k)
        for index in x.indices { x[index] = Float16(next() - 0.5) }
        guard let weights = device.makeBuffer(bytes: packed, length: packed.count, options: .storageModeShared),
              let scaleBuffer = device.makeBuffer(bytes: scales, length: scales.count * 2, options: .storageModeShared),
              let biasBuffer = device.makeBuffer(bytes: biases, length: biases.count * 2, options: .storageModeShared),
              let input = device.makeBuffer(bytes: x, length: x.count * 2, options: .storageModeShared),
              let narrowOut = device.makeBuffer(length: m * n * 2, options: .storageModeShared),
              let wideOut = device.makeBuffer(length: m * n * 2, options: .storageModeShared),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MPPPrefillInt4QMMError.invalidArguments("buffer allocation failed")
        }
        try narrow.encode(commandBuffer: commandBuffer, weights: weights, scales: scaleBuffer, biases: biasBuffer,
                          x: input, y: narrowOut, m: m, n: n, k: k, required: true)
        try wide.encode(commandBuffer: commandBuffer, weights: weights, scales: scaleBuffer, biases: biasBuffer,
                        x: input, y: wideOut, m: m, n: n, k: k, required: true)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let a = narrowOut.contents().assumingMemoryBound(to: Float16.self)
        let b = wideOut.contents().assumingMemoryBound(to: Float16.self)
        var mismatches = 0
        var maxAbs: Float = 0
        var maxRel: Float = 0
        var first = -1
        for index in 0..<(m * n) where a[index] != b[index] {
            mismatches += 1
            if first < 0 { first = index }
            let diff = abs(Float(a[index]) - Float(b[index]))
            maxAbs = max(maxAbs, diff)
            let magnitude = max(abs(Float(a[index])), 1e-6)
            maxRel = max(maxRel, diff / magnitude)
        }
        return TileComparison(m: m, n: n, k: k, bits: bits, pair: "\(narrowName)_vs_\(wideName)",
                              mismatches: mismatches, maxAbsDiff: maxAbs, maxRelDiff: maxRel,
                              firstMismatch: first)
    }
}
