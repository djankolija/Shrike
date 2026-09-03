import Foundation
import Metal

/// Lives in the module because `PrefillRouter` is internal.
public enum PrefillRouterBenchmark {
    public struct Result: Sendable {
        public let queryCount: Int
        public let d: Int
        public let numExperts: Int
        public let topK: Int
        public let weightBits: Int
        public let kind: String
        public let tokenBlock: Int
        public let threadgroupWidth: Int
        public let threadgroups: Int
        public let millisPerLaunch: Double
        public var weightBytesPerThreadgroup: Int { numExperts * d * weightBits / 8 }
        public var gflop: Double {
            2.0 * Double(queryCount) * Double(d) * Double(numExperts) / 1.0e9
        }
        public var tflops: Double { gflop / millisPerLaunch }
    }

    public static func environmentTokenBlock() -> Int {
        PrefillRouter.environmentTokenBlock()
    }

    /// Drives `PrefillRouter.encodeBlock` itself, so the pipeline, its function
    /// constant and the threadgroup width are production's and cannot drift.
    public static func run(context: MetalContext,
                           iterations: Int,
                           queryCount: Int = 4096,
                           d: Int = 2048,
                           numExperts: Int = 256,
                           topK: Int = 8,
                           weightBits: Int = 8,
                           kind kindName: String = "tiled",
                           tokenBlock: Int = 12) throws -> Result {
        let kind = PrefillRouter.Kind(rawValue: kindName) ?? .tiled
        let router = try PrefillRouter(context: context, weightBits: weightBits,
                                       kind: kind, tokenBlock: tokenBlock)
        let fixture = try Fixture(device: context.device, queryCount: queryCount, d: d,
                                  numExperts: numExperts, topK: topK, weightBits: weightBits)
        func encode(_ commandBuffer: MTLCommandBuffer) throws {
            try router.encodeBlock(commandBuffer: commandBuffer,
                                   weights: fixture.weights,
                                   scales: fixture.scales,
                                   biases: fixture.biases,
                                   hidden: fixture.hidden,
                                   effectiveScale: fixture.effectiveScale,
                                   perExpertScale: fixture.perExpertScale,
                                   logitBias: fixture.logitBias,
                                   outIndices: fixture.outIndices,
                                   outWeights: fixture.outWeights,
                                   queryCount: UInt32(queryCount),
                                   numExperts: UInt32(numExperts),
                                   d: UInt32(d),
                                   topK: UInt32(topK),
                                   hiddenStrideElements: UInt32(d))
        }
        func once() throws -> Double {
            guard let commandBuffer = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try encode(commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
            return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        }
        for _ in 0..<2 { _ = try once() }
        let runs = max(1, iterations)
        var total = 0.0
        for _ in 0..<runs { total += try once() }
        return Result(queryCount: queryCount, d: d, numExperts: numExperts, topK: topK,
                      weightBits: weightBits,
                      kind: kind.rawValue,
                      tokenBlock: kind == .tiled ? tokenBlock : 1,
                      threadgroupWidth: router.threadgroupWidth(numExperts: numExperts),
                      threadgroups: router.threadgroups(queryCount: queryCount),
                      millisPerLaunch: total / Double(runs) * 1000)
    }

    /// `effectiveScale` is indexed over `D`, not over the experts: the body
    /// walks it alongside the hidden row (`prefill.metal:404-405`).
    private struct Fixture {
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let hidden: MTLBuffer
        let effectiveScale: MTLBuffer
        let perExpertScale: MTLBuffer
        let logitBias: MTLBuffer
        let outIndices: MTLBuffer
        let outWeights: MTLBuffer

        init(device: MTLDevice, queryCount: Int, d: Int,
             numExperts: Int, topK: Int, weightBits: Int) throws {
            var state: UInt32 = 0x9E37_79B9
            func next() -> Float {
                state = state &* 1_664_525 &+ 1_013_904_223
                return Float(state >> 8) / Float(1 << 24)
            }
            func buffer<T>(_ values: [T], _ label: String) throws -> MTLBuffer {
                let byteCount = values.count * MemoryLayout<T>.stride
                guard let buffer = device.makeBuffer(length: byteCount,
                                                     options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed(label)
                }
                values.withUnsafeBytes { bytes in
                    buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
                }
                return buffer
            }
            func bf16(_ count: Int, _ lo: Float, _ hi: Float, _ label: String) throws -> MTLBuffer {
                try buffer((0..<count).map { _ in Quantization.bf16Bits(lo + (hi - lo) * next()) },
                           label)
            }
            let groups = d / Quantization.groupSize
            weights = try buffer((0..<(numExperts * d * weightBits / 8)).map { index in
                UInt8(truncatingIfNeeded: index &* 37 &+ 0x29)
            }, "router.weights")
            scales = try bf16(numExperts * groups, 0.0005, 0.0035, "router.scales")
            biases = try bf16(numExperts * groups, -0.02, 0.02, "router.biases")
            hidden = try buffer((0..<(queryCount * d)).map { _ in Float16(next() - 0.5) },
                                "router.hidden")
            effectiveScale = try bf16(d, 0.9, 1.1, "router.effectiveScale")
            perExpertScale = try bf16(numExperts, 0.9, 1.1, "router.perExpertScale")
            logitBias = try bf16(numExperts, -0.05, 0.05, "router.logitBias")
            outIndices = try buffer([UInt32](repeating: 0, count: queryCount * topK),
                                    "router.outIndices")
            outWeights = try buffer([Float16](repeating: 0, count: queryCount * topK),
                                    "router.outWeights")
        }
    }
}
