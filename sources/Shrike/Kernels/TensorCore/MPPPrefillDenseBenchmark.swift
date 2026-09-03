import Foundation
import Metal

/// Lives in the module because `MPPPrefillInt4QMM` is internal.
public enum MPPPrefillDenseBenchmark {
    public struct Result: Sendable {
        public let m: Int
        public let k: Int
        public let n: Int
        public let variant: String
        public let weightLoads: String
        public let launches: Int
        public let millisPerLaunch: Double
        public var gflop: Double { 2.0 * Double(m) * Double(n) * Double(k) / 1.0e9 }
        public var tflops: Double { gflop / millisPerLaunch }
    }

    /// The variant and load body arrive by name so one process can sweep every
    /// arm against one same-run ceiling, which the process-wide
    /// `SHRIKE_MPP_*` statics cannot do.
    public static func run(context: MetalContext,
                           iterations: Int,
                           m: Int,
                           k: Int,
                           n: Int,
                           variant variantName: String,
                           weightLoads loadsName: String) throws -> Result {
        guard let variant = MPPPrefillInt4QMM.TileVariant(rawValue: variantName) else {
            throw MPPPrefillInt4QMMError.invalidArguments("unknown tile variant \(variantName)")
        }
        guard let loads = MPPPrefillInt4QMM.WeightLoads(rawValue: loadsName) else {
            throw MPPPrefillInt4QMMError.invalidArguments("unknown weight loads \(loadsName)")
        }
        let bits = 4
        let mpp = MPPPrefillInt4QMM(context: context, weightBits: bits,
                                    variant: variant, weightLoads: loads)
        guard mpp.isAvailable else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(
                reason: "MPP prefill pipelines unavailable on this device")
        }
        let fixture = try Fixture(device: context.device, m: m, k: k, n: n, bits: bits)
        func encode(_ commandBuffer: MTLCommandBuffer) throws {
            try mpp.encode(commandBuffer: commandBuffer,
                           weights: fixture.weights, scales: fixture.scales,
                           biases: fixture.biases, x: fixture.x, y: fixture.y,
                           m: m, n: n, k: k, required: true)
        }
        for _ in 0..<2 {
            guard let warm = context.queue.makeCommandBuffer() else {
                throw MetalError.commandEncoderFailed
            }
            try encode(warm)
            warm.commit()
            warm.waitUntilCompleted()
            if let error = warm.error { throw error }
        }
        let flopsPerLaunch = 2.0 * Double(m) * Double(n) * Double(k)
        let launches = max(10, min(max(1, iterations), Int(3.0e12 / flopsPerLaunch)))
        guard let timed = context.queue.makeCommandBuffer() else {
            throw MetalError.commandEncoderFailed
        }
        for _ in 0..<launches { try encode(timed) }
        timed.commit()
        timed.waitUntilCompleted()
        if let error = timed.error { throw error }
        let millis = (timed.gpuEndTime - timed.gpuStartTime) * 1000 / Double(launches)
        return Result(m: m, k: k, n: n, variant: variant.rawValue,
                      weightLoads: loads.rawValue, launches: launches,
                      millisPerLaunch: millis)
    }

    /// Activations carry full fp16 mantissas: values whose partial sums are
    /// exact in fp32 do not exercise the real K reduction.
    private struct Fixture {
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let x: MTLBuffer
        let y: MTLBuffer

        init(device: MTLDevice, m: Int, k: Int, n: Int, bits: Int) throws {
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
            var scaleBits = [UInt16](repeating: 0, count: n * groups)
            var biasBits = [UInt16](repeating: 0, count: n * groups)
            for index in scaleBits.indices {
                scaleBits[index] = Quantization.bf16Bits(0.0005 + next() * 0.003)
                biasBits[index] = Quantization.bf16Bits(-0.02 + next() * 0.04)
            }
            var activations = [Float16](repeating: 0, count: m * k)
            for index in activations.indices { activations[index] = Float16(next() - 0.5) }
            guard let weights = device.makeBuffer(bytes: packed, length: packed.count,
                                                  options: .storageModeShared),
                  let scales = device.makeBuffer(bytes: scaleBits, length: scaleBits.count * 2,
                                                 options: .storageModeShared),
                  let biases = device.makeBuffer(bytes: biasBits, length: biasBits.count * 2,
                                                 options: .storageModeShared),
                  let x = device.makeBuffer(bytes: activations, length: activations.count * 2,
                                            options: .storageModeShared),
                  let y = device.makeBuffer(length: m * n * 2, options: .storageModeShared) else {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "buffer allocation failed at m=\(m) k=\(k) n=\(n)")
            }
            self.weights = weights
            self.scales = scales
            self.biases = biases
            self.x = x
            self.y = y
        }
    }
}
