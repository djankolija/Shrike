import Foundation
import Metal

/// Times the matrix-path causal attention (KV dequant, the q-group pack and
/// the causal kernel) for one prefill chunk of the ornith shape over synthetic
/// fp16 buffers. Lives in the module because `PrefillAttention` is internal.
public enum PrefillAttentionBenchmark {
    public struct Result: Sendable {
        public let tile: String
        public let start: Int
        public let chunk: Int
        public let kvValid: Int
        public let millisPerChunk: Double
        /// Causal (query, key) pairs over the 16 query heads.
        public var pairs: Double {
            let perHead = Double(chunk) * Double(start) + Double(chunk) * Double(chunk + 1) / 2
            return perHead * Double(Shape.qHeads)
        }
        /// QKᵀ and PV, two flops per multiply-add each, over head dim 256.
        public var gflop: Double { pairs * Double(Shape.headDim) * 4 / 1.0e9 }
        public var tflops: Double { gflop / millisPerChunk }
    }

    enum Shape {
        static let headDim = 256
        static let qHeads = 16
        static let kvHeads = 2
        static let scale: Float = 0.0625
    }

    public static func run(context: MetalContext,
                           iterations: Int,
                           tile: String,
                           start: Int,
                           chunk: Int) throws -> Result {
        guard let matrixTile = PrefillAttention.MatrixTile(rawValue: tile) else {
            throw MPPPrefillInt4QMMError.invalidArguments("unknown attention tile \(tile)")
        }
        let attention = try PrefillAttention(context: context, matrixTile: matrixTile)
        guard attention.matrixPathAvailable else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(reason: attention.matrixUnavailableReason)
        }
        let kvValid = start + chunk
        let qStride = Shape.qHeads * Shape.headDim
        let kvStride = Shape.kvHeads * Shape.headDim
        let params = PrefillAttentionParams(
            startPosition: UInt32(start),
            queryCount: UInt32(chunk),
            headDim: UInt32(Shape.headDim),
            numQHeads: UInt32(Shape.qHeads),
            numKVHeads: UInt32(Shape.kvHeads),
            kvValidCount: UInt32(kvValid),
            slidingWindow: 0,
            kvTokenStrideElements: UInt32(kvStride),
            qTokenStrideElements: UInt32(qStride),
            oTokenStrideElements: UInt32(qStride),
            scale: Shape.scale)
        guard PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 0, hasSinks: false) else {
            throw MPPPrefillInt4QMMError.invalidArguments("the matrix path rejects start \(start) chunk \(chunk)")
        }
        let device = context.device
        var seed: UInt64 = 0x5DEE_CE66_D1CE_5EED &+ UInt64(start)
        func randomHalves(_ count: Int, _ label: String) throws -> MTLBuffer {
            var values = [Float16](repeating: 0, count: count)
            for index in 0..<count {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                values[index] = Float16(Float(Int64(seed >> 40) - (1 << 23)) / Float(1 << 24))
            }
            guard let buffer = device.makeBuffer(bytes: values,
                                                 length: count * MemoryLayout<Float16>.stride,
                                                 options: .storageModeShared) else {
                throw MetalError.bufferAllocationFailed(label)
            }
            buffer.label = label
            return buffer
        }
        let q = try randomHalves(chunk * qStride, "attn.bench.q")
        let k = try randomHalves(kvValid * kvStride, "attn.bench.k")
        let v = try randomHalves(kvValid * kvStride, "attn.bench.v")
        guard let out = device.makeBuffer(length: chunk * qStride * MemoryLayout<Float16>.stride,
                                          options: .storageModePrivate) else {
            throw MetalError.bufferAllocationFailed("attn.bench.out")
        }
        func encode(_ commandBuffer: MTLCommandBuffer) throws {
            try attention.encodeCausal(commandBuffer: commandBuffer, q: q, k: k, v: v, out: out,
                                       params: params, path: .causalMatrix)
        }
        guard let warm = context.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        try encode(warm)
        warm.commit()
        warm.waitUntilCompleted()
        if let error = warm.error { throw error }
        guard let timed = context.queue.makeCommandBuffer() else { throw MetalError.commandEncoderFailed }
        for _ in 0..<iterations {
            try encode(timed)
        }
        timed.commit()
        timed.waitUntilCompleted()
        if let error = timed.error { throw error }
        let millis = (timed.gpuEndTime - timed.gpuStartTime) * 1000 / Double(iterations)
        return Result(tile: tile, start: start, chunk: chunk, kvValid: kvValid, millisPerChunk: millis)
    }
}
