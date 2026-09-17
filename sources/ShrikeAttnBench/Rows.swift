import Foundation
import Metal
import Shrike
import ShrikeValidationSupport

/// Quantised through the production quantizer so the rows carry the production layout.
final class SyntheticRows {
    let config: ArchConfig
    let maxSeq: Int
    let headDim: Int
    let numQHeads: Int
    let numKVHeads: Int
    let scale: Float
    let qBuf: MTLBuffer
    let keyView: KVView
    let valueView: KVView
    let outBuf: MTLBuffer
    private let cache: KVCacheManager

    static let cacheLayer = 3

    init(context: MetalContext, maxSeq: Int, seed: UInt64) throws {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let rowElements = numKVHeads * headDim
        let qCount = numQHeads * headDim

        var rng = SeedTree(seed).key("rows")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let kBuf = Fp16Buffer.make(context.device, halves: k),
              let vBuf = Fp16Buffer.make(context.device, halves: v),
              let outBuf = Fp16Buffer.make(context.device, count: qCount) else {
            throw BenchError.allocation("fp16 rows")
        }
        let cache = try KVCacheManager(device: context.device, config: config,
                                       maxContext: maxSeq, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        let keyView = cache.keyView(layer: Self.cacheLayer, validTokenCount: maxSeq)
        let valueView = cache.valueView(layer: Self.cacheLayer, validTokenCount: maxSeq)
        guard let cb = context.queue.makeCommandBuffer() else { throw BenchError.commandBuffer }
        try quantizer.encode(commandBuffer: cb, source: kBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: keyView, tokenCount: maxSeq,
                             elementCount: rowElements)
        try quantizer.encode(commandBuffer: cb, source: vBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: valueView, tokenCount: maxSeq,
                             elementCount: rowElements)
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw BenchError.gpu("quantize: \(cb.error.map { "\($0)" } ?? "status \(cb.status.rawValue)")")
        }

        self.config = config
        self.maxSeq = maxSeq
        self.headDim = headDim
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.scale = 1 / Float(headDim).squareRoot()
        self.qBuf = qBuf
        self.keyView = keyView
        self.valueView = valueView
        self.outBuf = outBuf
        self.cache = cache
    }

    var bytesPerPosition: Int { 2 * keyView.stride }
}
