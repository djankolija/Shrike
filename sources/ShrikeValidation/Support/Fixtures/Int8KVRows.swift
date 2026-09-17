import Metal
import Shrike

public enum Int8KVRowsError: Error {
    case allocationFailed(String)
}

/// Quantised through the production quantizer so the rows carry the production layout.
public struct Int8KVRows {
    public let q: [Float16]
    public let k: [Float16]
    public let v: [Float16]
    public let qBuf: MTLBuffer
    public let keyView: KVView
    public let valueView: KVView
    public let cache: KVCacheManager

    public static func make(context: MetalContext, config: ArchConfig, seqLen: Int,
                            seed: UInt64, layer: Int = 3) throws -> Int8KVRows {
        let rowElements = config.numFullKVHeads * config.fullHeadDim
        let qCount = config.numHeads * config.fullHeadDim
        var rng = SeedTree(seed).key("int8-rows-\(seqLen)")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<(seqLen * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<(seqLen * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let cache = try KVCacheManager(device: context.device, config: config,
                                       maxContext: seqLen, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let kBuf = Fp16Buffer.make(context.device, halves: k),
              let vBuf = Fp16Buffer.make(context.device, halves: v),
              let cb = context.queue.makeCommandBuffer() else {
            throw Int8KVRowsError.allocationFailed("int8 KV rows")
        }
        let keyView = cache.keyView(layer: layer, validTokenCount: seqLen)
        let valueView = cache.valueView(layer: layer, validTokenCount: seqLen)
        try quantizer.encode(commandBuffer: cb, source: kBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: keyView, tokenCount: seqLen,
                             elementCount: rowElements)
        try quantizer.encode(commandBuffer: cb, source: vBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: valueView, tokenCount: seqLen,
                             elementCount: rowElements)
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed else {
            throw Int8KVRowsError.allocationFailed("int8 KV rows: quantize failed")
        }
        return Int8KVRows(q: q, k: k, v: v, qBuf: qBuf, keyView: keyView,
                          valueView: valueView, cache: cache)
    }
}
