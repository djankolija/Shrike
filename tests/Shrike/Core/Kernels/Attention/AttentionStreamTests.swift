import Testing
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct AttentionStreamTests {
    struct Rows {
        let q: [Float16]
        let k: [Float16]
        let v: [Float16]
        let qBuf: MTLBuffer
        let keyView: KVView
        let valueView: KVView
        let cache: KVCacheManager
    }

    static let config = ArchConfig.qwen36_35B_A3B
    static var qCount: Int { config.numHeads * config.fullHeadDim }

    static func makeRows(context: MetalContext, seqLen: Int, seed: UInt64) throws -> Rows {
        let rowElements = config.numFullKVHeads * config.fullHeadDim
        var rng = SeedTree(seed).key("stream-\(seqLen)")
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
            throw MetalError.bufferAllocationFailed("stream rows")
        }
        let keyView = cache.keyView(layer: 3, validTokenCount: seqLen)
        let valueView = cache.valueView(layer: 3, validTokenCount: seqLen)
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
        return Rows(q: q, k: k, v: v, qBuf: qBuf, keyView: keyView, valueView: valueView,
                    cache: cache)
    }

    static func run(_ variant: Attention.PartialLoopVariant, context: MetalContext,
                    rows: Rows, seqLen: Int) throws -> [Float] {
        let attention = try Attention(context: context, partialLoopVariant: variant)
        guard let out = Fp16Buffer.make(context.device, count: qCount),
              let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.bufferAllocationFailed("stream output")
        }
        try attention.encodeFull(commandBuffer: cb,
                                 q: rows.qBuf,
                                 k: rows.keyView.buffer, kOffset: rows.keyView.offset,
                                 v: rows.valueView.buffer, vOffset: rows.valueView.offset,
                                 out: out,
                                 headDim: UInt32(config.fullHeadDim),
                                 numQHeads: UInt32(config.numHeads),
                                 numKVHeads: UInt32(config.numFullKVHeads),
                                 seqLen: UInt32(seqLen),
                                 kvFormat: rows.keyView)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        return Fp16Buffer.read(out, count: qCount)
    }

    @Test(arguments: [3, 17, 96, 500, 1100])
    func streamTracksTheReferenceOnInt8Rows(_ seqLen: Int) throws {
        let context = try MetalContext()
        let rows = try Self.makeRows(context: context, seqLen: seqLen, seed: 0x57E4)
        let actual = try Self.run(.stream, context: context, rows: rows, seqLen: seqLen)
        let reference = AttentionRef.apply(
            q: rows.q.map(Float.init), k: rows.k.map(Float.init), v: rows.v.map(Float.init),
            headDim: Self.config.fullHeadDim, numQHeads: Self.config.numHeads,
            numKVHeads: Self.config.numFullKVHeads, seqLen: seqLen)
        let relativeError = RelError.compute(actual: actual, reference: reference)
        #expect(relativeError < 0.02, "stream seq \(seqLen) rel=\(relativeError)")
    }

    @Test(arguments: [17, 500, 1100])
    func streamAgainstTheSharedKernel(_ seqLen: Int) throws {
        let context = try MetalContext()
        let rows = try Self.makeRows(context: context, seqLen: seqLen, seed: 0x57E5)
        let shared = try Self.run(.kvShared, context: context, rows: rows, seqLen: seqLen)
        let stream = try Self.run(.stream, context: context, rows: rows, seqLen: seqLen)
        let maxAbs = zip(shared, stream).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
        let relativeError = RelError.compute(actual: stream, reference: shared)
        print("stream against shared, seq \(seqLen): max |d| \(maxAbs), rel \(relativeError)")
        #expect(maxAbs < 1e-2, "stream vs shared seq \(seqLen) max |d| \(maxAbs)")
    }

    @Test func streamPipelineEngagesOnlyForTheServedShape() throws {
        let context = try MetalContext()
        let attention = try Attention(context: context, partialLoopVariant: .stream)
        let rows = try Self.makeRows(context: context, seqLen: 8, seed: 0x57E6)
        let first = attention.partialPipeline(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                              numChunks: 64, useGQAPartial: false,
                                              kvFormat: rows.keyView)
        let second = attention.partialPipeline(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                               numChunks: 64, useGQAPartial: false,
                                               kvFormat: rows.keyView)
        #expect(first === second)
        #expect(first !== attention.psoPartialShared)
        let fp16Rows = attention.partialPipeline(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                                 numChunks: 64, useGQAPartial: false,
                                                 kvFormat: nil)
        #expect(fp16Rows !== first)
        let otherShape = attention.partialPipeline(headDim: 128, numQHeads: 16, numKVHeads: 8,
                                                   numChunks: 64, useGQAPartial: false,
                                                   kvFormat: rows.keyView)
        #expect(otherShape !== first)
    }
}
