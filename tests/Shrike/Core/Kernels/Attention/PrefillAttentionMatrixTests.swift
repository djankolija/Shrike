import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct PrefillAttentionMatrixTests {
    private typealias Fixture = PrefillAttentionRef.Inputs
    private static let headDim = 256
    private static let qHeads = 16
    private static let kvHeads = 2
    private static let scale: Float = 0.0625
    private static let tolerance: Float = 2e-2

    @Test(arguments: [
        (label: "single-tile", start: 0, chunk: 64),
        (label: "ragged-rows-three-key-tiles", start: 0, chunk: 130),
        (label: "history-ragged-tail", start: 600, chunk: 40),
    ]) func matrixMatchesReferenceOnFP16Cache(c: (label: String, start: Int, chunk: Int)) throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        #expect(attention.matrixPathAvailable, "\(attention.matrixUnavailableReason)")
        let fixture = Self.makeFixture(start: c.start, chunk: c.chunk, seed: 0xB120)
        let actual = try Self.runFP16(fixture, attention: attention, context: ctx, path: .causalMatrix)
        let reference = PrefillAttentionRef.apply(fixture)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= Self.tolerance, "\(c.label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= Self.tolerance, "\(c.label) rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test(arguments: [
        (label: "single-chunk", start: 0, chunk: 96, bits: 8),
        (label: "second-chunk-history", start: 1024, chunk: 64, bits: 8),
        (label: "deep-history-ragged", start: 4000, chunk: 130, bits: 8),
        (label: "deep-history-short-block", start: 2048, chunk: 37, bits: 8),
        (label: "int4-history", start: 1024, chunk: 64, bits: 4),
    ]) func matrixMatchesTiledOnQuantizedCache(c: (label: String, start: Int, chunk: Int, bits: Int)) throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        #expect(attention.matrixPathAvailable, "\(attention.matrixUnavailableReason)")
        let fixture = Self.makeFixture(start: c.start, chunk: c.chunk, seed: 0xB121)
        let config = ArchConfig.qwen36_35B_A3B
        let cache = try KVCacheManager(device: ctx.device, config: config,
                                       maxContext: fixture.kvValid,
                                       precision: c.bits == 4 ? .int4 : .int8)
        let quantizer = try KVCacheQuantizer(context: ctx)
        let kvRow = Self.kvHeads * Self.headDim
        guard let kBuffer = Fp16Buffer.make(ctx.device, values: fixture.k),
              let vBuffer = Fp16Buffer.make(ctx.device, values: fixture.v),
              let commandBuffer = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        let keyView = cache.keyView(layer: 3, validTokenCount: fixture.kvValid)
        let valueView = cache.valueView(layer: 3, validTokenCount: fixture.kvValid)
        try quantizer.encode(commandBuffer: commandBuffer, source: kBuffer,
                             sourceTokenStrideElements: fixture.kvStride,
                             destination: keyView, tokenCount: fixture.kvValid,
                             elementCount: kvRow)
        try quantizer.encode(commandBuffer: commandBuffer, source: vBuffer,
                             sourceTokenStrideElements: fixture.kvStride,
                             destination: valueView, tokenCount: fixture.kvValid,
                             elementCount: kvRow)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var params = Self.params(fixture)
        params.kvBits = UInt32(c.bits)
        params.kvTokenStrideBytes = UInt32(keyView.stride)
        params.kvValueBytes = UInt32(keyView.valueBytes)
        params.kvGroupSize = UInt32(keyView.groupSize)
        params.kvTokenStrideElements = UInt32(kvRow)
        let tiled = try Self.run(fixture, attention: attention, context: ctx,
                                 k: keyView.buffer, v: valueView.buffer,
                                 params: params, path: .causalTiled)
        let matrix = try Self.run(fixture, attention: attention, context: ctx,
                                  k: keyView.buffer, v: valueView.buffer,
                                  params: params, path: .causalMatrix)
        let maxAbs = RelError.maxAbsDiff(matrix, tiled)
        let rel = RelError.compute(actual: matrix, reference: tiled)
        #expect(maxAbs <= Self.tolerance, "\(c.label) maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= Self.tolerance, "\(c.label) rel=\(rel) maxAbs=\(maxAbs)")
    }

    @Test func rejectedShapeRunsTheTiledKernelEndToEnd() throws {
        let ctx = try MetalContext()
        let attention = try PrefillAttention(context: ctx)
        var fixture = Self.makeFixture(start: 512, chunk: 64, seed: 0xB122)
        fixture.window = 256
        #expect(!PrefillAttention.matrixPathAccepts(Self.params(fixture), kvRingCapacity: 0, hasSinks: false))
        let viaMatrixRequest = try Self.runFP16(fixture, attention: attention, context: ctx, path: .causalMatrix)
        let tiled = try Self.runFP16(fixture, attention: attention, context: ctx, path: .causalTiled)
        #expect(viaMatrixRequest == tiled)
    }

    @Test func gateAcceptsOnlyTheMatrixShape() {
        let base = Self.params(Self.makeFixture(start: 512, chunk: 64, seed: 1))
        #expect(PrefillAttention.matrixPathAccepts(base, kvRingCapacity: 0, hasSinks: false))

        var window = base
        window.slidingWindow = 256
        #expect(!PrefillAttention.matrixPathAccepts(window, kvRingCapacity: 0, hasSinks: false))
        var fullWindow = base
        fullWindow.slidingWindow = base.kvValidCount
        #expect(PrefillAttention.matrixPathAccepts(fullWindow, kvRingCapacity: 0, hasSinks: false))

        var wide = base
        wide.headDim = 512
        #expect(!PrefillAttention.matrixPathAccepts(wide, kvRingCapacity: 0, hasSinks: false))

        var short = base
        short.queryCount = 8
        #expect(!PrefillAttention.matrixPathAccepts(short, kvRingCapacity: 0, hasSinks: false))

        #expect(!PrefillAttention.matrixPathAccepts(base, kvRingCapacity: 4096, hasSinks: false))
        #expect(!PrefillAttention.matrixPathAccepts(base, kvRingCapacity: 0, hasSinks: true))

        var tooLong = base
        tooLong.startPosition = 0
        tooLong.kvValidCount = PrefillAttention.matrixPathMaxContext + 1
        #expect(!PrefillAttention.matrixPathAccepts(tooLong, kvRingCapacity: 0, hasSinks: false))
    }

    private static func makeFixture(start: Int, chunk: Int, seed: UInt64) -> Fixture {
        let qStride = qHeads * headDim + 3
        let kvStride = kvHeads * headDim + 5
        let oStride = qHeads * headDim + 7
        let kvValid = start + chunk
        var rng = SeedTree(seed).key("prefill-attn-matrix-start\(start)-chunk\(chunk)")
        var q = [Float](repeating: 0, count: chunk * qStride)
        var k = [Float](repeating: 0, count: kvValid * kvStride)
        var v = [Float](repeating: 0, count: kvValid * kvStride)
        for t in 0..<chunk {
            for h in 0..<qHeads {
                for d in 0..<headDim {
                    q[t * qStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        for pos in 0..<kvValid {
            for h in 0..<kvHeads {
                for d in 0..<headDim {
                    k[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                    v[pos * kvStride + h * headDim + d] = rng.uniform(-0.35, 0.35)
                }
            }
        }
        return Fixture(q: q, k: k, v: v,
                       qStride: qStride, kvStride: kvStride, oStride: oStride,
                       headDim: headDim, qHeads: qHeads, kvHeads: kvHeads,
                       start: start, chunk: chunk, kvValid: kvValid,
                       window: 0, scale: scale)
    }

    private static func params(_ fixture: Fixture) -> PrefillAttentionParams {
        PrefillAttentionParams(
            startPosition: UInt32(fixture.start),
            queryCount: UInt32(fixture.chunk),
            headDim: UInt32(fixture.headDim),
            numQHeads: UInt32(fixture.qHeads),
            numKVHeads: UInt32(fixture.kvHeads),
            kvValidCount: UInt32(fixture.kvValid),
            slidingWindow: UInt32(fixture.window),
            kvTokenStrideElements: UInt32(fixture.kvStride),
            qTokenStrideElements: UInt32(fixture.qStride),
            oTokenStrideElements: UInt32(fixture.oStride),
            scale: fixture.scale)
    }

    private static let qPrefix = 17
    private static let kPrefix = 19
    private static let vPrefix = 23
    private static let oPrefix = 29

    private static func runFP16(_ fixture: Fixture,
                                attention: PrefillAttention,
                                context: MetalContext,
                                path: RuntimePrefillAttentionPath) throws -> [Float] {
        guard let kBuf = Fp16Buffer.make(context.device,
                                         values: [Float](repeating: 0, count: kPrefix) + fixture.k),
              let vBuf = Fp16Buffer.make(context.device,
                                         values: [Float](repeating: 0, count: vPrefix) + fixture.v) else {
            Issue.record("alloc failed")
            return []
        }
        let halfBytes = MemoryLayout<Float16>.size
        return try run(fixture, attention: attention, context: context,
                       k: kBuf, kOffset: kPrefix * halfBytes,
                       v: vBuf, vOffset: vPrefix * halfBytes,
                       params: params(fixture), path: path)
    }

    private static func run(_ fixture: Fixture,
                            attention: PrefillAttention,
                            context: MetalContext,
                            k: MTLBuffer, kOffset: Int = 0,
                            v: MTLBuffer, vOffset: Int = 0,
                            params: PrefillAttentionParams,
                            path: RuntimePrefillAttentionPath) throws -> [Float] {
        let outCount = fixture.chunk * fixture.oStride
        let halfBytes = MemoryLayout<Float16>.size
        guard let qBuf = Fp16Buffer.make(context.device,
                                         values: [Float](repeating: 0, count: qPrefix) + fixture.q),
              let outBuf = Fp16Buffer.make(context.device, count: oPrefix + outCount),
              let cb = context.queue.makeCommandBuffer() else {
            Issue.record("alloc failed")
            return []
        }
        try attention.encodeCausal(commandBuffer: cb,
                                   q: qBuf, qOffset: qPrefix * halfBytes,
                                   k: k, kOffset: kOffset,
                                   v: v, vOffset: vOffset,
                                   out: outBuf, outOffset: oPrefix * halfBytes,
                                   params: params, path: path)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let out = Fp16Buffer.read(outBuf, count: oPrefix + outCount)
        var compact = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        for t in 0..<fixture.chunk {
            for h in 0..<fixture.qHeads {
                for d in 0..<fixture.headDim {
                    compact[(t * fixture.qHeads + h) * fixture.headDim + d] =
                        out[oPrefix + t * fixture.oStride + h * fixture.headDim + d]
                }
            }
        }
        return compact
    }
}
