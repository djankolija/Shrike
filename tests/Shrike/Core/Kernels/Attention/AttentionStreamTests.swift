import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct AttentionStreamTests {
    static let config = ArchConfig.qwen36_35B_A3B
    static var qCount: Int { config.numHeads * config.fullHeadDim }
    static var rowElements: Int { config.numFullKVHeads * config.fullHeadDim }

    static func makeRows(context: MetalContext, seqLen: Int, seed: UInt64) throws -> Int8KVRows {
        try Int8KVRows.make(context: context, config: config, seqLen: seqLen, seed: seed)
    }

    static func run(_ attention: Attention, context: MetalContext,
                    rows: Int8KVRows, seqLen: Int) throws -> [Float] {
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

    static func run(_ variant: Attention.PartialLoopVariant, context: MetalContext,
                    rows: Int8KVRows, seqLen: Int) throws -> [Float] {
        let attention = try Attention(context: context, partialLoopVariant: variant)
        let out = try run(attention, context: context, rows: rows, seqLen: seqLen)
        #expect(attention.lastSplitChoice == (variant == .stream ? .stream : .shared),
                "the \(variant) run took \(String(describing: attention.lastSplitChoice))")
        return out
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

    /// The streaming pipeline is what the served shape on int8 rows runs, and the
    /// fallback on any other shape or on fp16 rows is the shared kernel, both read
    /// from the wrapper's recorded choice rather than inferred from pipeline identity.
    @Test func streamPipelineEngagesOnlyForTheServedShape() throws {
        let context = try MetalContext()
        let attention = try Attention(context: context, partialLoopVariant: .stream)
        let rows = try Self.makeRows(context: context, seqLen: 8, seed: 0x57E6)
        _ = try Self.run(attention, context: context, rows: rows, seqLen: 8)
        #expect(attention.lastSplitChoice == .stream)
        let first = attention.partialPipeline(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                              numChunks: 64, useGQAPartial: false,
                                              kvFormat: rows.keyView)
        let second = attention.partialPipeline(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                               numChunks: 64, useGQAPartial: false,
                                               kvFormat: rows.keyView)
        #expect(first === second)
        #expect(attention.sharedPartialChoice(headDim: 256, numQHeads: 16, numKVHeads: 2,
                                              useGQAPartial: false, ringCapacity: 0,
                                              kvFormat: nil) == .shared)
        #expect(attention.sharedPartialChoice(headDim: 128, numQHeads: 16, numKVHeads: 8,
                                              useGQAPartial: false, ringCapacity: 0,
                                              kvFormat: rows.keyView) == .shared)
        #expect(attention.sharedPartialChoice(headDim: 512, numQHeads: 16, numKVHeads: 2,
                                              useGQAPartial: false, ringCapacity: 0,
                                              kvFormat: rows.keyView) == .generic)

        let fp16 = try Attention(context: context, partialLoopVariant: .stream)
        guard let kBuf = Fp16Buffer.make(context.device, halves: rows.k),
              let vBuf = Fp16Buffer.make(context.device, halves: rows.v),
              let out = Fp16Buffer.make(context.device, count: Self.qCount),
              let cb = context.queue.makeCommandBuffer() else {
            throw MetalError.bufferAllocationFailed("fp16 rows")
        }
        try fp16.encodeFull(commandBuffer: cb, q: rows.qBuf, k: kBuf, v: vBuf, out: out,
                            headDim: 256, numQHeads: 16, numKVHeads: 2, seqLen: 8)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(fp16.lastSplitChoice == .shared)
    }

    @Test func streamServesExactlyTheServedShape() {
        #expect(Attention.streamServes(headDim: 256, numQHeads: 16, numKVHeads: 2, precision: .int8))
        #expect(!Attention.streamServes(headDim: 128, numQHeads: 16, numKVHeads: 2, precision: .int8))
        #expect(!Attention.streamServes(headDim: 256, numQHeads: 16, numKVHeads: 4, precision: .int8))
        #expect(!Attention.streamServes(headDim: 256, numQHeads: 16, numKVHeads: 2, precision: .fp16))
        #expect(!Attention.streamServes(headDim: 256, numQHeads: 16, numKVHeads: 2, precision: .int4))
        #expect(!Attention.streamServes(headDim: 256, numQHeads: 16, numKVHeads: 0, precision: .int8))
        #expect(Attention.streamServesShape(headDim: 256, numQHeads: 16, numKVHeads: 2))
        #expect(!Attention.streamServesShape(headDim: 256, numQHeads: 8, numKVHeads: 2))
    }

    /// The wrapper's `streamCount` and the kernel's `kAttnStreamCount` and
    /// `kAttnStreamHeads` are tied by hand; this reads the shipped source.
    @Test func streamConstantsMatchTheKernel() throws {
        let bundleDir = Bundle(for: Attention.self).bundleURL.deletingLastPathComponent()
        let source = try String(contentsOf: bundleDir
            .appendingPathComponent("Shrike_Shrike.bundle/Metal/Attention/attention.metal"),
                                encoding: .utf8)
        func constant(_ name: String) throws -> Int {
            let pattern = #"constant constexpr uint "# + name + #" = (\d+);"#
            let match = try #require(source.firstMatch(of: try Regex(pattern)))
            return try #require(Int(match.output[1].substring ?? ""))
        }
        let streams = try constant("kAttnStreamCount")
        let heads = try constant("kAttnStreamHeads")
        #expect(streams == Attention.streamCount)
        #expect((8 / heads) * streams == 8, "head sets times streams must fill the threadgroup")
    }

    // MARK: - The exact value

    enum Fp64Reference {
        static func dequantizedRows(view: KVView, seqLen: Int, rowElements: Int) -> [[Double]] {
            let base = view.buffer.contents().advanced(by: view.offset)
                .assumingMemoryBound(to: UInt8.self)
            let groups = (rowElements + view.groupSize - 1) / view.groupSize
            var rows: [[Double]] = []
            rows.reserveCapacity(seqLen)
            for pos in 0..<seqLen {
                let row = base + pos * view.stride
                let scales = UnsafeRawPointer(row + view.valueBytes)
                    .assumingMemoryBound(to: Float16.self)
                var values = [Double](repeating: 0, count: rowElements)
                for i in 0..<rowElements {
                    let g = i / view.groupSize
                    values[i] = Double(row[i]) * Double(Float(scales[g]))
                        + Double(Float(scales[groups + g]))
                }
                rows.append(values)
            }
            return rows
        }

        static func attention(q: [Float16], k: [[Double]], v: [[Double]],
                              headDim: Int, numQHeads: Int, numKVHeads: Int,
                              scale: Double) -> [Double] {
            let qPerKV = numQHeads / numKVHeads
            var out = [Double](repeating: 0, count: numQHeads * headDim)
            for h in 0..<numQHeads {
                let base = (h / qPerKV) * headDim
                let qh = (0..<headDim).map { Double(Float(q[h * headDim + $0])) }
                var scores = [Double](repeating: 0, count: k.count)
                var m = -Double.infinity
                for p in 0..<k.count {
                    var s = 0.0
                    for i in 0..<headDim { s += qh[i] * k[p][base + i] }
                    scores[p] = s * scale
                    m = max(m, scores[p])
                }
                var denom = 0.0
                for p in 0..<k.count {
                    scores[p] = exp(scores[p] - m)
                    denom += scores[p]
                }
                for i in 0..<headDim {
                    var acc = 0.0
                    for p in 0..<k.count { acc += scores[p] * v[p][base + i] }
                    out[h * headDim + i] = acc / denom
                }
            }
            return out
        }

        /// The combine in double, so the kernels' fp32 partials are the only rounding measured.
        static func combine(_ attention: Attention, numQHeads: Int, numChunks: Int,
                            headDim: Int) -> [Double] {
            let m = attention.mPartial.contents().assumingMemoryBound(to: Float.self)
            let d = attention.dPartial.contents().assumingMemoryBound(to: Float.self)
            let o = attention.oPartial.contents().assumingMemoryBound(to: Float.self)
            var out = [Double](repeating: 0, count: numQHeads * headDim)
            for h in 0..<numQHeads {
                let base = h * numChunks
                var mGlob = -Double.infinity
                for c in 0..<numChunks { mGlob = max(mGlob, Double(m[base + c])) }
                var weights = [Double](repeating: 0, count: numChunks)
                var denom = 0.0
                for c in 0..<numChunks where m[base + c] > -Float.infinity {
                    weights[c] = exp(Double(m[base + c]) - mGlob)
                    denom += Double(d[base + c]) * weights[c]
                }
                for i in 0..<headDim {
                    var acc = 0.0
                    for c in 0..<numChunks where weights[c] > 0 {
                        acc += Double(o[(base + c) * headDim + i]) * weights[c]
                    }
                    out[h * headDim + i] = acc / denom
                }
            }
            return out
        }

        static func errors(_ actual: [Double], _ exact: [Double]) -> (maxAbs: Double, relative: Double) {
            var maxAbs = 0.0, num = 0.0, den = 0.0
            for (a, e) in zip(actual, exact) {
                maxAbs = max(maxAbs, abs(a - e))
                num += (a - e) * (a - e)
                den += e * e
            }
            return (maxAbs, (num / den).squareRoot())
        }
    }

    /// Both kernels' own fp32 error against the exact value of the same
    /// dequantized rows, read from their fp32 partials so neither the int8
    /// quantization nor the fp16 output rounding is in the number.
    @Test(arguments: [500, 1100])
    func bothKernelsAgainstTheExactValue(_ seqLen: Int) throws {
        let context = try MetalContext()
        let rows = try Self.makeRows(context: context, seqLen: seqLen, seed: 0x57E7)
        let k = Fp64Reference.dequantizedRows(view: rows.keyView, seqLen: seqLen,
                                              rowElements: Self.rowElements)
        let v = Fp64Reference.dequantizedRows(view: rows.valueView, seqLen: seqLen,
                                              rowElements: Self.rowElements)
        let exact = Fp64Reference.attention(q: rows.q, k: k, v: v,
                                            headDim: Self.config.fullHeadDim,
                                            numQHeads: Self.config.numHeads,
                                            numKVHeads: Self.config.numFullKVHeads,
                                            scale: 1.0 / 16.0)
        for variant in [Attention.PartialLoopVariant.kvShared, .stream] {
            let attention = try Attention(context: context, partialLoopVariant: variant)
            _ = try Self.run(attention, context: context, rows: rows, seqLen: seqLen)
            let combined = Fp64Reference.combine(attention, numQHeads: Self.config.numHeads,
                                                 numChunks: Attention.maxChunks,
                                                 headDim: Self.config.fullHeadDim)
            let (maxAbs, relative) = Fp64Reference.errors(combined, exact)
            print("\(variant) against the exact value at \(seqLen): max |d| \(maxAbs), rel \(relative)")
            #expect(relative < 1e-5, "\(variant) at \(seqLen) rel \(relative)")
        }
    }
}
