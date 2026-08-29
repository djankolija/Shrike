import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

/// gpt-oss runtime integration over the synthetic toy fixture: Model.load
/// through the gpt-oss schema branch (biased attention, sinks, biased router,
/// 12-slice experts, no shared expert), runner construction with arch YaRN,
/// decode and chunked prefill over the alternating sliding/full layer graph,
/// and prefill-vs-decode consistency across the two attention paths.
@Suite struct GptOssRunnerTests {

    private func makeRunner(weightBits: Int = 4) throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try GptOssToySynthetic.write(weightBits: weightBits)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir,
                                   device: ctx.device,
                                   expecting: .gptOssToy())
        let runner = try RealForwardRunner(model: model,
                                           context: ctx,
                                           maxContext: 64)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    /// Runner init against the gpt-oss fixture: the schema branch accepts the
    /// biased/sinked layout, and init never touches Qwen-only tensors (QK
    /// norms, shared expert, scalar gate — absent here, any access throws).
    @Test func runnerInit_gptOssToy_acceptsSchemaAndSkipsQwenTensors() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.maxContext == 64)
    }

    /// Decode smoke over the alternating sliding/full graph: biased QKV,
    /// sinks in both softmax paths, YaRN rope, top-2-of-2 routing with
    /// additive expert and router biases, no shared expert.
    @Test func decodeSmoke_alternatingSwaFullLayers() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        try await runner.produce(token: 1, position: 0, into: logits)
        let first = runner.lastGreedyToken
        #expect(first < 256)
        try await runner.produce(token: Int32(first), position: 1, into: logits)
        #expect(runner.continuationPosition == 2)

        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 1, position: 0, into: logits)
        #expect(runner.lastGreedyToken == first)
    }

    /// Prefill/decode consistency: prefilling [t0] then decoding t1 must give
    /// the same argmax as decoding t0, t1 step by step — the chunked-prefill
    /// attention (biases + sinks + YaRN + window clamp) and the decode branch
    /// share KV state and must agree.
    @Test func prefillThenDecode_matchesPureDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        try await runner.produce(token: 11, position: 0, into: logits)
        try await runner.produce(token: 7, position: 1, into: logits)
        let reference = runner.lastGreedyToken

        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: logits,
            onProgress: { _ in })
        try await runner.produce(token: 7, position: 1, into: logits)
        #expect(runner.lastGreedyToken == reference)
    }

    /// Prefill and pure decode must produce the same logits at every length:
    /// 1 (single row), 8 (window edge), 9 (first sliding truncation), and 12
    /// (well past the 8-token window on the sliding layers). 8-bit like the
    /// qwen logits comparisons: the 4-bit fused greedy head never writes the
    /// vocab buffer, so a 4-bit decode reference would be stale memory.
    @Test(arguments: [1, 8, 9, 12])
    func multiTokenPrefillMatchesPureDecode(count: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: 8)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(!runner.usesFusedGreedyHead)
        let logits = try makeLogits(ctx, vocab: 256)
        let tokens = [Int32]([11, 7, 5, 3, 9, 2, 8, 4, 6, 10, 12, 1].prefix(count))

        for (position, token) in tokens.enumerated() {
            try await runner.produce(token: token, position: position, into: logits)
        }
        let reference = Fp16Buffer.read(logits, count: 256)

        runner.reset()
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(logits, count: 256)

        var maxAbs: Float = 0
        for (lhs, rhs) in zip(actual, reference) {
            maxAbs = max(maxAbs, abs(Float(lhs) - Float(rhs)))
        }
        #expect(maxAbs < 0.05, "count=\(count) maxAbs=\(maxAbs)")
        #expect(actual.allSatisfy { $0.isFinite })
    }

    /// The biased/roped K and V rows both paths write for the same token must
    /// match — a prefill-vs-decode divergence upstream of the softmax shows
    /// here first.
    @Test func singleTokenKVMatchesBetweenPrefillAndDecode() async throws {
        let (dir, ctx, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)

        try await runner.produce(token: 11, position: 0, into: logits)
        let decodeSnap = try runner.captureInferenceState()
        runner.reset()
        let tokens: [Int32] = [11]
        _ = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        let prefillSnap = try runner.captureInferenceState()

        #expect(decodeSnap.descriptor.kvSegmentLengths
                == prefillSnap.descriptor.kvSegmentLengths)
        #expect(decodeSnap.payload.count == prefillSnap.payload.count)
        var maxAbs: Float = 0
        decodeSnap.payload.withUnsafeBytes { a in
            prefillSnap.payload.withUnsafeBytes { b in
                let av = a.bindMemory(to: Float16.self)
                let bv = b.bindMemory(to: Float16.self)
                for i in 0..<min(av.count, bv.count) {
                    let d = abs(Float(av[i]) - Float(bv[i]))
                    if d.isFinite { maxAbs = max(maxAbs, d) }
                }
            }
        }
        #expect(maxAbs < 0.02, "KV payload maxAbs=\(maxAbs)")
    }

    @Test(arguments: [8])
    func decodeAndPrefillSupportHigherBitCheckpoints(bits: Int) async throws {
        let (dir, ctx, runner) = try makeRunner(weightBits: bits)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: 256)
        let tokens: [Int32] = [1, 2]
        let result = try await runner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: logits,
            onProgress: { _ in })
        #expect(result.newPosition == 2)
        try await runner.produce(token: 3, position: 2, into: logits)
        #expect(runner.continuationPosition == 3)
        let values = Fp16Buffer.read(logits, count: 256)
        #expect(values.allSatisfy { $0.isFinite })
    }
}
