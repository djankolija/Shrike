import Foundation
import Metal
import Testing

@testable import Shrike
import ShrikeValidationSupport

/// The pass committed ahead of a stop (v20 T3.3) on the Qwen toy: the state
/// the stop saw survives the extra pass and its drain, the last pass commits
/// nothing ahead, the runner is reusable after every kind of stop, and a
/// snapshot taken while the extra pass runs restores to the same state.
@Suite struct CommittedAheadTests {
    private struct Runner {
        let dir: URL
        let ctx: MetalContext
        let runner: RealForwardRunner
        let scratch: RawCompletionScratch
        let logits: MTLBuffer
    }

    private func makeRunner() throws -> Runner {
        let dir = try QwenToySynthetic.write(weightBits: 4)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device, expecting: .qwenToy())
        let runner = try RealForwardRunner(
            model: model, context: ctx, maxContext: 64,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true,
                                                       attentionFallbackAllowed: true))
        let scratch = try RawCompletionScratch(context: ctx, vocab: 1024)
        guard let logits = ctx.device.makeBuffer(length: 1024 * MemoryLayout<Float16>.stride,
                                                 options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return Runner(dir: dir, ctx: ctx, runner: runner, scratch: scratch, logits: logits)
    }

    private func greedy(_ toy: Runner) -> (MTLComputeCommandEncoder, Int, MTLBuffer) throws -> Void {
        let scratch = toy.scratch
        return { encoder, position, word in
            try scratch.sampler.sample(encoder: encoder, logits: scratch.logits,
                                       probs: scratch.probs, history: [],
                                       config: GenerationConfig(maxNewTokens: 8, temperature: 0),
                                       position: position, outToken: word)
        }
    }

    private func syncReference(_ toy: Runner, first: Int32, passes: Int) async throws -> ([Int32], [Float]) {
        toy.runner.reset()
        var fed: [Int32] = []
        var token = first
        var logits: [Float] = []
        for position in 0..<passes {
            try await toy.runner.produce(token: token, position: position, into: toy.logits)
            fed.append(token)
            logits = Fp16Buffer.read(toy.logits, count: 1024)
            token = Int32(argmax(logits))
        }
        return (fed, logits)
    }

    private func argmax(_ values: [Float]) -> Int {
        values.indices.max { values[$0] < values[$1] }!
    }

    private func runToAStop(_ toy: Runner, first: Int32) async throws -> [Int32] {
        toy.runner.reset()
        try await toy.runner.produce(token: first, position: 0, into: toy.scratch.logits,
                                     last: false, sample: greedy(toy))
        let sampled0 = try toy.runner.awaitBoundaryToken()
        try await toy.runner.produce(token: nil, position: 1, into: toy.scratch.logits,
                                     last: false, sample: greedy(toy))
        let sampled1 = try toy.runner.awaitBoundaryToken()
        #expect(toy.runner.isPassCommittedAhead)
        #expect(toy.runner.continuationPosition == 2)
        return [sampled0, sampled1]
    }

    @Test func theStateTheStopSawSurvivesTheExtraPassAndItsDrain() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let (fed, reference) = try await syncReference(toy, first: 11, passes: 3)

        let sampled = try await runToAStop(toy, first: 11)
        #expect(sampled == Array(fed[1...]))
        try toy.runner.prepareForContinuation(expectedPosition: 2)
        #expect(!toy.runner.isPassCommittedAhead)
        #expect(toy.runner.armedAgreedTokenCount == 0)
        #expect(toy.runner.leasedRingCellCount == 0)
        #expect(toy.runner.continuationPosition == 2)
        try await toy.runner.produce(token: fed[2], position: 2, into: toy.logits)
        #expect(Fp16Buffer.read(toy.logits, count: 1024) == reference)
    }

    @Test func theLastPassCommitsNothingAheadAndAContinuedPassRunsFreshFromTheWord() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let (fed, reference) = try await syncReference(toy, first: 11, passes: 3)

        toy.runner.reset()
        try await toy.runner.produce(token: 11, position: 0, into: toy.scratch.logits,
                                     last: true, sample: greedy(toy))
        #expect(try toy.runner.awaitBoundaryToken() == fed[1])
        #expect(!toy.runner.isPassCommittedAhead)
        #expect(toy.runner.armedAgreedTokenCount == 0)
        try await toy.runner.produce(token: nil, position: 1, into: toy.scratch.logits,
                                     last: true, sample: greedy(toy))
        #expect(try toy.runner.awaitBoundaryToken() == fed[2])
        #expect(!toy.runner.isPassCommittedAhead)
        #expect(toy.runner.continuationPosition == 2)
        try await toy.runner.produce(token: fed[2], position: 2, into: toy.logits)
        #expect(Fp16Buffer.read(toy.logits, count: 1024) == reference)
    }

    @Test func aContinuedPassWithNoCommandAheadAndNoWordIsRefused() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        toy.runner.reset()
        await #expect(throws: ModelError.self) {
            try await toy.runner.produce(token: nil, position: 0, into: toy.scratch.logits,
                                         last: true, sample: greedy(toy))
        }
    }

    @Test func aResetDuringTheExtraPassLeavesTheRunnerReusable() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let (fed, reference) = try await syncReference(toy, first: 11, passes: 2)

        _ = try await runToAStop(toy, first: 11)
        toy.runner.reset()
        #expect(!toy.runner.isPassCommittedAhead)
        #expect(toy.runner.armedAgreedTokenCount == 0)
        #expect(toy.runner.continuationPosition == 0)
        try await toy.runner.produce(token: 11, position: 0, into: toy.logits)
        try await toy.runner.produce(token: fed[1], position: 1, into: toy.logits)
        #expect(Fp16Buffer.read(toy.logits, count: 1024) == reference)
    }

    @Test func aSnapshotDuringTheExtraPassRestoresTheStateTheStopSaw() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let (fed, reference) = try await syncReference(toy, first: 11, passes: 3)

        _ = try await runToAStop(toy, first: 11)
        let snapshot = try toy.runner.captureInferenceState()
        #expect(snapshot.descriptor.position == 2)
        #expect(toy.runner.isPassCommittedAhead)

        toy.runner.reset()
        try toy.runner.restoreInferenceState(snapshot)
        #expect(toy.runner.continuationPosition == 2)
        try await toy.runner.produce(token: fed[2], position: 2, into: toy.logits)
        #expect(Fp16Buffer.read(toy.logits, count: 1024) == reference)
    }

    @Test func theReleaseLetsTheExtraPassRunThroughBeforeTheDrainWaits() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        let (fed, reference) = try await syncReference(toy, first: 11, passes: 3)

        _ = try await runToAStop(toy, first: 11)
        toy.runner.releasePassAhead()
        #expect(toy.runner.isPassCommittedAhead)
        #expect(toy.runner.armedAgreedTokenCount == 0)
        try toy.runner.prepareForContinuation(expectedPosition: 2)
        #expect(!toy.runner.isPassCommittedAhead)
        try await toy.runner.produce(token: fed[2], position: 2, into: toy.logits)
        #expect(Fp16Buffer.read(toy.logits, count: 1024) == reference)
    }

    @Test func settleDrainsTheExtraPassAndIsIdempotent() async throws {
        let toy = try makeRunner()
        defer { try? FileManager.default.removeItem(at: toy.dir) }
        _ = try await runToAStop(toy, first: 11)
        let drainedBefore = toy.runner.totalDrainedPasses
        toy.runner.settle()
        #expect(!toy.runner.isPassCommittedAhead)
        #expect(toy.runner.totalDrainedPasses == drainedBefore + 1)
        toy.runner.settle()
        #expect(toy.runner.totalDrainedPasses == drainedBefore + 1)
        #expect(toy.runner.continuationPosition == 2)
    }
}
