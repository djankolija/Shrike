import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

extension RawCompletionLoopTests {
    final class CollectingSink: LogitsSink, @unchecked Sendable {
        var rows: [[Float16]] = []
        var chosen: [Int32] = []

        func record(position: Int, logits: UnsafeBufferPointer<Float16>) {
            rows.append(Array(logits))
        }

        func chose(position: Int, token: Int32) {
            chosen.append(token)
        }
    }

    private func argmax(_ row: [Float16]) -> Int32 {
        var best = 0
        for i in row.indices where row[i] > row[best] { best = i }
        return Int32(best)
    }

    @Test func forcedTokensReplaceTheSamplerAndStopWhenTheyRunOut() async throws {
        var config = GenerationConfig(maxNewTokens: 10, temperature: 0)
        config.forcedTokens = [11, 12, 13]
        let sink = CollectingSink()
        config.logitsSink = sink
        let (collected, result) = try await runLoop(seq: [7, 8, 9], end: 99, config: config)
        #expect(collected.tokens.map { $0.1 } == [11, 12, 13])
        #expect(result.newTokens == 3)
        #expect(result.reason == .maxTokens)
        #expect(sink.chosen == [11, 12, 13])
        #expect(sink.rows.count == 3)
        #expect(sink.rows.map(argmax) == [7, 7, 7])
    }

    @Test func theSinkSeesTheLogitsEachTokenWasChosenFrom() async throws {
        var config = GenerationConfig(maxNewTokens: 3, temperature: 0)
        let sink = CollectingSink()
        config.logitsSink = sink
        let (collected, result) = try await runLoop(seq: [7, 8, 9], end: 99, config: config)
        #expect(collected.tokens.map { $0.1 } == [7, 8, 9])
        #expect(result.newTokens == 3)
        #expect(sink.chosen == [7, 8, 9])
        #expect(sink.rows.map(argmax) == [7, 8, 9])
    }

    @Test func emptyForcedTokensAreRefused() {
        var config = GenerationConfig(maxNewTokens: 3, temperature: 0)
        config.forcedTokens = []
        #expect(throws: GeneratorError.self) { try config.validate() }
    }
}
