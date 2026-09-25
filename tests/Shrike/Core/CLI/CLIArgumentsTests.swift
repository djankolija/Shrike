import ArgumentParser
import Foundation
import Shrike
import Testing
@testable import ShrikeCLICore

@Suite struct CLIArgumentsTests {
    private func rejection(_ argv: [String]) throws -> String {
        let error = #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse(argv)
        }
        return ShrikeGenerateCommand.message(for: try #require(error))
    }

    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 0.6)
        #expect(arguments.topK == .limit(20))
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func kvPrecisionAndYaRNOptionsParse() throws {
        let yarn = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--kv-bits", "4", "--rope-scaling", "yarn",
        ])
        #expect(yarn.kvCachePrecision == .int4)
        #expect(yarn.ropeScalingMode == .yarn)
        #expect(yarn.maxContext == 1_048_576)
        let halfMillion = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--rope-scaling", "yarn", "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(try rejection([
            "--model", "m.gturbo", "--prompt", "hi",
            "--rope-scaling", "yarn", "--max-context", "262144",
        ]).contains("--max-context"))
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == .limit(40))
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func contextArgumentAcceptsQwenMaximumAndRejectsLargerValues() throws {
        let maximum = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-context", "262144",
        ])
        #expect(maximum.maxContext == 262_144)
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--max-context", "262145",
            ])
        }
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == .off)
        #expect(disabled.topP == 1)

        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() throws {
        #expect(try rejection([
            "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
        ]).contains("--top-k"))
    }

    @Test func theTrimmedGenerateFlagsNoLongerParse() {
        for argv in [["--concise"], ["--force-tokens", "/tmp/ids.txt"],
                     ["--prefill-chunk", "4096"], ["--prefill-chunk", "auto"]] {
            #expect(throws: (any Error).self) {
                _ = try ShrikeGenerateCommand.parse(
                    ["--model", "m.gturbo", "--prompt", "hi"] + argv)
            }
        }
    }

    @Test func thinkingModeParsesOnlyTheOfficialBinaryValues() throws {
        let on = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        #expect(try rejection([
            "--model", "m.gturbo", "--prompt", "hi", "--thinking", "medium",
        ]).contains("medium"))
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--help",
            "--ram-budget",
            "--kv-bits", "--rope-scaling", "--thinking",
            "--dump-logits", "--dump-hidden", "--tokenize",
        ]
        let words = ShrikeGenerateCommand.helpMessage()
            .split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func theRAMBudgetAloneSizesTheExpertPool() throws {
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi"]).expertCacheBudgetBytes == nil)
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi", "--ram-budget", "2G"])
            .expertCacheBudgetBytes == 2 << 30)
        #expect(try ShrikeGenerateCommand.parse(["--prompt", "hi", "--ram-budget", "11324620800"])
            .expertCacheBudgetBytes == 11_324_620_800)
        for size in ["0", "8X"] {
            #expect(try rejection(["--prompt", "hi", "--ram-budget", size])
                .contains("--ram-budget must be a positive size such as 2G, 512M or a byte count"))
        }
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse(["--prompt", "hi", "--expert-cache-slots", "160"])
        }
    }

    @Test func generatesPoolFollowsItsRAMBudget() throws {
        let stride = 1_769_472
        let (dir, toy) = try ManifestReaderTests.writeToyManifest(["expertStride": stride])
        defer { try? FileManager.default.removeItem(at: dir) }
        let budgeted = try ShrikeGenerateCommand.parse(
            ["--prompt", "hi", "--ram-budget", String(160 * stride * toy.numLayers)])
        #expect(try budgeted.expertCacheSlots(modelDirectory: dir, expecting: toy) == 160)
        let bare = try ShrikeGenerateCommand.parse(["--prompt", "hi"])
        #expect(try bare.expertCacheSlots(modelDirectory: dir, expecting: toy)
            == RuntimeConfiguration.expertCacheSlots(expertStrideBytes: UInt64(stride),
                                                     layers: toy.numLayers))
    }

    @Test func bothHelpSpellingsExitZero() throws {
        for flag in ["--help", "-h"] {
            let error = #expect(throws: (any Error).self) {
                _ = try ShrikeGenerateCommand.parse([flag])
            }
            #expect(ShrikeGenerateCommand.exitCode(for: try #require(error)) == .success)
        }
    }

    @Test func unsupportedSelectorsAreRejectedWithANonZeroExit() throws {
        for flag in ["--runtime-profile", "--experiment-id"] {
            let error = #expect(throws: (any Error).self) {
                _ = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
            #expect(ShrikeGenerateCommand.exitCode(for: try #require(error)) != .success)
        }
    }

    @Test func theGoldensRetiredImitationFlagsAreRejected() throws {
        for extra in [["--logits-head"], ["--follow-up", "and a semaphore?"]] {
            let error = #expect(throws: (any Error).self) {
                _ = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi"] + extra)
            }
            #expect(ShrikeGenerateCommand.exitCode(for: try #require(error)) != .success)
        }
    }

    @Test func aPromptIsRequiredButTheModelNeedNotBeNamed() throws {
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse(["--model", "m.gturbo"])
        }
        let arguments = try ShrikeGenerateCommand.parse(["--prompt", "hi"])
        #expect(arguments.model == nil)
    }

    @Test func anOptionValueMayBeginWithADash() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "-- explain this",
            "--stop", "-->",
        ])
        #expect(arguments.prompt == "-- explain this")
        #expect(arguments.stops == ["-->"])
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
}
