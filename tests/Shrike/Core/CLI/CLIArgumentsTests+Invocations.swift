import Testing
@testable import ShrikeCLICore

/// Copied from the tools that issue them: a migration needing one of these
/// edited has changed the contract, not the parser.
extension CLIArgumentsTests {
    @Test func goldenBaselineShortProfileParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--max-new", "96", "--temperature", "0", "--seed", "1234", "--quiet",
        ])
        #expect(arguments.model == "/models/ornith15.gturbo")
        #expect(arguments.prompt == "Explain quantization.")
        #expect(arguments.maxNew == 96)
        #expect(arguments.temperature == 0)
        #expect(arguments.seed == 1_234)
        #expect(arguments.quiet)
        #expect(!arguments.logitsHead)
    }

    @Test func goldenBaselineLogitsHeadProfileParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--max-new", "128", "--temperature", "0", "--seed", "1234", "--quiet",
            "--logits-head",
        ])
        #expect(arguments.maxNew == 128)
        #expect(arguments.logitsHead)
    }

    @Test func goldenBaselineTurnsProfileParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--messages-file", "/tmp/golden-turns.json",
            "--follow-up", "And in one sentence?",
            "--max-new", "128", "--temperature", "0", "--seed", "1234", "--quiet",
            "--logits-head",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "/tmp/golden-turns.json")
        #expect(arguments.followUp == "And in one sentence?")
        #expect(arguments.logitsHead)
    }

    @Test func goldenBaselineExtraArgumentsParse() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--max-new", "96", "--temperature", "0", "--seed", "1234", "--quiet",
            "--expert-cache-slots", "160",
        ])
        #expect(arguments.expertCacheSlots == 160)
    }

    @Test func classTwoGateInstrumentParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--force-tokens", "/tmp/ids.txt", "--dump-logits", "/tmp/out.f16",
        ])
        #expect(arguments.forceTokensPath == "/tmp/ids.txt")
        #expect(arguments.dumpLogitsPath == "/tmp/out.f16")
    }

    @Test func hiddenStateInstrumentParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--dump-hidden", "/tmp/hidden.f16",
        ])
        #expect(arguments.dumpHiddenPath == "/tmp/hidden.f16")
    }

    @Test func tokenizerOnlyInvocationParses() throws {
        let arguments = try ShrikeCLICommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain quantization.",
            "--tokenize", "/tmp/pieces.json",
        ])
        #expect(arguments.tokenizePath == "/tmp/pieces.json")
    }
}
