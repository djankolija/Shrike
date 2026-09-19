import Testing
@testable import ShrikeCLICore

/// Copied from the tools that issue them: a migration needing one of these
/// edited has changed the contract, not the parser.
extension CLIArgumentsTests {
    /// `TURNS_FOLLOW_UP` from `tools/golden-baseline.sh:81`, verbatim. It opens on a
    /// newline and carries chat-template tokens, which is the point of pinning it.
    static let turnsFollowUp = """
        \n<|im_start|>user\nAnd a semaphore, in one sentence?<|im_end|>\n        <|im_start|>assistant\n<think>\n\n</think>\n\n
        """

    @Test func goldenBaselineShortProfileParses() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain what a mutex is and when you would use one.",
            "--max-new", "96", "--temperature", "0", "--seed", "1234", "--quiet",
        ])
        #expect(arguments.model == "/models/ornith15.gturbo")
        #expect(arguments.prompt == "Explain what a mutex is and when you would use one.")
        #expect(arguments.maxNew == 96)
        #expect(arguments.temperature == 0)
        #expect(arguments.seed == 1_234)
        #expect(arguments.quiet)
        #expect(!arguments.logitsHead)
    }

    @Test func goldenBaselineLogitsHeadProfileParses() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain what a mutex is and when you would use one.",
            "--max-new", "128", "--temperature", "0", "--seed", "1234", "--quiet",
            "--logits-head",
        ])
        #expect(arguments.maxNew == 128)
        #expect(arguments.logitsHead)
    }

    @Test func goldenBaselineTurnsProfileParses() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--messages-file", "/tmp/golden-turns.json",
            "--follow-up", Self.turnsFollowUp,
            "--max-new", "128", "--temperature", "0", "--seed", "1234", "--quiet",
            "--logits-head",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "/tmp/golden-turns.json")
        #expect(arguments.followUp == Self.turnsFollowUp)
        #expect(arguments.logitsHead)
    }

    @Test func goldenBaselineExtraArgumentsParse() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain what a mutex is and when you would use one.",
            "--max-new", "96", "--temperature", "0", "--seed", "1234", "--quiet",
            "--expert-cache-slots", "160",
        ])
        #expect(arguments.expertCacheSlots == 160)
    }

    @Test func hiddenStateInstrumentParses() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain what a mutex is and when you would use one.",
            "--dump-hidden", "/tmp/hidden.f16",
        ])
        #expect(arguments.dumpHiddenPath == "/tmp/hidden.f16")
    }

    @Test func tokenizerOnlyInvocationParses() throws {
        let arguments = try ShrikeGenerateCommand.parse([
            "--model", "/models/ornith15.gturbo", "--prompt", "Explain what a mutex is and when you would use one.",
            "--tokenize", "/tmp/pieces.json",
        ])
        #expect(arguments.tokenizePath == "/tmp/pieces.json")
    }
}
