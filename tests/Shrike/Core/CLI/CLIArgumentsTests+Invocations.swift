import Testing
@testable import ShrikeCLICore

/// Copied from the tools that issue them: a migration needing one of these
/// edited has changed the contract, not the parser.
extension CLIArgumentsTests {
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
