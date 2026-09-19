import Testing
@testable import ShrikeCLICore

extension CLIArgumentsTests {
    @Test func instrumentFlagsParse() throws {
        let instrumented = try ShrikeGenerateCommand.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--force-tokens", "ids.txt", "--dump-logits", "out.f16", "--logits-head",
            "--tokenize", "pieces.json",
        ])
        #expect(instrumented.forceTokensPath == "ids.txt")
        #expect(instrumented.dumpLogitsPath == "out.f16")
        #expect(instrumented.logitsHead)
        #expect(instrumented.tokenizePath == "pieces.json")
        let plain = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(plain.forceTokensPath == nil)
        #expect(plain.dumpLogitsPath == nil)
        #expect(!plain.logitsHead)
        #expect(plain.tokenizePath == nil)
        #expect(throws: (any Error).self) {
            _ = try ShrikeGenerateCommand.parse(["--model", "m.gturbo", "--prompt", "hi", "--force-tokens"])
        }
    }
}
