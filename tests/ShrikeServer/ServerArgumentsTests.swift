import Testing
@testable import ShrikeServerCore

/// Copied from the tools that issue them and from CLAUDE.md's launch line: a
/// migration needing one of these edited has changed the contract, not the parser.
@Suite struct ServerInvocationTests {
    @Test func rigLaunchLineParses() throws {
        let arguments = try ShrikeServerCommand.parse([
            "--model", "./models/ornith15.gturbo", "--model-id", "ornith15",
            "--port", "8081", "--max-context", "32768",
            "--ram-budget", "8G", "--thinking", "off",
        ])
        #expect(arguments.model == "./models/ornith15.gturbo")
        #expect(arguments.modelIDOverride == "ornith15")
        #expect(arguments.port == 8081)
        #expect(arguments.maxContext == 32_768)
        #expect(arguments.expertCacheBudgetBytes == 8 << 30)
        #expect(arguments.expertCacheSlots == nil)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func productionLaunchLineParses() throws {
        let arguments = try ShrikeServerCommand.parse([
            "--model", "./models/ornith15.gturbo", "--model-id", "ornith15",
            "--port", "8081", "--max-context", "32768",
            "--ram-budget", "11324620800", "--thinking", "off",
        ])
        #expect(arguments.expertCacheBudgetBytes == 11_324_620_800)
        #expect(arguments.maxContext == 32_768)
        #expect(arguments.port == 8081)
    }

    @Test func readmeModelOnlyLaunchParses() throws {
        let arguments = try ShrikeServerCommand.parse(["--model", "models/ornith15.gturbo"])
        #expect(arguments.model == "models/ornith15.gturbo")
        #expect(arguments.port == 8080)
        #expect(arguments.modelIDOverride == nil)
    }
}
