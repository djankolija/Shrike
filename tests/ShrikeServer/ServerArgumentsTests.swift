import Testing
@testable import ShrikeServerCore

/// Copied from the tools that issue them and from CLAUDE.md's launch line: a
/// migration needing one of these edited has changed the contract, not the parser.
@Suite struct ServerInvocationTests {
    @Test func productionLaunchLineParses() throws {
        let arguments = try ShrikeServerCommand.parse([
            "--model", "./models/ornith15.gturbo",
            "--port", "8081", "--max-context", "32768",
            "--ram-budget", "11324620800", "--thinking", "off",
        ])
        #expect(arguments.model == "./models/ornith15.gturbo")
        #expect(arguments.port == 8081)
        #expect(arguments.maxContext == 32_768)
        #expect(arguments.expertCacheBudgetBytes == 11_324_620_800)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func readmeModelOnlyLaunchParses() throws {
        let arguments = try ShrikeServerCommand.parse(["--model", "models/ornith15.gturbo"])
        #expect(arguments.model == "models/ornith15.gturbo")
        #expect(arguments.port == 8080)
    }
}

@Suite struct ServerPoolArgumentTests {
    @Test func theRAMBudgetIsTheOnlyPoolKnob() throws {
        #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--model", "m.gturbo", "--expert-cache-slots", "160"])
        }
        let error = #expect(throws: (any Error).self) {
            _ = try ShrikeServerCommand.parse(["--model", "m.gturbo", "--ram-budget", "8X"])
        }
        #expect(ShrikeServerCommand.message(for: try #require(error))
            .contains("--ram-budget must be a positive size such as 2G, 512M or a byte count"))
    }
}
