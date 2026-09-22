import ArgumentParser
import Testing

import ShrikeAttnBenchCore
import ShrikeCLICore
import ShrikeExpertBenchCore
import ShrikeRepackCore
import ShrikeRootCore
import ShrikeServerCore

@Suite struct RootCommandTests {
    private func parse(_ arguments: [String]) throws -> any ParsableCommand {
        try ShrikeRootCommand.parseAsRoot(arguments)
    }

    @Test func aBareInvocationAsksForTheOneThingItCannotResolve() throws {
        do {
            _ = try ShrikeRootCommand.parseAsRoot([])
            Issue.record("a bare invocation should not parse to a runnable command")
        } catch {
            #expect(ShrikeRootCommand.exitCode(for: error) == ExitCode.validationFailure)
            let message = ShrikeRootCommand.message(for: error)
            #expect(message == "one of --prompt or --messages-file is required")
            #expect(!message.contains("--model"))
        }
    }

    @Test func generationNeedsNoVerb() throws {
        let command = try parse(["--model", "m.gturbo", "--prompt", "hi"])
        let generate = try #require(command as? ShrikeGenerateCommand)
        #expect(generate.model == "m.gturbo")
        #expect(generate.prompt == "hi")
    }

    @Test func generateIsAlsoReachableByName() throws {
        let command = try parse(["generate", "--model", "m.gturbo", "--prompt", "hi"])
        #expect(command is ShrikeGenerateCommand)
    }

    @Test func serveKeepsItsOwnModel() throws {
        let command = try parse(["serve", "--model", "m.gturbo"])
        let serve = try #require(command as? ShrikeServerCommand)
        #expect(serve.model == "m.gturbo")
    }

    @Test func serveKeepsEveryFlagItSharesWithGenerate() throws {
        let command = try parse([
            "serve", "--model", "m.gturbo", "--max-context", "32768",
            "--thinking", "off", "--kv-bits", "4", "--expert-cache-slots", "160",
        ])
        let serve = try #require(command as? ShrikeServerCommand)
        #expect(serve.model == "m.gturbo")
        #expect(serve.maxContext == 32768)
        #expect(serve.thinkingMode == .off)
        #expect(serve.kvCachePrecision == .int4)
        #expect(serve.expertCacheSlots == 160)
    }

    @Test func serveStillRefusesAnUnsupportedContext() {
        do {
            _ = try parse(["serve", "--model", "m.gturbo", "--max-context", "50000"])
            Issue.record("an unsupported --max-context should not parse")
        } catch {
            #expect(ShrikeRootCommand.message(for: error).hasPrefix("--max-context must be one of"))
        }
    }

    @Test func theMinisProductionLaunchLineParses() throws {
        let command = try parse([
            "serve",
            "--model", "./models/ornith15.gturbo",
            "--port", "8081",
            "--max-context", "32768",
            "--ram-budget", "11324620800",
            "--thinking", "off",
        ])
        let serve = try #require(command as? ShrikeServerCommand)
        #expect(serve.model == "./models/ornith15.gturbo")
        #expect(serve.port == 8081)
        #expect(serve.maxContext == 32768)
        #expect(serve.thinkingMode == .off)
    }

    @Test func repackResolvesTwoLevelsDeep() throws {
        let command = try parse(["repack", "verify-install", "--input-gturbo", "m.gturbo"])
        #expect(command is ShrikeRepackCommand.VerifyInstall)
    }

    @Test func benchResolvesEitherChild() throws {
        #expect(try parse(["bench", "attention"]) is AttnBenchCommand)
        #expect(try parse(["bench", "expert", "--model", "m.gturbo"]) is ExpertBenchCommand)
    }

    @Test func anUnknownVerbIsRejected() {
        do {
            _ = try parse(["srve", "--model", "m.gturbo", "--prompt", "hi"])
            Issue.record("a mistyped verb should not reach generate")
        } catch {
            #expect(ShrikeRootCommand.message(for: error).contains("srve"))
        }
    }
}
