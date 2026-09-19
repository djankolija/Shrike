import ArgumentParser
import Testing
@testable import ShrikeRepackCore

/// The command lines that exist, including CLAUDE.md's and README's re-issue
/// line: a change needing one of these edited has changed the contract.
@Suite struct RepackInvocationTests {
    private func parse<T: ParsableCommand>(_ argv: [String], as type: T.Type) throws -> T {
        let command = try ShrikeRepackCommand.parseAsRoot(argv)
        return try #require(command as? T)
    }

    @Test func receiptReissueLineParses() throws {
        let verify = try parse(["verify-install", "--input-gturbo", "/models/ornith15.gturbo"],
                               as: ShrikeRepackCommand.VerifyInstall.self)
        #expect(verify.inputGTurbo == "/models/ornith15.gturbo")
    }

    @Test func installDefaultsToTheDefaultCheckpoint() throws {
        let install = try parse(["install", "--output", "/models/out.gturbo"],
                                as: ShrikeRepackCommand.Install.self)
        #expect(install.model == SupportedModelSource.default)
        #expect(install.output == "/models/out.gturbo")
    }

    @Test func installAcceptsACheckpointName() throws {
        let install = try parse(
            ["install", "--model", "ornith15", "--output", "/models/out.gturbo"],
            as: ShrikeRepackCommand.Install.self)
        #expect(install.model.name == "ornith15")
    }

    @Test func theTrimmedInstallFlagsNoLongerParse() {
        for flag in ["--overwrite", "--resume"] {
            #expect(throws: (any Error).self) {
                _ = try ShrikeRepackCommand.parseAsRoot(
                    ["install", "--output", "/models/out.gturbo", flag])
            }
        }
    }

    @Test func installRejectsAnUnknownCheckpointName() {
        #expect(throws: (any Error).self) {
            _ = try ShrikeRepackCommand.parseAsRoot(
                ["install", "--model", "llama", "--output", "/models/out.gturbo"])
        }
    }

    @Test func importSnapshotParses() throws {
        let snapshot = try parse(
            ["import-snapshot", "--input-snapshot", "/snap", "--model-id", "ornith15-mtp",
             "--output", "/models/out.gturbo"],
            as: ShrikeRepackCommand.ImportSnapshot.self)
        #expect(snapshot.inputSnapshot == "/snap")
        #expect(snapshot.modelID == "ornith15-mtp")
        #expect(snapshot.output == "/models/out.gturbo")
    }

    @Test func discardPartialParses() throws {
        let discard = try parse(["discard-partial", "--output", "/models/out.gturbo"],
                                as: ShrikeRepackCommand.DiscardPartial.self)
        #expect(discard.output == "/models/out.gturbo")
    }

    @Test func eachSubcommandRequiresItsOwnRequiredOption() {
        for argv in [["verify-install"], ["install"], ["discard-partial"],
                     ["import-snapshot", "--input-snapshot", "/snap"]] {
            #expect(throws: (any Error).self) {
                _ = try ShrikeRepackCommand.parseAsRoot(argv)
            }
        }
    }

    @Test func aSubcommandRejectsAnotherSubcommandsOptions() {
        #expect(throws: (any Error).self) {
            _ = try ShrikeRepackCommand.parseAsRoot(
                ["verify-install", "--input-gturbo", "/m.gturbo", "--input-snapshot", "/snap"])
        }
    }

    @Test func theRetiredFlagSpellingNoLongerParses() {
        #expect(throws: (any Error).self) {
            _ = try ShrikeRepackCommand.parseAsRoot(
                ["--verify-install", "--input-gturbo", "/models/ornith15.gturbo"])
        }
    }

    // A root with subcommands answers help by returning ArgumentParser's help
    // command, where a leaf command throws; both exit zero.
    @Test func helpIsAcceptedInEverySpellingAndListsEverySubcommand() {
        for argv in [["--help"], ["-h"], ["verify-install", "--help"], ["help"]] {
            #expect(throws: Never.self) {
                _ = try ShrikeRepackCommand.parseAsRoot(argv)
            }
        }
        let help = ShrikeRepackCommand.helpMessage()
        for subcommand in ["install", "import-snapshot", "verify-install", "discard-partial"] {
            #expect(help.contains(subcommand))
        }
    }
}
