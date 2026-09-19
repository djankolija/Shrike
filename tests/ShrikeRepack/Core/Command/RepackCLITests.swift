import ArgumentParser
import Foundation
import Testing

@Suite(.serialized)
struct RepackCLITests {
    private let parseFailure = ExitCode.validationFailure.rawValue
    private let operationalFailure = ExitCode.failure.rawValue

    @Test func discardPartialIsNotReachableFromInstall() throws {
        let output = temporaryOutput("exclusive")
        defer { clean(output) }
        let result = try run([
            "install",
            "--output", output,
            "--discard-partial",
        ])

        #expect(result.status == parseFailure)
        #expect(result.stderr.contains("Unknown option '--discard-partial'"))
    }

    @Test func resumeWithoutStateFailsBeforeNetwork() throws {
        let output = temporaryOutput("missing-resume")
        defer { clean(output) }
        let result = try run([
            "install",
            "--output", output,
            "--resume",
        ])

        #expect(result.status == operationalFailure)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func discardWithoutStateReportsAnError() throws {
        let output = temporaryOutput("missing-discard")
        defer { clean(output) }
        let result = try run([
            "discard-partial",
            "--output", output,
        ])

        #expect(result.status == operationalFailure)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func unknownModelSelectorIsRejectedAndNamesTheSupportedOnes() throws {
        let output = temporaryOutput("bad-model")
        defer { clean(output) }
        let result = try run([
            "install",
            "--model", "bogus",
            "--output", output,
        ])

        #expect(result.status == parseFailure)
        #expect(result.stderr.contains("invalid for '--model"))
        #expect(result.stderr.contains("qwen36"))
        #expect(result.stderr.contains("ornith15"))
    }

    @Test func qwenModelSelectorIsAccepted() throws {
        let output = temporaryOutput("qwen-model")
        defer { clean(output) }
        // --resume without saved state fails fast after argument parsing,
        // proving the selector itself is accepted without touching the network.
        let result = try run([
            "install",
            "--model", "qwen36",
            "--output", output,
            "--resume",
        ])

        #expect(result.status == operationalFailure)
        #expect(result.stderr.contains("no resumable install state exists"))
    }

    @Test func theRetiredFlagSpellingIsRejectedByTheBinary() throws {
        let output = temporaryOutput("retired")
        defer { clean(output) }
        let result = try run([
            "--discard-partial",
            "--output", output,
        ])

        #expect(result.status == parseFailure)
        #expect(result.stderr.contains("--discard-partial"))
    }

    private func run(_ arguments: [String]) throws
        -> (status: Int32, stdout: String, stderr: String) {
        let executable = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/debug/shrike")
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = executable
        process.arguments = ["repack"] + arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        let out = stdout.fileHandleForReading.readDataToEndOfFile()
        let err = stderr.fileHandleForReading.readDataToEndOfFile()
        return (
            process.terminationStatus,
            String(decoding: out, as: UTF8.self),
            String(decoding: err, as: UTF8.self))
    }

    private func temporaryOutput(_ tag: String) -> String {
        (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("shrikerepack-\(tag)-\(UUID().uuidString).gturbo")
    }

    private func clean(_ output: String) {
        for path in [
            output,
            output + ".partial",
            output + ".install-state",
            output + ".install-state.cleanup",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
