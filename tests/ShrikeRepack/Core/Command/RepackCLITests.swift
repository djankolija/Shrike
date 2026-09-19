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

    @Test func aHalfWrittenSavedDownloadIsRefusedBeforeNetwork() throws {
        let output = temporaryOutput("half-written")
        defer { clean(output) }
        try Data("{}".utf8).write(to: URL(fileURLWithPath: output + ".resume.json"))
        let result = try run([
            "install",
            "--output", output,
        ])

        #expect(result.status == operationalFailure)
        #expect(result.stderr.contains(
            "partial directory and checkpoint must exist together"))
    }

    @Test func theRetiredInstallFlagsAreRejectedByTheBinary() throws {
        let output = temporaryOutput("retired-install")
        defer { clean(output) }
        for flag in ["--resume", "--overwrite"] {
            let result = try run(["install", "--output", output, flag])
            #expect(result.status == parseFailure)
            #expect(result.stderr.contains("Unknown option '\(flag)'"))
        }
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
        // Reaching the existing-output refusal is what proves the selector parsed,
        // since that check runs after parsing and before any network traffic.
        try FileManager.default.createDirectory(atPath: output,
                                                withIntermediateDirectories: true)
        let result = try run([
            "install",
            "--model", "qwen36",
            "--output", output,
        ])

        #expect(result.status == operationalFailure)
        #expect(result.stderr.contains("output directory already exists"))
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
            output + ".resume.json",
            output + ".install.lock",
        ] {
            try? FileManager.default.removeItem(atPath: path)
        }
    }
}
