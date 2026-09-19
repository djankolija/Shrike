import ArgumentParser
import Foundation

extension SupportedModelSource: ExpressibleByArgument {
    public init?(argument: String) {
        guard let source = SupportedModelSource.named(argument) else { return nil }
        self = source
    }

    public var defaultValueDescription: String { name }

    public static var allValueStrings: [String] { all.map(\.name) }
}

public struct ShrikeRepackCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "ShrikeRepack",
        abstract: "Repackage a checkpoint into the .gturbo format, and verify an install.",
        discussion: """
            The .gturbo format stores routed experts in a layout that can be read \
            a single expert at a time. Set HF_TOKEN only if Hugging Face requests \
            authentication.
            """,
        subcommands: [Install.self, ImportSnapshot.self, VerifyInstall.self,
                      DiscardPartial.self])

    public init() {}
}

extension ShrikeRepackCommand {
    public struct Install: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            abstract: "Stream a supported checkpoint from Hugging Face and repack it.",
            discussion: """
                Repackages without materializing the source checkpoint on disk. A \
                cancelled or interrupted download can be continued with --resume or \
                removed with the discard-partial subcommand.
                """)

        @Option(help: ArgumentHelp("Checkpoint to install.", valueName: "name"))
        public var model: SupportedModelSource = .default

        @Option(help: ArgumentHelp("Destination bundle.", valueName: "model.gturbo"))
        public var output: String

        @Flag(help: "Replace an existing bundle at --output.")
        public var overwrite = false

        @Flag(help: "Continue a previously interrupted download.")
        public var resume = false

        public init() {}

        public func run() async throws {
            let options = model.installOptions(
                outputDirectory: URL(fileURLWithPath: output),
                overwrite: overwrite,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                resume: resume)
            let result = try await RemoteStreamingRepacker(options: options).run()
            print("Installed \(model.displayName)")
            print("Source revision: \(result.resolvedCommit)")
            print("Model: \(result.outputDir)")
        }
    }

    public struct ImportSnapshot: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "import-snapshot",
            abstract: "Import a completed local MLX-affine safetensors snapshot.",
            discussion: """
                Intended for reproducibly derived sidecars such as Ornith's native \
                MTP draft. There is no --resume because no network payload is involved.
                """)

        @Option(name: .customLong("input-snapshot"),
                help: ArgumentHelp("Snapshot to import.", valueName: "affine-safetensors-dir"))
        public var inputSnapshot: String

        @Option(name: .customLong("model-id"),
                help: ArgumentHelp("Identifier recorded in the bundle.", valueName: "id"))
        public var modelID: String

        @Option(help: ArgumentHelp("Destination bundle.", valueName: "model.gturbo"))
        public var output: String

        @Flag(help: "Replace an existing bundle at --output.")
        public var overwrite = false

        public init() {}

        public func run() async throws {
            let result = try await RemoteStreamingRepacker.runLocalSnapshot(
                options: LocalSnapshotRepackOptions(
                    inputSnapshotDir: inputSnapshot,
                    outputDir: output,
                    modelID: modelID,
                    overwrite: overwrite))
            print("Imported local snapshot")
            print("Source fingerprint: \(result.resolvedCommit)")
            print("Model: \(result.outputDir)")
        }
    }

    public struct VerifyInstall: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "verify-install",
            abstract: "Re-issue an install receipt in place.",
            discussion: """
                A receipt is bound to the absolute path it was installed to, so moving \
                or renaming an installed model makes it fail to load. Re-issue the \
                receipt rather than editing it.
                """)

        @Option(name: .customLong("input-gturbo"),
                help: ArgumentHelp("Bundle to verify.", valueName: "model.gturbo"))
        public var inputGTurbo: String

        public init() {}

        public func run() throws {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputGTurbo: inputGTurbo))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
        }
    }

    public struct DiscardPartial: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "discard-partial",
            abstract: "Remove a saved partial download.")

        @Option(help: ArgumentHelp("Bundle whose partial download to remove.",
                                   valueName: "model.gturbo"))
        public var output: String

        public init() {}

        public func run() throws {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
        }
    }
}
