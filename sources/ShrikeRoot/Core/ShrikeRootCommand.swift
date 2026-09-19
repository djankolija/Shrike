import ArgumentParser
import ShrikeAttnBenchCore
import ShrikeCLICore
import ShrikeExpertBenchCore
import ShrikeRepackCore
import ShrikeServerCore

/// The `shrike` root.
///
/// The root declares no options of its own: a flag it shared with a subcommand
/// would bind to the root in either position, and the subcommand would silently
/// never see it. `generate` is reached without typing it via `defaultSubcommand`.
public struct ShrikeRootCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "shrike",
        abstract: "Mixture-of-experts inference for models larger than RAM.",
        subcommands: [
            ShrikeGenerateCommand.self,
            ShrikeServerCommand.self,
            ShrikeRepackCommand.self,
            BenchCommand.self,
        ],
        defaultSubcommand: ShrikeGenerateCommand.self)

    public init() {}
}

public struct BenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Kernel benchmarks.",
        subcommands: [AttnBenchCommand.self, ExpertBenchCommand.self])

    public init() {}
}
