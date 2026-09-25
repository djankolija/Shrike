import ArgumentParser
import ShrikeCLICore
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
        ],
        defaultSubcommand: ShrikeGenerateCommand.self)

    public init() {}
}
