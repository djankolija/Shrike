import ArgumentParser
import ShrikeAttnBenchCore
import ShrikeExpertBenchCore

public struct ShrikeBenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "shrike-bench",
        abstract: "Kernel benchmarks, a development tool built beside shrike.",
        subcommands: [AttnBenchCommand.self, ExpertBenchCommand.self])

    public init() {}
}
