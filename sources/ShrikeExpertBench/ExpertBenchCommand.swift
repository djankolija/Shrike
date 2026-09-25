import ArgumentParser
import Foundation
import ShrikeArgumentSupport

public struct ExpertBenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "expert",
        abstract: "Decode's phase-1 gate/up math on real experts, through the kernel entry point without the expert pool.")

    @Option(help: ArgumentHelp("The .gturbo directory.", valueName: "dir"))
    public var model: String

    @Option(help: ArgumentHelp("The layer whose experts are read.", valueName: "n"))
    public var layer = 20

    @Option(help: ArgumentHelp("""
        Distinct experts read, 1 to 8. A pass always runs the routed top-k of eight, \
        repeating the last expert read.
        """,
        valueName: "n"))
    public var experts = 8

    @Option(help: ArgumentHelp("Timed command buffers; the median is reported.",
                               valueName: "n"))
    public var repeats = 15

    @Option(help: ArgumentHelp("Untimed command buffers before the repeats.", valueName: "n"))
    public var warmup = 3

    @Option(help: ArgumentHelp("""
        Dispatches per command buffer, so the GPU holds its clock; the time \
        reported is per dispatch.
        """,
        valueName: "n"))
    public var batch = 20

    @Option(help: ArgumentHelp("The activation vector's seed.", valueName: "n"))
    public var seed = BenchSeed(0x5EED_0021)

    public init() {}

    public func validate() throws {
        guard !model.isEmpty else {
            throw ValidationError("--model must not be empty")
        }
        guard layer >= 0 else {
            throw ValidationError("--layer must be zero or more")
        }
        guard (1...8).contains(experts) else {
            throw ValidationError("--experts is 1 to 8")
        }
        guard repeats > 0 else {
            throw ValidationError("--repeats needs a positive count")
        }
        guard warmup >= 0 else {
            throw ValidationError("--warmup needs a count of zero or more")
        }
        guard batch > 0 else {
            throw ValidationError("--batch needs a positive count")
        }
    }

    public func run() throws {
        try BenchRunner(args: self).run()
    }
}
