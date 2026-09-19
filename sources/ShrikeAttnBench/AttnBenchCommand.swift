import ArgumentParser
import Foundation
import ShrikeArgumentSupport

public struct AttnBenchCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "ShrikeAttnBench",
        abstract: "The decode attention scan on synthetic rows at the served shape.",
        discussion: "A composite arm joins switches with '+', e.g. qregs+dbuf+load8.",
        subcommands: [ListArms.self])

    @Option(help: ArgumentHelp("Arms to run; the list subcommand names them.",
                               valueName: "a,b,..."))
    public var arms = CommaSeparatedNames(Arm.defaultLadder)

    @Option(help: ArgumentHelp("Context lengths.", valueName: "n,..."))
    public var positions = CommaSeparatedCounts([1_024, 4_096, 8_192])

    @Option(help: ArgumentHelp("Timed command buffers per arm and length; the median is reported.",
                               valueName: "n"))
    public var repeats = 7

    @Option(help: ArgumentHelp("Untimed command buffers before the repeats.", valueName: "n"))
    public var warmup = 2

    @Option(help: ArgumentHelp("Row generator seed.", valueName: "n"))
    public var seed = BenchSeed(0x5EED_0019)

    public init() {}

    public func validate() throws {
        guard repeats > 0 else {
            throw ValidationError("--repeats needs a positive count")
        }
        guard warmup >= 0 else {
            throw ValidationError("--warmup needs a count of zero or more")
        }
    }

    public func run() throws {
        try BenchRunner(args: self).run()
    }
}

extension AttnBenchCommand {
    public struct ListArms: ParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "list",
            abstract: "Print the arm names and exit.")

        public init() {}

        public func run() {
            for (name, meaning) in Arm.switchHelp {
                print("\(name.padding(toLength: 12, withPad: " ", startingAt: 0)) \(meaning)")
            }
        }
    }
}
