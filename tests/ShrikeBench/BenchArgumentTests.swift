import ArgumentParser
import Testing
@testable import ShrikeAttnBenchCore
@testable import ShrikeExpertBenchCore
import ShrikeBenchCore

@Suite struct AttnBenchArgumentTests {
    @Test func defaultsMatchTheStepZeroLadder() throws {
        let bench = try AttnBenchCommand.parse([])
        #expect(bench.arms.names == Arm.defaultLadder)
        #expect(bench.positions.counts == [1_024, 4_096, 8_192])
        #expect(bench.repeats == 7)
        #expect(bench.warmup == 2)
        #expect(bench.seed.value == 0x5EED_0019)
    }

    @Test func everyOptionParses() throws {
        let bench = try AttnBenchCommand.parse([
            "--arms", "qregs,dbuf,qregs+dbuf+load8",
            "--positions", "1024,2048",
            "--repeats", "3", "--warmup", "0", "--seed", "0x1234",
        ])
        #expect(bench.arms.names == ["qregs", "dbuf", "qregs+dbuf+load8"])
        #expect(bench.positions.counts == [1_024, 2_048])
        #expect(bench.repeats == 3)
        #expect(bench.warmup == 0)
        #expect(bench.seed.value == 0x1234)
    }

    @Test func listIsASubcommandRatherThanAFlag() throws {
        let listed = try AttnBenchCommand.parseAsRoot(["list"])
        #expect(listed is AttnBenchCommand.ListArms)
        #expect(throws: (any Error).self) {
            _ = try AttnBenchCommand.parseAsRoot(["--list"])
        }
    }

    @Test func aBareInvocationStillRunsTheBenchItself() throws {
        let bare = try AttnBenchCommand.parseAsRoot([])
        #expect(bare is AttnBenchCommand)
    }

    @Test func badCountsAndEmptyListsAreRejected() {
        for argv in [["--repeats", "0"], ["--positions", "1024,0"],
                     ["--positions", "nope"], ["--arms", ""], ["--seed", "0xZZ"]] {
            #expect(throws: (any Error).self) {
                _ = try AttnBenchCommand.parse(argv)
            }
        }
    }

    @Test func theDefaultLadderMeasuresTheRunnersKernel() throws {
        #expect(try AttnBenchCommand.parse([]).arms.names.contains("prodstream"))
    }
}

@Suite struct ExpertBenchArgumentTests {
    @Test func modelIsRequiredAndTheRestDefault() throws {
        let bench = try ExpertBenchCommand.parse(["--model", "/models/ornith15.gturbo"])
        #expect(bench.model == "/models/ornith15.gturbo")
        #expect(bench.layer == 20)
        #expect(bench.experts == 8)
        #expect(bench.repeats == 15)
        #expect(bench.warmup == 3)
        #expect(bench.batch == 20)
        #expect(bench.seed.value == 0x5EED_0021)

        #expect(throws: (any Error).self) {
            _ = try ExpertBenchCommand.parse([])
        }
    }

    @Test func everyOptionParses() throws {
        let bench = try ExpertBenchCommand.parse([
            "--model", "/models/ornith15.gturbo",
            "--layer", "0", "--experts", "4",
            "--repeats", "5", "--warmup", "0", "--batch", "1", "--seed", "99",
        ])
        #expect(bench.layer == 0)
        #expect(bench.experts == 4)
        #expect(bench.repeats == 5)
        #expect(bench.warmup == 0)
        #expect(bench.batch == 1)
        #expect(bench.seed.value == 99)
    }

    @Test func outOfRangeValuesAreRejected() {
        for argv in [["--experts", "9"], ["--experts", "0"], ["--layer=-1"],
                     ["--repeats", "0"], ["--batch", "0"]] {
            #expect(throws: (any Error).self) {
                _ = try ExpertBenchCommand.parse(["--model", "/m.gturbo"] + argv)
            }
        }
    }

    @Test func armsIsNotAnOptionOnceProductionIsTheOnlyArm() {
        #expect(throws: (any Error).self) {
            _ = try ExpertBenchCommand.parse(["--model", "/m.gturbo", "--arms", "plain"])
        }
    }
}

@Suite struct BenchSeedTests {
    @Test func bothBenchesTakeHexAndDecimalAlike() throws {
        for text in ["0x5EED0019", "0X5eed0019"] {
            #expect(try AttnBenchCommand.parse(["--seed", text]).seed.value == 0x5EED_0019)
            #expect(try ExpertBenchCommand.parse(
                ["--model", "/m.gturbo", "--seed", text]).seed.value == 0x5EED_0019)
        }
        #expect(try AttnBenchCommand.parse(["--seed", "42"]).seed.value == 42)
        #expect(try ExpertBenchCommand.parse(
            ["--model", "/m.gturbo", "--seed", "42"]).seed.value == 42)
    }
}

@Suite struct ShrikeBenchCommandTests {
    @Test func theBenchBinaryResolvesEitherChild() throws {
        #expect(try ShrikeBenchCommand.parseAsRoot(["attention"]) is AttnBenchCommand)
        #expect(try ShrikeBenchCommand.parseAsRoot(["expert", "--model", "m.gturbo"])
            is ExpertBenchCommand)
    }
}
