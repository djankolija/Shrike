import Testing

@testable import Shrike

@Suite struct PrefetchRaceSplitTests {
    @Test func adoptedPredictionsSplitAgainstTheTailCommandsGPUWindow() {
        let split = RealForwardRunner.prefetchRaceSplit(
            completions: [1: 100_000, 2: 700_000, 3: 460_000, 5: 690_000, 6: 300_000, 7: 50],
            adopted: [1, 2, 3, 4, 5, 6],
            gpuStartNanos: 200_000, gpuEndNanos: 700_000)
        #expect(split.before == 1)
        #expect(split.during == 3)
        #expect(split.duringLastFifty == 1)
        #expect(split.duringFiftyToOneFifty == 0)
        #expect(split.duringEarlier == 2)
        #expect(split.after == 1)
        #expect(split.unknown == 1)
    }

    @Test func theBoundariesFallOnTheLaterSide() {
        let split = RealForwardRunner.prefetchRaceSplit(
            completions: [1: 200_000, 2: 700_000, 3: 650_000, 4: 550_000],
            adopted: [1, 2, 3, 4],
            gpuStartNanos: 200_000, gpuEndNanos: 700_000)
        #expect(split.before == 0)
        #expect(split.during == 3)
        #expect(split.duringLastFifty == 0)
        #expect(split.duringFiftyToOneFifty == 1)
        #expect(split.duringEarlier == 2)
        #expect(split.after == 1)
    }

    @Test func anUnreportedGPUWindowOrAMissingStampCountsAsUnknown() {
        let onlyStart = RealForwardRunner.prefetchRaceSplit(
            completions: [1: 100, 2: 300], adopted: [1, 2, 3],
            gpuStartNanos: 200, gpuEndNanos: 0)
        #expect(onlyStart.before == 0 && onlyStart.during == 0 && onlyStart.after == 0)
        #expect(onlyStart.unknown == 3)
        let onlyEnd = RealForwardRunner.prefetchRaceSplit(
            completions: [1: 100], adopted: [1], gpuStartNanos: 0, gpuEndNanos: 300)
        #expect(onlyEnd.unknown == 1)
    }
}
