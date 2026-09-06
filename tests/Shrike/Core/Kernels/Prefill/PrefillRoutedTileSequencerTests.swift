import Testing
@testable import Shrike

@Suite struct PrefillRoutedTileSequencerTests {
    enum Call: Equatable {
        case plan(Int, avoidingInFlight: Int?)
        case begin(Int)
        case abandonPlan(Int)
        case encode(Int)
        case commit
        case drain
        case abandonBegun
    }

    struct EncodeFailure: Error, Equatable {
        let tile: Int
    }

    struct BeginFailure: Error, Equatable {
        let tile: Int
    }

    /// Records only a commit that seals a non-empty batch, so a trace reads as the sequencer's requests.
    final class RecordingDriver: PrefillRoutedTileDriver {
        var calls: [Call] = []
        var planRefusals: [Int: Int] = [:]
        var failingEncodes: Set<Int> = []
        var failingBegins: Set<Int> = []
        var slotsPerTile = 2

        private var openTiles: [Int] = []
        private var pending: [[Int]] = []
        private(set) var kept: Set<Int> = []
        private(set) var begun: Set<Int> = []
        private(set) var abandonedBegun: Set<Int> = []
        private(set) var abandonedKept: Set<Int> = []

        var openBatchTiles: Int { openTiles.count }
        var openBatchSlots: [Int] { openTiles.flatMap(slots) }
        var pendingDepth: Int { pending.count }
        var pendingAssignedSlots: [Int] { pending.flatMap { $0 } }

        func plan(tile: Int, avoidingInFlight inFlight: Int?) throws -> Bool {
            calls.append(.plan(tile, avoidingInFlight: inFlight))
            if let left = planRefusals[tile], left > 0 {
                planRefusals[tile] = left - 1
                return false
            }
            kept.insert(tile)
            return true
        }

        /// Like the runner's driver, a begin that fails has already dropped the
        /// kept plan (released inline there) and holds no begun fetch.
        func begin(tile: Int) throws {
            calls.append(.begin(tile))
            kept.remove(tile)
            if failingBegins.contains(tile) {
                throw BeginFailure(tile: tile)
            }
            begun.insert(tile)
        }

        func abandonPlan(tile: Int) throws {
            calls.append(.abandonPlan(tile))
            kept.remove(tile)
        }

        func encode(tile: Int) async throws {
            calls.append(.encode(tile))
            if failingEncodes.contains(tile) {
                throw EncodeFailure(tile: tile)
            }
            begun.remove(tile)
            openTiles.append(tile)
        }

        func commitOpenBatch() {
            guard !openTiles.isEmpty else { return }
            calls.append(.commit)
            pending.append(openBatchSlots)
            openTiles = []
        }

        func drainOldestBatch() throws {
            calls.append(.drain)
            if !pending.isEmpty {
                pending.removeFirst()
            }
        }

        func abandonBegunFetches() {
            calls.append(.abandonBegun)
            abandonedBegun = begun
            abandonedKept = kept
            begun = []
            kept = []
        }

        private func slots(ofTile tile: Int) -> [Int] {
            (0..<slotsPerTile).map { tile * slotsPerTile + $0 }
        }
    }

    private func run(_ driver: RecordingDriver,
                     tiles: Int,
                     depth: Int = 2,
                     width: Int = 1,
                     lookahead: Int = 1) async throws {
        let config = PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: depth, tilesPerCommandBuffer: width, fetchLookahead: lookahead)
        try await PrefillRoutedTileSequencer(scheduler: PrefillRoutedTileScheduler(config: config))
            .run(tileCount: tiles, driver: driver)
    }

    @Test func theLookaheadBeginsTheSuccessorBeforeTheTileIsAwaited() async throws {
        let driver = RecordingDriver()

        try await run(driver, tiles: 3)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0),
            .plan(1, avoidingInFlight: 0), .begin(1),
            .encode(0), .commit,
            .plan(2, avoidingInFlight: 1), .begin(2),
            .encode(1), .commit,
            .encode(2), .commit, .drain,
            .drain, .drain,
        ])
    }

    @Test func withoutALookaheadEachTileIsBegunAtItsOwnTurn() async throws {
        let driver = RecordingDriver()

        try await run(driver, tiles: 3, lookahead: 0)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0), .encode(0), .commit,
            .plan(1, avoidingInFlight: nil), .begin(1), .encode(1), .commit,
            .plan(2, avoidingInFlight: nil), .begin(2), .encode(2), .commit, .drain,
            .drain, .drain,
        ])
    }

    @Test func theLastTileBeginsNoSuccessor() async throws {
        let single = RecordingDriver()
        try await run(single, tiles: 1)
        #expect(single.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0), .encode(0), .commit, .drain,
        ])

        let none = RecordingDriver()
        try await run(none, tiles: 0)
        #expect(none.calls.isEmpty)
    }

    @Test func anUnplaceableTileCommitsTheOpenBatchAndDrainsItBeforeReplanning() async throws {
        let driver = RecordingDriver()
        driver.planRefusals = [1: 1]

        try await run(driver, tiles: 2, width: 2, lookahead: 0)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0), .encode(0),
            .plan(1, avoidingInFlight: nil), .commit, .drain, .begin(1),
            .encode(1), .commit,
            .drain,
        ])
    }

    @Test func theValveHoldsAtTheLookaheadDepthToo() async throws {
        let driver = RecordingDriver()
        driver.planRefusals = [1: 2]

        try await run(driver, tiles: 2, width: 2)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0),
            .plan(1, avoidingInFlight: 0),
            .encode(0),
            .plan(1, avoidingInFlight: nil), .commit, .drain, .begin(1),
            .encode(1), .commit,
            .drain,
        ])
    }

    @Test func aDeclinedLookaheadIsPlannedAgainAtItsOwnTurn() async throws {
        let driver = RecordingDriver()
        driver.planRefusals = [1: 1]

        try await run(driver, tiles: 2)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0),
            .plan(1, avoidingInFlight: 0),
            .encode(0), .commit,
            .plan(1, avoidingInFlight: nil), .begin(1),
            .encode(1), .commit,
            .drain, .drain,
        ])
    }

    @Test func theOpenBatchCountsAgainstTheDepthWithTheCommittedOnes() async throws {
        let driver = RecordingDriver()

        try await run(driver, tiles: 3, depth: 1, width: 2, lookahead: 0)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0), .encode(0),
            .plan(1, avoidingInFlight: nil), .begin(1), .encode(1), .commit,
            .plan(2, avoidingInFlight: nil), .begin(2), .encode(2), .commit, .drain,
            .drain,
        ])
    }

    @Test func aPendingBatchWithoutSlotsIsDrainedAndTheKeptPlanAbandoned() async throws {
        let driver = RecordingDriver()
        driver.slotsPerTile = 0

        try await run(driver, tiles: 2, lookahead: 0)

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0), .encode(0), .commit,
            .plan(1, avoidingInFlight: nil), .abandonPlan(1), .drain, .begin(1),
            .encode(1), .commit,
            .drain,
        ])
    }

    @Test func aFailureWaitsOutEveryBegunFetchAndPropagates() async {
        let driver = RecordingDriver()
        driver.failingEncodes = [0]

        await #expect(throws: EncodeFailure(tile: 0)) {
            try await run(driver, tiles: 3)
        }

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0),
            .plan(1, avoidingInFlight: 0), .begin(1),
            .encode(0), .abandonBegun,
        ])
        #expect(driver.abandonedBegun == [0, 1])
        #expect(driver.abandonedKept.isEmpty)
        #expect(driver.begun.isEmpty && driver.kept.isEmpty)
    }

    @Test func aFailureInsideABeginStillWaitsOutTheBegunPredecessor() async {
        let driver = RecordingDriver()
        driver.failingBegins = [1]

        await #expect(throws: BeginFailure(tile: 1)) {
            try await run(driver, tiles: 3)
        }

        #expect(driver.calls == [
            .plan(0, avoidingInFlight: nil), .begin(0),
            .plan(1, avoidingInFlight: 0), .begin(1),
            .abandonBegun,
        ])
        #expect(driver.abandonedBegun == [0])
        #expect(driver.abandonedKept.isEmpty)
        #expect(driver.begun.isEmpty && driver.kept.isEmpty)
    }
}
