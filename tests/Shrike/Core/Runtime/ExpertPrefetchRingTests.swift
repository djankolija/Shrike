import Foundation
import Testing

@testable import Shrike

@Suite struct ExpertPrefetchRingTests {
    private final class Drops: @unchecked Sendable {
        private let lock = NSLock()
        private var _dropped: [(layer: Int, expert: Int, cell: Int)] = []
        var dropped: [(layer: Int, expert: Int, cell: Int)] { lock.withLock { _dropped } }
        func record(_ layer: Int, _ expert: Int, _ cell: Int) {
            lock.withLock { _dropped.append((layer, expert, cell)) }
        }
    }

    private func makeRing(cells: [Int] = [10, 11, 12, 13],
                          drops: Drops = Drops()) throws -> ExpertPrefetchRing {
        try ExpertPrefetchRing(cells: cells) { layer, expert, cell in
            drops.record(layer, expert, cell)
        }
    }

    @Test func beginKeepsTheInFlightBudgetAndFreesItWhenAReadCompletes() throws {
        let ring = try makeRing()
        var issued: [[Int]] = []
        let first = ExpertLoadOperation()
        try ring.begin(layer: 3, experts: [7, 9, 11], resident: []) { experts, cells in
            issued.append(experts)
            #expect(cells.count == experts.count)
            #expect(cells.allSatisfy { (10...13).contains($0) })
            return first
        }
        #expect(issued == [[7]])
        #expect(ring.inFlightCount == 1)

        let second = ExpertLoadOperation()
        try ring.begin(layer: 4, experts: [1, 2], resident: []) { experts, _ in
            issued.append(experts)
            return second
        }
        #expect(issued == [[7]])
        #expect(ring.statistics.issued == 1)

        first.markInFlight()
        first.finish(.success(()))
        try ring.begin(layer: 4, experts: [1, 2], resident: []) { experts, _ in
            issued.append(experts)
            return second
        }
        #expect(issued == [[7], [1]])
        #expect(ring.statistics.issued == 2)
        #expect(ring.statistics.reclaimed == 1)
    }

    @Test func beginDedupesResidentAndActivePredictions() throws {
        let ring = try makeRing()
        var issued: [[Int]] = []
        let first = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1, 2, 3], resident: [2]) { experts, _ in
            issued.append(experts)
            return first
        }
        first.finish(.success(()))
        try ring.begin(layer: 2, experts: [1, 3, 4, 4], resident: []) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        #expect(issued == [[1], [3]])
        #expect(ring.statistics.refused == 2)
    }

    @Test func readyCellsWaitsForAnInFlightPredictionAndCountsALateOne() throws {
        let ring = try makeRing()
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.markInFlight()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(20)) {
            operation.finish(.success(()))
        }
        let ready = ring.readyCells(layer: 2, experts: [1, 5], joinNanos: 1_000_000)
        #expect(ready == [1: 10])
        #expect(ring.statistics.late == 1)
        #expect(ring.statistics.joined == 0)
    }

    @Test func readyCellsJoinsAPredictionThatFinishesWithinTheBound() throws {
        let ring = try makeRing()
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.markInFlight()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(2)) {
            operation.finish(.success(()))
        }
        let ready = ring.readyCells(layer: 2, experts: [1, 2], joinNanos: 500_000_000)
        #expect(ready == [1: 10])
        #expect(ring.statistics.joined == 1)
        #expect(ring.statistics.late == 0)
    }

    @Test func readyCellsSkipsAFailedPredictionAndAClaimedSlotAwaitingItsAttach() throws {
        let ring = try makeRing()
        let failed = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in failed }
        failed.markInFlight()
        failed.finish(.failure(CocoaError(.fileReadUnknown)))
        #expect(ring.readyCells(layer: 2, experts: [1], joinNanos: 1_000_000).isEmpty)
        #expect(ring.statistics.late == 0)

        var lateDuringIssue: UInt64 = 0
        try ring.begin(layer: 3, experts: [4], resident: []) { _, _ in
            #expect(ring.readyCells(layer: 3, experts: [4], joinNanos: 1_000_000).isEmpty)
            lateDuringIssue = ring.statistics.late
            return ExpertLoadOperation()
        }
        #expect(lateDuringIssue == 1)
    }

    @Test func consumeMovesTheEntryToTheCellThePoolFreed() throws {
        let ring = try makeRing()
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        #expect(ring.readyCells(layer: 2, experts: [1]) == [1: 10])

        ring.consume(layer: 2, experts: [1], freedCells: [1: 77])
        #expect(ring.statistics.adopted == 1)
        var issuedCells: [Int] = []
        try ring.begin(layer: 3, experts: [5], resident: []) { _, cells in
            issuedCells = cells
            return ExpertLoadOperation()
        }
        #expect(issuedCells == [77])
    }

    @Test func consumeWithoutAFreedCellKeepsTheRingsCell() throws {
        let ring = try makeRing()
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        _ = ring.readyCells(layer: 2, experts: [1])
        ring.consume(layer: 2, experts: [1], freedCells: [:])
        var issuedCells: [Int] = []
        try ring.begin(layer: 3, experts: [5], resident: []) { _, cells in
            issuedCells = cells
            return ExpertLoadOperation()
        }
        #expect(issuedCells == [10])
    }

    @Test func consumeWithoutAFreedCellDropsTheLanding() throws {
        let drops = Drops()
        let ring = try makeRing(drops: drops)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        _ = ring.readyCells(layer: 2, experts: [1])
        ring.consume(layer: 2, experts: [1], freedCells: [:])
        let drop = try #require(drops.dropped.first)
        #expect(drop.layer == 2 && drop.expert == 1 && drop.cell == 10)
        #expect(drops.dropped.count == 1)
    }

    @Test func aRefusedClaimIsCountedAndItsSlotsFreed() throws {
        let ring = try makeRing()
        try ring.begin(layer: 2, experts: [1, 2], resident: []) { experts, cells in
            throw PrefetchClaimRefused(expert: experts[0], cell: cells[0])
        }
        #expect(ring.statistics.refused == 2)
        #expect(ring.statistics.issued == 0)
        #expect(ring.statistics.hookFailures == 0)
        #expect(ring.inFlightCount == 0)
        var issued: [[Int]] = []
        try ring.begin(layer: 2, experts: [1, 2], resident: []) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        #expect(issued == [[1]])
    }

    @Test func theReclaimKeepsTheLayerWhosePlanIsStillToCome() throws {
        let drops = Drops()
        let ring = try makeRing(drops: drops)
        let past = ExpertLoadOperation()
        try ring.begin(layer: 1, experts: [1], resident: []) { _, _ in past }
        past.finish(.success(()))
        let pending = ExpertLoadOperation()
        try ring.begin(layer: 3, experts: [3], resident: []) { _, _ in pending }
        pending.finish(.success(()))

        try ring.begin(layer: 3, experts: [4], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(drops.dropped.count == 1)
        #expect(drops.dropped.first?.layer == 1)
        #expect(ring.statistics.reclaimed == 1)
        #expect(ring.readyCells(layer: 3, experts: [3]) == [3: 10])
    }

    @Test func aLeasedSlotIsNeverReclaimedByAnotherLayersBegin() throws {
        let drops = Drops()
        let ring = try makeRing(drops: drops)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        let leased = try #require(ring.readyCells(layer: 2, experts: [1])[1])

        var issuedCells: [Int] = []
        let next = ExpertLoadOperation()
        try ring.begin(layer: 3, experts: [5], resident: []) { _, cells in
            issuedCells = cells
            return next
        }
        #expect(issuedCells == [11])
        #expect(!issuedCells.contains(leased))
        #expect(ring.statistics.reclaimed == 0)
        #expect(drops.dropped.isEmpty)

        ring.consume(layer: 2, experts: [1], freedCells: [1: 40])
        next.finish(.success(()))
        try ring.begin(layer: 4, experts: [8], resident: []) { _, cells in
            issuedCells = cells
            return ExpertLoadOperation()
        }
        #expect(issuedCells == [40])
    }

    @Test func theReclaimDropsACompletedLandingBeforeItsCellIsReused() throws {
        let drops = Drops()
        let ring = try makeRing(cells: [10], drops: drops)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))

        var issuedCells: [Int] = []
        try ring.begin(layer: 3, experts: [5], resident: []) { _, cells in
            issuedCells = cells
            #expect(drops.dropped.count == 1)
            return ExpertLoadOperation()
        }
        #expect(issuedCells == [10])
        #expect(ring.statistics.reclaimed == 1)
        let drop = try #require(drops.dropped.first)
        #expect(drop.layer == 2 && drop.expert == 1 && drop.cell == 10)
    }

    @Test func theReclaimOfAFailedPredictionDropsNothing() throws {
        let drops = Drops()
        let ring = try makeRing(cells: [10], drops: drops)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.markInFlight()
        operation.finish(.failure(CocoaError(.fileReadUnknown)))
        try ring.begin(layer: 3, experts: [5], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(drops.dropped.isEmpty)
        #expect(ring.statistics.reclaimed == 0)
        #expect(ring.statistics.failed == 1)
        #expect(ring.statistics.issued == 2)
    }

    @Test func unleaseReturnsAThrowingPlansPredictionsToTheRing() throws {
        let drops = Drops()
        let ring = try makeRing(drops: drops)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        let leased = try #require(ring.readyCells(layer: 2, experts: [1])[1])
        ring.unlease(layer: 2, experts: [1])

        var issuedCells: [Int] = []
        try ring.begin(layer: 3, experts: [5], resident: []) { _, cells in
            issuedCells = cells
            return ExpertLoadOperation()
        }
        #expect(issuedCells == [leased])
        #expect(ring.statistics.reclaimed == 1)
        #expect(ring.statistics.adopted == 0)
        #expect(drops.dropped.count == 1)
    }

    @Test func demandSubmissionsCountTheReadsTheyOverlap() throws {
        let ring = try makeRing()
        ring.noteDemandSubmission()
        #expect(ring.statistics.overlapped == 0)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        ring.noteDemandSubmission()
        #expect(ring.statistics.overlapped == 1)
        operation.finish(.success(()))
        ring.noteDemandSubmission()
        #expect(ring.statistics.overlapped == 1)
    }

    @Test func beginRecordsDeferredIssuesAndTheirWall() throws {
        let ring = try makeRing()
        let first = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: [], deferred: true) { _, _ in first }
        first.finish(.success(()))
        try ring.begin(layer: 3, experts: [2], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.deferred == 1)
        #expect(ring.statistics.issued == 2)
        #expect(ring.statistics.beginNanos > 0)
    }

    @Test func completionNanosReportsCompletedPredictionsOnly() throws {
        let ring = try makeRing()
        let first = ExpertLoadOperation()
        let second = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in first }
        let before = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        first.finish(.success(()))
        try ring.begin(layer: 2, experts: [5], resident: []) { _, _ in second }

        let stamps = ring.completionNanos(layer: 2, experts: [1, 5, 9])
        #expect(stamps.keys.sorted() == [1])
        #expect(stamps[1]! >= before)
        #expect(ring.completionNanos(layer: 3, experts: [1]).isEmpty)

        second.finish(.failure(CocoaError(.fileReadUnknown)))
        #expect(ring.completionNanos(layer: 2, experts: [5]).isEmpty)
        ring.consume(layer: 2, experts: [1], freedCells: [:])
        #expect(ring.completionNanos(layer: 2, experts: [1]).isEmpty)
    }

    @Test func predictionsBeyondTheBudgetOrTheCellsAreCountedAsRefused() throws {
        let ring = try makeRing()
        try ring.begin(layer: 2, experts: [1, 2, 3], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.refused == 2)
        try ring.begin(layer: 3, experts: [4], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.refused == 3)
        #expect(ring.statistics.issued == 1)
    }

    @Test func theFirstHookFailureIsTheOneToLog() throws {
        let ring = try makeRing(cells: [10, 11])
        #expect(ring.noteHookFailure())
        #expect(!ring.noteHookFailure())
        #expect(ring.statistics.hookFailures == 2)
    }

    @Test func beginRollsTheSlotsBackWhenTheIssueThrows() throws {
        struct ReaderRefused: Error {}
        let ring = try makeRing()
        #expect(throws: ReaderRefused.self) {
            try ring.begin(layer: 2, experts: [1, 2], resident: []) { _, _ in throw ReaderRefused() }
        }
        #expect(ring.inFlightCount == 0)
        #expect(ring.statistics.issued == 0)
        var issued: [[Int]] = []
        try ring.begin(layer: 2, experts: [1, 2], resident: []) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        #expect(issued == [[1]])
    }

    @Test func anEmptyRingIsRefused() {
        #expect(throws: (any Error).self) {
            try ExpertPrefetchRing(cells: []) { _, _, _ in }
        }
    }
}
