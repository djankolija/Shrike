import Foundation
import Metal
import Testing

@testable import Shrike

@Suite struct ExpertPrefetchRingTests {
    private func makeRing(slots: Int, budget: Int) throws -> ExpertPrefetchRing {
        let context = try MetalContext()
        return try ExpertPrefetchRing(device: context.device, expertStride: 16_384,
                                      slotCount: slots, inFlightBudget: budget)
    }

    @Test func beginKeepsTheInFlightBudgetAndFreesItWhenAReadCompletes() throws {
        let ring = try makeRing(slots: 4, budget: 1)
        var issued: [[Int]] = []
        let first = ExpertLoadOperation()
        try ring.begin(layer: 3, experts: [7, 9, 11], resident: []) { experts, buffers in
            issued.append(experts)
            #expect(buffers.count == experts.count)
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
        #expect(ring.statistics.reclaimedUnadopted == 1)
    }

    @Test func beginIssuesTheBestScoredPredictionsWithinTheBudget() throws {
        let ring = try makeRing(slots: 4, budget: 2)
        var issued: [[Int]] = []
        try ring.begin(layer: 2, experts: [5, 6, 7], resident: []) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        #expect(issued == [[5, 6]])
        #expect(ring.inFlightCount == 2)
    }

    @Test func beginDedupesResidentAndActivePredictions() throws {
        let ring = try makeRing(slots: 4, budget: 4)
        var issued: [[Int]] = []
        try ring.begin(layer: 2, experts: [1, 2, 3], resident: [2]) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        try ring.begin(layer: 2, experts: [1, 3, 4, 4], resident: []) { experts, _ in
            issued.append(experts)
            return ExpertLoadOperation()
        }
        #expect(issued == [[1, 3], [4]])
    }

    @Test func readyBuffersCountsAnInFlightPredictionAsLate() throws {
        let ring = try makeRing(slots: 4, budget: 2)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1, 2], resident: []) { _, _ in operation }
        operation.markInFlight()
        #expect(ring.readyBuffers(layer: 2, experts: [1, 5]).isEmpty)
        #expect(ring.statistics.late == 1)

        operation.finish(.success(()))
        let ready = ring.readyBuffers(layer: 2, experts: [1, 5])
        #expect(Set(ready.keys) == [1])
        #expect(ring.statistics.late == 1)
    }

    @Test func demandSubmissionsCountTheReadsTheyOverlap() throws {
        let ring = try makeRing(slots: 4, budget: 2)
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
        let ring = try makeRing(slots: 4, budget: 2)
        try ring.begin(layer: 2, experts: [1], resident: [], deferred: true) { _, _ in
            ExpertLoadOperation()
        }
        try ring.begin(layer: 3, experts: [2], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.deferred == 1)
        #expect(ring.statistics.issued == 2)
        #expect(ring.statistics.beginNanos > 0)
    }

    @Test func aLeasedSlotIsNeverReclaimedByAnotherLayersBegin() throws {
        let ring = try makeRing(slots: 4, budget: 4)
        let operation = ExpertLoadOperation()
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in operation }
        operation.finish(.success(()))
        let leased = try #require(ring.readyBuffers(layer: 2, experts: [1])[1])

        var issuedBuffers: [MTLBuffer] = []
        try ring.begin(layer: 3, experts: [5, 6, 7], resident: []) { _, buffers in
            issuedBuffers = buffers
            return ExpertLoadOperation()
        }
        #expect(issuedBuffers.count == 3)
        #expect(!issuedBuffers.contains { $0 === leased })
        #expect(ring.statistics.reclaimedUnadopted == 0)

        ring.consume(layer: 2, experts: [1])
        #expect(ring.statistics.adopted == 1)
        try ring.begin(layer: 4, experts: [8], resident: []) { _, buffers in
            issuedBuffers = buffers
            return ExpertLoadOperation()
        }
        #expect(issuedBuffers.first === leased)
    }

    @Test func predictionsBeyondTheBudgetOrTheSlotsAreCountedAsRefused() throws {
        let ring = try makeRing(slots: 4, budget: 1)
        try ring.begin(layer: 2, experts: [1, 2, 3], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.refused == 2)
        try ring.begin(layer: 3, experts: [4], resident: []) { _, _ in ExpertLoadOperation() }
        #expect(ring.statistics.refused == 3)
        #expect(ring.statistics.issued == 1)
    }

    @Test func readyBuffersCountsAClaimedSlotAwaitingItsAttachAsLate() throws {
        let ring = try makeRing(slots: 4, budget: 2)
        var lateDuringIssue: UInt64 = 0
        try ring.begin(layer: 2, experts: [1], resident: []) { _, _ in
            #expect(ring.readyBuffers(layer: 2, experts: [1]).isEmpty)
            lateDuringIssue = ring.statistics.late
            return ExpertLoadOperation()
        }
        #expect(lateDuringIssue == 1)
    }

    @Test func theFirstHookFailureIsTheOneToLog() throws {
        let ring = try makeRing(slots: 2, budget: 1)
        #expect(ring.noteHookFailure())
        #expect(!ring.noteHookFailure())
        #expect(ring.statistics.hookFailures == 2)
    }

    @Test func beginRollsTheSlotsBackWhenTheIssueThrows() throws {
        struct ReaderRefused: Error {}
        let ring = try makeRing(slots: 4, budget: 2)
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
        #expect(issued == [[1, 2]])
    }
}
