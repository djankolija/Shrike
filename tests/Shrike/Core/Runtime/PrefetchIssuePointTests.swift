import Foundation
import Testing

@testable import Shrike

@Suite struct PrefetchIssuePointTests {
    @Test func afterPlacementDefersUntilTheDemandBatchCompletes() {
        let demand = ExpertLoadOperation()
        demand.markInFlight()
        var calls: [Bool] = []
        RealForwardRunner.schedulePrefetchIssue(placement: .after, demand: demand) { calls.append($0) }
        #expect(calls.isEmpty)
        demand.finish(.success(()))
        #expect(calls == [true])
    }

    @Test func afterPlacementIssuesAtOnceWithoutAnInFlightBatch() {
        var calls: [Bool] = []
        RealForwardRunner.schedulePrefetchIssue(placement: .after, demand: nil) { calls.append($0) }
        #expect(calls == [false])
        let finished = ExpertLoadOperation()
        finished.finish(.success(()))
        RealForwardRunner.schedulePrefetchIssue(placement: .after, demand: finished) { calls.append($0) }
        #expect(calls == [false, false])
    }

    @Test func besidePlacementIssuesAtOnceBesideTheBatch() {
        let demand = ExpertLoadOperation()
        demand.markInFlight()
        var calls: [Bool] = []
        RealForwardRunner.schedulePrefetchIssue(placement: .beside, demand: demand) { calls.append($0) }
        #expect(calls == [false])
        demand.finish(.success(()))
        #expect(calls == [false])
    }
}
