import Foundation
import Testing

@testable import Shrike

@Suite struct PrefetchIssuePointTests {
    @Test func theIssueDefersUntilTheDemandBatchCompletes() {
        let demand = ExpertLoadOperation()
        demand.markInFlight()
        var calls: [Bool] = []
        RealForwardRunner.schedulePrefetchIssue(demand: demand) { calls.append($0) }
        #expect(calls.isEmpty)
        demand.finish(.success(()))
        #expect(calls == [true])
    }

    @Test func theIssueRunsAtOnceWithoutAnInFlightBatch() {
        var calls: [Bool] = []
        RealForwardRunner.schedulePrefetchIssue(demand: nil) { calls.append($0) }
        #expect(calls == [false])
        let finished = ExpertLoadOperation()
        finished.finish(.success(()))
        RealForwardRunner.schedulePrefetchIssue(demand: finished) { calls.append($0) }
        #expect(calls == [false, false])
    }
}
