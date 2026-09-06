import Foundation
import Testing

@testable import Shrike

@Suite struct ExpertLoadOperationTests {
    @Test func operationPublishesStateAndResumesAsyncWaiter() async throws {
        let operation = ExpertLoadOperation()
        #expect(operation.state == .submitted)

        let waiter = Task { try await operation.completion() }
        operation.markInFlight()
        #expect(operation.state == .inFlight)
        operation.finish(.success(()))

        try await waiter.value
        #expect(operation.state == .completed)
        #expect(operation.completedNanos >= operation.startedNanos)
    }

    @Test func operationPropagatesFailureToBlockingAndAsyncWaiters() async {
        struct ExpectedFailure: Error {}
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        operation.finish(.failure(ExpectedFailure()))

        await #expect(throws: ExpectedFailure.self) {
            try await operation.completion()
        }
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
        #expect(operation.state == .failed)
    }

    @Test func completionHookRunsOnceAfterTheTerminalTransition() {
        let operation = ExpertLoadOperation()
        let seen = HookRecorder()
        operation.onCompletion { seen.record(operation.state) }
        #expect(seen.states.isEmpty)
        operation.markInFlight()
        #expect(seen.states.isEmpty)
        operation.finish(.success(()))
        #expect(seen.states == [.completed])
    }

    @Test func completionHooksRunExactlyOnceAcrossThreadsRacingTheFinish() {
        for _ in 0..<50 {
            let operation = ExpertLoadOperation()
            operation.markInFlight()
            let seen = HookRecorder()
            let group = DispatchGroup()
            for _ in 0..<4 {
                group.enter()
                DispatchQueue.global().async {
                    operation.onCompletion { seen.record(operation.state) }
                    group.leave()
                }
            }
            DispatchQueue.global().async { operation.finish(.success(())) }
            group.wait()
            _ = try? operation.wait()
            while seen.states.count < 4 { usleep(50) }
            #expect(seen.states == [.completed, .completed, .completed, .completed])
        }
    }

    @Test func boundedWaitReportsWhetherTheOperationFinishedInTime() {
        let finishing = ExpertLoadOperation()
        finishing.markInFlight()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(2)) {
            finishing.finish(.success(()))
        }
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 500_000_000
        #expect(finishing.wait(untilNanos: deadline))
        #expect(finishing.state == .completed)

        let stalled = ExpertLoadOperation()
        stalled.markInFlight()
        let soon = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 3_000_000
        #expect(!stalled.wait(untilNanos: soon))
        #expect(stalled.state == .inFlight)
        #expect(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) >= soon)
        stalled.finish(.success(()))
    }

    @Test func completionHookRunsAtOnceWhenTheOperationIsAlreadyTerminal() {
        struct ExpectedFailure: Error {}
        let operation = ExpertLoadOperation()
        operation.markInFlight()
        operation.finish(.failure(ExpectedFailure()))
        let seen = HookRecorder()
        operation.onCompletion { seen.record(operation.state) }
        #expect(seen.states == [.failed])
    }

    @Test func sharedEventDoesNotAdvancePastOutOfOrderBatch() throws {
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let firstToken = try coordinator.reserve()
        let secondToken = try coordinator.reserve()
        let first = ExpertLoadOperation(completionToken: firstToken,
                                        eventCoordinator: coordinator)
        let second = ExpertLoadOperation(completionToken: secondToken,
                                         eventCoordinator: coordinator)

        second.finish(.success(()))
        #expect(secondToken.status.contents().advanced(by: secondToken.statusOffset)
            .load(as: UInt32.self) == 1)
        #expect(secondToken.event.signaledValue == 0)

        first.finish(.success(()))
        #expect(firstToken.event.signaledValue == secondToken.value)
        #expect(firstToken.value != secondToken.value)
    }

    @Test func failedBatchSignalsTerminalStatusInsteadOfDeadlocking() throws {
        struct ExpectedFailure: Error {}
        let context = try MetalContext()
        let coordinator = try #require(ExpertIOEventCoordinator(device: context.device))
        let token = try coordinator.reserve()
        let operation = ExpertLoadOperation(completionToken: token,
                                            eventCoordinator: coordinator)

        operation.finish(.failure(ExpectedFailure()))

        #expect(token.status.contents().advanced(by: token.statusOffset)
            .load(as: UInt32.self) == 2)
        #expect(token.event.signaledValue == token.value)
        #expect(throws: ExpectedFailure.self) { try operation.wait() }
    }
}

private final class HookRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [ExpertLoadOperationState] = []
    var states: [ExpertLoadOperationState] { lock.withLock { recorded } }
    func record(_ state: ExpertLoadOperationState) { lock.withLock { recorded.append(state) } }
}
