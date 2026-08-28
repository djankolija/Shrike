import Foundation
import Testing
@testable import NVMAIServerCore

private actor TestGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

private final class OrderRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _values: [String] = []

    var values: [String] { lock.withLock { _values } }

    func append(_ value: String) { lock.withLock { _values.append(value) } }
}

@Suite("Server coordinator")
struct ServerCoordinatorTests {
    @Test func boundsFIFOAndRecoversAfterCancellation() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1)
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }

    @Test func admitsQueuedSameModelRequestsBeforeASwap() async throws {
        let coordinator = ServerCoordinator(queueLimit: 4)
        let gate = TestGate()
        let order = OrderRecorder()
        let active = Task {
            try await coordinator.run(modelID: "alpha") {
                await gate.wait()
                order.append("alpha-1")
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let swap = Task {
            try await coordinator.run(modelID: "beta") { order.append("beta-1"); return 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        let same = Task {
            try await coordinator.run(modelID: "alpha") { order.append("alpha-2"); return 3 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 2 }

        await gate.open()
        _ = try await active.value
        _ = try await swap.value
        _ = try await same.value
        // The queued alpha request runs before the earlier-queued beta one:
        // one swap per batch, not one per request.
        #expect(order.values == ["alpha-1", "alpha-2", "beta-1"])
    }

    @Test func fallsBackToFIFOWhenNoWaiterMatches() async throws {
        let coordinator = ServerCoordinator(queueLimit: 4)
        let gate = TestGate()
        let order = OrderRecorder()
        let active = Task {
            try await coordinator.run(modelID: "alpha") {
                await gate.wait()
                order.append("alpha-1")
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let beta = Task {
            try await coordinator.run(modelID: "beta") { order.append("beta-1"); return 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        let gamma = Task {
            try await coordinator.run(modelID: "gamma") { order.append("gamma-1"); return 3 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 2 }

        await gate.open()
        _ = try await active.value
        _ = try await beta.value
        _ = try await gamma.value
        #expect(order.values == ["alpha-1", "beta-1", "gamma-1"])
    }

    @Test func affinitySkipsUntaggedWaiters() async throws {
        let coordinator = ServerCoordinator(queueLimit: 4)
        let gate = TestGate()
        let order = OrderRecorder()
        let active = Task {
            try await coordinator.run(modelID: "alpha") {
                await gate.wait()
                order.append("alpha-1")
                return 1
            }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.isActive }
        let untagged = Task {
            try await coordinator.run { order.append("untagged"); return 2 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 1 }
        let same = Task {
            try await coordinator.run(modelID: "alpha") { order.append("alpha-2"); return 3 }
        }
        try await waitUntil(timeout: .seconds(5)) { await coordinator.queuedCount == 2 }

        await gate.open()
        _ = try await active.value
        _ = try await untagged.value
        _ = try await same.value
        #expect(order.values == ["alpha-1", "alpha-2", "untagged"])
    }

    /// Bounded poll so a state that never reaches `condition` fails fast
    /// instead of spinning forever.
    private func waitUntil(timeout: Duration,
                           _ condition: () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while await !condition() {
            guard ContinuousClock.now < deadline else {
                throw CoordinatorTimeout()
            }
            await Task.yield()
        }
    }

    private struct CoordinatorTimeout: Error {}
}
