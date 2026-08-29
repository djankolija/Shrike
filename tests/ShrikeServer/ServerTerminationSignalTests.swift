import Darwin
import Foundation
import Testing
@testable import ShrikeServerCore

@Suite("Server termination signals", .serialized)
struct ServerTerminationSignalTests {
    /// unchecked-invariant: `value` is guarded by `lock`; set from the signal
    /// source's handler thread, read from the test's polling loop.
    private final class ForcedExitFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func set() { lock.withLock { value = true } }
        var isSet: Bool { lock.withLock { value } }
    }

    @Test func dispatchSignalCrossesIntoAsyncCodeWithoutExecutorTrap() async {
        let signals = ServerTerminationSignals([SIGUSR1])
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)

        #expect(await waiter.value == SIGUSR1)
        await signals.cancel()
    }

    /// The second kill comes only after the first signal is consumed:
    /// back-to-back kills can coalesce into one handler invocation, and a
    /// non-coalesced pair used to exit(1) the whole test-runner process.
    @Test func secondSignalDeliveryForcesExitAfterTheFirstIsKept() async {
        let exited = ForcedExitFlag()
        let signals = ServerTerminationSignals([SIGUSR1]) { exited.set() }
        let waiter = Task {
            await signals.wait()
        }

        kill(getpid(), SIGUSR1)
        #expect(await waiter.value == SIGUSR1)

        kill(getpid(), SIGUSR1)
        for _ in 0..<400 where !exited.isSet {
            try? await Task.sleep(for: .milliseconds(5))
        }
        #expect(exited.isSet)
        await signals.cancel()
    }
}
