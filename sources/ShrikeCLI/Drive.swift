import Darwin
import Foundation

/// unchecked-invariant: a hand-off box between the top-level task and the
/// blocking signal wait, published exactly once before the semaphore is
/// signalled and read only after -- the semaphore is the ordering.
final class RunBox: @unchecked Sendable {
    var code: Int32 = 0
    var task: Task<Void, Never>?
}

// Keep the cancellable task off the top-level executor so the blocking signal
// bridge cannot prevent it from starting.
func drive(_ args: Args) -> Int32 {
    let box = RunBox()
    let sem = DispatchSemaphore(value: 0)
    box.task = Task {
        let result = await run(args: args)
        box.code = result.exitCode
        sem.signal()
    }

    signal(SIGINT, SIG_IGN)
    let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    sigintSource.setEventHandler { box.task?.cancel() }
    sigintSource.resume()

    sem.wait()
    return box.code
}
