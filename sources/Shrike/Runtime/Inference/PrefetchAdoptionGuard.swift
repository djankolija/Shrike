import Foundation

/// Owns an adoption's cleanup from the plan onward, so a throw anywhere before
/// the pending command takes the transfer still empties the reserved slots and
/// returns the ring's buffers: `abandon` runs `fail` and either the transfer's
/// release or the direct one exactly once; `commit` hands both to the command.
final class PrefetchAdoptionGuard {
    private let fail: () -> Void
    private var release: (() -> Void)?
    private(set) var transfer: PrefetchAdoptionTransfer?
    private var settled = false

    init(fail: @escaping () -> Void, release: @escaping () -> Void) {
        self.fail = fail
        self.release = release
    }

    func attach(_ transfer: PrefetchAdoptionTransfer) {
        precondition(self.transfer == nil, "an adoption transfer was attached twice")
        self.transfer = transfer
        release = nil
    }

    func commit() {
        settled = true
    }

    func abandon() {
        guard !settled else { return }
        settled = true
        fail()
        if let transfer {
            transfer.release(adopted: false)
        } else {
            release?()
        }
        release = nil
    }
}
