import Foundation

/// Lifetime totals of the predictive prefetch ring, printed on the runner
/// line: `adopted` landings a plan wanted (swapped into its pool, or a hit
/// the pool already held), `reclaimed` completed landings no plan wanted,
/// `deferred` batches issued from a demand batch's completion, `overlapped`
/// demand batches submitted with a ring read in flight, `late` predictions
/// still in flight past the join bound when the exact route asked for them,
/// `refused` predictions that found no budget or cell or an expert the pool
/// already held, `joined` predictions in flight that finished inside the join
/// bound, `failed` reads reclaimed failed.
public struct ExpertPrefetchStatistics: Sendable, Equatable {
    public var issued: UInt64 = 0
    public var adopted: UInt64 = 0
    public var reclaimed: UInt64 = 0
    public var deferred: UInt64 = 0
    public var overlapped: UInt64 = 0
    public var late: UInt64 = 0
    public var joined: UInt64 = 0
    public var refused: UInt64 = 0
    public var failed: UInt64 = 0
    public var hookFailures: UInt64 = 0
    public var beginNanos: UInt64 = 0

    public init() {}
}

/// The predictive routed-expert reads, landing in cells of the shared expert
/// arena the ring owns, where the target layer's classifier can hit them. A
/// landing the layer's plan wants is swapped into that layer's pool and the
/// ring takes the freed cell; one the plan does not want is dropped from the
/// layer's table when the ring reclaims the cell. At most `inFlightBudget`
/// reads are in flight across all layers (v15 step zero: a read still in
/// flight shares the drive with the next demand read).
/// unchecked-invariant: slot ownership, operation association and the
/// statistics are guarded by `lock`; `begin` may run on a storage thread and
/// on the decode thread at once; `drop` runs under `lock` and takes the
/// layer's cache lock inside it, and nothing takes the two the other way.
final class ExpertPrefetchRing: @unchecked Sendable {
    typealias Issue = (_ experts: [Int], _ cells: [Int]) throws -> ExpertLoadOperation
    typealias Drop = (_ layer: Int, _ expert: Int, _ cell: Int) -> Void

    /// How long `readyCells` waits for a claimed slot's operation to attach
    /// (the window between the ring's claim and the issue's return).
    static let attachSpinNanos: UInt64 = 5_000_000

    private struct Slot {
        var cell: Int
        var layer = -1
        var expert = -1
        var operation: ExpertLoadOperation?
        /// Handed to a plan by `readyCells` and not yet consumed: the plan may
        /// still swap it, so no `begin` on another thread may reclaim it.
        var leased = false

        /// A claimed slot whose operation is not yet attached is in flight:
        /// its submission is between the claim and the attach.
        var isInFlight: Bool {
            guard expert >= 0 else { return false }
            switch operation?.state {
            case .none, .submitted, .inFlight: return true
            case .completed, .failed: return false
            }
        }
    }

    private let lock = NSLock()
    private var slots: [Slot]
    private let inFlightBudget: Int
    private let drop: Drop
    private var stats = ExpertPrefetchStatistics()

    var statistics: ExpertPrefetchStatistics {
        lock.withLock { stats }
    }

    var inFlightCount: Int {
        lock.withLock { inFlightCountUnlocked }
    }

    private var inFlightCountUnlocked: Int {
        slots.reduce(0) { $0 + ($1.isInFlight ? 1 : 0) }
    }

    init(cells: [Int], inFlightBudget: Int, drop: @escaping Drop) throws {
        guard !cells.isEmpty, inFlightBudget > 0, Set(cells).count == cells.count else {
            throw ModelError.internalInconsistency(detail: "invalid prefetch ring geometry")
        }
        slots = cells.map { Slot(cell: $0) }
        self.inFlightBudget = inFlightBudget
        self.drop = drop
    }

    /// Begins missing reads for `layer` in free cells, best-scored first,
    /// within the in-flight budget, issued while the decode stands at `from`
    /// (the layer whose plan or demand completion issues them). Finished
    /// entries of the layers between, whose plans are still to come, are kept;
    /// the rest are reclaimed. Already queued predictions are deduplicated; a
    /// batch the streamer refuses (the pool already holds an expert) is
    /// counted refused, not thrown.
    func begin(layer: Int, from issuing: Int? = nil, experts: [Int], resident: Set<Int>,
               deferred: Bool = false, issue: Issue) throws {
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        defer {
            let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            lock.withLock { stats.beginNanos &+= elapsed }
        }
        lock.lock()
        reclaimTerminalSlotsUnlocked(after: issuing ?? layer - 1, upTo: layer)
        let active = Set(slots.compactMap { slot in
            slot.layer == layer && slot.expert >= 0 ? slot.expert : nil
        })
        var seen: Set<Int> = []
        let wanted = experts.filter {
            !resident.contains($0) && !active.contains($0) && seen.insert($0).inserted
        }
        let free = slots.indices.filter { slots[$0].expert < 0 }
        let budgetLeft = max(0, inFlightBudget - inFlightCountUnlocked)
        let count = min(wanted.count, free.count, budgetLeft)
        stats.refused &+= UInt64(wanted.count - count)
        guard count > 0 else {
            lock.unlock()
            return
        }
        let selectedSlots = Array(free.prefix(count))
        let selectedExperts = Array(wanted.prefix(count))
        let cells = selectedSlots.map { slots[$0].cell }
        for (slot, expert) in zip(selectedSlots, selectedExperts) {
            slots[slot].layer = layer
            slots[slot].expert = expert
            slots[slot].operation = nil
        }
        lock.unlock()

        do {
            let operation = try issue(selectedExperts, cells)
            lock.lock()
            for slot in selectedSlots { slots[slot].operation = operation }
            stats.issued &+= UInt64(count)
            if deferred { stats.deferred &+= 1 }
            lock.unlock()
        } catch {
            lock.lock()
            for slot in selectedSlots {
                slots[slot].layer = -1
                slots[slot].expert = -1
                slots[slot].operation = nil
            }
            if error is PrefetchClaimRefused {
                stats.refused &+= UInt64(count)
                lock.unlock()
                return
            }
            lock.unlock()
            throw error
        }
    }

    /// The landed predictions one exact route wants, by cell, leased until
    /// `consume`. A wanted prediction still in flight is awaited to completion,
    /// since its cell is claimed and a demand read would only duplicate it;
    /// the wait counts `joined` inside `joinNanos` and `late` past it. A claim
    /// whose issue has not returned yet is given `attachSpinNanos` to attach.
    func readyCells(layer: Int, experts: [Int], joinNanos: UInt64 = 0) -> [Int: Int] {
        let requested = Set(experts)
        let deadline = joinNanos > 0 ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + joinNanos : 0
        lock.lock()
        defer { lock.unlock() }
        var result: [Int: Int] = [:]
        // One operation carries a batch; every prediction it carried counts.
        var joinedOperations: Set<ObjectIdentifier> = []
        var lateOperations: Set<ObjectIdentifier> = []
        for index in slots.indices where slots[index].layer == layer
            && requested.contains(slots[index].expert) {
            if slots[index].isInFlight {
                guard let operation = attachedOperationUnlocked(index) else {
                    stats.late &+= 1
                    continue
                }
                lock.unlock()
                let joined = deadline > 0 && operation.wait(untilNanos: deadline)
                if !joined { _ = try? operation.wait() }
                lock.lock()
                if joined {
                    joinedOperations.insert(ObjectIdentifier(operation))
                } else {
                    lateOperations.insert(ObjectIdentifier(operation))
                }
                guard slots[index].layer == layer, requested.contains(slots[index].expert) else {
                    continue
                }
            }
            guard let operation = slots[index].operation else { continue }
            if joinedOperations.contains(ObjectIdentifier(operation)) {
                stats.joined &+= 1
            } else if lateOperations.contains(ObjectIdentifier(operation)) {
                stats.late &+= 1
            }
            if operation.state == .completed {
                result[slots[index].expert] = slots[index].cell
                slots[index].leased = true
            }
        }
        return result
    }

    /// The slot's operation, waiting briefly for an issue in progress on
    /// another thread to attach it; nil when it does not.
    private func attachedOperationUnlocked(_ index: Int) -> ExpertLoadOperation? {
        if let operation = slots[index].operation { return operation }
        let layer = slots[index].layer, expert = slots[index].expert
        let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + Self.attachSpinNanos
        while slots[index].operation == nil, clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline {
            lock.unlock()
            usleep(50)
            lock.lock()
            guard slots[index].layer == layer, slots[index].expert == expert else { return nil }
        }
        return slots[index].operation
    }

    /// Frees the entries a plan resolved; an entry whose expert the pool
    /// swapped in takes the cell the pool freed, any other is dropped from
    /// the layer's table before its cell can be read into again.
    func consume(layer: Int, experts: Set<Int>, freedCells: [Int: Int]) {
        lock.lock()
        defer { lock.unlock() }
        for index in slots.indices where slots[index].layer == layer
            && experts.contains(slots[index].expert) {
            if let freed = freedCells[slots[index].expert] {
                slots[index].cell = freed
            } else {
                drop(layer, slots[index].expert, slots[index].cell)
            }
            slots[index].layer = -1
            slots[index].expert = -1
            slots[index].operation = nil
            slots[index].leased = false
            stats.adopted &+= 1
        }
    }

    func completionNanos(layer: Int, experts: Set<Int>) -> [Int: UInt64] {
        lock.withLock {
            var stamps: [Int: UInt64] = [:]
            for index in slots.indices where slots[index].layer == layer
                && experts.contains(slots[index].expert) {
                if let operation = slots[index].operation, operation.state == .completed {
                    stamps[slots[index].expert] = operation.completedNanos
                }
            }
            return stamps
        }
    }

    func unlease(layer: Int, experts: Set<Int>) {
        lock.withLock {
            for index in slots.indices where slots[index].layer == layer
                && experts.contains(slots[index].expert) {
                slots[index].leased = false
            }
        }
    }

    func noteDemandSubmission() {
        lock.withLock {
            if inFlightCountUnlocked > 0 { stats.overlapped &+= 1 }
        }
    }

    /// Returns true on the first failure, so the caller can log it once.
    func noteHookFailure() -> Bool {
        lock.withLock {
            stats.hookFailures &+= 1
            return stats.hookFailures == 1
        }
    }

    /// A completed landing no plan took is dropped from its layer's table
    /// before its cell can be read into again; a failed one was dropped by
    /// the read's own failure path. Entries for the layers whose plans are
    /// still to come, `issuing` exclusive to `target` inclusive, are kept.
    private func reclaimTerminalSlotsUnlocked(after issuing: Int, upTo target: Int) {
        for index in slots.indices where !slots[index].leased
            && !(slots[index].layer > issuing && slots[index].layer <= target) {
            switch slots[index].operation?.state {
            case .completed:
                drop(slots[index].layer, slots[index].expert, slots[index].cell)
                stats.reclaimed &+= 1
                slots[index].layer = -1
                slots[index].expert = -1
                slots[index].operation = nil
            case .failed:
                stats.failed &+= 1
                slots[index].layer = -1
                slots[index].expert = -1
                slots[index].operation = nil
            case .none, .submitted, .inFlight:
                break
            }
        }
    }
}
