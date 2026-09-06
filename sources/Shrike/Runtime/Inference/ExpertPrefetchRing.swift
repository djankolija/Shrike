import Foundation
import Metal

/// Lifetime totals of the predictive prefetch ring, printed on the runner
/// line: `deferred` batches were issued from a demand batch's completion,
/// `overlapped` demand batches were submitted with a ring read in flight,
/// `late` predictions were still in flight when the exact route asked for
/// them, `refused` predictions found no budget or slot.
public struct ExpertPrefetchStatistics: Sendable, Equatable {
    public var issued: UInt64 = 0
    public var adopted: UInt64 = 0
    public var reclaimedUnadopted: UInt64 = 0
    public var deferred: UInt64 = 0
    public var overlapped: UInt64 = 0
    public var late: UInt64 = 0
    public var refused: UInt64 = 0
    public var hookFailures: UInt64 = 0
    public var beginNanos: UInt64 = 0

    public init() {}
}

/// A fixed raw-byte staging ring for v4.3 predictive routed-expert reads.
///
/// Slots are intentionally outside the authoritative expert cache. A completed
/// entry becomes a cache resident only when the exact router selects it and the
/// ordinary cache planner adopts its bytes. Failed or incorrect predictions
/// are simply discarded without changing cache mappings or slot generations.
/// At most `inFlightBudget` reads are in flight across all layers (v15 step
/// zero: a read still in flight shares the drive with the next demand read).
/// unchecked-invariant: slot ownership, operation association and the
/// statistics are guarded by `lock`; backing buffers are retained while an
/// operation can write; `begin` may run on a storage thread and on the decode
/// thread at once.
final class ExpertPrefetchRing: @unchecked Sendable {
    typealias Issue = (_ experts: [Int], _ buffers: [MTLBuffer]) throws -> ExpertLoadOperation

    private struct Slot {
        let buffer: MTLBuffer
        var layer = -1
        var expert = -1
        var operation: ExpertLoadOperation?
        /// Handed to a plan by `readyBuffers` and not yet consumed: the decode
        /// thread may still be copying its bytes, so no `begin` on another
        /// thread may reclaim it.
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

    init(device: MTLDevice, expertStride: Int, slotCount: Int, inFlightBudget: Int) throws {
        guard expertStride > 0, slotCount > 0, inFlightBudget > 0 else {
            throw ModelError.internalInconsistency(detail: "invalid prefetch ring geometry")
        }
        var allocated: [Slot] = []
        allocated.reserveCapacity(slotCount)
        for index in 0..<slotCount {
            guard let buffer = device.makeBuffer(length: expertStride,
                                                 options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = "decode.prefetch.\(index)"
            allocated.append(Slot(buffer: buffer))
        }
        slots = allocated
        self.inFlightBudget = inFlightBudget
    }

    /// Begins missing next-layer reads in unused staging slots, best-scored
    /// first, within the in-flight budget. Already queued predictions are
    /// deduplicated; finished entries are reclaimed only when they have not
    /// been selected by a later exact route.
    func begin(layer: Int, experts: [Int], resident: Set<Int>,
               deferred: Bool = false, issue: Issue) throws {
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        defer {
            let elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            lock.withLock { stats.beginNanos &+= elapsed }
        }
        lock.lock()
        reclaimTerminalSlotsUnlocked(exceptLayer: layer)
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
        let buffers = selectedSlots.map { slots[$0].buffer }
        for (slot, expert) in zip(selectedSlots, selectedExperts) {
            slots[slot].layer = layer
            slots[slot].expert = expert
            slots[slot].operation = nil
        }
        lock.unlock()

        do {
            let operation = try issue(selectedExperts, buffers)
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
            lock.unlock()
            throw error
        }
    }

    /// Completed raw bytes for one exact route, leased until `consume`.
    /// In-flight reads are not awaited: a demand miss remains authoritative
    /// and may start immediately.
    func readyBuffers(layer: Int, experts: [Int]) -> [Int: MTLBuffer] {
        let requested = Set(experts)
        lock.lock()
        defer { lock.unlock() }
        var result: [Int: MTLBuffer] = [:]
        for index in slots.indices where slots[index].layer == layer
            && requested.contains(slots[index].expert) {
            if slots[index].operation?.state == .completed {
                result[slots[index].expert] = slots[index].buffer
                slots[index].leased = true
            } else if slots[index].isInFlight {
                stats.late &+= 1
            }
        }
        return result
    }

    func consume(layer: Int, experts: Set<Int>) {
        lock.lock()
        defer { lock.unlock() }
        for index in slots.indices where slots[index].layer == layer
            && experts.contains(slots[index].expert) {
            slots[index].layer = -1
            slots[index].expert = -1
            slots[index].operation = nil
            slots[index].leased = false
            stats.adopted &+= 1
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

    private func reclaimTerminalSlotsUnlocked(exceptLayer: Int) {
        for index in slots.indices where slots[index].layer != exceptLayer && !slots[index].leased {
            switch slots[index].operation?.state {
            case .completed, .failed:
                if slots[index].operation?.state == .completed {
                    stats.reclaimedUnadopted &+= 1
                }
                slots[index].layer = -1
                slots[index].expert = -1
                slots[index].operation = nil
            case .none, .submitted, .inFlight:
                break
            }
        }
    }
}
