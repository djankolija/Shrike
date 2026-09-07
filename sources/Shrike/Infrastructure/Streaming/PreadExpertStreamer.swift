import Darwin
import Foundation
import Metal
import ShrikeKernelsC

public struct ExpertCachePlan: Sendable, Equatable {
    /// K11: the layer the plan's pread offsets are computed against.
    ///
    /// NOTE: the streamer is bound to ONE layer file at construction, and
    /// `StreamLayout.expertOffset(layer: 0, ...)` is the branch that consults
    /// that file's per-layer `expertOffsets` table — so 0 is the correct value
    /// for the current per-layer design (passing the real layer would select
    /// the dense cross-layer formula and mis-offset). Callers must only pass a
    /// nonzero layer if the streamer ever serves a multi-layer file.
    public let layer: Int
    public let experts: [Int]
    public let assignedSlots: [Int]
    /// Slot incarnation captured when this plan reserved or hit each slot.
    /// A command may use the slot only while this generation still matches.
    public let assignedGenerations: [UInt64]
    public let misses: [Int]
    public let hits: Int
    /// Indices into `experts` counted as hits because a landed prediction was
    /// swapped in for them after the GPU's residency classification had seen
    /// them as misses; the fixup computes them from their cell.
    public let adopted: [Int]
    /// Per landed expert, the pool cell its swap handed back to the ring.
    public let freedCells: [Int: Int]

    public init(experts: [Int], assignedSlots: [Int],
                assignedGenerations: [UInt64], misses: [Int], hits: Int,
                layer: Int = 0, adopted: [Int] = [], freedCells: [Int: Int] = [:]) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.assignedGenerations = assignedGenerations
        self.misses = misses
        self.hits = hits
        self.layer = layer
        self.adopted = adopted
        self.freedCells = freedCells
    }
}

/// A prediction the pool holds, is loading, or has landed already: the ring
/// counts it refused, nothing is read.
public struct PrefetchClaimRefused: Error, Sendable {
    public let expert: Int
    public let cell: Int
}

public struct ExpertStreamingStatistics: Sendable, Equatable {
    public let plans: UInt64
    public let requestedExperts: UInt64
    public let hits: UInt64
    public let misses: UInt64
    public let bytesRead: UInt64
    public let readOperations: UInt64
    public let evictions: UInt64
    public let reloads: UInt64
    public let loadBatches: UInt64
    public let totalLoadNanos: UInt64
    /// The portion of `totalLoadNanos` spent inside the read calls
    /// themselves; the remainder is scheduler handoff and slot bookkeeping.
    public let fetchNanos: UInt64
    public let maximumLoadNanos: UInt64
    public let latencyHistogram: [UInt64]
    public let residentSlots: Int
    public let loadingSlots: Int
    public let pinnedSlots: Int
    public let peakLoadingSlots: Int

    public var hitRate: Double {
        requestedExperts == 0 ? 0 : Double(hits) / Double(requestedExperts)
    }

    /// An upper-bound estimate from a fixed, bounded power-of-two histogram.
    public func loadLatencyPercentile(_ percentile: Double) -> UInt64 {
        guard loadBatches > 0 else { return 0 }
        let clamped = min(1, max(0, percentile))
        let rank = max(UInt64(1), UInt64(ceil(clamped * Double(loadBatches))))
        var cumulative: UInt64 = 0
        for (index, count) in latencyHistogram.enumerated() {
            cumulative &+= count
            if cumulative >= rank {
                return PreadExpertStreamer.latencyBucketUpperBound(index: index)
            }
        }
        return UInt64.max
    }

    public static let zero = ExpertStreamingStatistics(
        plans: 0, requestedExperts: 0, hits: 0, misses: 0,
        bytesRead: 0, readOperations: 0, evictions: 0, reloads: 0,
        loadBatches: 0, totalLoadNanos: 0, fetchNanos: 0, maximumLoadNanos: 0,
        latencyHistogram: [UInt64](repeating: 0, count: 17),
        residentSlots: 0, loadingSlots: 0, pinnedSlots: 0, peakLoadingSlots: 0)

    func adding(_ other: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        ExpertStreamingStatistics(
            plans: plans &+ other.plans,
            requestedExperts: requestedExperts &+ other.requestedExperts,
            hits: hits &+ other.hits,
            misses: misses &+ other.misses,
            bytesRead: bytesRead &+ other.bytesRead,
            readOperations: readOperations &+ other.readOperations,
            evictions: evictions &+ other.evictions,
            reloads: reloads &+ other.reloads,
            loadBatches: loadBatches &+ other.loadBatches,
            totalLoadNanos: totalLoadNanos &+ other.totalLoadNanos,
            fetchNanos: fetchNanos &+ other.fetchNanos,
            maximumLoadNanos: max(maximumLoadNanos, other.maximumLoadNanos),
            latencyHistogram: zip(latencyHistogram, other.latencyHistogram)
                .map { $0 &+ $1 },
            residentSlots: residentSlots + other.residentSlots,
            loadingSlots: loadingSlots + other.loadingSlots,
            pinnedSlots: pinnedSlots + other.pinnedSlots,
            peakLoadingSlots: max(peakLoadingSlots, other.peakLoadingSlots))
    }

    public func subtracting(_ baseline: ExpertStreamingStatistics) -> ExpertStreamingStatistics {
        func delta(_ value: UInt64, _ base: UInt64) -> UInt64 {
            value >= base ? value - base : 0
        }
        return ExpertStreamingStatistics(
            plans: delta(plans, baseline.plans),
            requestedExperts: delta(requestedExperts, baseline.requestedExperts),
            hits: delta(hits, baseline.hits),
            misses: delta(misses, baseline.misses),
            bytesRead: delta(bytesRead, baseline.bytesRead),
            readOperations: delta(readOperations, baseline.readOperations),
            evictions: delta(evictions, baseline.evictions),
            reloads: delta(reloads, baseline.reloads),
            loadBatches: delta(loadBatches, baseline.loadBatches),
            totalLoadNanos: delta(totalLoadNanos, baseline.totalLoadNanos),
            fetchNanos: delta(fetchNanos, baseline.fetchNanos),
            maximumLoadNanos: maximumLoadNanos,
            latencyHistogram: zip(latencyHistogram, baseline.latencyHistogram)
                .map { delta($0, $1) },
            residentSlots: residentSlots,
            loadingSlots: loadingSlots,
            pinnedSlots: pinnedSlots,
            peakLoadingSlots: peakLoadingSlots)
    }
}

enum BoundedReaderConfiguration {
    static let defaultThreads = 4
    /// Two published batches, the v13 T2 winner.
    static let defaultBatchDepth = 2
}

/// unchecked-invariant: the destinations are `ExpertCellArena` cells (v16),
/// alive for the model's lifetime, so the raw pointers outlive this request.
private final class PrefetchDestinations: @unchecked Sendable {
    let values: [UnsafeMutableRawPointer]

    init(_ values: [UnsafeMutableRawPointer]) {
        self.values = values
    }
}

/// SSD-backed routed-expert streamer with a fixed per-layer slot cache.
/// unchecked-invariant: the expert cache bookkeeping is guarded by `cacheLock`.
/// Slot state is published only after a read finishes, so concurrent planners
/// never treat partial bytes as resident.
public final class PreadExpertStreamer: @unchecked Sendable {
    public let layout: StreamLayout
    public let slotCount: Int
    public let poolSlotStride: Int

    private let fd: Int32

    /// Bounded-footprint reader.
    ///
    /// Opens its own F_NOCACHE descriptors so expert reads never enter the unified
    /// buffer cache. That makes the slot budget the machine's true footprint,
    /// which is the whole point of streaming a 35B model on 24 GB.
    ///
    /// It is not free. Measured against the page-cache path it costs 15-30% of
    /// decode throughput, because every miss becomes a real device read instead of
    /// a cache hit -- and the cost is worst exactly where the hit rate is lowest
    /// (-40% at the 8-slot floor against -18% at 16 slots).
    ///
    /// It is the default anyway. The page-cache path is faster only by borrowing
    /// memory it never declares: process RSS looks smaller while the OS holds the
    /// difference, so "a 35B model in 1 GB" stops being true. A footprint you can
    /// account for is the product; throughput is what is being traded for it.
    private let boundedReader: ParallelExpertReader
    private let eventCoordinator: ExpertIOEventCoordinator?
    private var slotPointers: [UnsafeMutableRawPointer]
    private var slotBuffers: [MTLBuffer]
    private var slotBufferOffsets: [UInt64]
    private let arena: ExpertCellArena
    private let residencyTable: MTLBuffer

    private struct Landing {
        let cell: Int
        var resident: Bool
        let generation: UInt64
    }

    private var landings: [Int: Landing] = [:]
    private var landingGeneration: UInt64 = 0

    private var nextSlot = 0

    private enum SlotState: UInt8 {
        case empty
        case loading
        case resident
    }

    private var reservedSlots: [Bool]
    private var victimSlotsScratch: [Int]
    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var slotState: [SlotState]
    private var slotGeneration: [UInt64]
    private var slotPinCount: [Int]
    private var expertUseCount: [Int]
    private var expertLoadCount: [Int]
    private var useClock = 0
    private var statisticsPlans: UInt64 = 0
    private var statisticsRequestedExperts: UInt64 = 0
    private var statisticsHits: UInt64 = 0
    private var statisticsMisses: UInt64 = 0
    private var statisticsBytesRead: UInt64 = 0
    private var statisticsReadOperations: UInt64 = 0
    private var statisticsEvictions: UInt64 = 0
    private var statisticsReloads: UInt64 = 0
    private var statisticsLoadBatches: UInt64 = 0
    private var statisticsTotalLoadNanos: UInt64 = 0
    private var statisticsFetchNanos: UInt64 = 0
    private var statisticsMaximumLoadNanos: UInt64 = 0
    private var statisticsLatencyHistogram = [UInt64](repeating: 0, count: 17)
    private var statisticsPeakLoadingSlots = 0
    private let cacheLock = NSLock()

    public init(layout: StreamLayout,
                device: MTLDevice,
                slotCount: Int,
                eventCoordinator: ExpertIOEventCoordinator? = nil,
                arena: ExpertCellArena? = nil,
                cellRange: Range<Int>? = nil) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.reservedSlots = Array(repeating: false, count: slotCount)
        self.victimSlotsScratch = Array(repeating: -1, count: slotCount)
        self.slotCount = slotCount
        self.eventCoordinator = eventCoordinator
        let pageSize = Int(getpagesize())

        let openedFD = open(layout.path, O_RDONLY)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD

        var fileStats = stat()
        // K9: fstat failure must not silently skip size validation — a
        // truncated file would then be read out of bounds by pread.
        guard fstat(openedFD, &fileStats) == 0 else {
            let statErrno = errno
            close(openedFD)
            throw ModelError.posixFailed(call: "fstat(\(layout.path))", errno: statErrno)
        }
        let required = layout.streamOffset + layout.streamSize
        if UInt64(fileStats.st_size) < required {
            close(openedFD)
            throw StreamerError.sizeMismatch(
                expected: required,
                actual: UInt64(fileStats.st_size))
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        // The pool base retains the validated 2 MiB allocation alignment.
        // Individual offsets need only VM-page alignment for pread and Metal;
        // rounding every slot to 2 MiB inflated the 8-bit pool by several GiB.
        self.poolSlotStride = allocationSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        var bufferOffsets: [UInt64] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)
        bufferOffsets.reserveCapacity(slotCount)
        guard let residencyTable = device.makeBuffer(
            length: max(1, layout.expertsPerLayer)
                * MemoryLayout<ExpertResidencyEntry>.stride,
            options: .storageModeShared)
        else {
            close(openedFD)
            throw StreamerError.bufferWrapFailed
        }
        self.residencyTable = residencyTable
        let residencyEntries = residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: max(1, layout.expertsPerLayer))
        for expert in 0..<max(1, layout.expertsPerLayer) {
            residencyEntries[expert] = ExpertResidencyEntry()
        }

        let cells: ExpertCellArena
        let range: Range<Int>
        if let arena {
            guard arena.stride == poolSlotStride else {
                close(openedFD)
                throw ModelError.internalInconsistency(
                    detail: "expert cell arena stride \(arena.stride) differs from the pool's \(poolSlotStride)")
            }
            guard let cellRange, cellRange.count == slotCount,
                  cellRange.lowerBound >= 0, cellRange.upperBound <= arena.cellCount else {
                close(openedFD)
                throw ModelError.internalInconsistency(
                    detail: "expert cell range \(String(describing: cellRange)) does not fit \(slotCount) slots of a \(arena.cellCount)-cell arena")
            }
            cells = arena
            range = cellRange
        } else {
            do {
                cells = try ExpertCellArena(device: device, cellCount: slotCount, stride: poolSlotStride)
            } catch {
                close(openedFD)
                throw error
            }
            range = 0..<slotCount
        }
        self.arena = cells
        for cell in range {
            pointers.append(cells.pointer(cell: cell))
            buffers.append(cells.buffer)
            bufferOffsets.append(cells.offset(cell: cell))
        }

        do {
            self.boundedReader = try ParallelExpertReader(
                path: layout.path,
                expertStride: Int(layout.expertStride),
                threads: BoundedReaderConfiguration.defaultThreads,
                batchDepth: BoundedReaderConfiguration.defaultBatchDepth)
        } catch {
            close(openedFD)
            throw error
        }

        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotBufferOffsets = bufferOffsets
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.slotState = [SlotState](repeating: .empty, count: slotCount)
        self.slotGeneration = [UInt64](repeating: 0, count: slotCount)
        self.slotPinCount = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        self.expertLoadCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        // K12: slot selection and fill share one critical section so the
        // round-robin path never lands on a slot a concurrent plan reserved
        // (`loading`) and no fill can interleave with another pread.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        var candidate = nextSlot
        var scanned = 0
        while (slotState[candidate] == .loading || slotPinCount[candidate] > 0)
            && scanned < slotCount {
            candidate = (candidate + 1) % slotCount
            scanned += 1
        }
        guard scanned < slotCount else {
            throw ModelError.expertCacheUnplaceable(
                detail: "all \(slotCount) expert-cache slots are loading or pinned")
        }
        nextSlot = (candidate + 1) % slotCount
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: candidate)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        // K12: the pread fill and the slot bookkeeping share one critical
        // section so a concurrent plan/execute or another load cannot write
        // into this slot while the pread is in flight.
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard slotState[slot] != .loading, slotPinCount[slot] == 0 else {
            throw ModelError.expertCacheUnplaceable(
                detail: "expert-cache slot \(slot) is loading or pinned")
        }
        return try loadExpertUnlocked(layer: layer, expert: expert, slot: slot)
    }

    /// Fill `slot` with `expert` and update bookkeeping. Callers hold
    /// `cacheLock`.
    private func loadExpertUnlocked(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        slotGeneration[slot] &+= 1
        let previousExpert = slotExpert[slot]
        if previousExpert >= 0 {
            publishResidencyUnlocked(expert: previousExpert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: slotGeneration[slot])
        }
        slotExpert[slot] = expert
        slotState[slot] = .loading
        publishResidencyUnlocked(expert: expert,
                                 slot: slot,
                                 state: ExpertResidencyEntry.loading,
                                 generation: slotGeneration[slot])
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try readFull(
                into: slotPointers[slot],
                fileOffset: layout.streamOffset + regionOffset,
                count: Int(layout.expertStride))
            slotState[slot] = .resident
            publishResidencyUnlocked(expert: expert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.resident,
                                     generation: slotGeneration[slot])
            slotLastUse[slot] = useClock
            recordSuccessfulLoadsUnlocked(
                experts: [expert], elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started)
        } catch {
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(expert: expert,
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: slotGeneration[slot])
            throw error
        }
        return (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    /// `leasedLandings` are the experts whose landings the ring leased to
    /// this plan; only they are swapped in, so the cell exchange and the
    /// ring's lease are one transaction (any other planner reads the expert
    /// into the pool and the landing is dropped when the ring reclaims it).
    /// `gpuMissedExperts` is the residency classifier's view when it ran
    /// before this plan: a landed prediction it missed is swapped in all the
    /// same but reported `adopted`, so the fixup computes it.
    public func planExpertsCached(experts: [Int],
                                  layer: Int = 0,
                                  avoidingSlots: Set<Int> = [],
                                  protectedExperts: [Bool]? = nil,
                                  gpuMissedExperts: Set<Int>? = nil,
                                  leasedLandings: Set<Int> = []) throws
        -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(layer: layer,
                                             experts: experts,
                                             avoidingSlots: avoidingSlots,
                                             protectedExperts: protectedExperts,
                                             gpuMissedExperts: gpuMissedExperts,
                                             leasedLandings: leasedLandings) else {
            // K10: config-triggered placement failure (too few slots for the
            // requested expert set) is recoverable — throw instead of
            // crashing; the runner already handles thrown errors.
            throw ModelError.expertCacheUnplaceable(
                detail: "\(experts.count) experts do not fit in \(slotCount) cache slots (avoiding \(avoidingSlots.count) slots)")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            layer: Int = 0,
                                            avoidingSlots: Set<Int> = [],
                                            protectedExperts: [Bool]? = nil,
                                            gpuMissedExperts: Set<Int>? = nil,
                                            leasedLandings: Set<Int> = [])
        -> ExpertCachePlan? {
        makeExpertCachePlan(layer: layer, experts: experts, avoidingSlots: avoidingSlots,
                             protectedExperts: protectedExperts, gpuMissedExperts: gpuMissedExperts,
                             leasedLandings: leasedLandings)
    }

    private func makeExpertCachePlan(layer: Int,
                                     experts: [Int],
                                     avoidingSlots rawAvoidingSlots: consuming Set<Int>,
                                     protectedExperts: [Bool]?,
                                     gpuMissedExperts: Set<Int>?,
                                     leasedLandings: Set<Int>)
        -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")

        var avoidingSlots = Set<Int>(minimumCapacity: slotCount)
        for slot in rawAvoidingSlots {
            if slot >= 0 && slot < slotCount {
                avoidingSlots.insert(slot)
            }
        }

        cacheLock.lock()
        defer { cacheLock.unlock() }

        reservedSlots.withUnsafeMutableBufferPointer { buffer in
            buffer.update(repeating: false)
        }

        let clock = useClock + 1
        if statisticsPlans > 0,
           statisticsPlans.isMultiple(of: 1_024) {
            for i in 0..<expertUseCount.count {
                expertUseCount[i] >>= 1
            }
        }

        // Loading slots are not valid hits and cannot be reassigned.
        for slot in 0..<slotCount where slotState[slot] == .loading {
            reservedSlots[slot] = true
        }

        var assignedSlots = [Int](repeating: -1, count: experts.count)

        var missCount = 0
        for index in 0..<experts.count {
            for slot in 0..<slotCount where !reservedSlots[slot] && slotState[slot] == .resident && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reservedSlots[slot] = true
                break
            }
            if assignedSlots[index] == -1 { missCount &+= 1 }
        }

        for slot in avoidingSlots {
            reservedSlots[slot] = true
        }

        if missCount > 0 {
            let protectedSucceeded = protectedExperts != nil
                && selectVictimSlots(missCount: missCount, protectedExperts: protectedExperts)
            if !protectedSucceeded {
                guard selectVictimSlots(missCount: missCount) else { return nil }
            }
        }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        var misses: [Int] = []
        var landedExperts: [Int] = []
        var adoptedIndices: [Int] = []
        var freedCells: [Int: Int] = [:]
        var victimOffset = 0
        for index in 0..<experts.count where assignedSlots[index] == -1 {
            let slot = victimSlotsScratch[victimOffset]
            victimOffset += 1
            if slotState[slot] == .resident { statisticsEvictions &+= 1 }
            let previousExpert = slotExpert[slot]
            assignedSlots[index] = slot
            reservedSlots[slot] = true
            let nextGeneration = slotGeneration[slot] &+ 1
            if previousExpert >= 0 {
                publishResidencyUnlocked(expert: previousExpert,
                                         slot: slot,
                                         state: ExpertResidencyEntry.empty,
                                         generation: nextGeneration)
            }
            if leasedLandings.contains(experts[index]),
               let landing = landings[experts[index]], landing.resident {
                landings[experts[index]] = nil
                freedCells[experts[index]] = cellIndexUnlocked(slot)
                slotPointers[slot] = arena.pointer(cell: landing.cell)
                slotBufferOffsets[slot] = arena.offset(cell: landing.cell)
                slotGeneration[slot] = nextGeneration
                slotExpert[slot] = experts[index]
                slotLastUse[slot] = clock
                slotState[slot] = .resident
                // The same cell the classifier may have read, under the slot's generation.
                publishResidencyUnlocked(expert: experts[index],
                                         slot: slot,
                                         state: ExpertResidencyEntry.resident,
                                         generation: nextGeneration)
                landedExperts.append(experts[index])
                if gpuMissedExperts?.contains(experts[index]) == true {
                    adoptedIndices.append(index)
                }
                continue
            }
            slotGeneration[slot] = nextGeneration
            slotExpert[slot] = experts[index]
            slotLastUse[slot] = clock
            slotState[slot] = .loading
            publishResidencyUnlocked(expert: experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.loading,
                                     generation: slotGeneration[slot])
            misses.append(index)
        }

        recordPrefetchLandingsUnlocked(landedExperts)

        statisticsPlans &+= 1
        statisticsRequestedExperts &+= UInt64(experts.count)
        statisticsHits &+= UInt64(experts.count - misses.count)
        statisticsMisses &+= UInt64(misses.count)
        statisticsPeakLoadingSlots = max(
            statisticsPeakLoadingSlots,
            slotState.count(where: { $0 == .loading }))

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            assignedGenerations: assignedSlots.map { slotGeneration[$0] },
            misses: misses,
            hits: experts.count - misses.count,
            layer: layer,
            adopted: adoptedIndices,
            freedCells: freedCells)
    }

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        precondition(plan.assignedGenerations.count == plan.experts.count,
                     "expert cache plan generation count mismatch")

        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        var succeeded = false
        var fetchNanos: UInt64 = 0
        defer {
            finishPlanExecution(
                plan,
                succeeded: succeeded,
                elapsedNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started,
                fetchNanos: fetchNanos)
        }

        if !plan.misses.isEmpty {
            let fetchStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try executeBoundedReads(plan, reader: boundedReader)
            fetchNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - fetchStarted
            try markPlanMissesResident(plan)
        }

        succeeded = true
        return expertCachePlanBuffers(plan)
    }

    /// Submits the plan to the persistent storage service and returns before
    /// any read has to complete. Reserved generations are already pinned by
    /// the caller, so the destination pointers remain valid for the operation.
    public func beginExpertCachePlan(
        _ plan: ExpertCachePlan,
        eventDriven: Bool = false
    ) throws -> ExpertLoadOperation {
        let token: ExpertIOCompletionToken?
        if eventDriven {
            guard let eventCoordinator else {
                throw ModelError.internalInconsistency(
                    detail: "event-driven expert I/O requested without a shared event")
            }
            token = try eventCoordinator.reserve()
        } else {
            token = nil
        }
        guard !plan.misses.isEmpty else {
            let operation = ExpertLoadOperation(
                completionToken: token,
                eventCoordinator: eventCoordinator)
            operation.finish(.success(()))
            return operation
        }
        let operation = ExpertLoadOperation(
            completionToken: token,
            eventCoordinator: eventCoordinator)
        ExpertIOScheduler.shared.submit { [self, operation] in
            operation.markInFlight()
            do {
                _ = try executeExpertCachePlan(plan)
                operation.finish(.success(()))
            } catch {
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    private func executeBoundedReads(_ plan: ExpertCachePlan,
                                     reader: ParallelExpertReader) throws {
        var offsets: [UInt64] = []
        var destinations: [UnsafeMutableRawPointer] = []
        offsets.reserveCapacity(plan.misses.count)
        destinations.reserveCapacity(plan.misses.count)
        for index in plan.misses {
            offsets.append(try fileOffset(plan: plan, index: index))
            destinations.append(slotPointers[plan.assignedSlots[index]])
        }
        try reader.fetch(offsets: offsets, into: destinations)
    }

    private func fileOffset(plan: ExpertCachePlan, index: Int) throws -> UInt64 {
        let regionOffset = layout.expertOffset(
            layer: plan.layer,
            expert: plan.experts[index])
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        return layout.streamOffset + regionOffset
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], slotBufferOffsets[slot], layout.expertStride)
        }
    }

    public func expertResidencyResources() -> ExpertResidencyResources {
        ExpertResidencyResources(
            table: residencyTable,
            expertPool: arena.buffer,
            poolSlotStride: UInt64(poolSlotStride),
            expertStride: layout.expertStride,
            expertCount: layout.expertsPerLayer)
    }

    public func residencyEntry(expert: Int) -> ExpertResidencyEntry {
        precondition(expert >= 0 && expert < layout.expertsPerLayer)
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: layout.expertsPerLayer)[expert]
    }

    /// Fills the first `missCount` entries of `victimSlotsScratch` with the
    /// best eviction victims in comparator order, without the old
    /// filter+sort's allocations; an all-hit plan never calls this. Ties
    /// resolve to the lower slot index, which the unstable sort left
    /// unspecified. Returns false when fewer than `missCount` slots are
    /// eligible. `protectedExperts`, a `[Bool]` sized `expertsPerLayer`,
    /// excludes a slot whose expert reads `true`, exactly like loading,
    /// pinned, and already-reserved slots, at one array read and no hashing;
    /// the caller retries with `nil` once this returns false. Caller must
    /// hold `cacheLock`.
    private func selectVictimSlots(missCount: Int, protectedExperts: [Bool]? = nil) -> Bool {
        var victimCount = 0
        var eligibleCount = 0
        for slot in 0..<slotCount {
            guard !reservedSlots[slot], slotState[slot] != .loading,
                  slotPinCount[slot] == 0 else { continue }
            if let protectedExperts, slotExpert[slot] >= 0, slotExpert[slot] < protectedExperts.count,
               protectedExperts[slotExpert[slot]] {
                continue
            }
            eligibleCount &+= 1
            if victimCount < missCount {
                var at = victimCount
                while at > 0, shouldEvictSlot(slot, before: victimSlotsScratch[at - 1]) {
                    victimSlotsScratch[at] = victimSlotsScratch[at - 1]
                    at -= 1
                }
                victimSlotsScratch[at] = slot
                victimCount += 1
            } else if shouldEvictSlot(slot, before: victimSlotsScratch[victimCount - 1]) {
                var at = victimCount - 1
                while at > 0, shouldEvictSlot(slot, before: victimSlotsScratch[at - 1]) {
                    victimSlotsScratch[at] = victimSlotsScratch[at - 1]
                    at -= 1
                }
                victimSlotsScratch[at] = slot
            }
        }
        return missCount <= eligibleCount
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    func pin(_ plan: ExpertCachePlan) throws -> ExpertCacheLease {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard plan.assignedSlots.count == plan.experts.count,
              plan.assignedGenerations.count == plan.experts.count else {
            throw ModelError.internalInconsistency(
                detail: "cannot pin an incomplete expert-cache plan")
        }
        for index in plan.experts.indices {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                  slotGeneration[slot] == plan.assignedGenerations[index],
                  slotExpert[slot] == plan.experts[index],
                  slotState[slot] != .empty else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache plan became stale before GPU pin")
            }
        }
        for slot in plan.assignedSlots { slotPinCount[slot] &+= 1 }
        return ExpertCacheLease(
            streamer: self,
            slots: plan.assignedSlots,
            generations: plan.assignedGenerations)
    }

    fileprivate func unpin(slots: [Int], generations: [UInt64]) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for (slot, generation) in zip(slots, generations)
            where slot >= 0 && slot < slotCount && slotGeneration[slot] == generation {
            precondition(slotPinCount[slot] > 0, "expert-cache slot pin underflow")
            slotPinCount[slot] -= 1
        }
    }

    public func statistics() -> ExpertStreamingStatistics {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return ExpertStreamingStatistics(
            plans: statisticsPlans,
            requestedExperts: statisticsRequestedExperts,
            hits: statisticsHits,
            misses: statisticsMisses,
            bytesRead: statisticsBytesRead,
            readOperations: statisticsReadOperations,
            evictions: statisticsEvictions,
            reloads: statisticsReloads,
            loadBatches: statisticsLoadBatches,
            totalLoadNanos: statisticsTotalLoadNanos,
            fetchNanos: statisticsFetchNanos,
            maximumLoadNanos: statisticsMaximumLoadNanos,
            latencyHistogram: statisticsLatencyHistogram,
            residentSlots: slotState.count(where: { $0 == .resident }),
            loadingSlots: slotState.count(where: { $0 == .loading }),
            pinnedSlots: slotPinCount.count(where: { $0 > 0 }),
            peakLoadingSlots: statisticsPeakLoadingSlots)
    }

    /// A stable snapshot of authoritative cache entries. Loading slots are
    /// intentionally omitted: their bytes must not be consumed or treated as
    /// available by a predictor until a successful demand load publishes them.
    /// Diagnostic and policy code use this before a cache plan reserves slots.
    public func residentExperts() -> [Int] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return zip(slotExpert, slotState).compactMap { expert, state in
            state == .resident && expert >= 0 ? expert : nil
        }.sorted()
    }

    /// Claims a ring cell for a prediction's read: `loading` for the
    /// classifier, neither a hit nor a victim for a plan. Refused when the
    /// pool holds or is loading the expert, or a landing for it exists.
    public func claimLanding(expert: Int, cell: Int) -> Bool {
        guard expert >= 0, expert < layout.expertsPerLayer,
              cell >= 0, cell < arena.cellCount else { return false }
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard landings[expert] == nil, !slotExpert.contains(expert),
              !landings.values.contains(where: { $0.cell == cell }) else { return false }
        landingGeneration &+= 1
        landings[expert] = Landing(cell: cell, resident: false, generation: landingGeneration)
        writeResidencyEntryUnlocked(expert: expert, cell: cell,
                                    state: ExpertResidencyEntry.loading,
                                    generation: landingGeneration)
        return true
    }

    /// The read's completion, on the storage thread: `resident` at the cell
    /// so the classifier can hit it. A landing the pool overtook with its own
    /// read of the expert is discarded, the pool's entry standing.
    public func completeLanding(expert: Int, cell: Int) -> Bool {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard let landing = landings[expert], landing.cell == cell, !landing.resident else {
            return false
        }
        if slotExpert.contains(expert) {
            landings[expert] = nil
            return false
        }
        landings[expert]?.resident = true
        writeResidencyEntryUnlocked(expert: expert, cell: cell,
                                    state: ExpertResidencyEntry.resident,
                                    generation: landing.generation)
        return true
    }

    public func failLanding(expert: Int, cell: Int) {
        dropLanding(expert: expert, cell: cell)
    }

    /// The ring's reclaim of a landing no plan wanted: the entry is emptied
    /// unless the pool owns the expert, whose entry then stands.
    public func dropLanding(expert: Int, cell: Int) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard let landing = landings[expert], landing.cell == cell else { return }
        landings[expert] = nil
        if !slotExpert.contains(expert) {
            writeResidencyEntryUnlocked(expert: expert, cell: cell,
                                        state: ExpertResidencyEntry.empty,
                                        generation: landing.generation)
        }
    }

    /// Whether `cell` backs one of this layer's pool slots.
    private func ownsCell(_ cell: Int) -> Bool {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return slotBufferOffsets.contains(arena.offset(cell: cell))
    }

    /// Starts a speculative read of `experts` into the ring's `cells`, each a
    /// landing published `resident` from the storage thread when its bytes
    /// land. An expert the pool holds or is loading, or has landed already,
    /// refuses the batch with `PrefetchClaimRefused` (the ring counts it; a
    /// batch is one read at the production budget); a cell the pool owns is
    /// an inconsistency and throws. Demand work is always scheduled at
    /// higher priority.
    public func beginPrefetch(experts: [Int], cells: [Int]) throws -> ExpertLoadOperation {
        guard experts.count == cells.count else {
            throw ModelError.internalInconsistency(
                detail: "prefetch experts and cells differ in count")
        }
        for expert in experts where expert < 0 || expert >= layout.expertsPerLayer {
            throw ModelError.internalInconsistency(detail: "invalid prefetched expert")
        }
        for cell in cells where ownsCell(cell) {
            throw ModelError.internalInconsistency(
                detail: "a prefetch landing was offered cell \(cell), which the pool owns")
        }
        var landed: [(expert: Int, cell: Int)] = []
        var offsets: [UInt64] = []
        var destinations: [UnsafeMutableRawPointer] = []
        for (expert, cell) in zip(experts, cells) {
            guard claimLanding(expert: expert, cell: cell) else {
                for entry in landed { dropLanding(expert: entry.expert, cell: entry.cell) }
                throw PrefetchClaimRefused(expert: expert, cell: cell)
            }
            landed.append((expert, cell))
            offsets.append(layout.streamOffset + layout.expertOffset(layer: 0, expert: expert))
            destinations.append(arena.pointer(cell: cell))
        }
        let operation = ExpertLoadOperation()
        let safeDestinations = PrefetchDestinations(destinations)
        let readOffsets = offsets
        let landings = landed
        ExpertIOScheduler.shared.submit(priority: .speculative) { [self, operation, safeDestinations] in
            operation.markInFlight()
            do {
                try boundedReader.fetch(offsets: readOffsets, into: safeDestinations.values)
                for entry in landings { _ = completeLanding(expert: entry.expert, cell: entry.cell) }
                operation.finish(.success(()))
            } catch {
                for entry in landings { failLanding(expert: entry.expert, cell: entry.cell) }
                operation.finish(.failure(error))
            }
        }
        return operation
    }

    private func finishPlanExecution(_ plan: ExpertCachePlan,
                                     succeeded: Bool,
                                     elapsedNanos: UInt64,
                                     fetchNanos: UInt64 = 0) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if succeeded {
            recordSuccessfulLoadsUnlocked(
                experts: plan.misses.map { plan.experts[$0] },
                elapsedNanos: elapsedNanos,
                fetchNanos: fetchNanos)
            return
        }
        resetLoadingMissesUnlocked(plan)
    }

    private func markPlanMissesResident(_ plan: ExpertCachePlan) throws {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slotGeneration[slot] == plan.assignedGenerations[index] else {
                throw ModelError.internalInconsistency(
                    detail: "expert-cache slot generation changed during expert load")
            }
        }
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            slotState[slot] = .resident
            slotExpert[slot] = plan.experts[index]
            slotLastUse[slot] = useClock
            publishResidencyUnlocked(expert: plan.experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.resident,
                                     generation: plan.assignedGenerations[index])
        }
    }

    /// Planning reserves miss slots as `.loading`, so a plan discarded without
    /// execution must be abandoned or those slots stay un-loadable and
    /// un-evictable for the life of the streamer.
    public func abandonExpertCachePlan(_ plan: ExpertCachePlan) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        resetLoadingMissesUnlocked(plan)
    }

    private func resetLoadingMissesUnlocked(_ plan: ExpertCachePlan) {
        for index in plan.misses {
            let slot = plan.assignedSlots[index]
            guard slot >= 0, slot < slotCount,
                  slotGeneration[slot] == plan.assignedGenerations[index],
                  slotState[slot] == .loading else { continue }
            slotState[slot] = .empty
            slotExpert[slot] = -1
            publishResidencyUnlocked(expert: plan.experts[index],
                                     slot: slot,
                                     state: ExpertResidencyEntry.empty,
                                     generation: plan.assignedGenerations[index])
        }
    }

    /// The slot's global cell, what the table names and the kernels address.
    private func cellIndexUnlocked(_ slot: Int) -> Int {
        Int(slotBufferOffsets[slot]) / poolSlotStride
    }

    /// CPU publication occurs under the cache lock. A loading entry is visible
    /// immediately after reservation; resident is written only after every
    /// byte lands. Event-driven consumers additionally wait on the batch's
    /// shared-event value, which is the CPU/GPU release/acquire boundary.
    private func publishResidencyUnlocked(expert: Int,
                                          slot: Int,
                                          state: UInt32,
                                          generation: UInt64) {
        writeResidencyEntryUnlocked(expert: expert, cell: cellIndexUnlocked(slot),
                                    state: state, generation: generation)
    }

    private func writeResidencyEntryUnlocked(expert: Int,
                                             cell: Int,
                                             state: UInt32,
                                             generation: UInt64) {
        guard expert >= 0 && expert < layout.expertsPerLayer else { return }
        let entries = residencyTable.contents()
            .bindMemory(to: ExpertResidencyEntry.self,
                        capacity: layout.expertsPerLayer)
        entries[expert] = ExpertResidencyEntry(
            slot: state == ExpertResidencyEntry.empty
                ? ExpertResidencyEntry.notResidentSlot : UInt32(cell),
            state: state,
            generation: generation)
    }

    private func recordSuccessfulLoadsUnlocked(experts: [Int], elapsedNanos: UInt64,
                                               fetchNanos: UInt64 = 0) {
        guard !experts.isEmpty else { return }
        statisticsBytesRead &+= UInt64(experts.count) * layout.expertStride
        statisticsReadOperations &+= UInt64(experts.count)
        statisticsLoadBatches &+= 1
        statisticsTotalLoadNanos &+= elapsedNanos
        statisticsFetchNanos &+= fetchNanos
        statisticsMaximumLoadNanos = max(statisticsMaximumLoadNanos, elapsedNanos)
        let bucket = Self.latencyBucketIndex(nanos: elapsedNanos)
        statisticsLatencyHistogram[bucket] &+= 1
        for expert in experts where expert >= 0 && expert < expertLoadCount.count {
            if expertLoadCount[expert] > 0 { statisticsReloads &+= 1 }
            expertLoadCount[expert] &+= 1
        }
    }

    private func recordPrefetchLandingsUnlocked(_ experts: [Int]) {
        for expert in experts where expert >= 0 && expert < expertLoadCount.count {
            if expertLoadCount[expert] > 0 { statisticsReloads &+= 1 }
            expertLoadCount[expert] &+= 1
        }
    }

    private static func latencyBucketIndex(nanos: UInt64) -> Int {
        var bound: UInt64 = 125_000
        for index in 0..<16 {
            if nanos <= bound { return index }
            bound &*= 2
        }
        return 16
    }

    static func latencyBucketUpperBound(index: Int) -> UInt64 {
        guard index < 16 else { return UInt64.max }
        return 125_000 << UInt64(index)
    }

    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        var filled = 0
        while filled < count {
            let readCount = pread(
                fd,
                destination.advanced(by: filled),
                count - filled,
                off_t(fileOffset) + off_t(filled))
            if readCount < 0 {
                throw StreamerError.preadFailed(errno: errno)
            }
            if readCount == 0 {
                throw StreamerError.sizeMismatch(expected: UInt64(count), actual: UInt64(filled))
            }
            filled += readCount
        }
    }
}

/// Pins exact slot generations until every GPU command using them completes.
/// Release is idempotent so error cleanup and normal command completion can
/// safely converge on the same lifetime operation.
/// unchecked-invariant: immutable slot metadata is published at init and the
/// only mutable release flag is guarded by `lock`; streamer state has its own lock.
final class ExpertCacheLease: @unchecked Sendable {
    private weak var streamer: PreadExpertStreamer?
    private let slots: [Int]
    private let generations: [UInt64]
    private let lock = NSLock()
    private var released = false

    fileprivate init(streamer: PreadExpertStreamer,
                     slots: [Int],
                     generations: [UInt64]) {
        self.streamer = streamer
        self.slots = slots
        self.generations = generations
    }

    func release() {
        lock.lock()
        guard !released else {
            lock.unlock()
            return
        }
        released = true
        lock.unlock()
        streamer?.unpin(slots: slots, generations: generations)
    }

    deinit { release() }
}
