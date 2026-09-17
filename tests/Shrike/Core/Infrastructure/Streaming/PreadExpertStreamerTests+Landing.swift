import Darwin
import Foundation
import Metal
import Testing

@testable import Shrike

extension PreadExpertStreamerTests {
    private struct Landed {
        let url: URL
        let arena: ExpertCellArena
        let streamer: PreadExpertStreamer
        let ringCells: [Int]
    }

    private static func makeLanded(slotCount: Int = 4, cellRange: Range<Int>? = nil,
                                   arenaCells: Int? = nil,
                                   withCoordinator: Bool = false) throws -> Landed {
        let url = try writeSyntheticLayer()
        let device = try MetalContext().device
        let cells = arenaCells ?? slotCount + 2
        let arena = try ExpertCellArena(device: device, cellCount: cells, stride: expertStride)
        let range = cellRange ?? 0..<slotCount
        let coordinator = withCoordinator ? ExpertIOEventCoordinator(device: device) : nil
        let streamer = try PreadExpertStreamer(
            layout: makeLayout(path: url.path), device: device, slotCount: slotCount,
            eventCoordinator: coordinator, arena: arena, cellRange: range)
        let ringCells = Array((cells - 2)..<cells)
        return Landed(url: url, arena: arena, streamer: streamer, ringCells: ringCells)
    }

    private static func land(_ fixture: Landed, expert: Int, cell: Int) throws {
        let operation = try fixture.streamer.beginPrefetch(experts: [expert], cells: [cell])
        try operation.wait()
    }

    private static func cell(of plan: ExpertCachePlan, index: Int, in fixture: Landed) -> Int {
        let buffers = fixture.streamer.expertCachePlanBuffers(plan)
        return fixture.arena.cell(atOffset: buffers[index].offset)
    }

    private static func table(of fixture: Landed) -> [ExpertResidencyEntry] {
        let resources = fixture.streamer.expertResidencyResources()
        return (0..<resources.expertCount).map { expert in
            resources.table.contents().load(
                fromByteOffset: expert * MemoryLayout<ExpertResidencyEntry>.stride,
                as: ExpertResidencyEntry.self)
        }
    }

    private static func word(of fixture: Landed, expert: Int) -> UInt64 {
        fixture.streamer.expertResidencyResources().table.contents()
            .load(fromByteOffset: expert * MemoryLayout<UInt64>.stride, as: UInt64.self)
    }

    private static func resident(_ cell: Int) -> ExpertResidencyEntry {
        ExpertResidencyEntry(slot: UInt32(cell), state: ExpertResidencyEntry.resident)
    }

    private static func loading(_ cell: Int) -> ExpertResidencyEntry {
        ExpertResidencyEntry(slot: UInt32(cell), state: ExpertResidencyEntry.loading)
    }

    private static let empty = ExpertResidencyEntry()

    @Test func aClaimedLandingIsLoadingAtItsCellAndNeitherAHitNorAVictim() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        let entry = fixture.streamer.residencyEntry(expert: 1)
        #expect(entry.state == ExpertResidencyEntry.loading)
        #expect(entry.slot == UInt32(ringCell))
        #expect(!fixture.streamer.residentExperts().contains(1))

        let plan = try fixture.streamer.planExpertsCached(experts: [1])
        #expect(plan.misses == [0])
        #expect(Self.cell(of: plan, index: 0, in: fixture) < 4)
        fixture.streamer.abandonExpertCachePlan(plan)
        #expect(!fixture.streamer.claimLanding(expert: 1, cell: fixture.ringCells[1]))
    }

    private static func statusWord(_ token: ExpertIOCompletionToken) -> UInt32 {
        token.status.contents().advanced(by: token.statusOffset).load(as: UInt32.self)
    }

    @Test func anOverflowSlotIsAVictimOfThePoolLoadingAtItsCell() throws {
        let fixture = try Self.makeLanded(slotCount: 2)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 1])

        let first = try #require(fixture.streamer.reserveOverflowSlot(expert: 2))
        #expect((0..<2).contains(first))
        #expect(fixture.streamer.residencyEntry(expert: 2) == Self.loading(first))
        #expect(fixture.streamer.residentExperts().count == 1)
        let evicted = fixture.streamer.residentExperts() == [0] ? 1 : 0
        #expect(fixture.streamer.residencyEntry(expert: evicted) == Self.empty)

        let second = try #require(fixture.streamer.reserveOverflowSlot(expert: 3))
        #expect(second != first)
        #expect(fixture.streamer.residentExperts().isEmpty)
        #expect(fixture.streamer.reserveOverflowSlot(expert: evicted) == nil)
    }

    @Test func theOverflowVictimIsNeverOneOfTheRouteItServes() throws {
        let fixture = try Self.makeLanded(slotCount: 2)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 1])

        #expect(fixture.streamer.reserveOverflowSlot(expert: 2, protecting: [0, 1]) == nil)
        #expect(fixture.streamer.residentExperts() == [0, 1])

        let cell = try #require(fixture.streamer.reserveOverflowSlot(expert: 2, protecting: [0, 5]))
        #expect(fixture.streamer.residencyEntry(expert: 2) == Self.loading(cell))
        #expect(fixture.streamer.residentExperts() == [0])
        #expect(fixture.streamer.residencyEntry(expert: 1) == Self.empty)
    }

    @Test func agreedReadsLandInARingCellAndAPoolCellAndPublishTheToken() throws {
        let fixture = try Self.makeLanded(slotCount: 2, withCoordinator: true)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 1])
        let token = try fixture.streamer.reserveCompletionToken()
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        let poolCell = try #require(fixture.streamer.reserveOverflowSlot(expert: 2))

        let operation = try fixture.streamer.beginAgreedReads(
            experts: [3, 2], cells: [ringCell, poolCell], token: token)
        try operation.wait()

        #expect(fixture.streamer.residencyEntry(expert: 3) == Self.resident(ringCell))
        #expect(fixture.streamer.residencyEntry(expert: 2) == Self.resident(poolCell))
        #expect(fixture.streamer.residentExperts().contains(2))
        #expect(!fixture.streamer.residentExperts().contains(3))
        for (expert, cell) in [(3, ringCell), (2, poolCell)] {
            let bytes = Self.bytes(of: fixture.arena.buffer, offset: fixture.arena.offset(cell: cell),
                                   count: Self.expertStride)
            #expect(bytes.allSatisfy { $0 == Self.tagByte(expert) })
        }
        #expect(Self.statusWord(token) == 1)
        #expect(token.event.signaledValue == token.value)
        #expect(fixture.streamer.statistics().readOperations == 4)

        let plan = try fixture.streamer.planExpertsCached(
            experts: [3, 2], leasedLandings: [3], missesCounted: 2)
        #expect(plan.misses.isEmpty)
        #expect(plan.freedCells[3] != nil)
        #expect(Self.cell(of: plan, index: 0, in: fixture) == ringCell)
        #expect(Self.cell(of: plan, index: 1, in: fixture) == poolCell)
        let stats = fixture.streamer.statistics()
        #expect(stats.misses == 4)
        #expect(stats.hits == 0)
    }

    @Test func anEmptyAgreedBatchPublishesItsTokenAtOnce() throws {
        let fixture = try Self.makeLanded(withCoordinator: true)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let token = try fixture.streamer.reserveCompletionToken()
        let operation = try fixture.streamer.beginAgreedReads(experts: [], cells: [], token: token)
        #expect(operation.state == .completed)
        #expect(Self.statusWord(token) == 1)
        #expect(token.event.signaledValue == token.value)
    }

    @Test func aFailedAgreedReadDropsItsLandingEmptiesItsPoolCellAndStillPublishes() throws {
        let fixture = try Self.makeLanded(slotCount: 2, withCoordinator: true)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 1])
        let token = try fixture.streamer.reserveCompletionToken()
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        let poolCell = try #require(fixture.streamer.reserveOverflowSlot(expert: 2))
        #expect(truncate(fixture.url.path, 0) == 0)

        let operation = try fixture.streamer.beginAgreedReads(
            experts: [3, 2], cells: [ringCell, poolCell], token: token)
        #expect(throws: (any Error).self) { try operation.wait() }

        #expect(fixture.streamer.residencyEntry(expert: 3) == Self.empty)
        #expect(fixture.streamer.residencyEntry(expert: 2) == Self.empty)
        #expect(Self.statusWord(token) == 2)
        #expect(token.event.signaledValue == token.value)
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        #expect(fixture.streamer.reserveOverflowSlot(expert: 2) != nil)
    }

    @Test func aCompletedLandingIsAHitThatSwapsItsCellIntoThePool() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)

        let landed = fixture.streamer.residencyEntry(expert: 1)
        #expect(landed.state == ExpertResidencyEntry.resident)
        #expect(landed.slot == UInt32(ringCell))
        #expect(fixture.streamer.residentExperts() == [0, 2])
        let bytes = Self.bytes(of: fixture.arena.buffer, offset: fixture.arena.offset(cell: ringCell),
                               count: Self.expertStride)
        #expect(bytes.allSatisfy { $0 == Self.tagByte(1) })

        let plan = try fixture.streamer.planExpertsCached(experts: [0, 1, 2], leasedLandings: [1])
        #expect(plan.hits == 3)
        #expect(plan.misses.isEmpty)
        #expect(plan.adopted.isEmpty)
        #expect(Self.cell(of: plan, index: 1, in: fixture) == ringCell)
        let freed = try #require(plan.freedCells[1])
        #expect((0..<4).contains(freed))
        #expect(plan.freedCells.count == 1)
        let swapped = fixture.streamer.residencyEntry(expert: 1)
        #expect(swapped.state == ExpertResidencyEntry.resident)
        #expect(swapped.slot == UInt32(ringCell))
        #expect(fixture.arena.cellGeneration(ringCell) == plan.assignedGenerations[1])
        #expect(fixture.streamer.residentExperts() == [0, 1, 2])

        let lease = try fixture.streamer.pin(plan)
        lease.release()
        let again = try fixture.streamer.planExpertsCached(experts: [1])
        #expect(again.hits == 1)
        #expect(again.freedCells.isEmpty)
        #expect(Self.cell(of: again, index: 0, in: fixture) == ringCell)
    }

    @Test func aCompletedLandingTheGPUMissedIsAdoptedAndSwappedTheSame() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)

        let plan = try fixture.streamer.planExpertsCached(experts: [0, 1, 2], gpuMissedExperts: [1],
                                                          leasedLandings: [1])
        #expect(plan.hits == 3)
        #expect(plan.misses.isEmpty)
        #expect(plan.adopted == [1])
        #expect(Self.cell(of: plan, index: 1, in: fixture) == ringCell)
        #expect(plan.freedCells[1] != nil)
    }

    @Test func theSwapEvictsTheLeastUsedResidentWhenThePoolIsFull() throws {
        let fixture = try Self.makeLanded(slotCount: 3)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2, 3])
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let victimCell = Self.cell(
            of: try fixture.streamer.planExpertsCached(experts: [3]), index: 0, in: fixture)
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)

        let plan = try fixture.streamer.planExpertsCached(experts: [0, 1, 2], leasedLandings: [1])
        #expect(plan.hits == 3)
        #expect(plan.freedCells[1] == victimCell)
        #expect(fixture.streamer.residentExperts() == [0, 1, 2])
        #expect(fixture.streamer.residencyEntry(expert: 3).state == ExpertResidencyEntry.empty)
    }

    @Test func anUnwantedCompletedLandingStaysResidentUntilDropped() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)

        let plan = try fixture.streamer.planExpertsCached(experts: [0, 2])
        #expect(plan.freedCells.isEmpty)
        #expect(fixture.streamer.residencyEntry(expert: 1).state == ExpertResidencyEntry.resident)

        fixture.streamer.dropLanding(expert: 1, cell: ringCell)
        #expect(fixture.streamer.residencyEntry(expert: 1).state == ExpertResidencyEntry.empty)
        #expect(fixture.streamer.residentExperts() == [0, 2])
        #expect(try fixture.streamer.planExpertsCached(experts: [1]).misses == [0])
    }

    @Test func aFailedLandingIsEmptyAndAMissAgain() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        fixture.streamer.failLanding(expert: 1, cell: ringCell)
        #expect(fixture.streamer.residencyEntry(expert: 1).state == ExpertResidencyEntry.empty)
        #expect(try fixture.streamer.planExpertsCached(experts: [1]).misses == [0])
        #expect(throws: (any Error).self) {
            try fixture.streamer.beginPrefetch(experts: [Self.numExperts], cells: [ringCell])
        }
    }

    @Test func aLandingCompletingAfterThePoolLoadedItsExpertIsDiscarded() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        _ = try fixture.streamer.loadExpertsCached(experts: [1])
        let pooled = fixture.streamer.residencyEntry(expert: 1)
        #expect(pooled.state == ExpertResidencyEntry.resident)
        #expect((0..<4).contains(Int(pooled.slot)))

        #expect(!fixture.streamer.completeLanding(expert: 1, cell: ringCell))
        #expect(fixture.streamer.residencyEntry(expert: 1) == pooled)
        fixture.streamer.dropLanding(expert: 1, cell: ringCell)
        #expect(fixture.streamer.residencyEntry(expert: 1) == pooled)
        #expect(try fixture.streamer.planExpertsCached(experts: [1]).hits == 1)
    }

    @Test func aPredictionThePoolHoldsRefusesTheBatch() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [1])
        let pooled = fixture.streamer.residencyEntry(expert: 1)
        #expect(throws: PrefetchClaimRefused.self) {
            try fixture.streamer.beginPrefetch(
                experts: [3, 1], cells: [fixture.ringCells[0], fixture.ringCells[1]])
        }
        #expect(fixture.streamer.residencyEntry(expert: 1) == pooled)
        #expect(fixture.streamer.residencyEntry(expert: 3).state == ExpertResidencyEntry.empty)
        #expect(fixture.streamer.claimLanding(expert: 3, cell: fixture.ringCells[0]))
    }

    @Test func aLandingNotLeasedToThePlanIsReadIntoThePoolAndDroppedLater() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)

        let plan = try fixture.streamer.planExpertsCached(experts: [0, 1, 2])
        #expect(plan.misses == [0, 1, 2])
        #expect(plan.freedCells.isEmpty)
        _ = try fixture.streamer.executeExpertCachePlan(plan)
        let pooled = fixture.streamer.residencyEntry(expert: 1)
        #expect(pooled.state == ExpertResidencyEntry.resident)
        #expect((0..<4).contains(Int(pooled.slot)))

        fixture.streamer.dropLanding(expert: 1, cell: ringCell)
        #expect(fixture.streamer.residencyEntry(expert: 1) == pooled)
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
    }

    @Test func aCellAnotherLandingHoldsOrThePoolOwnsIsRefused() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        #expect(!fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        #expect(throws: ModelError.self) {
            try fixture.streamer.beginPrefetch(experts: [3], cells: [0])
        }
        #expect(fixture.streamer.residencyEntry(expert: 3).state == ExpertResidencyEntry.empty)
    }

    @Test func residencyEntriesNameTheArenasGlobalCells() throws {
        let fixture = try Self.makeLanded(slotCount: 4, cellRange: 4..<8, arenaCells: 10)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [2])
        let entry = fixture.streamer.residencyEntry(expert: 2)
        #expect((4..<8).contains(Int(entry.slot)))
        let plan = try fixture.streamer.planExpertsCached(experts: [2])
        #expect(Int(entry.slot) == Self.cell(of: plan, index: 0, in: fixture))
        #expect(fixture.streamer.expertResidencyResources().expertPool === fixture.arena.buffer)
    }

    @Test func theFreedCellIsTheRingsNextLanding() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)
        let plan = try fixture.streamer.planExpertsCached(experts: [0, 1, 2], leasedLandings: [1])
        let freed = try #require(plan.freedCells[1])

        try Self.land(fixture, expert: 3, cell: freed)
        let landed = fixture.streamer.residencyEntry(expert: 3)
        #expect(landed.state == ExpertResidencyEntry.resident)
        #expect(landed.slot == UInt32(freed))
        let bytes = Self.bytes(of: fixture.arena.buffer, offset: fixture.arena.offset(cell: freed),
                               count: Self.expertStride)
        #expect(bytes.allSatisfy { $0 == Self.tagByte(3) })
        #expect(fixture.streamer.residentExperts() == [0, 1, 2])
        #expect(try fixture.streamer.planExpertsCached(experts: [3], leasedLandings: [3]).hits == 1)
    }

    @Test func aLandingThePoolOvertookIsDiscardedAtItsCompletion() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        #expect(fixture.arena.cellGeneration(ringCell) == 1)
        #expect(Self.table(of: fixture) == [Self.empty, Self.loading(ringCell), Self.empty, Self.empty])

        _ = try fixture.streamer.loadExpertsCached(experts: [1])
        let poolCell = Int(fixture.streamer.residencyEntry(expert: 1).slot)
        #expect((0..<4).contains(poolCell))
        #expect(fixture.arena.cellGeneration(poolCell) == 2)
        #expect(Self.table(of: fixture) == [Self.empty, Self.resident(poolCell), Self.empty, Self.empty])

        #expect(!fixture.streamer.completeLanding(expert: 1, cell: ringCell))
        #expect(Self.table(of: fixture) == [Self.empty, Self.resident(poolCell), Self.empty, Self.empty])
        #expect(fixture.arena.cellGeneration(ringCell) == 1)
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        #expect(fixture.arena.cellGeneration(ringCell) == 3)
    }

    @Test func aDemandCompletionAgainstABumpedCellGenerationThrows() throws {
        let fixture = try Self.makeLanded(slotCount: 1)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let stale = try fixture.streamer.planExpertsCached(experts: [1])
        #expect(stale.assignedSlots == [0])
        #expect(stale.assignedGenerations == [1])
        #expect(Self.table(of: fixture) == [Self.empty, Self.loading(0), Self.empty, Self.empty])

        fixture.streamer.abandonExpertCachePlan(stale)
        #expect(fixture.arena.cellGeneration(0) == 1)
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.empty, Self.empty])

        let current = try fixture.streamer.planExpertsCached(experts: [2])
        #expect(current.assignedSlots == [0])
        #expect(current.assignedGenerations == [2])
        #expect(fixture.arena.cellGeneration(0) == 2)
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.loading(0), Self.empty])

        #expect {
            _ = try fixture.streamer.executeExpertCachePlan(stale)
        } throws: { error in
            guard case ModelError.internalInconsistency(let detail) = error else { return false }
            return detail.contains("generation changed")
        }
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.loading(0), Self.empty])
        #expect(fixture.streamer.statistics().loadingSlots == 1)

        _ = try fixture.streamer.executeExpertCachePlan(current)
        #expect(fixture.arena.cellGeneration(0) == 2)
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.resident(0), Self.empty])
        let bytes = Self.bytes(of: fixture.arena.buffer, offset: fixture.arena.offset(cell: 0),
                               count: Self.expertStride)
        #expect(bytes.allSatisfy { $0 == Self.tagByte(2) })
    }

    @Test func aSwapThenAnEvictionOfTheSameSlotPublishesEmptyOnceAtTheCell() throws {
        let fixture = try Self.makeLanded(slotCount: 2)
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        _ = try fixture.streamer.loadExpertsCached(experts: [0, 2])
        let ringCell = fixture.ringCells[0]
        try Self.land(fixture, expert: 1, cell: ringCell)
        let landed = Self.word(of: fixture, expert: 1)
        #expect(Self.table(of: fixture) == [Self.resident(0), Self.resident(ringCell), Self.resident(1), Self.empty])

        let swap = try fixture.streamer.planExpertsCached(experts: [0, 1], leasedLandings: [1])
        #expect(swap.hits == 2)
        #expect(swap.assignedSlots == [0, 1])
        #expect(swap.freedCells == [1: 1])
        #expect(swap.assignedGenerations == [1, 3])
        #expect(fixture.arena.cellGeneration(1) == 4)
        #expect(fixture.arena.cellGeneration(ringCell) == 3)
        #expect(Self.word(of: fixture, expert: 1) == landed)
        #expect(Self.table(of: fixture) == [Self.resident(0), Self.resident(ringCell), Self.empty, Self.empty])

        let evict = try fixture.streamer.planExpertsCached(experts: [3])
        #expect(evict.assignedSlots == [1])
        #expect(evict.assignedGenerations == [5])
        #expect(fixture.arena.cellGeneration(ringCell) == 5)
        #expect(Self.table(of: fixture) == [Self.resident(0), Self.empty, Self.empty, Self.loading(ringCell)])

        _ = try fixture.streamer.executeExpertCachePlan(evict)
        #expect(fixture.arena.cellGeneration(ringCell) == 5)
        #expect(Self.table(of: fixture) == [Self.resident(0), Self.empty, Self.empty, Self.resident(ringCell)])
    }

    @Test func aStaleCompletionAfterADropPublishesNothing() throws {
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        let ringCell = fixture.ringCells[0]
        #expect(fixture.streamer.claimLanding(expert: 1, cell: ringCell))
        #expect(fixture.arena.cellGeneration(ringCell) == 1)
        fixture.streamer.dropLanding(expert: 1, cell: ringCell)
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.empty, Self.empty])

        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        #expect(fixture.arena.cellGeneration(ringCell) == 2)
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.empty, Self.loading(ringCell)])

        #expect(!fixture.streamer.completeLanding(expert: 1, cell: ringCell))
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.empty, Self.loading(ringCell)])
        #expect(fixture.streamer.completeLanding(expert: 3, cell: ringCell))
        #expect(Self.table(of: fixture) == [Self.empty, Self.empty, Self.empty, Self.resident(ringCell)])
        #expect(fixture.arena.cellGeneration(ringCell) == 2)
    }

    @Test func anEntryIsOneEightByteWordWithTheStateAboveTheCell() throws {
        #expect(MemoryLayout<ExpertResidencyEntry>.size == 8)
        #expect(MemoryLayout<ExpertResidencyEntry>.stride == 8)
        let fixture = try Self.makeLanded()
        defer { try? FileManager.default.removeItem(at: fixture.url) }
        #expect(Self.word(of: fixture, expert: 0) == UInt64(ExpertResidencyEntry.notResidentSlot))

        _ = try fixture.streamer.loadExpertsCached(experts: [2])
        let cell = Int(fixture.streamer.residencyEntry(expert: 2).slot)
        #expect(Self.word(of: fixture, expert: 2)
                == UInt64(ExpertResidencyEntry.resident) << 32 | UInt64(cell))

        let ringCell = fixture.ringCells[1]
        #expect(fixture.streamer.claimLanding(expert: 3, cell: ringCell))
        #expect(Self.word(of: fixture, expert: 3)
                == UInt64(ExpertResidencyEntry.loading) << 32 | UInt64(ringCell))
        fixture.streamer.dropLanding(expert: 3, cell: ringCell)
        #expect(Self.word(of: fixture, expert: 3) == UInt64(ExpertResidencyEntry.notResidentSlot))
    }
}
