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
                                   arenaCells: Int? = nil) throws -> Landed {
        let url = try writeSyntheticLayer()
        let device = try MetalContext().device
        let cells = arenaCells ?? slotCount + 2
        let arena = try ExpertCellArena(device: device, cellCount: cells, stride: expertStride)
        let range = cellRange ?? 0..<slotCount
        let streamer = try PreadExpertStreamer(
            layout: makeLayout(path: url.path), device: device, slotCount: slotCount,
            arena: arena, cellRange: range)
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
        #expect(swapped.generation == plan.assignedGenerations[1])
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
}
