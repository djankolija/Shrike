import Foundation
import Metal
import Testing

@testable import Shrike

@Suite struct PrefetchAdoptionGuardTests {
    @Test func abandonBeforeATransferFailsTheSlotsAndReleasesTheRingOnce() {
        var failed = 0
        var released = 0
        let guardObject = PrefetchAdoptionGuard(fail: { failed += 1 }, release: { released += 1 })
        guardObject.abandon()
        guardObject.abandon()
        #expect(failed == 1)
        #expect(released == 1)
    }

    @Test func abandonAfterATransferReleasesThroughTheTransferAsNoAdoption() throws {
        let context = try MetalContext()
        let buffer = try #require(context.device.makeBuffer(length: 16_384, options: .storageModeShared))
        var failed = 0
        var directReleases = 0
        var transferReleases: [Bool] = []
        let guardObject = PrefetchAdoptionGuard(fail: { failed += 1 }, release: { directReleases += 1 })
        let transfer = PrefetchAdoptionTransfer(
            sources: [buffer], destinations: [buffer], destinationOffsets: [0],
            byteCount: 16_384) { transferReleases.append($0) }
        guardObject.attach(transfer)
        guardObject.abandon()
        transfer.release()
        #expect(failed == 1)
        #expect(directReleases == 0)
        #expect(transferReleases == [false])
    }

    @Test func aCommittedAdoptionIsNeverAbandoned() {
        var failed = 0
        var released = 0
        let guardObject = PrefetchAdoptionGuard(fail: { failed += 1 }, release: { released += 1 })
        guardObject.commit()
        guardObject.abandon()
        #expect(failed == 0)
        #expect(released == 0)
    }
}
