import Testing
import Foundation
import ShrikeKernelsC
@testable import Shrike

/// The FIFO claim and free-slot rules are pure C functions over a slot table,
/// asserted directly here rather than through timing.
extension ParallelExpertReaderTests {
    private static func claim(_ slots: [shrike_expert_batch_slot]) -> Int32 {
        slots.withUnsafeBufferPointer {
            shrike_expert_reader_claim_slot($0.baseAddress, Int32($0.count))
        }
    }

    private static func freeSlot(_ slots: [shrike_expert_batch_slot]) -> Int32 {
        slots.withUnsafeBufferPointer {
            shrike_expert_reader_free_slot($0.baseAddress, Int32($0.count))
        }
    }

    @Test func twoConcurrentFetchesBothLandTheirOwnBytes() async throws {
        let url = try Self.makeFixture(count: 64)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(path: url.path,
                                             expertStride: Self.stride,
                                             threads: 4,
                                             batchDepth: 2)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<2 {
                group.addTask {
                    // Disjoint id ranges per task, so there is no shared byte
                    // pattern either task could accidentally match against.
                    let base = worker * 32
                    try Self.withDestinations(4) { buffers in
                        for round in 0..<50 {
                            let ids = (0..<4).map { UInt32(base + (round * 4 + $0) % 32) }
                            try reader.fetch(experts: ids, into: buffers)
                            for (index, expert) in ids.enumerated() {
                                let byte = buffers[index]
                                    .assumingMemoryBound(to: UInt8.self)[0]
                                guard byte == UInt8(Int(expert) % 251) else {
                                    throw ParallelExpertReader.Failure.readFailed(errno: EIO)
                                }
                            }
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    @Test func claimTakesTheOlderBatchBeforeTheNewer() {
        let bothActiveOlderFirst: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 1, sequence: 5, next_index: 0, count: 2),
            shrike_expert_batch_slot(active: 1, sequence: 6, next_index: 0, count: 2),
        ]
        #expect(Self.claim(bothActiveOlderFirst) == 0)

        let olderExhausted: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 1, sequence: 5, next_index: 2, count: 2),
            shrike_expert_batch_slot(active: 1, sequence: 6, next_index: 0, count: 2),
        ]
        #expect(Self.claim(olderExhausted) == 1)

        let neitherClaimable: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 1, sequence: 5, next_index: 2, count: 2),
            shrike_expert_batch_slot(active: 1, sequence: 6, next_index: 2, count: 2),
        ]
        #expect(Self.claim(neitherClaimable) == -1)

        let inactiveNeverClaimed: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 0, sequence: 1, next_index: 0, count: 2),
            shrike_expert_batch_slot(active: 1, sequence: 6, next_index: 0, count: 2),
        ]
        #expect(Self.claim(inactiveNeverClaimed) == 1)
    }

    @Test func publishFindsNoSlotWhileBothAreActive() {
        let bothActive: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 1, sequence: 1, next_index: 0, count: 1),
            shrike_expert_batch_slot(active: 1, sequence: 2, next_index: 0, count: 1),
        ]
        #expect(Self.freeSlot(bothActive) == -1)

        let secondReaped: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 1, sequence: 1, next_index: 0, count: 1),
            shrike_expert_batch_slot(active: 0, sequence: 0, next_index: 0, count: 0),
        ]
        #expect(Self.freeSlot(secondReaped) == 1)

        let firstReaped: [shrike_expert_batch_slot] = [
            shrike_expert_batch_slot(active: 0, sequence: 0, next_index: 0, count: 0),
            shrike_expert_batch_slot(active: 1, sequence: 1, next_index: 0, count: 1),
        ]
        #expect(Self.freeSlot(firstReaped) == 0)
    }

    @Test func oneBatchsReadErrorDoesNotFailTheOtherCaller() async throws {
        let count = 16
        let url = try Self.makeFixture(count: count)
        defer { try? FileManager.default.removeItem(at: url) }
        let reader = try ParallelExpertReader(path: url.path,
                                             expertStride: Self.stride,
                                             threads: 2,
                                             batchDepth: 2)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try Self.withDestinations(4) { buffers in
                    for round in 0..<50 {
                        let ids = (0..<4).map { UInt32((round * 4 + $0) % count) }
                        try reader.fetch(experts: ids, into: buffers)
                        for (index, expert) in ids.enumerated() {
                            let byte = buffers[index]
                                .assumingMemoryBound(to: UInt8.self)[0]
                            guard byte == UInt8(Int(expert) % 251) else {
                                throw ParallelExpertReader.Failure.readFailed(errno: EIO)
                            }
                        }
                    }
                }
            }
            group.addTask {
                for _ in 0..<50 {
                    try Self.withDestinations(1) { buffers in
                        #expect(throws: ParallelExpertReader.Failure.self) {
                            try reader.fetch(experts: [UInt32(count + 5)], into: buffers)
                        }
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    @Test func cancellingAPublishedBatchAccountsOnlyItsUnclaimedReads() {
        var withUnclaimedReads = shrike_expert_batch_cancel_state(
            count: 5, next_index: 2, outstanding: 5, first_errno: 0)
        shrike_expert_reader_cancel_slot(&withUnclaimedReads)
        #expect(withUnclaimedReads.first_errno == ECANCELED)
        #expect(withUnclaimedReads.outstanding == 2)
        #expect(withUnclaimedReads.next_index == 5)

        var withAnExistingError = shrike_expert_batch_cancel_state(
            count: 4, next_index: 1, outstanding: 4, first_errno: EIO)
        shrike_expert_reader_cancel_slot(&withAnExistingError)
        #expect(withAnExistingError.first_errno == EIO)
        #expect(withAnExistingError.outstanding == 1)
        #expect(withAnExistingError.next_index == 4)

        var withNoUnclaimedReads = shrike_expert_batch_cancel_state(
            count: 3, next_index: 3, outstanding: 1, first_errno: 0)
        shrike_expert_reader_cancel_slot(&withNoUnclaimedReads)
        #expect(withNoUnclaimedReads.first_errno == 0)
        #expect(withNoUnclaimedReads.outstanding == 1)
        #expect(withNoUnclaimedReads.next_index == 3)
    }

    @Test func depthTwoReadsTheSameBytesAsDepthOne() throws {
        let count = 32
        let url = try Self.makeFixture(count: count)
        defer { try? FileManager.default.removeItem(at: url) }
        let wanted: [UInt32] = [31, 0, 15, 15, 20, 3, 30, 1]
        func firstBytes(threads: Int, batchDepth: Int) throws -> [UInt8] {
            let reader = try ParallelExpertReader(path: url.path,
                                                 expertStride: Self.stride,
                                                 threads: threads,
                                                 batchDepth: batchDepth)
            return try Self.withDestinations(wanted.count) { bufs in
                try reader.fetch(experts: wanted, into: bufs)
                return bufs.map { $0.assumingMemoryBound(to: UInt8.self)[0] }
            }
        }
        #expect(try firstBytes(threads: 1, batchDepth: 1) == firstBytes(threads: 8, batchDepth: 2))
    }

    @Test func batchDepthIsClampedToTheSupportedRange() throws {
        let url = try Self.makeFixture(count: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let low = try ParallelExpertReader(path: url.path,
                                          expertStride: Self.stride, batchDepth: 0)
        #expect(low.batchDepth == 1)
        let high = try ParallelExpertReader(path: url.path,
                                           expertStride: Self.stride, batchDepth: 9)
        #expect(high.batchDepth == 2)
    }
}
