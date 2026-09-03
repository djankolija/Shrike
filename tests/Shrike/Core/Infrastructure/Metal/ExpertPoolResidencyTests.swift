import Foundation
import Metal
import Testing
@testable import Shrike

@Suite struct ExpertPoolResidencyTests {
    private static func makePool(device: MTLDevice, length: Int,
                                 onFree: @escaping @Sendable () -> Void) -> MTLBuffer? {
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, 16_384, length) == 0, let pointer = raw else { return nil }
        // unchecked-invariant: only the deallocator below ever touches this pointer, and Metal
        // calls it at most once for the buffer it was handed to.
        nonisolated(unsafe) let capturedPointer = pointer
        return device.makeBuffer(
            bytesNoCopy: pointer, length: length, options: .storageModeShared,
            deallocator: { _, _ in
                free(capturedPointer)
                onFree()
            })
    }

    @Test func expertPoolResidencySetHoldsThePoolBuffer() throws {
        let context = try MetalContext()
        let length = 4 * 16_384
        guard let pool = Self.makePool(device: context.device, length: length, onFree: {}),
              let other = context.device.makeBuffer(length: 16_384, options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            return
        }
        let residency = try ExpertPoolResidency(device: context.device, queue: context.queue)
        residency.include(pool)
        residency.include(pool)
        #expect(residency.allocationCount == 1)
        #expect(residency.allocatedBytes >= UInt64(length))
        residency.include(other)
        #expect(residency.allocationCount == 2)
        #expect(residency.allocatedBytes >= UInt64(length + 16_384))
    }

    @Test func expertPoolResidencyReleasesThePoolWhenItGoesAway() throws {
        let context = try MetalContext()
        let freed = Freed()
        do {
            guard let pool = Self.makePool(device: context.device, length: 16_384,
                                           onFree: { freed.mark() }) else {
                Issue.record("buffer allocation failed")
                return
            }
            let residency = try ExpertPoolResidency(device: context.device, queue: context.queue)
            residency.include(pool)
            #expect(residency.allocationCount == 1)
            #expect(!freed.value)
        }
        #expect(freed.value, "the pool's deallocator did not run after the holder and the buffer went away")
    }

    // unchecked-invariant: `flag` is only ever read or written under `lock`, so the type is
    // safe to share across the deallocator's thread and the asserting test thread.
    private final class Freed: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        func mark() { lock.lock(); flag = true; lock.unlock() }
        var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    }
}
