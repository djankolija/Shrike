import Foundation
import Metal

/// The GPU-side copy of adopted predictions from the ring's buffers into their
/// reserved cache slots, encoded at the head of the command that computes them
/// so it runs under the demand read that command waits for. The ring's slots
/// stay leased until `release`, after that command has completed.
/// unchecked-invariant: immutable transfer resources; `release` runs its
/// closure once, serialized by `lock`.
final class PrefetchAdoptionTransfer: @unchecked Sendable {
    let plan: RoutedExpertFetchPlan?
    private let sources: [MTLBuffer]
    private let destinations: [MTLBuffer]
    private let destinationOffsets: [Int]
    private let byteCount: Int
    private let lock = NSLock()
    private var onRelease: ((Bool) -> Void)?

    var expertCount: Int { sources.count }

    init(plan: RoutedExpertFetchPlan? = nil,
         sources: [MTLBuffer],
         destinations: [MTLBuffer],
         destinationOffsets: [Int],
         byteCount: Int,
         onRelease: @escaping (Bool) -> Void) {
        precondition(sources.count == destinations.count)
        precondition(destinations.count == destinationOffsets.count)
        precondition(byteCount > 0)
        self.plan = plan
        self.sources = sources
        self.destinations = destinations
        self.destinationOffsets = destinationOffsets
        self.byteCount = byteCount
        self.onRelease = onRelease
    }

    func encodeCopy(commandBuffer: MTLCommandBuffer) throws {
        for index in destinations.indices {
            let destinationOffset = destinationOffsets[index]
            guard destinationOffset >= 0,
                  destinations[index].length - destinationOffset >= byteCount,
                  sources[index].length >= byteCount else {
                throw ModelError.internalInconsistency(
                    detail: "prefetch adoption transfer range is out of bounds")
            }
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        for index in destinations.indices {
            blit.copy(from: sources[index], sourceOffset: 0,
                      to: destinations[index], destinationOffset: destinationOffsets[index],
                      size: byteCount)
        }
        blit.endEncoding()
    }

    func release(adopted: Bool = true) {
        lock.lock()
        let closure = onRelease
        onRelease = nil
        lock.unlock()
        closure?(adopted)
    }
}
