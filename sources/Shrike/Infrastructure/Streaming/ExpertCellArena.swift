import Darwin
import Foundation
import Metal
import Synchronization

/// Every expert cell the classifier can name in one buffer, so a cell can
/// change owner at a landing's swap without a byte moving.
/// unchecked-invariant: the handles are immutable after init; a cell's bytes
/// are owned by the streamer or the ring holding that cell, under their own
/// locks, and its generation is read and written only under the cache lock of
/// the layer that owns the cell, ownership moving through the ring's lock
/// (ring then cache, never the reverse).
public final class ExpertCellArena: @unchecked Sendable {
    public static let allocationAlignment = 2 * 1024 * 1024

    public let cellCount: Int
    public let stride: Int
    public let buffer: MTLBuffer
    private let base: UnsafeMutableRawPointer
    private let generations: UnsafeMutablePointer<UInt64>
    private let clock = Atomic<UInt64>(0)

    public init(device: MTLDevice, cellCount: Int, stride: Int) throws {
        let pageSize = Int(getpagesize())
        guard cellCount > 0, stride > 0, stride % pageSize == 0 else {
            throw ModelError.internalInconsistency(
                detail: "invalid expert cell arena geometry: \(cellCount) cells of \(stride) bytes")
        }
        let (bytes, overflow) = stride.multipliedReportingOverflow(by: cellCount)
        guard !overflow else { throw StreamerError.allocFailed(errno: EOVERFLOW) }
        var raw: UnsafeMutableRawPointer?
        let result = posix_memalign(&raw, Self.allocationAlignment, bytes)
        guard result == 0, let pointer = raw else {
            throw StreamerError.allocFailed(errno: result)
        }
        nonisolated(unsafe) let capturedPointer = pointer
        guard let buffer = device.makeBuffer(
            bytesNoCopy: pointer,
            length: bytes,
            options: .storageModeShared,
            deallocator: { _, _ in free(capturedPointer) })
        else {
            free(pointer)
            throw StreamerError.bufferWrapFailed
        }
        buffer.label = "expert.cells"
        let generations = UnsafeMutablePointer<UInt64>.allocate(capacity: cellCount)
        generations.initialize(repeating: 0, count: cellCount)
        self.cellCount = cellCount
        self.stride = stride
        self.buffer = buffer
        self.base = pointer
        self.generations = generations
    }

    deinit {
        generations.deallocate()
    }

    public func cellGeneration(_ cell: Int) -> UInt64 {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        return generations[cell]
    }

    @discardableResult
    func bumpCellGeneration(_ cell: Int) -> UInt64 {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        generations[cell] = clock.add(1, ordering: .relaxed).newValue
        return generations[cell]
    }

    public func pointer(cell: Int) -> UnsafeMutableRawPointer {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        return base.advanced(by: cell * stride)
    }

    public func offset(cell: Int) -> UInt64 {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        return UInt64(cell * stride)
    }

    public func cell(atOffset offset: UInt64) -> Int {
        precondition(offset % UInt64(stride) == 0 && offset < UInt64(stride) * UInt64(cellCount),
                     "offset is not an expert cell of this arena")
        return Int(offset / UInt64(stride))
    }
}
