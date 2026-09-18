import Darwin
import Foundation
import Metal
import Synchronization

/// Every expert cell the classifier can name, numbered globally, in as many
/// buffers as the device's `maxBufferLength` needs, so a cell can change owner
/// at a landing's swap without a byte moving and the pool can outgrow one
/// buffer (8.88 GiB on the mini). The kernels reach a cell through its chunk's
/// base, the `PoolBases` struct in moe.metal that `bases` holds.
/// unchecked-invariant: the handles are immutable after init; a cell's bytes
/// are owned by the streamer or the ring holding that cell, under their own
/// locks, and its generation is read and written only under the cache lock of
/// the layer that owns the cell, ownership moving through the ring's lock
/// (ring then cache, never the reverse).
public final class ExpertCellArena: @unchecked Sendable {
    public static let allocationAlignment = 2 * 1024 * 1024
    /// `PoolBases.base` in moe.metal has this many entries.
    public static let maxChunks = 8

    public let cellCount: Int
    public let stride: Int
    public let cellsPerChunk: Int
    public let chunkBuffers: [MTLBuffer]
    public let bases: MTLBuffer
    private let chunkPointers: [UnsafeMutableRawPointer]
    private let generations: UnsafeMutablePointer<UInt64>
    private let clock = Atomic<UInt64>(0)

    public init(device: MTLDevice, cellCount: Int, stride: Int, chunkBytes: Int? = nil) throws {
        let pageSize = Int(getpagesize())
        guard cellCount > 0, stride > 0, stride % pageSize == 0 else {
            throw ModelError.internalInconsistency(
                detail: "invalid expert cell arena geometry: \(cellCount) cells of \(stride) bytes")
        }
        let limit = chunkBytes ?? device.maxBufferLength
        let cellsPerChunk = max(1, min(cellCount, limit / stride))
        let chunkCount = (cellCount + cellsPerChunk - 1) / cellsPerChunk
        guard chunkCount <= Self.maxChunks else {
            throw ModelError.internalInconsistency(
                detail: "expert cell arena of \(cellCount) cells needs \(chunkCount) chunks of "
                    + "\(cellsPerChunk); the kernels address at most \(Self.maxChunks)")
        }
        var buffers: [MTLBuffer] = []
        var pointers: [UnsafeMutableRawPointer] = []
        for chunk in 0..<chunkCount {
            let cells = min(cellsPerChunk, cellCount - chunk * cellsPerChunk)
            let (bytes, overflow) = stride.multipliedReportingOverflow(by: cells)
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
            buffer.label = "expert.cells.\(chunk)"
            buffers.append(buffer)
            pointers.append(pointer)
        }
        guard let bases = device.makeBuffer(length: Self.maxChunks * 8 + 8, options: .storageModeShared) else {
            throw StreamerError.bufferWrapFailed
        }
        bases.label = "expert.cells.bases"
        let addresses = bases.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<Self.maxChunks {
            addresses[i] = buffers[min(i, buffers.count - 1)].gpuAddress
        }
        bases.contents().storeBytes(of: UInt32(cellsPerChunk), toByteOffset: Self.maxChunks * 8,
                                    as: UInt32.self)
        let generations = UnsafeMutablePointer<UInt64>.allocate(capacity: cellCount)
        generations.initialize(repeating: 0, count: cellCount)
        self.cellCount = cellCount
        self.stride = stride
        self.cellsPerChunk = cellsPerChunk
        self.chunkBuffers = buffers
        self.bases = bases
        self.chunkPointers = pointers
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
        return chunkPointers[cell / cellsPerChunk].advanced(by: (cell % cellsPerChunk) * stride)
    }

    public func buffer(cell: Int) -> MTLBuffer {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        return chunkBuffers[cell / cellsPerChunk]
    }

    /// The cell's offset inside its chunk's buffer, for binding.
    public func bufferOffset(cell: Int) -> UInt64 {
        precondition(cell >= 0 && cell < cellCount, "expert cell out of range")
        return UInt64((cell % cellsPerChunk) * stride)
    }

    /// The cell's global offset, `cell × stride`, the identity the streamer keys on.
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
