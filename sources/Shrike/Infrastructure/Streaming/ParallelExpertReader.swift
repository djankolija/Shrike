import Foundation
import ShrikeKernelsC

/// Reads routed-expert blocks from a packed layer file, in parallel.
///
/// Device ceiling, measured against a 16.88 GiB working set -- deliberately larger
/// than this machine's page cache, so the numbers are the disk's and not RAM's:
///
///     1 thread   2.43 GB/s
///     4 threads  3.92 GB/s
///     8 threads  3.92 GB/s   (saturated)
///
/// Four readers is the knee.
///
/// `batchDepth` lets a second caller publish its batch while the first is
/// still outstanding, instead of parking until the first clears -- see
/// `include/shrike_expert_io.h` for the two-slot ring and the FIFO claim that
/// keeps an older batch's reads from being starved by a newer one. This
/// initializer's own default is 1 (a single-batch reader); production's
/// default is 2, set by `BoundedReaderConfiguration` in `PreadExpertStreamer.swift`.
///
/// Expert reads bypass the unified buffer cache: streaming 16.88 GiB that way
/// left the machine at 78% free memory, so the slot cache stays the only cache
/// and a declared RAM budget means what it says.
/// unchecked-invariant: the C reader serializes batch publication under its
/// mutex and this wrapper stores no mutable Swift state.
public final class ParallelExpertReader: @unchecked Sendable {
    /// The C reader owns a fixed pool created in `init` and
    /// serialises every batch behind its own mutex, so concurrent `fetch` calls
    /// are safe at the C level. This type adds no Swift mutable state -- every
    /// stored property is immutable after init -- so there is nothing here for
    /// a second caller to corrupt.
    private let handle: OpaquePointer

    public let threadCount: Int
    public let batchDepth: Int
    public let expertStride: Int

    public enum Failure: Error, CustomStringConvertible {
        case openFailed(path: String, errno: Int32)
        case readFailed(errno: Int32)

        public var description: String {
            switch self {
            case .openFailed(let path, let code):
                return "parallel expert reader could not open \(path): "
                    + String(cString: strerror(code))
            case .readFailed(let code):
                return "parallel expert read failed: \(String(cString: strerror(code)))"
            }
        }
    }

    /// - Parameters:
    ///   - threads: readers to run concurrently; clamped to 1...16 by the C layer.
    ///     Four saturates the development machine.
    ///   - batchDepth: published batches held at once; clamped to
    ///     1...SHRIKE_IO_MAX_BATCHES by the C layer. Default 1; production's
    ///     default is 2, set by `BoundedReaderConfiguration`.
    public init(path: String,
                expertStride: Int,
                threads: Int = 4,
                batchDepth: Int = 1) throws {
        precondition(expertStride > 0, "expertStride must be positive")
        var failure: Int32 = 0
        guard let handle = shrike_expert_reader_create(path,
                                                     expertStride,
                                                     Int32(threads),
                                                     Int32(batchDepth),
                                                     1,
                                                     &failure) else {
            throw Failure.openFailed(path: path, errno: failure)
        }
        self.handle = handle
        self.expertStride = expertStride
        self.threadCount = Int(shrike_expert_reader_threads(handle))
        self.batchDepth = Int(shrike_expert_reader_batch_depth(handle))
    }

    deinit {
        shrike_expert_reader_destroy(handle)
    }

    /// Reads `experts[i]` into `destinations[i]` and returns once all have landed.
    ///
    /// Each destination must be at least `expertStride` bytes. On failure the
    /// destinations hold undefined bytes and must not be used -- a partially
    /// filled expert slot would otherwise be indistinguishable from a valid one.
    public func fetch(experts: [UInt32],
                      into destinations: [UnsafeMutableRawPointer]) throws {
        precondition(experts.count == destinations.count,
                     "experts and destinations must be the same length")
        guard !experts.isEmpty else { return }
        let status = destinations.withUnsafeBufferPointer { dst in
            // `void *const *` imports with an optional element type. A
            // non-optional pointer has identical layout to its optional, so the
            // rebind is a type-level adjustment and moves no bytes.
            dst.withMemoryRebound(to: UnsafeMutableRawPointer?.self) { rebound in
                experts.withUnsafeBufferPointer { ids in
                    shrike_expert_reader_fetch(handle,
                                              ids.baseAddress,
                                              rebound.baseAddress,
                                              experts.count)
                }
            }
        }
        if status != 0 {
            throw Failure.readFailed(errno: status)
        }
    }

    /// As `fetch(experts:into:)` but with absolute byte offsets.
    ///
    /// The streamer's regions carry a per-layer base and a container offset, so an
    /// expert index alone would address the wrong layer.
    public func fetch(offsets: [UInt64],
                      into destinations: [UnsafeMutableRawPointer]) throws {
        precondition(offsets.count == destinations.count,
                     "offsets and destinations must be the same length")
        guard !offsets.isEmpty else { return }
        let status = destinations.withUnsafeBufferPointer { dst in
            dst.withMemoryRebound(to: UnsafeMutableRawPointer?.self) { rebound in
                offsets.withUnsafeBufferPointer { offs in
                    shrike_expert_reader_fetch_offsets(handle,
                                                      offs.baseAddress,
                                                      rebound.baseAddress,
                                                      offsets.count)
                }
            }
        }
        if status != 0 {
            throw Failure.readFailed(errno: status)
        }
    }
}
