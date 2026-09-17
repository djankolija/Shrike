import Foundation
import Metal

/// GPU dependency for one storage batch. Value zero is never issued, and a
/// value is never reused for the lifetime of the coordinator.
/// unchecked-invariant: token fields are immutable Metal references and values.
public struct ExpertIOCompletionToken: @unchecked Sendable {
    public let event: MTLSharedEvent
    public let value: UInt64
    /// 0 = loading, 1 = complete, 2 = failed. GPU consumers wait on `event`
    /// before reading this shared word.
    public let status: MTLBuffer
    public let statusOffset: Int
}

/// Owns one shared Metal timeline for all expert reads in a model.
///
/// Out-of-order operations are deliberately held until every preceding value
/// is terminal. Advancing a timeline directly from N to N+2 would also release
/// a GPU wait for N+1, potentially before that batch's bytes were valid.
/// unchecked-invariant: allocation, status publication, and timeline advance
/// are serialized by `lock`.
public final class ExpertIOEventCoordinator: @unchecked Sendable {
    private let device: MTLDevice
    private let event: MTLSharedEvent
    private let lock = NSLock()
    private var nextValue: UInt64 = 1
    private var publishedValue: UInt64 = 0
    private var terminalValues: Set<UInt64> = []
    /// The status words: one ring of `statusWordCount`, a value's word at
    /// `(value - 1) % statusWordCount`, rewritten that many values later.
    /// Every routed layer reserves a value at its encode (v20 T3.1), forty per
    /// token, and a token's command drains within a few tokens; a wait older
    /// than the ring would have hung the runner long before its word turned.
    static let statusWordCount = 4_096
    private var statusWords: MTLBuffer?

    init?(device: MTLDevice) {
        guard let event = device.makeSharedEvent() else { return nil }
        event.label = "Shrike expert I/O completion"
        self.device = device
        self.event = event
    }

    func reserve() throws -> ExpertIOCompletionToken {
        lock.lock()
        defer { lock.unlock() }
        guard nextValue != 0 && nextValue < UInt64.max else {
            throw ModelError.internalInconsistency(
                detail: "expert I/O shared-event value space exhausted")
        }
        if statusWords == nil {
            guard let words = device.makeBuffer(
                length: Self.statusWordCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            words.label = "Shrike expert I/O status words"
            statusWords = words
        }
        let status = statusWords!
        let statusOffset = Int((nextValue - 1) % UInt64(Self.statusWordCount))
            * MemoryLayout<UInt32>.stride
        status.contents().advanced(by: statusOffset)
            .storeBytes(of: UInt32(0), as: UInt32.self)
        let token = ExpertIOCompletionToken(
            event: event,
            value: nextValue,
            status: status,
            statusOffset: statusOffset)
        nextValue &+= 1
        return token
    }

    func publish(_ token: ExpertIOCompletionToken, succeeded: Bool) {
        lock.lock()
        token.status.contents().advanced(by: token.statusOffset).storeBytes(
            of: succeeded ? UInt32(1) : UInt32(2),
            as: UInt32.self)
        terminalValues.insert(token.value)
        while terminalValues.remove(publishedValue &+ 1) != nil {
            publishedValue &+= 1
        }
        // A CPU signal after the slot bytes and status word are written is the
        // release operation paired with encodeWaitForEvent on the GPU queue.
        event.signaledValue = publishedValue
        lock.unlock()
    }
}
