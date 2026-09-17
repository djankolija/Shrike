import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

/// A decode boundary without a host round trip: the pass ends with the head, the
/// caller's sampler and the next embed in one command, and the caller waits for
/// the sampled id's word instead of the command. The pass after this one is
/// committed behind it before its token is known (v20 T3.3), so a stop the
/// caller sees at the word is one pass late: the producer drains that pass at
/// its next entry point, and the caller passes `last` when no pass after this
/// one is wanted.
public protocol BoundaryLogitProducer: LogitProducer {
    /// Runs the pass for `position`; a nil `token` continues the pass the
    /// previous call committed ahead. `sample` encodes the caller's sampler at
    /// a boundary, given the position of the pass it ends and the word its
    /// token goes into: this pass's when the pass is fresh, and the next
    /// pass's at the end of every pass unless `last`.
    func produce(token: Int32?, position: Int, into logits: MTLBuffer, last: Bool,
                 sample: @escaping (MTLComputeCommandEncoder, Int, MTLBuffer) throws -> Void)
        async throws
    /// The id the last boundary's sampler wrote, once the host can see it.
    func awaitBoundaryToken() throws -> Int32
    /// The caller wants no more passes: everything the pass committed ahead
    /// waits on is published, so it runs through on its own while the caller
    /// finishes; the producer's next entry point waits it out.
    func releasePassAhead()
}

/// The class-2 gate's instrument (docs/v19-scan-rewrite.md, Task 1).
public protocol LogitsSink: AnyObject, Sendable {
    func record(position: Int, logits: UnsafeBufferPointer<Float16>)
    func chose(position: Int, token: Int32)
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}
