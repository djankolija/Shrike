import Foundation
import Metal

/// Streaming callbacks from `runRawCompletion`. `.prefill` reports monotonic
/// producer-defined prompt progress; scalar replay reports per token, while a
/// prefill-capable producer may report per internal chunk. `.token` fires per
/// decoded non-stop token; `.tail` carries the detokenizer flush remainder at a
/// stop boundary.
public enum RawDecodeProgress: Sendable {
    case prefill(done: Int, total: Int)
    case token(index: Int, id: Int32, delta: String)
    case tail(String)
}

public enum RawCompletionStart: Sendable, Equatable {
    case reset
    case resume(cachedPromptTokens: Int)
}

public struct RawDecodeResult: Sendable {
    public let prefillTokens: Int
    public let cachedPromptTokens: Int
    public let computedPrefillTokens: Int
    public let prefillSeconds: Double
    public let newTokens: Int
    public let decodeSeconds: Double
    public let reason: StopReason
    public let kvPosition: Int
    public let kvBackedTokenIDs: [Int32]
    public let uncommittedBoundaryTokenIDs: [Int32]
}

/// Preallocated per-generation buffers (two 512 KiB vocab buffers plus a token
/// slot) and sampler. A warm session reuses them for every token, avoiding
/// per-token Metal buffer allocation.
///
/// unchecked-invariant: the buffers and sampler are exclusively owned by one
/// generation at a time — the single-in-flight guard upstream is the contract.
public struct RawCompletionScratch: @unchecked Sendable {
    /// Public so a caller inside the same single-in-flight window can prefill
    /// through it without allocating a second vocab-sized buffer.
    public let logits: MTLBuffer
    let probs: MTLBuffer
    let outToken: MTLBuffer
    let sampler: Sampler

    public init(context: MetalContext, vocab: Int, logitSoftcap: Float = 0.0) throws {
        guard let logits = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                     options: .storageModeShared),
              let probs = context.device.makeBuffer(length: vocab * MemoryLayout<Float16>.size,
                                                    options: .storageModeShared),
              let outToken = context.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                                       options: .storageModeShared)
        else {
            throw ModelError.residentBufferWrapFailed
        }
        self.logits = logits
        self.probs = probs
        self.outToken = outToken
        self.sampler = try Sampler(context: context, vocab: vocab,
                                   logitSoftcap: logitSoftcap)
    }
}

extension GenerationConfig {
    /// A pure-greedy config can use the fused head's GPU argmax
    /// (`RealForwardRunner.lastGreedyToken`) instead of sampling from the
    /// logits buffer. Anything else needs real logits.
    public var isPureGreedy: Bool {
        temperature == 0 && presencePenalty == 0 && repetitionPenalty == 1
    }

}

/// Raw-completion prefill + decode loop shared by the CLI and the Mac app.
/// Consumes pre-encoded `promptIds` (BOS + verbatim encode upstream — no chat
/// template). Stop handling, detokenizer flush ordering, and history append
/// ordering are shared by both front ends.
///
/// When the producer runs the fused lm_head (`RealForwardRunner` default) the
/// logits buffer is never written; the loop then requires a pure-greedy config
/// and reads `lastGreedyToken`. Callers with sampling configs must construct
/// the runner with `forceLogitsHead: true`.
public func runRawCompletion(producer: any LogitProducer,
                             tokenizer: GFTokenizer,
                             promptIds: [Int32],
                             config: GenerationConfig,
                             context: MetalContext,
                             scratch: RawCompletionScratch,
                             prefillConfig: PrefillRuntimeConfig = .defaultChunked,
                             start: RawCompletionStart = .reset,
                             shouldStop: () -> Bool = { false },
                             onProgress: (RawDecodeProgress) -> Void) async throws -> RawDecodeResult {
    try config.validate()
    guard !promptIds.isEmpty else {
        throw GeneratorError.emptyPrompt
    }
    let fusedRunner = producer as? RealForwardRunner
    let fusedGreedy = fusedRunner?.usesFusedGreedyHead == true
    guard !fusedGreedy || config.isPureGreedy else {
        throw PrefillError.unsupportedPrefillSeed(
            "the fused-head producer cannot serve this sampling configuration; use a logits head")
    }

    let cachedPromptTokens: Int
    switch start {
    case .reset:
        cachedPromptTokens = 0
    case .resume(let count):
        guard count > 0, count < promptIds.count else {
            throw GeneratorError.invalidContinuation(
                "cached prompt token count must be greater than zero and less than the effective prompt")
        }
        guard producer is any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        cachedPromptTokens = count
    }
    let computedPrefillTokens = promptIds.count - cachedPromptTokens
    fusedRunner?.recordRouteTraceRequestStart(cachedTokens: cachedPromptTokens,
                                              promptTokens: promptIds.count)

    var detok = GFDetokenizer(tokenizer: tokenizer)
    var history = Array(promptIds.prefix(cachedPromptTokens))
    history.reserveCapacity(promptIds.count + config.maxNewTokens)

    if let context = producer as? any ContextWindowReporting {
        // A resume already occupies `cachedPromptTokens` KV rows, so only the
        // uncached prompt plus the response is new work — it must fit the
        // remaining capacity. Algebraically this is the final-KV-position
        // bound (`promptIds.count + maxNewTokens <= maxContext`); written in
        // remaining-capacity form so near-maxContext continuations are not
        // over-rejected (R9).
        let newRows = (promptIds.count - cachedPromptTokens) + config.maxNewTokens
        let remainingCapacity = context.maxContext - cachedPromptTokens
        if newRows > remainingCapacity {
            throw GeneratorError.contextOverflow(prompt: promptIds.count,
                                                 maxNew: config.maxNewTokens,
                                                 maxContext: context.maxContext)
        }
    }
    switch start {
    case .reset:
        producer.reset()
    case .resume:
        // Re-derive the conformance rather than force-cast on the guard 30
        // lines above: a trap here would take down the server process, and the
        // invariant is far enough away to be broken by an unrelated edit.
        guard let continuable = producer as? any ContinuableLogitProducer else {
            throw GeneratorError.invalidContinuation(
                "producer does not support continuation")
        }
        try continuable.prepareForContinuation(expectedPosition: cachedPromptTokens)
    }
    let prefillStart = Date()
    var position = cachedPromptTokens
    let prefillSeed = try await runPrefill(producer: producer,
                                           promptIds: promptIds,
                                           cachedPromptTokens: cachedPromptTokens,
                                           config: config,
                                           scratch: scratch,
                                           prefillConfig: prefillConfig,
                                           fusedGreedy: fusedGreedy,
                                           position: &position,
                                           history: &history,
                                           onProgress: onProgress)

    let decodeStart = Date()
    let prefillSeconds = decodeStart.timeIntervalSince(prefillStart)
    let outcome = try await runDecodeLoop(producer: producer,
                                          tokenizer: tokenizer,
                                          config: config,
                                          context: context,
                                          scratch: scratch,
                                          fusedRunner: fusedRunner,
                                          fusedGreedy: fusedGreedy,
                                          prefillSeed: prefillSeed,
                                          detok: &detok,
                                          history: &history,
                                          position: &position,
                                          shouldStop: shouldStop,
                                          onProgress: onProgress)

    return RawDecodeResult(prefillTokens: promptIds.count,
                           cachedPromptTokens: cachedPromptTokens,
                           computedPrefillTokens: computedPrefillTokens,
                           prefillSeconds: prefillSeconds,
                           newTokens: outcome.generated,
                           decodeSeconds: Date().timeIntervalSince(decodeStart),
                           reason: outcome.reason,
                           kvPosition: position,
                           kvBackedTokenIDs: history,
                           uncommittedBoundaryTokenIDs: outcome.uncommittedBoundaryTokenIDs)
}

private func runPrefill(producer: any LogitProducer,
                        promptIds: [Int32],
                        cachedPromptTokens: Int,
                        config: GenerationConfig,
                        scratch: RawCompletionScratch,
                        prefillConfig: PrefillRuntimeConfig,
                        fusedGreedy: Bool,
                        position: inout Int,
                        history: inout [Int32],
                        onProgress: (RawDecodeProgress) -> Void) async throws -> PrefillSeed? {
    let prefillTokens = promptIds[cachedPromptTokens...]
    switch prefillConfig.mode {
    case .chunked where producer is any ChunkedPrefillRunner:
        // The `where` clause one line above is the guard; a producer without
        // the conformance falls through to plain `.chunked`.
        // swiftlint:disable:next force_cast
        let chunked = producer as! any ChunkedPrefillRunner
        let mode: PrefillOutputMode = fusedGreedy ? .greedyIfAvailable : .logits
        let result = try await chunked.prefillChunked(tokens: prefillTokens,
                                                      startPosition: position,
                                                      outputMode: mode,
                                                      config: prefillConfig,
                                                      into: scratch.logits) { done in
            onProgress(.prefill(done: cachedPromptTokens + done, total: promptIds.count))
        }
        if mode == .logits, result.seed != .logitsWritten {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill requested logits but producer returned \(result.seed)")
        }
        if case .greedyToken = result.seed, !config.isPureGreedy {
            throw PrefillError.unsupportedPrefillSeed(
                "RawCompletion chunked prefill returned a greedy token for a sampling config")
        }
        position = result.newPosition
        let seed = result.seed
        history.append(contentsOf: prefillTokens)
        return seed
    case .chunked:
        throw PrefillError.chunkedUnsupported(
            PrefillError.chunkedRequiresChunkedRunnerReason)
    case .off:
        for t in prefillTokens {
            try Task.checkCancellation()
            try await producer.produce(token: t, position: position, into: scratch.logits)
            position += 1
            history.append(t)
            onProgress(.prefill(done: position, total: promptIds.count))
        }
        return nil
    }
}

private struct DecodeLoopOutcome {
    let generated: Int
    let reason: StopReason
    let uncommittedBoundaryTokenIDs: [Int32]
}

private func runDecodeLoop(producer: any LogitProducer,
                           tokenizer: GFTokenizer,
                           config: GenerationConfig,
                           context: MetalContext,
                           scratch: RawCompletionScratch,
                           fusedRunner: RealForwardRunner?,
                           fusedGreedy: Bool,
                           prefillSeed: PrefillSeed?,
                           detok: inout GFDetokenizer,
                           history: inout [Int32],
                           position: inout Int,
                           shouldStop: () -> Bool,
                           onProgress: (RawDecodeProgress) -> Void) async throws -> DecodeLoopOutcome {
    // The scratch sampler persists across generations; its incremental
    // repetition-penalty history is per-generation (R25).
    scratch.sampler.resetPenaltyHistory()
    var stopMatcher = StreamingStopMatcher(stops: config.stopStrings)
    var generated = 0
    var reason: StopReason = .maxTokens
    var uncommittedBoundaryTokenIDs: [Int32] = []
    let boundaryProducer = producer as? any BoundaryLogitProducer
    let useBoundary = boundaryProducer != nil && !fusedGreedy && config.repetitionPenalty == 1.0
    var boundaryPending = false

    while true {
        try Task.checkCancellation()

        let tLoopStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let tokenID: Int32
        if generated == 0, let seed = prefillSeed {
            switch seed {
            case .greedyToken(let token):
                tokenID = Int32(bitPattern: token)
            case .logitsWritten:
                tokenID = try sampleOnce(scratch: scratch, context: context,
                                     history: history, config: config, position: generated,
                                     timing: fusedRunner)
            }
        } else if fusedGreedy {
            tokenID = Int32(bitPattern: fusedRunner!.lastGreedyToken)
        } else if boundaryPending, let boundaryProducer {
            tokenID = try boundaryProducer.awaitBoundaryToken()
        } else {
            tokenID = try sampleOnce(scratch: scratch, context: context,
                                 history: history, config: config, position: generated,
                                 timing: fusedRunner)
        }
        let tSampled = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        generated += 1
        uncommittedBoundaryTokenIDs = [tokenID]

        if tokenizer.stopTokenIDs.contains(tokenID) || config.extraStopTokens.contains(tokenID) {
            if tokenID == tokenizer.endOfTurnID {
                reason = .endOfTurn
            } else if tokenID == tokenizer.toolResponseID {
                reason = .toolCalls
            } else {
                reason = .eos
            }
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            break
        }

        let delta = try detok.push(tokenID)
        let visible = stopMatcher.push(delta)
        let tDetok = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        onProgress(.token(index: generated - 1, id: tokenID, delta: visible))
        let tProgress = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        let hitStopString = stopMatcher.isStopped || shouldStop()
        let hitMax = generated >= config.maxNewTokens
        if hitStopString || hitMax {
            let tail = stopMatcher.push(detok.flush()) + stopMatcher.finish()
            if !tail.isEmpty { onProgress(.tail(tail)) }
            if hitStopString {
                // A configured stop string truncates output; the caller's
                // external stop signal reports `.external` instead (R35).
                reason = stopMatcher.isStopped ? .stopString : .external
            } else {
                reason = .maxTokens
            }
            break
        }

        history.append(tokenID)
        let tProduceStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        if useBoundary, let boundaryProducer {
            let samplePosition = generated
            try await boundaryProducer.produce(token: boundaryPending ? nil : tokenID,
                                               position: position, into: scratch.logits,
                                               tokenWord: scratch.outToken) { cb in
                try scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits,
                                           probs: scratch.probs, history: [],
                                           config: config, position: samplePosition,
                                           outToken: scratch.outToken)
            }
            boundaryPending = true
        } else {
            try await producer.produce(token: tokenID, position: position, into: scratch.logits)
        }
        fusedRunner?.recordDecodeLoopPhases(
            sample: tSampled - tLoopStart,
            detok: tDetok - tSampled,
            progress: tProgress - tDetok,
            produce: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tProduceStart)
        position += 1
        uncommittedBoundaryTokenIDs.removeAll(keepingCapacity: true)
    }

    return DecodeLoopOutcome(generated: generated, reason: reason,
                             uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs)
}

/// Samples one token id.
///
/// `timing` is the runner whose `SHRIKE_KERNEL_STATS` timeline this command
/// buffer joins, when the producer is one. Without it the sampler's GPU span
/// is invisible to the role summary *and* to the gap accounting, so it lands
/// inside the `head_logits->embed` transition and inflates what reads as idle.
/// That is not hypothetical: it hid a 15.45 ms/token Top-K kernel until the
/// gap was traced by hand.
private func sampleOnce(scratch: RawCompletionScratch, context: MetalContext,
                        history: [Int32], config: GenerationConfig, position: Int,
                        timing: RealForwardRunner? = nil) throws -> Int32 {
    guard let cb = context.queue.makeCommandBuffer() else {
        throw ModelError.residentBufferWrapFailed
    }
    try scratch.sampler.sample(commandBuffer: cb, logits: scratch.logits, probs: scratch.probs,
                               history: history, config: config, position: position,
                               outToken: scratch.outToken)
    cb.commit(); cb.waitUntilCompleted()
    timing?.recordKernelGPU(role: "sample", cb)
    return Int32(bitPattern: scratch.outToken.contents().load(as: UInt32.self))
}
