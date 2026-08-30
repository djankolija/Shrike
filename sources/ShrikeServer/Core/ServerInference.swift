import CryptoKit
import Foundation
import Shrike

public enum ServerInferenceEvent: Equatable, Sendable {
    case content(String)
    case thinking(String)
    case toolCall(ParsedToolCall)
}

public struct ServerCompletion: Equatable, Sendable {
    public let content: String
    /// Harmony analysis text (`reasoning_content` on the wire); nil when the
    /// dialect has no thinking channel or none was produced.
    public let reasoningContent: String?
    public let toolCalls: [ParsedToolCall]
    public let finishReason: String
    public let usage: OpenAIUsage

    public init(content: String,
                reasoningContent: String? = nil,
                toolCalls: [ParsedToolCall],
                finishReason: String,
                usage: OpenAIUsage) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
        self.finishReason = finishReason
        self.usage = usage
    }
}

// MARK: - Structured Output Diagnostics (#90)

/// Kinds of structured-output failures that can be diagnosed.
enum StructuredOutputFailureKind: String, Equatable, Sendable {
    case decoderConsume = "decoder_consume"
    case decoderFinish = "decoder_finish"
    case orphanToolResponse = "orphan_tool_response"
}

/// Classifies the root cause of a structured-output failure.
enum StructuredOutputFailureCause: String, Equatable, Sendable {
    case malformed
    case unknownTool = "unknown_tool"
    case oversized
    case unexpected
    case none

    static func classify(_ error: Error) -> Self {
        guard let parserError = error as? ToolCallParserError else {
            return .unexpected
        }
        switch parserError {
        case .malformed: return .malformed
        case .unknownTool: return .unknownTool
        case .oversized: return .oversized
        }
    }

    static func unknownToolName(_ error: Error) -> String? {
        if case ToolCallParserError.unknownTool(let name) = error { return name }
        return nil
    }
}

/// Opt-in generated-token dump: set SHRIKE_GEN_DIAG=1 to log every
/// completion's generated token IDs to stderr, so channel-marker questions
/// (which 2000xx token preceded a text region) are answerable post-hoc.
enum ShrikeGenDiag {
    static let enabled =
        ProcessInfo.processInfo.environment["SHRIKE_GEN_DIAG"] != nil

    static func line(prefillTokens: Int,
                     kvBackedTokenIDs: [Int32],
                     boundaryTokenIDs: [Int32]) -> String {
        let prefill = min(max(prefillTokens, 0), kvBackedTokenIDs.count)
        let generated = Array(kvBackedTokenIDs.dropFirst(prefill))
            + boundaryTokenIDs
        return "Shrike gen_diag prefill=\(prefill) "
            + "generated=\(generated.count) ids=\(generated)"
    }
}

/// Rich diagnostic snapshot collected at structured-output failure time.
/// Includes SHA-256 hashes of token sequences for forensic comparison.
struct StructuredOutputFailureDiagnostics: Equatable, Sendable {
    let renderedPromptTokens: Int
    let effectivePromptTokens: Int
    let resultPromptTokens: Int
    let cachedPromptTokens: Int
    let computedPrefillTokens: Int
    let completionTokens: Int
    let maxCompletionTokens: Int
    let rawStop: String
    let kvPosition: Int
    let kvBackedTokens: Int
    let boundaryTokens: Int
    let decodedCalls: Int
    let visibleBytes: Int
    let stopStringMatched: Bool
    let toolStartCount: Int
    let toolEndCount: Int
    let toolResponseCount: Int
    let toolResponseEndCount: Int
    let lastToolStartOffset: Int
    let lastToolEndOffset: Int
    let lastToolResponseOffset: Int
    let lastToolResponseEndOffset: Int
    let effectiveCountMatchesResult: Bool
    let effectivePrefixMatchesKV: Bool
    let kvPositionMatchesHistory: Bool
    let completionCountMatchesHistory: Bool
    let prefillAccountingMatches: Bool
    let renderedPromptHash: String
    let effectivePromptHash: String
    let generatedHash: String

    init(
        renderedPromptIDs: [Int32],
        effectivePromptIDs: [Int32],
        result: RawDecodeResult,
        maxCompletionTokens: Int,
        decodedCalls: Int,
        visibleBytes: Int,
        stopStringMatched: Bool,
        toolStartID: Int32?,
        toolEndID: Int32?,
        toolResponseID: Int32?,
        toolResponseEndID: Int32?
    ) {
        let safePrefillCount = min(
            max(result.prefillTokens, 0),
            result.kvBackedTokenIDs.count)
        let committedGenerated = result.kvBackedTokenIDs.dropFirst(safePrefillCount)
        let boundary = result.uncommittedBoundaryTokenIDs[...]
        let generatedSegments = [committedGenerated, boundary]

        var toolStartCount = 0
        var toolEndCount = 0
        var toolResponseCount = 0
        var toolResponseEndCount = 0
        var lastToolStartOffset = -1
        var lastToolEndOffset = -1
        var lastToolResponseOffset = -1
        var lastToolResponseEndOffset = -1
        var offset = 0
        for segment in generatedSegments {
            for tokenID in segment {
                if tokenID == toolStartID {
                    toolStartCount += 1
                    lastToolStartOffset = offset
                }
                if tokenID == toolEndID {
                    toolEndCount += 1
                    lastToolEndOffset = offset
                }
                if tokenID == toolResponseID {
                    toolResponseCount += 1
                    lastToolResponseOffset = offset
                }
                if tokenID == toolResponseEndID {
                    toolResponseEndCount += 1
                    lastToolResponseEndOffset = offset
                }
                offset += 1
            }
        }

        let (prefillAccounted, prefillOverflow) = result.cachedPromptTokens
            .addingReportingOverflow(result.computedPrefillTokens)

        self.renderedPromptTokens = renderedPromptIDs.count
        self.effectivePromptTokens = effectivePromptIDs.count
        self.resultPromptTokens = result.prefillTokens
        self.cachedPromptTokens = result.cachedPromptTokens
        self.computedPrefillTokens = result.computedPrefillTokens
        self.completionTokens = result.newTokens
        self.maxCompletionTokens = maxCompletionTokens
        self.rawStop = Self.rawStop(result.reason)
        self.kvPosition = result.kvPosition
        self.kvBackedTokens = result.kvBackedTokenIDs.count
        self.boundaryTokens = result.uncommittedBoundaryTokenIDs.count
        self.decodedCalls = decodedCalls
        self.visibleBytes = visibleBytes
        self.stopStringMatched = stopStringMatched
        self.toolStartCount = toolStartCount
        self.toolEndCount = toolEndCount
        self.toolResponseCount = toolResponseCount
        self.toolResponseEndCount = toolResponseEndCount
        self.lastToolStartOffset = lastToolStartOffset
        self.lastToolEndOffset = lastToolEndOffset
        self.lastToolResponseOffset = lastToolResponseOffset
        self.lastToolResponseEndOffset = lastToolResponseEndOffset
        self.effectiveCountMatchesResult = effectivePromptIDs.count == result.prefillTokens
        self.effectivePrefixMatchesKV = result.kvBackedTokenIDs.count >= effectivePromptIDs.count
            && result.kvBackedTokenIDs.prefix(effectivePromptIDs.count)
                .elementsEqual(effectivePromptIDs)
        self.kvPositionMatchesHistory = result.kvPosition == result.kvBackedTokenIDs.count
        self.completionCountMatchesHistory = offset == result.newTokens
        self.prefillAccountingMatches = !prefillOverflow
            && prefillAccounted == result.prefillTokens
        self.renderedPromptHash = Self.i32leSHA256([renderedPromptIDs[...]])
        self.effectivePromptHash = Self.i32leSHA256([effectivePromptIDs[...]])
        self.generatedHash = Self.i32leSHA256(generatedSegments)
    }

    /// SHA-256 over little-endian UInt32 byte representations of token IDs.
    static func i32leSHA256(_ segments: [ArraySlice<Int32>]) -> String {
        var hasher = SHA256()
        var bytes = Data()
        bytes.reserveCapacity(4_096)
        for segment in segments {
            for tokenID in segment {
                var littleEndian = UInt32(bitPattern: tokenID).littleEndian
                withUnsafeBytes(of: &littleEndian) {
                    bytes.append(contentsOf: $0)
                }
                if bytes.count == 4_096 {
                    hasher.update(data: bytes)
                    bytes.removeAll(keepingCapacity: true)
                }
            }
        }
        if !bytes.isEmpty { hasher.update(data: bytes) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    var logDescription: String {
        [
            "rendered_prompt_tokens=\(renderedPromptTokens)",
            "effective_prompt_tokens=\(effectivePromptTokens)",
            "result_prompt_tokens=\(resultPromptTokens)",
            "cached_prompt_tokens=\(cachedPromptTokens)",
            "computed_prefill_tokens=\(computedPrefillTokens)",
            "completion_tokens=\(completionTokens)",
            "max_completion_tokens=\(maxCompletionTokens)",
            "raw_stop=\(rawStop)",
            "kv_position=\(kvPosition)",
            "kv_backed_tokens=\(kvBackedTokens)",
            "boundary_tokens=\(boundaryTokens)",
            "decoded_calls=\(decodedCalls)",
            "visible_bytes=\(visibleBytes)",
            "stop_string_matched=\(stopStringMatched)",
            "tool_start_count=\(toolStartCount)",
            "tool_end_count=\(toolEndCount)",
            "tool_response_count=\(toolResponseCount)",
            "tool_response_end_count=\(toolResponseEndCount)",
            "last_tool_start_offset=\(lastToolStartOffset)",
            "last_tool_end_offset=\(lastToolEndOffset)",
            "last_tool_response_offset=\(lastToolResponseOffset)",
            "last_tool_response_end_offset=\(lastToolResponseEndOffset)",
            "effective_count_matches_result=\(effectiveCountMatchesResult)",
            "effective_prefix_matches_kv=\(effectivePrefixMatchesKV)",
            "kv_position_matches_history=\(kvPositionMatchesHistory)",
            "completion_count_matches_history=\(completionCountMatchesHistory)",
            "prefill_accounting_matches=\(prefillAccountingMatches)",
            "rendered_prompt_i32le_sha256=\(renderedPromptHash)",
            "effective_prompt_i32le_sha256=\(effectivePromptHash)",
            "generated_i32le_sha256=\(generatedHash)",
        ].joined(separator: " ")
    }

    private static func rawStop(_ reason: StopReason) -> String {
        switch reason {
        case .eos: "eos"
        case .endOfTurn: "end_of_turn"
        case .maxTokens: "max_tokens"
        case .stopString: "stop_string"
        case .toolCalls: "tool_calls"
        case .external: "external"
        }
    }
}

/// A structured-output failure with full diagnostic context.
struct StructuredOutputFailure: Error, CustomDebugStringConvertible, Sendable {
    let kind: StructuredOutputFailureKind
    let cause: StructuredOutputFailureCause
    let unknownToolName: String?
    let diagnostics: StructuredOutputFailureDiagnostics

    var debugDescription: String {
        let name = unknownToolName.map { " unknown_tool_name=\($0)" } ?? ""
        return "structured_output_failure kind=\(kind.rawValue) "
            + "cause=\(cause.rawValue)\(name) \(diagnostics.logDescription)"
    }
}

// MARK: - End of Structured Output Diagnostics

public protocol ServerInferenceBackend: Sendable {
    /// The backend's configured context window, used to validate
    /// max_tokens/max_completion_tokens against the session's maxContext (S11).
    var maximumContext: Int { get }
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion
}

public extension ServerInferenceBackend {
    var maximumContext: Int {
        RuntimeConfiguration.supportedContextTokens.max() ?? 262_144
    }
}

public actor ServerCoordinator {
    private struct Waiter {
        let id: UUID
        let modelID: String?
        let continuation: CheckedContinuation<Void, Error>
    }

    private let queueLimit: Int
    private var admittedCount = 0
    private var active = false
    private var waiters: [Waiter] = []
    private var shuttingDown = false
    /// The model of the most recently admitted request, so `release` can
    /// prefer a waiter that will not force a model swap.
    private var lastAdmittedModelID: String?

    public init(queueLimit: Int) {
        self.queueLimit = queueLimit
    }

    public func run<T: Sendable>(
        modelID: String? = nil,
        onQueued: @escaping @Sendable () -> Void = {},
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await runPreparing(
            modelID: modelID,
            onQueued: onQueued,
            prepare: { () },
            operation: { _ in try await operation() })
    }

    func runPreparing<Prepared: Sendable, T: Sendable>(
        modelID: String? = nil,
        onQueued: @escaping @Sendable () -> Void = {},
        prepare: @escaping @Sendable () async throws -> Prepared,
        operation: @escaping @Sendable (Prepared) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        // S6: at most queueLimit queued behind one active request, i.e. up to
        // queueLimit + 1 admitted.
        guard admittedCount <= queueLimit else {
            // Shed load rather than queue without bound.
            throw ServerRequestError.queueFull
        }
        admittedCount += 1
        defer { admittedCount -= 1 }

        let prepared = try await prepare()
        try Task.checkCancellation()
        try await acquire(modelID: modelID, onQueued: onQueued)
        defer { release() }
        return try await operation(prepared)
    }

    private func acquire(modelID: String?,
                         onQueued: @escaping @Sendable () -> Void) async throws {
        try Task.checkCancellation()
        guard !shuttingDown else { throw CancellationError() }
        if !active {
            active = true
            lastAdmittedModelID = modelID
            return
        }
        guard waiters.count < queueLimit else { throw ServerRequestError.queueFull }
        onQueued()
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(Waiter(id: id, modelID: modelID, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
        if Task.isCancelled {
            release()
            throw CancellationError()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            active = false
            return
        }
        // Model affinity: admit every queued request for the model that just
        // ran before one that would force a swap, FIFO otherwise. A continuous
        // same-model stream can starve a waiter for another model; accepted,
        // and bounded only by the queue limit (design: multi-model-serving).
        let index = waiters.firstIndex { $0.modelID != nil && $0.modelID == lastAdmittedModelID } ?? 0
        let waiter = waiters.remove(at: index)
        if let modelID = waiter.modelID {
            lastAdmittedModelID = modelID
        }
        waiter.continuation.resume()
    }

    public func shutdown() {
        shuttingDown = true
        let queued = waiters
        waiters.removeAll()
        for waiter in queued {
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    public var queuedCount: Int { waiters.count }
    public var isActive: Bool { active }
}

/// Snapshot of the runner's lifetime stage counters at request start, so the
/// SHRIKE_RUNNER_STATS footer can report this request's per-stage deltas.
private struct RunnerCounterSnapshot {
    let cb1: UInt64
    let io: UInt64
    let cb2: UInt64
    let head: UInt64
    let headFused: UInt64
    let rdadvise: UInt64
    let rdadviseCalls: UInt64
    let rdadviseBytes: UInt64
    let wait: UInt64
    let body: UInt64
    let missIo: UInt64
    let exposedIo: UInt64
    let hitFixupLayers: UInt64
    let routerReadback: UInt64
    let cachePlan: UInt64
    let ioQueue: UInt64
    let ioCompletionToFixup: UInt64
    let ioHostWaits: UInt64
    let ioHostWaitsAvoided: UInt64
    let gpuClassifiedHits: UInt64
    let gpuClassifiedMisses: UInt64
    let gpuAllHitLayers: UInt64
    let expertStreaming: ExpertStreamingStatistics
}

/// Prefill a KV rewrite's tokens, discarding the head output. A free function
/// for the same reason `runRawCompletion` is one: `MTLBuffer` is not Sendable,
/// so the logits scratch must stay inside a single non-isolated region.
private func prefillRewrite(runner: RealForwardRunner,
                            scratch: RawCompletionScratch,
                            tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            config: PrefillRuntimeConfig) async throws {
    _ = try await runner.prefillChunked(
        tokens: tokens,
        startPosition: startPosition,
        outputMode: .greedyIfAvailable,
        config: config,
        into: scratch.logits) { _ in }
}

/// What a completed generation's KV rewrite left behind.
private enum KVNormalization: Sendable, Equatable {
    /// No rewrite ran; the KV holds what the generation left in it.
    case unchanged
    /// The KV now holds exactly these tokens.
    case rewritten([Int32])
}

/// What the response path leaves for the KV: either a rewrite already done —
/// both are cursor moves — or a forward pass still to run between requests.
private enum KVNormalizationPlan: Sendable, Equatable {
    case done(KVNormalization)
    /// Make the KV hold this sequence: the target, and the cursor a runner that
    /// can rewind goes back to before prefilling the remainder. A settle's
    /// target and a degenerate turn's truncation are both sequences, so both
    /// arrive here. The target is what the next request arbitrates against.
    ///
    /// `announcement` is the line this plan owes only once something dispatches
    /// it. A drop's `drop_emission` asserts the KV was cut to its target, which
    /// stays false until the rewrite is in flight — and a turn whose publish is
    /// refused has nothing to rewrite, so the plan is discarded and the line
    /// would have described a truncation that never happened. A settle's line is
    /// a record of the decision and is already out by the time it gets here.
    case reconstruct(target: [Int32], rewindTo: Int, announcement: String?)

    /// What the KV holds at publish time. A pending reconstruction has not moved
    /// it yet, so its entry is published against the bytes the generation left.
    var completedNormalization: KVNormalization {
        switch self {
        case .done(let normalization): return normalization
        case .reconstruct: return .unchanged
        }
    }
}

public actor ServerModelSession: ServerInferenceBackend {
    /// The session's configured context window; the HTTP layer validates
    /// max_tokens against it (S11).
    public nonisolated var maximumContext: Int { maxContext }
    private nonisolated let modelFamily: ModelFamily

    private let context: MetalContext
    private let model: Model
    private let tokenizer: GFTokenizer
    private let defaultReasoningEffort: ReasoningEffort
    private let runner: RealForwardRunner
    private let mtpDecoder: StreamingMTPDecoder?
    private let scratch: RawCompletionScratch
    private let prefillConfig: PrefillRuntimeConfig
    // Long prompts are prefilled chunk by chunk — small enough to keep expert
    // reads tight.
    public nonisolated let prefillChunkTokens: Int
    /// Routed-expert slots per layer actually in force, so the ready banner can
    /// report the streaming budget rather than leaving the user to infer it.
    public nonisolated let expertCacheSlots: Int
    private let maxContext: Int
    public nonisolated let promptCacheMode: ServerPromptCacheMode
    private let promptCacheDomain: ServerPromptCacheDomain
    private var promptCache: ServerPromptCache
    private let promptStateStore: ServerPromptStateStore?
    private var activePromptCacheEntryID: UUID?
    /// A settle prefilling between requests, and the sequence it is prefilling
    /// toward. Nothing else may touch the runner or the scratch while this is
    /// set, so every request arbitrates it — join or abort — before it starts.
    private var pendingRewrite: PendingRewrite?
    /// The snapshot write for the entry a pending rewrite will replace. Both
    /// writes carry the same entry id and only the store's disk queue orders
    /// them, so the rewrite waits for this one before saving its own.
    private var pendingSnapshotSave: Task<Void, Never>?

    private struct PendingRewrite {
        let target: [Int32]
        let task: Task<Void, Never>
    }
    /// Concise-mode system prompt injected into every completion, or nil when
    /// concise mode is off. Selected per quantization (see ConcisePrompt).
    private nonisolated let concisePrompt: String?

    /// A pure function of its two arguments, so a caller can reproduce the
    /// effective cache mode for the startup banner without loading a model.
    public static func effectivePromptCacheMode(
        requested: ServerPromptCacheMode,
        mtpEnabled: Bool
    ) -> ServerPromptCacheMode {
        // A target-only snapshot cannot restore the draft stream. Keeping a
        // cache allocated while MTP is active would spend memory on entries
        // that must never be consumed or published.
        mtpEnabled ? .off : requested
    }

    static func defaultReasoningEffort(
        explicit: ReasoningEffort?,
        dialect: ChatDialect,
        thinkingMode: ModelThinkingMode
    ) -> (effort: ReasoningEffort, warning: String?) {
        if let explicit { return (explicit, nil) }
        guard dialect == .harmony, thinkingMode == .off else {
            return (.medium, nil)
        }
        return (.low, "Shrike reasoning_effort: harmony cannot disable thinking; "
            + "--thinking off maps to reasoning effort low")
    }

    /// lint:allow-long a sequential construction pipeline: tokenizer, Metal
    /// context, runtime config, model, optional MTP sidecar, runner, scratch.
    /// Each step consumes the last, so extracting any of them would return a
    /// tuple straight back into the next -- the same shape as Model.load.
    public static func load(modelDirectory: URL,
                            maxContext: Int,
                            promptCacheMode: ServerPromptCacheMode = .multiPrefix,
                            promptCacheMaximumEntries: Int = 4,
                            promptCacheMemoryLimitBytes: Int = 256 * 1_048_576,
                            promptCacheDiskDirectory: URL? = nil,
                            promptCacheDiskLimitBytes: Int = 8_192 * 1_048_576,
                            prefillChunkTokens requestedPrefillChunkTokens: Int? = nil,
                            kvCachePrecision: KVCachePrecision = .int8,
                            ropeScalingMode: RuntimeRoPEScalingMode = .none,
                            thinkingMode: ModelThinkingMode = .off,
                            reasoningEffort: ReasoningEffort? = nil,
                            expertCacheSlots requestedExpertCacheSlots: Int? = nil,
                            expertCacheBudgetBytes: Int? = nil,
                            mtpModelDirectory: URL? = nil,
                            mtpMemoryMiB: Int = StreamingMTPMemoryPlan.defaultBudgetMiB,
                            reusingContext: MetalContext? = nil) async throws -> ServerModelSession {
        let tokenizerFolder = GFTokenizer.tokenizerFolder(forModelDirectory: modelDirectory)
        guard let tokenizerFolder else {
            throw GFTokenizerError.missingToolTemplate
        }
        let templateURL = tokenizerFolder.appendingPathComponent("chat_template.jinja")
        guard FileManager.default.fileExists(atPath: templateURL.path) else {
            throw GFTokenizerError.missingToolTemplate
        }
        let tokenizer = try await GFTokenizer.load(
            from: tokenizerFolder,
            thinkingMode: thinkingMode)
        let resolvedReasoningEffort = Self.defaultReasoningEffort(
            explicit: reasoningEffort,
            dialect: tokenizer.dialect,
            thinkingMode: thinkingMode)
        if let warning = resolvedReasoningEffort.warning {
            FileHandle.standardError.write(Data((warning + "\n").utf8))
        }
        // A caller managing model residency supplies its own context so one
        // MTLCommandQueue and one compiled shader library survive across
        // unload/reload cycles (MetalContext.deinit documents that queue
        // teardown is not deinit-safe). Nil for every ordinary caller.
        let context = try reusingContext ?? MetalContext()
        let loadRuntime = try RuntimeConfiguration(
            forceLogitsHead: true,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue())
        let slotOverride = ProcessInfo.processInfo.environment["SHRIKE_EXPERT_CACHE_SLOTS"]
            .flatMap(Int.init)
        // Precedence: --expert-cache-slots flag, then the env override, then a
        // count derived from the model's own expert stride against a 1 GiB budget.
        //
        // Derived rather than fixed because the right count depends on the
        // quantisation: 1 GiB is 16 slots at 4-bit and 8 at 8-bit, which are the
        // measured optima for each. The previous fixed default of 64 was slower
        // *and* larger than either -- benchmarked at the shipped 262144 context,
        // 4-bit managed 9.85 tok/s at 64 slots against 13.61 at 16.
        let expectedArch: ArchConfig
        do {
            let family = try ManifestReader.peekFamily(directoryURL: modelDirectory)
            guard let baseline = ArchConfig.knownArchitectures[family] else {
                throw ModelError.unsupportedArchitecture(
                    detail: "no compiled baseline for family \(family.rawValue)")
            }
            expectedArch = baseline
        }
        let derivedSlots: Int
        if let manifest = try? ManifestReader.load(directoryURL: modelDirectory,
                                                  expecting: expectedArch) {
            derivedSlots = RuntimeConfiguration.expertCacheSlots(
                expertStrideBytes: manifest.expertStride,
                layers: manifest.arch.numLayers,
                budgetBytes: expertCacheBudgetBytes
                    ?? RuntimeConfiguration.defaultExpertCacheBudgetBytes)
        } else {
            // Unreadable manifest means the load below will fail with a better
            // message than anything this could throw, so pick the safe small end.
            derivedSlots = RuntimeConfiguration.allowedExpertCacheSlots.first ?? 8
        }
        let loadSlots = requestedExpertCacheSlots ?? slotOverride ?? derivedSlots
        let model = try Model.load(
            directoryURL: modelDirectory,
            device: context.device,
            expecting: expectedArch,
            streamingMode: .pread(slotCount: loadSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .resolved(directoryURL: modelDirectory))
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: ProcessInfo.processInfo.environment["SHRIKE_RDADVISE_POLICY"]
                .map(RDAdvicePolicyMode.parse)
                ?? loadRuntime.rdadvisePolicy,
            prefillChunkTokens: requestedPrefillChunkTokens
                ?? (model.config.family == .qwen36
                    ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                    : loadRuntime.prefillChunkTokens),
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: true,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        let mtpDecoder: StreamingMTPDecoder?
        let runner: RealForwardRunner
        if let mtpModelDirectory {
            let sidecar = try Model.load(
                directoryURL: mtpModelDirectory,
                device: context.device,
                expecting: .qwen36MTP,
                streamingMode: .pread(slotCount: StreamingMTPMemoryPlan.expertSlots),
                expertCachePolicy: runtime.modelExpertCachePolicy,
                integrityPolicy: .resolved(directoryURL: mtpModelDirectory))
            let decoder = try StreamingMTPDecoder(
                targetModel: model,
                mtpSidecar: sidecar,
                context: context,
                maxContext: maxContext,
                memoryBudgetMiB: mtpMemoryMiB,
                runtimeConfiguration: runtime)
            mtpDecoder = decoder
            runner = decoder.target
        } else {
            mtpDecoder = nil
            runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: maxContext,
                                           runtimeConfiguration: runtime)
        }
        let scratch = try RawCompletionScratch(context: context, vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let templateDigest = SHA256.hash(data: try Data(contentsOf: templateURL))
            .map { String(format: "%02x", $0) }
            .joined()
        let runtimeIdentity = [
            String(runtime.expertCacheSlots),
            runtime.expertCachePolicy.rawValue,
            runtime.rdadvisePolicy.rawValue,
            runtime.prefillPolicy.rawValue,
            String(runtime.prefillChunkTokens),
            runtime.headPath.rawValue,
            String(runtime.kvCachePrecision.rawValue),
            runtime.ropeScalingMode.rawValue,
            String(runtime.yarnContextTokens),
        ].joined(separator: ":")
        let runtimeDigest = SHA256.hash(data: Data(runtimeIdentity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        let promptCacheDomain = ServerPromptCacheDomain(
            modelID: model.modelID,
            sourceSnapshotHash: model.sourceSnapshotHash,
            runtimeProfileHash: runtimeDigest,
            maximumContext: maxContext,
            kvStorage: runtime.kvCachePrecision.label,
            fp16RingEnabled: runtime.fp16RingEnabled,
            templateSHA256: templateDigest)
        let effectivePromptCacheMode = Self.effectivePromptCacheMode(
            requested: promptCacheMode,
            mtpEnabled: mtpDecoder != nil)
        let promptStateStore: ServerPromptStateStore?
        let promptCache: ServerPromptCache
        if effectivePromptCacheMode == .multiPrefix {
            let store = try ServerPromptStateStore(
                configuration: ServerPromptCacheStorageConfiguration(
                    memoryLimitBytes: promptCacheMemoryLimitBytes,
                    diskDirectory: promptCacheDiskDirectory,
                    diskLimitBytes: promptCacheDiskLimitBytes))
            let persisted = store.loadEntries(domain: promptCacheDomain)
            if persisted.count > promptCacheMaximumEntries {
                store.remove(entryIDs: persisted
                    .dropLast(promptCacheMaximumEntries)
                    .map(\.id))
            }
            promptStateStore = store
            promptCache = ServerPromptCache(
                maximumEntries: promptCacheMaximumEntries,
                entries: persisted,
                allowsPartialSalvage: runner.supportsPartialRewind)
        } else {
            promptStateStore = nil
            promptCache = ServerPromptCache(
                maximumEntries: 1,
                allowsPartialSalvage: runner.supportsPartialRewind)
        }
        return ServerModelSession(context: context,
                                  model: model,
                                  tokenizer: tokenizer,
                                  defaultReasoningEffort: resolvedReasoningEffort.effort,
                                  runner: runner,
                                  mtpDecoder: mtpDecoder,
                                  scratch: scratch,
                                  prefillConfig: runtime.prefillConfig,
                                  expertCacheSlots: loadSlots,
                                  maxContext: maxContext,
                                  promptCacheMode: effectivePromptCacheMode,
                                  promptCacheDomain: promptCacheDomain,
                                  promptCache: promptCache,
                                  promptStateStore: promptStateStore,
                                  concisePrompt: conciseModeEnabled()
                                    ? ConcisePrompt.prompt(for: model) : nil)
    }

    private init(context: MetalContext,
                 model: Model,
                 tokenizer: GFTokenizer,
                 defaultReasoningEffort: ReasoningEffort,
                 runner: RealForwardRunner,
                 mtpDecoder: StreamingMTPDecoder?,
                 scratch: RawCompletionScratch,
                 prefillConfig: PrefillRuntimeConfig,
                 expertCacheSlots: Int,
                 maxContext: Int,
                 promptCacheMode: ServerPromptCacheMode,
                 promptCacheDomain: ServerPromptCacheDomain,
                 promptCache: ServerPromptCache,
                 promptStateStore: ServerPromptStateStore?,
                 concisePrompt: String?) {
        self.context = context
        self.model = model
        self.tokenizer = tokenizer
        self.defaultReasoningEffort = defaultReasoningEffort
        self.modelFamily = model.config.family
        self.runner = runner
        self.mtpDecoder = mtpDecoder
        self.scratch = scratch
        self.prefillConfig = prefillConfig
        self.prefillChunkTokens = prefillConfig.chunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheDomain = promptCacheDomain
        self.promptCache = promptCache
        self.promptStateStore = promptStateStore
        self.concisePrompt = concisePrompt
    }

    /// SHRIKE_CONCISE_MODE=1 (or "on") enables concise mode; the per-quant
    /// system prompt is then injected into every completion.
    private static func conciseModeEnabled() -> Bool {
        switch ProcessInfo.processInfo.environment["SHRIKE_CONCISE_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return true
        default: return false
        }
    }

    /// Render a validated request into prompt tokens.
    ///
    /// SHRIKE_STRIP_CLI_PROMPT: drop the coding-CLI's system/developer guidance,
    /// tool definitions, tool-call history, and in-message <system-reminder>
    /// scaffolding, keeping only the real user/assistant conversation (see
    /// CLIStrip). Guards ensure the real prompt can never be stripped into an
    /// empty turn or an empty request. Runs when SHRIKE_STRIP_CLI_PROMPT is
    /// set — an operator lever, not something a client can reach.
    ///
    /// Returns the encoded prompt alongside the `cacheRequest` — the post-strip
    /// view the prompt cache must key on. Cache entries describe a KV range
    /// prefilled from the filtered messages, and the cache's text-continuation
    /// path re-renders the tail with the same template, so matching or
    /// publishing against the raw request would splice an unstripped tail onto
    /// a stripped prefix, silently losing the strip on every cached
    /// continuation turn.
    private func preparePrompt(
        _ request: ValidatedChatRequest
    ) throws -> (promptIDs: [Int32],
                 cacheRequest: ValidatedChatRequest,
                 effectiveMessages: [GFTokenizer.Message],
                 needsToolTemplate: Bool) {
        let filteredMessages: [GFTokenizer.Message]
        let filteredTools: [GFTokenizer.FunctionDefinition]
        var stripStats: CLIStrip.Stats?
        if CLIStrip.isEnabled() {
            let filtered = CLIStrip.filter(
                messages: request.messages,
                tools: request.tools)
            filteredMessages = filtered.messages
            filteredTools = filtered.tools
            stripStats = filtered.stats
        } else {
            filteredMessages = request.messages
            filteredTools = request.tools
        }
        let cacheRequest = request.replacingMessages(
            filteredMessages,
            tools: filteredTools)
        let needsToolTemplate = GFTokenizer.usesToolTemplate(
            messages: filteredMessages,
            tools: filteredTools)
        let effectiveMessages = concisePrompt.map {
            ConcisePrompt.appendingSystemPrompt($0, to: filteredMessages)
        } ?? filteredMessages
        let promptIDs = try encodePrompt(
            messages: effectiveMessages,
            tools: filteredTools,
            usesToolTemplate: needsToolTemplate,
            reasoningEffort: defaultReasoningEffort)
        if let stats = stripStats {
            ServerLog.strip(stats: stats,
                            promptTokens: promptIDs.count)
        }
        guard promptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return (promptIDs, cacheRequest, effectiveMessages, needsToolTemplate)
    }

    private func cacheDiag(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    /// Say which check left the KV unrewritten. Only chain breaks come here: a
    /// conversation whose settles start failing has one of these lines at its
    /// root, and without them the whole contagion is invisible.
    private func declined(_ reason: KVNormalizationDecline) {
        cacheDiag("Shrike prompt_cache normalize kind=declined "
                + "reason=\(reason.rawValue)")
    }

    /// Decide where this request's prefill starts: from scratch, or resumed on
    /// a cache entry whose KV is live or restorable.
    ///
    /// Mutates the cache and `activePromptCacheEntryID`, so it must run on the
    /// actor and before any generation begins.
    private func resolveCacheStart(
        cacheRequest: ValidatedChatRequest,
        promptIDs: [Int32]
    ) async throws -> (effectivePromptIDs: [Int32], start: RawCompletionStart) {
        let effectivePromptIDs: [Int32]
        var completionStart: RawCompletionStart
        if promptCacheMode == .singlePrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: cacheRequest,
                renderedPromptIDs: promptIDs) {
            case .miss:
                promptCache.invalidate()
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(_, let effective, let cached):
                if runner.continuationPosition != cached {
                    // S15: the live KV no longer sits at the cached entry's
                    // position; re-prefill instead of resuming from a stale
                    // or mismatched in-memory state.
                    promptCache.invalidate()
                    effectivePromptIDs = promptIDs
                    completionStart = .reset
                } else {
                    effectivePromptIDs = effective
                    completionStart = .resume(cachedPromptTokens: cached)
                }
            }
        } else if promptCacheMode == .multiPrefix {
            switch promptCache.match(
                domain: promptCacheDomain,
                request: cacheRequest,
                renderedPromptIDs: promptIDs) {
            case .miss:
                activePromptCacheEntryID = nil
                effectivePromptIDs = promptIDs
                completionStart = .reset
            case .hit(let entryID, let effective, let cached):
                if entryID == activePromptCacheEntryID,
                   runner.continuationPosition == cached {
                    // S15: tier=live is only trusted when the in-memory KV
                    // still matches the entry (same entry id and the KV
                    // cursor sits exactly at the request's expected
                    // position). Anything else falls through to a snapshot
                    // restore or a full prefill instead of resuming from a
                    // stale or mismatched KV.
                    cacheDiag(
                        "Shrike prompt_cache hit tier=live "
                            + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                } else {
                    do {
                        guard let promptStateStore else {
                            throw ServerPromptStateStoreError.missing(entryID)
                        }
                        let tier = try await promptStateStore.restore(
                            entryID: entryID,
                            into: runner)
                        // The snapshot seats the KV where the entry was
                        // published; a partial-prefix salvage kept less than
                        // that, so the cursor moves back to what still matches.
                        if runner.continuationPosition != cached {
                            try runner.rewind(to: cached)
                        }
                        cacheDiag(
                            "Shrike prompt_cache hit tier=\(tier) "
                                + "cached_tokens=\(cached) entry=\(entryID.uuidString.lowercased())")
                    } catch {
                        // Drop the stale entry and prefill from scratch rather
                        // than trust it.
                        FileHandle.standardError.write(Data(
                            ("Shrike prompt_cache restore_failed "
                                + "entry=\(entryID.uuidString.lowercased()) error=\(error)\n").utf8))
                        promptStateStore?.remove(entryIDs: [entryID])
                        promptCache.remove(entryIDs: [entryID])
                        activePromptCacheEntryID = nil
                        effectivePromptIDs = promptIDs
                        completionStart = .reset
                        break
                    }
                }
                activePromptCacheEntryID = entryID
                effectivePromptIDs = effective
                completionStart = .resume(cachedPromptTokens: cached)
            }
        } else {
            promptCache.invalidate()
            activePromptCacheEntryID = nil
            effectivePromptIDs = promptIDs
            completionStart = .reset
        }
        // S12: an identical-prompt replay whose render equals the entry's
        // KV-backed prefix has nothing to prefill (cached == prompt count).
        // The continuation API requires cached < prompt count (it must
        // prefill at least one token), so resume as a full prefill; the
        // entry stays active for later extending requests.
        if case .resume(let cached) = completionStart,
           cached >= effectivePromptIDs.count {
            completionStart = .reset
        }
        guard effectivePromptIDs.count < maxContext else {
            throw ServerRequestError.invalid(
                message: "effective prompt exceeds the configured context",
                param: "messages",
                code: "context_length_exceeded")
        }
        return (effectivePromptIDs, completionStart)
    }

    /// lint:allow-long the request orchestrator: prompt preparation, cache
    /// resolution, decode, publish, and the completion. Each of those is its
    /// own method; what remains is the sequence plus a nested failure builder
    /// that closes over eight locals -- hoisting it would mean an
    /// eight-parameter signature for a twenty-line body.
    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        // Rendered before anything is awaited, because a rewrite still running
        // between requests is arbitrated on this render: one that it is not
        // prefilling for must cancel it rather than queue behind it. The reset
        // guard below stays under this, so a request rejected here cannot reset
        // a runner the rewrite is still driving.
        let prepared = try preparePrompt(request)
        await arbitratePendingRewrite(renderedPromptIDs: prepared.promptIDs)
        // Stage-split measurement (SHRIKE_RUNNER_STATS): snapshot the runner's
        // lifetime counters so the footer can report this request's delta.
        let runnerSnapshot = RunnerCounterSnapshot(
            cb1: runner.totalCb1Nanos,
            io: runner.totalIoNanos,
            cb2: runner.totalCb2Nanos,
            head: runner.totalHeadNanos,
            headFused: runner.totalHeadFusedNanos,
            rdadvise: runner.totalRDAdviseNanos,
            rdadviseCalls: runner.totalRDAdviseCalls,
            rdadviseBytes: runner.totalRDAdviseBytes,
            wait: runner.totalWaitNanos,
            body: runner.totalBodyNanos,
            missIo: runner.totalMissIoNanos,
            exposedIo: runner.totalExposedIoNanos,
            hitFixupLayers: runner.totalHitFixupLayers,
            routerReadback: runner.totalRouterReadbackNanos,
            cachePlan: runner.totalCachePlanNanos,
            ioQueue: runner.totalIOQueueNanos,
            ioCompletionToFixup: runner.totalIOCompletionToFixupSubmitNanos,
            ioHostWaits: runner.totalExpertIOHostWaits,
            ioHostWaitsAvoided: runner.totalExpertIOHostWaitsAvoided,
            gpuClassifiedHits: runner.totalGPUClassifiedHits,
            gpuClassifiedMisses: runner.totalGPUClassifiedMisses,
            gpuAllHitLayers: runner.totalGPUResidencyAllHitLayers,
            expertStreaming: runner.expertStreamingStatistics())
        runner.resetKernelGPUTimings()
        var completed = false
        defer {
            if !completed {
                if promptCacheMode == .singlePrefix {
                    promptCache.invalidate()
                }
                activePromptCacheEntryID = nil
                runner.reset()
                mtpDecoder?.reset()
            }
        }
        let promptIDs = prepared.promptIDs
        let cacheRequest = prepared.cacheRequest
        let needsToolTemplate = prepared.needsToolTemplate

        let resolved = try await resolveCacheStart(
            cacheRequest: cacheRequest,
            promptIDs: promptIDs)
        let effectivePromptIDs = resolved.effectivePromptIDs
        let completionStart = resolved.start

        var config = request.generationConfig
        config.maxNewTokens = min(
            request.maximumCompletionTokens,
            maxContext - effectivePromptIDs.count)
        config.stopStrings = []

        // Harmony always decodes structurally: without the decoder, analysis
        // text would leak into visible content on tool-free requests. ChatML
        // needs the same whenever thinking may occur — an injected-open or
        // model-opened <think> must route to reasoning, not content.
        let decoder = needsToolTemplate || tokenizer.dialect == .harmony
            || (tokenizer.dialect == .chatml && tokenizer.thinkingMode != .off)
            ? StructuredAssistantDecoder(
                tokenizer: tokenizer,
                allowedTools: Set(request.tools.map(\.name)))
            : nil
        var stopMatcher = StreamingStopMatcher(stops: request.generationConfig.stopStrings)
        var content = ""
        var reasoning = ""
        var calls: [ParsedToolCall] = []
        var decodingError: Error?
        var shouldStop = false

        let activeProducer: any LogitProducer = if config.isPureGreedy,
                                                   let mtpDecoder,
                                                   promptIDs.count + config.maxNewTokens
                                                    <= mtpDecoder.draftMaxContext {
            mtpDecoder
        } else {
            runner
        }
        let activeStart: RawCompletionStart = activeProducer is StreamingMTPDecoder
            ? .reset : completionStart
        let activePromptIDs = activeProducer is StreamingMTPDecoder
            ? promptIDs : effectivePromptIDs
        func publish(_ events: [StructuredAssistantEvent]) {
            for event in events {
                switch event {
                case .content(let text):
                    let visible = stopMatcher.push(text)
                    if !visible.isEmpty {
                        content += visible
                        onEvent(.content(visible))
                    }
                    if stopMatcher.isStopped { shouldStop = true }
                case .thinking(let text):
                    reasoning += text
                    onEvent(.thinking(text))
                case .toolCall(let call):
                    calls.append(call)
                    onEvent(.toolCall(call))
                }
            }
        }
        let result = try await runRawCompletion(
            producer: activeProducer,
            tokenizer: tokenizer,
            promptIds: activePromptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig,
            start: activeStart,
            shouldStop: { shouldStop }) { progress in
                guard decodingError == nil else { return }
                do {
                    switch progress {
                    case .prefill:
                        break
                    case .token(_, let tokenID, let delta):
                        let events = if let decoder {
                            try decoder.consume(tokenID: tokenID, delta: delta)
                        } else {
                            delta.isEmpty ? [] : [StructuredAssistantEvent.content(delta)]
                        }
                        publish(events)
                    case .tail(let text):
                        let events = if let decoder {
                            try decoder.consumeTail(text)
                        } else {
                            text.isEmpty ? [] : [StructuredAssistantEvent.content(text)]
                        }
                        publish(events)
                    }
                } catch {
                    decodingError = error
                    shouldStop = true
                }
        }
        emitGenerationDiagnostics(activeProducer: activeProducer,
                                  result: result,
                                  snapshot: runnerSnapshot)
        // Harmony ends at a stop token the generation loop never forwards
        // (`<|return|>` or `<|call|>`, both mapped to `.eos`); the decoder
        // needs it to finalize a buffered tool call or close the turn, so
        // replay the uncommitted boundary token here.
        if tokenizer.dialect == .harmony, let decoder, decodingError == nil,
           result.reason == .eos,
           let boundary = result.uncommittedBoundaryTokenIDs.first {
            do {
                publish(try decoder.consume(tokenID: boundary, delta: ""))
            } catch {
                decodingError = error
            }
        }
        if ShrikeGenDiag.enabled {
            cacheDiag(ShrikeGenDiag.line(
                prefillTokens: result.prefillTokens,
                kvBackedTokenIDs: result.kvBackedTokenIDs,
                boundaryTokenIDs: result.uncommittedBoundaryTokenIDs))
        }
        func structuredFailure(
            kind: StructuredOutputFailureKind,
            cause: StructuredOutputFailureCause,
            unknownToolName: String? = nil
        ) -> StructuredOutputFailure {
            StructuredOutputFailure(
                kind: kind,
                cause: cause,
                unknownToolName: unknownToolName,
                diagnostics: StructuredOutputFailureDiagnostics(
                    renderedPromptIDs: promptIDs,
                    effectivePromptIDs: effectivePromptIDs,
                    result: result,
                    maxCompletionTokens: config.maxNewTokens,
                    decodedCalls: calls.count,
                    visibleBytes: content.utf8.count,
                    stopStringMatched: stopMatcher.isStopped,
                    toolStartID: tokenizer.toolCallStartID,
                    toolEndID: tokenizer.toolCallEndID,
                    toolResponseID: tokenizer.toolResponseID,
                    toolResponseEndID: tokenizer.toolResponseEndID))
        }
        if let decodingError {
            throw structuredFailure(
                kind: .decoderConsume,
                cause: .classify(decodingError),
                unknownToolName: StructuredOutputFailureCause
                    .unknownToolName(decodingError))
        }
        do {
            try decoder?.finish()
        } catch {
            throw structuredFailure(
                kind: .decoderFinish,
                cause: .classify(error),
                unknownToolName: StructuredOutputFailureCause
                    .unknownToolName(error))
        }
        if needsToolTemplate, result.reason == .toolCalls, calls.isEmpty {
            throw structuredFailure(kind: .orphanToolResponse, cause: .none)
        }
        let tail = stopMatcher.finish()
        if !tail.isEmpty {
            content += tail
            onEvent(.content(tail))
        }
        let reason: String
        if !calls.isEmpty {
            reason = "tool_calls"
        } else if result.reason == .maxTokens {
            reason = "length"
        } else {
            reason = "stop"
        }
        let plan = normalizeCompletedKV(
            messages: prepared.effectiveMessages,
            tools: cacheRequest.tools,
            content: content,
            result: result,
            promptTokenCount: effectivePromptIDs.count,
            thoughtChannelClosed: decoder?.thoughtChannelClosed ?? true,
            emittedToolCalls: !calls.isEmpty,
            stopStringFiltered: stopMatcher.isStopped)
        let publishedEntryID = publishCacheEntry(
            cacheRequest: cacheRequest,
            result: result,
            stopStringFiltered: stopMatcher.isStopped,
            normalization: plan.completedNormalization)
        // Nothing suspends between the publish and this, so no request can see
        // the entry before the rewrite that will replace it is arbitrable.
        if case .reconstruct(let target, let rewindTo, let announcement) = plan,
           let publishedEntryID {
            if let announcement {
                cacheDiag(announcement
                    + " entry=\(publishedEntryID.uuidString.lowercased())")
            }
            startRewrite(target: target, rewindTo: rewindTo, entryID: publishedEntryID)
        }
        completed = true
        return ServerCompletion(
            content: content,
            reasoningContent: reasoning.isEmpty ? nil : reasoning,
            toolCalls: calls,
            finishReason: reason,
            // S26: completion_tokens reports the number of GENERATED tokens,
            // matching OpenAI's "completion_tokens = tokens in the generated
            // completion". A stop-string-hidden suffix is therefore counted as
            // generated even though it is filtered from the visible content.
            usage: OpenAIUsage(promptTokens: result.prefillTokens,
                               completionTokens: result.newTokens,
                               totalTokens: result.prefillTokens + result.newTokens,
                               cachedTokens: result.cachedPromptTokens))
    }

    /// The entry as the KV rewrite left it, or as published when the rewrite
    /// declined. Its bytes are the cursor actually reached, never the target
    /// the rewrite was computing toward.
    @discardableResult
    private func applying(_ normalization: KVNormalization,
                          to entry: ServerPromptCacheEntry) -> ServerPromptCacheEntry {
        guard case .rewritten(let kvBackedTokenIDs) = normalization,
              let rewritten = promptCache.rewrite(
                entryID: entry.id,
                kvBackedTokenIDs: kvBackedTokenIDs) else {
            return entry
        }
        return rewritten
    }

    /// Rewrite the completed generation's KV into the bytes the next request
    /// will render, so the prompt cache stays a byte comparison.
    ///
    /// Decides and renders on the response path — the target has to be recorded
    /// before this returns, since the next request arbitrates on it — but the
    /// forward pass a settle needs is left to `startRewrite` to run between
    /// requests, so the stream closes on the response rather than on the
    /// rewrite.
    private func normalizeCompletedKV(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        content: String,
        result: RawDecodeResult,
        promptTokenCount: Int,
        thoughtChannelClosed: Bool,
        emittedToolCalls: Bool,
        stopStringFiltered: Bool
    ) -> KVNormalizationPlan {
        guard promptCacheMode != .off,
              mtpDecoder == nil,
              prefillConfig.mode == .chunked,
              result.kvPosition == result.kvBackedTokenIDs.count,
              result.uncommittedBoundaryTokenIDs.count == 1,
              runner.continuationPosition == result.kvPosition else {
            return .done(.unchanged)
        }
        switch KVRewrite.forCompletion(
            reason: result.reason,
            thoughtChannelClosed: thoughtChannelClosed,
            emittedToolCalls: emittedToolCalls,
            stopStringFiltered: stopStringFiltered,
            supportsRewind: runner.supportsPartialRewind,
            canRestore: canSnapshotForRestore) {
        case .none:
            if KVRewrite.degenerateDeclinedForCapability(
                reason: result.reason,
                thoughtChannelClosed: thoughtChannelClosed,
                emittedToolCalls: emittedToolCalls,
                stopStringFiltered: stopStringFiltered,
                supportsRewind: runner.supportsPartialRewind,
                canRestore: canSnapshotForRestore) {
                declined(.degenerateUnsupported)
            }
            return .done(.unchanged)
        case .dropEmission:
            return dropEmission(result: result,
                                promptTokenCount: promptTokenCount)
        case .settleLiveRegion:
            let completed = messages
                + [GFTokenizer.Message(role: .assistant, content: content)]
            return settleLiveRegion(messages: completed,
                                    tools: tools,
                                    result: result)
        }
    }

    /// Drop this generation's suffix and emission, keeping the request history
    /// the next render still reproduces.
    ///
    /// A cursor move where the runner can rewind, and the line goes out with it.
    /// Where it cannot, the truncation is a target like any other and the
    /// background rewrite reaches it — a degenerate turn left standing keeps its
    /// blob under every later boundary in the conversation, so every downstream
    /// settle would splice onto bytes no render produces — but the line then
    /// belongs to whoever dispatches the plan, since the cut has not happened
    /// yet and a refused publish means it never will.
    private func dropEmission(result: RawDecodeResult,
                              promptTokenCount: Int) -> KVNormalizationPlan {
        guard let target = KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            kvPosition: result.kvPosition,
            promptTokenCount: promptTokenCount,
            generationSuffix: tokenizer.encode(tokenizer.generationSuffix,
                                               addBOS: false)) else {
            declined(.suffixMismatch)
            return .done(.unchanged)
        }
        let dropped = Array(result.kvBackedTokenIDs.prefix(target))
        let line = "Shrike prompt_cache normalize kind=drop_emission "
            + "target=\(target) kv=\(result.kvPosition)"
        guard runner.supportsPartialRewind else {
            return .reconstruct(target: dropped,
                                rewindTo: target,
                                announcement: line)
        }
        do {
            try runner.rewind(to: target)
        } catch {
            return .done(.unchanged)
        }
        cacheDiag(line)
        return .done(.rewritten(dropped))
    }

    /// Rewind to where the KV and the settled render part company and prefill
    /// the rest. The parting point is at or after the settled boundary, so this
    /// is the spec's rewind with the work the two forms already agree on left
    /// standing — a dialect that retains reasoning then costs nothing.
    ///
    /// The rewind itself is deferred with the prefill: it moves the cursor off
    /// the position the entry is about to be published at, which is the
    /// position that entry's snapshot has to be captured from.
    private func settleLiveRegion(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        result: RawDecodeResult
    ) -> KVNormalizationPlan {
        guard let boundary = try? tokenizer.settledBoundaryTokens(messages: messages,
                                                                 tools: tools),
              let live = try? tokenizer.settledLiveRegionTokens(messages: messages,
                                                                tools: tools) else {
            declined(.renderFailed)
            return .done(.unchanged)
        }
        guard let settled = KVRewrite.settledSequence(
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            boundaryTokens: boundary,
            liveRegionTokens: live) else {
            declined(.spliceMismatch)
            return .done(.unchanged)
        }
        guard settled.count < maxContext else {
            declined(.overContext)
            return .done(.unchanged)
        }
        guard settled != result.kvBackedTokenIDs else { return .done(.unchanged) }
        let comparable = min(result.kvPosition, settled.count)
        let common = (0..<comparable).first {
            result.kvBackedTokenIDs[$0] != settled[$0]
        } ?? comparable
        let line = "Shrike prompt_cache normalize kind=settle "
            + "boundary=\(boundary.count) rewind=\(common) "
            + "kv=\(result.kvPosition) settled=\(settled.count)"
        guard common < settled.count else {
            // The settled form is a prefix of what the KV holds, so the rewind
            // alone is the whole rewrite and nothing has to run in the
            // background — where the cursor can go back at all. Where it
            // cannot, the truncation is reconstructed like any other target,
            // with an empty remainder to prefill.
            guard runner.supportsPartialRewind else {
                cacheDiag(line)
                return .reconstruct(target: settled,
                                    rewindTo: common,
                                    announcement: nil)
            }
            do {
                try runner.rewind(to: common)
            } catch {
                return .done(.unchanged)
            }
            cacheDiag(line)
            return .done(.rewritten(settled))
        }
        cacheDiag(line)
        return .reconstruct(target: settled, rewindTo: common, announcement: nil)
    }

    /// Run a reconstruction's forward pass between requests, so the response the
    /// rewrite belongs to has already closed. Cancelled before it seats
    /// anything, it skips and leaves the KV as the generation left it; cancelled
    /// after, it lands in a `checkCancellation` or in `prefillChunked`'s catch,
    /// which resets the runner — an aborted rewrite then leaves no live KV it
    /// claims to describe, and the entry published before it started is what
    /// the aborting request falls back to.
    private func startRewrite(target: [Int32], rewindTo: Int, entryID: UUID) {
        pendingRewrite = PendingRewrite(
            target: target,
            task: Task { await self.runRewrite(target: target,
                                               rewindTo: rewindTo,
                                               entryID: entryID) })
    }

    /// Whether this session can build a restore chain at all: a store that is
    /// present and whose budgets leave room for one snapshot. A store that can
    /// hold nothing captures nothing, so every entry ends up unbacked and the
    /// inventory is permanently empty — the same condition as having no store.
    private var canSnapshotForRestore: Bool {
        (promptStateStore?.maximumSnapshotBytes ?? 0) > 0
    }

    /// Every snapshot a settle could seat on. The entry under rewrite is not
    /// among them: its snapshot describes the bytes this rewrite is replacing,
    /// and its write is the one `pendingSnapshotSave` is still ordering.
    ///
    /// `entry.kvPosition` stands in for the snapshot's seated position because
    /// salvage — the only truncation that skips recapturing it — is gated to
    /// rewind-capable runners, on which `KVReconstruction.plan` never chooses
    /// `.restore`.
    private func reconstructionSnapshots(
        target: [Int32],
        excluding entryID: UUID
    ) -> [KVReconstruction.Snapshot] {
        guard let promptStateStore else { return [] }
        return promptCache.entries.compactMap { entry in
            guard entry.id != entryID,
                  entry.kvPosition == entry.kvBackedTokenIDs.count,
                  promptStateStore.contains(entry.id) else { return nil }
            return KVReconstruction.Snapshot(
                entryID: entry.id,
                position: entry.kvPosition,
                isPrefixOfTarget: entry.kvPosition <= target.count
                    && target.prefix(entry.kvPosition)
                        .elementsEqual(entry.kvBackedTokenIDs))
        }
    }

    /// Put the KV at a prefix of `target` and return the position reached, so
    /// the prefill that follows starts from the achieved cursor and never from
    /// the one the mechanism intended. Every failure throws: a reconstruction
    /// that does not land leaves the rewrite exactly where a failed prefill
    /// leaves it.
    private func seatForRewrite(_ plan: KVReconstruction,
                                target: [Int32],
                                entryID: UUID) async throws -> Int {
        let settling = entryID.uuidString.lowercased()
        switch plan {
        case .rewind(let position):
            if runner.continuationPosition != position {
                try runner.rewind(to: position)
            }
            cacheDiag("Shrike prompt_cache normalize kind=settle_rewind "
                    + "at=\(position) settled=\(target.count) entry=\(settling)")
        case .restore(let source, let position):
            guard let promptStateStore else {
                throw ServerPromptStateStoreError.missing(source)
            }
            cacheDiag("Shrike prompt_cache normalize kind=settle_restore "
                    + "from=\(source.uuidString.lowercased()) at=\(position) "
                    + "settled=\(target.count) entry=\(settling)")
            do {
                _ = try await promptStateStore.restore(entryID: source, into: runner)
            } catch {
                // As in `resolveCacheStart`: a snapshot that will not seat is
                // dropped with the entry that describes it, so the pair stays
                // whole and neither is chosen again.
                promptStateStore.remove(entryIDs: [source])
                promptCache.remove(entryIDs: [source])
                throw error
            }
        case .reset(let reason):
            cacheDiag("Shrike prompt_cache normalize kind=settle_reset "
                    + "reason=\(reason.rawValue) settled=\(target.count) "
                    + "entry=\(settling)")
            runner.reset()
        }
        let reached = runner.continuationPosition
        guard reached == plan.seatedPosition else {
            if case .restore(let source, _) = plan {
                // Mirrors the restore-failure arm above: a source seated at
                // the wrong position is no more trustworthy than one that
                // failed to restore outright.
                promptStateStore?.remove(entryIDs: [source])
                promptCache.remove(entryIDs: [source])
            }
            throw PrefillError.prefillCursorMismatch(
                "settle seated the KV at \(reached), expected \(plan.seatedPosition)")
        }
        return reached
    }

    private func runRewrite(target: [Int32], rewindTo: Int, entryID: UUID) async {
        defer { pendingRewrite = nil }
        // Nothing has been seated, so the KV still holds what the generation
        // left and the entry published against it still describes it exactly.
        // Disowning that would cost the aborting request its live tier — and in
        // single-prefix the whole cache — for work that never began.
        guard !Task.isCancelled else {
            cacheDiag("Shrike prompt_cache normalize kind=settle_skipped "
                    + "reason=cancelled settled=\(target.count) "
                    + "entry=\(entryID.uuidString.lowercased())")
            return
        }
        var lost = false
        do {
            let start = try await seatForRewrite(
                KVReconstruction.plan(
                    supportsRewind: runner.supportsPartialRewind,
                    livePosition: runner.continuationPosition,
                    rewindTo: rewindTo,
                    snapshots: reconstructionSnapshots(target: target,
                                                       excluding: entryID),
                    targetCount: target.count),
                target: target,
                entryID: entryID)
            try Task.checkCancellation()
            try await prefillRewrite(runner: runner,
                                     scratch: scratch,
                                     tokens: target[start...],
                                     startPosition: start,
                                     config: prefillConfig)
        } catch {
            // A failure inside the chunk loop resets the runner; the guards
            // that reject the call before it leave the KV alone. Nothing here
            // can tell which happened, so the KV counts as gone.
            FileHandle.standardError.write(Data(
                ("Shrike prompt_cache normalize_failed error=\(error) "
                    + "entry=\(entryID.uuidString.lowercased())\n").utf8))
            lost = true
        }
        // The pre-rewrite pair has to be on disk before its replacement is
        // written: both carry this entry's id, and only the store's disk queue
        // orders the two writes.
        await pendingSnapshotSave?.value
        pendingSnapshotSave = nil
        guard !lost else {
            // The entry stands as published, still described by the snapshot
            // captured before the rewind; only the claim that the live KV
            // matches it is withdrawn.
            if promptCacheMode == .singlePrefix { promptCache.invalidate() }
            activePromptCacheEntryID = nil
            cacheDiag("Shrike prompt_cache normalize kind=settle_lost "
                    + "settled=\(target.count) entry=\(entryID.uuidString.lowercased())")
            return
        }
        await finishRewrite(target: target, entryID: entryID)
    }

    /// Move the entry and its snapshot onto the rewritten bytes together. Both
    /// happen before the arbitrating request is let go, so no match ever pairs
    /// a rewritten entry with the snapshot of what it used to hold.
    private func finishRewrite(target: [Int32], entryID: UUID) async {
        guard let entry = promptCache.rewrite(entryID: entryID,
                                              kvBackedTokenIDs: target) else {
            activePromptCacheEntryID = nil
            return
        }
        cacheDiag("Shrike prompt_cache normalize kind=settle_done "
                + "settled=\(target.count) entry=\(entryID.uuidString.lowercased())")
        guard promptCacheMode == .multiPrefix, let promptStateStore else { return }
        do {
            let snapshot = try runner.captureInferenceState(
                maximumBytes: promptStateStore.maximumSnapshotBytes)
            guard snapshot.descriptor.position == entry.kvPosition else {
                throw InferenceStateSnapshotError.invalidPosition(
                    snapshot.descriptor.position)
            }
            let saved = await promptStateStore.save(entry: entry, snapshot: snapshot)
            if let diskError = saved.diskError {
                FileHandle.standardError.write(Data(
                    ("Shrike prompt_cache disk_write_failed error=\(diskError) "
                        + "entry=\(entry.id.uuidString.lowercased())\n").utf8))
            }
            cacheDiag("Shrike prompt_cache stored "
                    + "tokens=\(entry.kvPosition) "
                    + "state_bytes=\(snapshot.payload.count) "
                    + "ram_bytes=\(saved.memoryBytes) "
                    + "disk_bytes=\(saved.diskBytes) "
                    + "entry=\(entry.id.uuidString.lowercased())")
        } catch {
            // S24: the entry now describes bytes the stored snapshot does not,
            // so both go rather than leave a restore that would seat the wrong
            // context. Naming the entry is what lets a log reader tie the next
            // settle's `settle_reset` back to the capture that broke the chain.
            FileHandle.standardError.write(Data(
                ("Shrike prompt_cache snapshot_failed "
                    + "entry=\(entryID.uuidString.lowercased()) error=\(error)\n").utf8))
            promptStateStore.remove(entryIDs: [entryID])
            promptCache.remove(entryIDs: [entryID])
            activePromptCacheEntryID = nil
        }
    }

    /// Settle with a rewrite still running: decide on this request's render
    /// first, then wait. `RewriteArbitration.arbitrate` holds that order, which
    /// is what keeps an aborting request from queueing behind the work its own
    /// arrival invalidated.
    private func arbitratePendingRewrite(renderedPromptIDs: [Int32]) async {
        guard let pending = pendingRewrite else { return }
        let decision = await RewriteArbitration.arbitrate(
            target: pending.target,
            render: renderedPromptIDs,
            cancel: { pending.task.cancel() },
            wait: { await pending.task.value })
        cacheDiag("Shrike prompt_cache arbitrate decision=\(decision.rawValue) "
                + "target=\(pending.target.count) render=\(renderedPromptIDs.count)")
    }

    /// Publish this turn's KV range to the prompt cache, and persist a snapshot
    /// so a later request can resume from it without re-prefilling.
    ///
    /// Every failure path here degrades to "no cache entry" rather than to a
    /// broken one: an entry whose snapshot cannot be captured or verified is
    /// removed again, so the next hit re-prefills instead of attempting a
    /// doomed restore.
    ///
    /// Returns the published entry's id, or nil when nothing survived — which
    /// is what tells a deferred settle whether it has an entry to rewrite.
    private func publishCacheEntry(
        cacheRequest: ValidatedChatRequest,
        result: RawDecodeResult,
        stopStringFiltered: Bool,
        normalization: KVNormalization
    ) -> UUID? {
        if mtpDecoder != nil {
            // Native MTP keeps a second KV stream. Until both states are
            // persisted atomically, do not publish target-only cache entries.
            promptCache.invalidate()
            activePromptCacheEntryID = nil
        } else if promptCacheMode == .singlePrefix {
            guard let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                result: result,
                stopStringFiltered: stopStringFiltered) else {
                promptCache.invalidate()
                return nil
            }
            applying(normalization, to: publication.entry)
            return publication.entry.id
        } else if promptCacheMode == .multiPrefix {
            let previousActive = activePromptCacheEntryID
            if let publication = promptCache.publish(
                domain: promptCacheDomain,
                request: cacheRequest,
                result: result,
                stopStringFiltered: stopStringFiltered) {
                promptStateStore?.remove(entryIDs: publication.evictedEntryIDs)
                let published = applying(normalization, to: publication.entry)
                var backed = true
                do {
                    guard let promptStateStore else {
                        throw ServerPromptStateStoreError.missing(published.id)
                    }
                    // S2: capture is bounded by the store's hard snapshot cap;
                    // the payload is a plain Data copy, so the disk write can
                    // proceed off the actor (dedicated store disk queue) while
                    // the next request starts. Concurrent saves serialize on
                    // the queue, so a later generation's snapshot can never
                    // clobber an in-flight write. The entry is already in the
                    // in-memory cache; a request that races the write simply
                    // misses and re-prefills (restore failure self-heals).
                    let snapshot = try runner.captureInferenceState(
                        maximumBytes: promptStateStore.maximumSnapshotBytes)
                    guard snapshot.descriptor.position == published.kvPosition else {
                        throw InferenceStateSnapshotError.invalidPosition(
                            snapshot.descriptor.position)
                    }
                    let entry = published
                    pendingSnapshotSave = Task.detached(priority: .utility) {
                        [promptStateStore] in
                        let saved = await promptStateStore.save(
                            entry: entry,
                            snapshot: snapshot)
                        if let diskError = saved.diskError {
                            FileHandle.standardError.write(Data(
                                ("Shrike prompt_cache disk_write_failed error=\(diskError) "
                                    + "entry=\(entry.id.uuidString.lowercased())\n").utf8))
                        }
                        FileHandle.standardError.write(Data(
                            ("Shrike prompt_cache stored "
                                + "tokens=\(entry.kvPosition) "
                                + "state_bytes=\(snapshot.payload.count) "
                                + "ram_bytes=\(saved.memoryBytes) "
                                + "disk_bytes=\(saved.diskBytes) "
                                + "entry=\(entry.id.uuidString.lowercased())\n").utf8))
                    }
                } catch {
                    // S24: a snapshot that cannot be captured or verified is
                    // never left published without backing; drop the entry so
                    // the next hit re-prefills instead of a doomed restore.
                    FileHandle.standardError.write(Data(
                        ("Shrike prompt_cache snapshot_failed "
                            + "entry=\(publication.entry.id.uuidString.lowercased()) "
                            + "error=\(error)\n").utf8))
                    promptCache.remove(entryIDs: [publication.entry.id])
                    activePromptCacheEntryID = nil
                    backed = false
                }
                if let previousActive,
                   previousActive != publication.entry.id,
                   promptStateStore?.contains(previousActive) != true {
                    promptCache.remove(entryIDs: [previousActive])
                }
                activePromptCacheEntryID = publication.entry.id
                return backed ? publication.entry.id : nil
            } else {
                activePromptCacheEntryID = nil
            }
        }
        return nil
    }

    private func encodePrompt(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition],
        usesToolTemplate: Bool,
        reasoningEffort: ReasoningEffort
    ) throws -> [Int32] {
        if usesToolTemplate {
            return try tokenizer.encodeToolChat(
                messages: messages, tools: tools, reasoningEffort: reasoningEffort)
        }
        let rendered = try tokenizer.applyChatTemplate(
            messages, reasoningEffort: reasoningEffort)
        return tokenizer.encode(rendered, addBOS: false)
    }

    /// Optional per-request diagnostics: MTP acceptance, the SHRIKE_RUNNER_STATS
    /// stage split, and the SHRIKE_KERNEL_STATS GPU breakdown. All three are
    /// env-gated and read-only, so they stay out of the generation path proper.
    private func emitGenerationDiagnostics(
        activeProducer: any LogitProducer,
        result: RawDecodeResult,
        snapshot runnerSnapshot: RunnerCounterSnapshot
    ) {
        if let activeMTP = activeProducer as? StreamingMTPDecoder {
            let stats = activeMTP.statistics
            let decodeRate = result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            cacheDiag(String(format:
                "Shrike mtp drafted=%d accepted=%d acceptance=%.1f%% "
                    + "target_passes=%d emitted_per_pass=%.3f "
                    + "prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f "
                    + "memory_required_mib=%.1f memory_budget_mib=%.1f",
                stats.draftedTokens,
                stats.acceptedTokens,
                stats.acceptanceRate * 100,
                stats.targetBackbonePasses,
                stats.emittedTokensPerTargetPass,
                result.prefillSeconds,
                result.decodeSeconds,
                decodeRate,
                Double(activeMTP.memoryPlan.requiredBytes) / 1_048_576,
                Double(activeMTP.memoryPlan.budgetBytes) / 1_048_576))
            if ProcessInfo.processInfo.environment["SHRIKE_RUNNER_STATS"] != nil,
               stats.targetBackbonePasses > 0 {
                // Per-pass phase attribution for the Track B1 investigation:
                // where a verify pass's wall time actually goes. Milliseconds
                // averaged over the request's target passes.
                let passes = Double(stats.targetBackbonePasses)
                let ms: (UInt64) -> Double = { Double($0) / passes / 1_000_000 }
                cacheDiag(String(format:
                    "Shrike mtp-phases per_pass_ms proposal=%.3f checkpoint=%.3f "
                        + "verify=%.3f verify_backbone=%.3f verify_head=%.3f "
                        + "verify_argmax=%.3f commit=%.3f rollback=%.3f passes=%d",
                    ms(stats.proposalNanos),
                    ms(stats.checkpointNanos),
                    ms(stats.verifyNanos),
                    ms(stats.verifyBackboneNanos),
                    ms(stats.verifyHeadNanos),
                    ms(stats.verifyArgmaxNanos),
                    ms(stats.commitNanos),
                    ms(stats.rollbackNanos),
                    stats.targetBackbonePasses))
            }
        } else {
            let decodeRate = result.decodeSeconds > 0
                ? Double(result.newTokens) / result.decodeSeconds : 0
            cacheDiag(String(format:
                "Shrike generation prefill_s=%.3f decode_s=%.3f decode_tok_s=%.3f",
                result.prefillSeconds,
                result.decodeSeconds,
                decodeRate))
        }
        if ProcessInfo.processInfo.environment["SHRIKE_RUNNER_STATS"] != nil {
            emitRunnerDiagnostics(result: result, snapshot: runnerSnapshot)
        }
        if ProcessInfo.processInfo.environment["SHRIKE_KERNEL_STATS"] != nil {
            emitKernelDiagnostics(result: result)
        }
    }

    private func emitRunnerDiagnostics(
        result: RawDecodeResult,
        snapshot: RunnerCounterSnapshot
    ) {
        let tokens = max(1, result.newTokens)
        let ms: (UInt64, UInt64) -> Double = { delta, base in
            Double(delta > base ? delta - base : 0) / Double(tokens) / 1_000_000
        }
        let missIoNanos = runner.totalMissIoNanos - snapshot.missIo
        let exposedIoNanos = runner.totalExposedIoNanos - snapshot.exposedIo
        let hiddenPercent = missIoNanos == 0 ? 100.0
            : 100 * (1 - Double(exposedIoNanos) / Double(missIoNanos))
        let expert = runner.expertStreamingStatistics()
            .subtracting(snapshot.expertStreaming)
        let gpuHits = runner.totalGPUClassifiedHits - snapshot.gpuClassifiedHits
        let gpuMisses = runner.totalGPUClassifiedMisses - snapshot.gpuClassifiedMisses
        let gpuAllHit = runner.totalGPUResidencyAllHitLayers - snapshot.gpuAllHitLayers
        cacheDiag(String(
            format: "Shrike runner cb1_ms=%.3f io_ms=%.3f cb2_ms=%.3f "
                + "head_ms=%.3f head_fused_ms=%.3f rdadvise_ms=%.3f "
                + "wait_ms=%.3f body_ms=%.3f rdadvise_calls=%llu rdadvise_mib=%.1f "
                + "expert_hit_rate=%.4f expert_hits=%llu expert_misses=%llu "
                + "expert_evictions=%llu expert_reloads=%llu expert_read_mib=%.1f "
                + "expert_load_p50_ms=%.3f expert_load_p95_ms=%.3f "
                + "expert_load_p99_ms=%.3f io_hidden_pct=%.2f hit_fixup_layers=%llu "
                + "router_readback_ms=%.4f cache_plan_ms=%.4f io_queue_ms=%.4f "
                + "io_completion_to_fixup_ms=%.4f io_host_waits=%llu "
                + "io_host_waits_avoided=%llu gpu_classified_hits=%llu "
                + "gpu_classified_misses=%llu gpu_all_hit_layers=%llu",
            ms(runner.totalCb1Nanos, snapshot.cb1),
            ms(runner.totalIoNanos, snapshot.io),
            ms(runner.totalCb2Nanos, snapshot.cb2),
            ms(runner.totalHeadNanos, snapshot.head),
            ms(runner.totalHeadFusedNanos, snapshot.headFused),
            ms(runner.totalRDAdviseNanos, snapshot.rdadvise),
            ms(runner.totalWaitNanos, snapshot.wait),
            ms(runner.totalBodyNanos, snapshot.body),
            runner.totalRDAdviseCalls - snapshot.rdadviseCalls,
            Double(runner.totalRDAdviseBytes - snapshot.rdadviseBytes) / 1_048_576,
            expert.hitRate, expert.hits, expert.misses, expert.evictions,
            expert.reloads, Double(expert.bytesRead) / 1_048_576,
            Double(expert.loadLatencyPercentile(0.50)) / 1_000_000,
            Double(expert.loadLatencyPercentile(0.95)) / 1_000_000,
            Double(expert.loadLatencyPercentile(0.99)) / 1_000_000,
            hiddenPercent, runner.totalHitFixupLayers - snapshot.hitFixupLayers,
            ms(runner.totalRouterReadbackNanos, snapshot.routerReadback),
            ms(runner.totalCachePlanNanos, snapshot.cachePlan),
            ms(runner.totalIOQueueNanos, snapshot.ioQueue),
            ms(runner.totalIOCompletionToFixupSubmitNanos, snapshot.ioCompletionToFixup),
            runner.totalExpertIOHostWaits - snapshot.ioHostWaits,
            runner.totalExpertIOHostWaitsAvoided - snapshot.ioHostWaitsAvoided,
            gpuHits, gpuMisses, gpuAllHit))
    }

    private func emitKernelDiagnostics(result: RawDecodeResult) {
        let tokens = max(1, result.newTokens)
        let summary = runner.kernelGPUTimingSummary()
        let totalGPU = summary.reduce(0) { $0 + $1.millis }
        for entry in summary {
            cacheDiag(String(
                format: "Shrike kernel role=%@ gpu_ms=%.3f per_token_ms=%.3f count=%d",
                entry.role, entry.millis, entry.millis / Double(tokens), entry.count))
        }
        // Role sums overlap by design. Merged busy/span is the actual queue
        // occupancy and distinguishes useful concurrency from idle gaps.
        let occupancy = runner.kernelGPUOccupancy()
        cacheDiag(String(format: "Shrike kernel total_gpu_ms=%.3f gpu_share_of_decode=%.1f%%",
            totalGPU,
            result.decodeSeconds > 0
                ? totalGPU / (result.decodeSeconds * 1000) * 100 : 0))
        for gap in runner.kernelGPUGaps().prefix(8) {
            cacheDiag(String(
                format: "Shrike gap %@ total_ms=%.1f per_token_ms=%.3f count=%d",
                gap.transition, gap.millis, gap.millis / Double(tokens), gap.count))
        }
        cacheDiag(String(format: "Shrike kernel busy_ms=%.3f span_ms=%.3f "
            + "occupancy=%.1f%% busy_share_of_decode=%.1f%% busy_per_token_ms=%.3f",
            occupancy.busyMillis, occupancy.spanMillis,
            occupancy.spanMillis > 0
                ? occupancy.busyMillis / occupancy.spanMillis * 100 : 0,
            result.decodeSeconds > 0
                ? occupancy.busyMillis / (result.decodeSeconds * 1000) * 100 : 0,
            occupancy.busyMillis / Double(tokens)))
    }
}
