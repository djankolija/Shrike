import Foundation
import NVMAI

public enum ServerPromptCacheMode: String, Codable, Sendable, Equatable {
    case off
    case singlePrefix = "single-prefix"
    case multiPrefix = "multi-prefix"
}

struct ServerPromptCacheDomain: Codable, Sendable, Equatable {
    let modelID: String
    let sourceSnapshotHash: String?
    let runtimeProfileHash: String
    let maximumContext: Int
    let kvStorage: String
    let fp16RingEnabled: Bool
    let templateSHA256: String
}

struct CachedAssistantTurn: Codable, Sendable, Equatable {
    let message: GFTokenizer.Message
    let rawStopReason: StopReason
}

struct ServerPromptCacheEntry: Codable, Sendable, Equatable {
    let id: UUID
    let domain: ServerPromptCacheDomain
    let inputMessages: [GFTokenizer.Message]
    let tools: [GFTokenizer.FunctionDefinition]
    let assistantTurn: CachedAssistantTurn?
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int

    /// The structural description goes with the bytes it described: a bridge
    /// spliced onto rewritten bytes would be a render nothing produced.
    func rewritten(kvBackedTokenIDs rewritten: [Int32]) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: id,
            domain: domain,
            inputMessages: [],
            tools: tools,
            assistantTurn: nil,
            kvBackedTokenIDs: rewritten,
            uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
            kvPosition: rewritten.count)
    }

    func truncated(to position: Int) -> ServerPromptCacheEntry {
        rewritten(kvBackedTokenIDs: Array(kvBackedTokenIDs.prefix(position)))
    }
}

/// The rewrite a completed generation's KV takes before its entry is published,
/// so that the cached bytes are the ones the next request will render.
enum KVRewrite: Sendable, Equatable {
    /// Settle the live region: the turns the template will stop rendering
    /// reasoning for are re-prefilled in the form it renders them.
    case settleLiveRegion
    /// Drop this generation's suffix and emission, keeping the request history
    /// the next render still reproduces.
    case dropEmission
    /// Leave the KV as it stands.
    case none

    /// The spec's trigger table. Both signals are decoder state — the stop
    /// reason is a special token id, the thought channel the running parse.
    ///
    /// The settle names a sequence rather than a cursor move, so it is offered
    /// wherever `KVReconstruction` has a mechanism that reaches one — a rewind,
    /// or a restore onto a snapshot. With neither, the only way left is to
    /// reset and re-prefill the conversation whole, and that pays for itself
    /// only if it finishes: aborted, it takes the live KV with it and leaves an
    /// entry no snapshot backs, which is strictly worse than never settling. A
    /// session that can neither rewind nor snapshot therefore declines.
    ///
    /// `dropEmission` is a truncation of the live KV and nothing else expresses
    /// it, so it stays rewind-gated whatever the store can do; the settled
    /// entries below a skipped degenerate turn still carry the conversation.
    static func forCompletion(reason: StopReason,
                              thoughtChannelClosed: Bool,
                              emittedToolCalls: Bool,
                              stopStringFiltered: Bool,
                              supportsRewind: Bool,
                              canRestore: Bool) -> KVRewrite {
        // `publish` rejects a stop-string-filtered turn, and the settle path is
        // the most expensive operation here — a rewrite for an entry that will
        // never exist is minutes spent on nothing.
        guard !stopStringFiltered else { return .none }
        // The settle path rebuilds the turn as a content-only message, whose
        // initializer hard-sets `toolCalls` empty: settling a tool hop would
        // re-prefill the KV to a render its calls had been deleted from.
        guard !emittedToolCalls else { return .none }
        let settle: KVRewrite = supportsRewind || canRestore
            ? .settleLiveRegion : .none
        let degenerate: KVRewrite = supportsRewind ? .dropEmission : .none
        switch reason {
        case .endOfTurn, .eos:
            return thoughtChannelClosed ? settle : degenerate
        case .toolCalls:
            return .none
        case .maxTokens, .stopString, .external:
            return degenerate
        }
    }

    /// The token sequence a settled rewrite leaves in the KV: the bytes below
    /// the boundary as they stand, the whole live region in settled form after
    /// them.
    ///
    /// Nil when the KV's own bytes below the boundary are not the ones the
    /// settled render produces — the Harmony template embeds today's date, so a
    /// conversation spanning midnight drifts — because splicing settled bytes
    /// onto drifted ones builds a prefix no render reproduces.
    static func settledSequence(kvBackedTokenIDs: [Int32],
                                boundaryTokens: [Int32],
                                liveRegionTokens: [Int32]) -> [Int32]? {
        guard !boundaryTokens.isEmpty,
              boundaryTokens.count <= kvBackedTokenIDs.count,
              kvBackedTokenIDs.prefix(boundaryTokens.count)
                .elementsEqual(boundaryTokens) else {
            return nil
        }
        return boundaryTokens + liveRegionTokens
    }
}

/// How a settle seats the KV on a prefix of its target before prefilling the
/// remainder.
///
/// The rewrite's job is to make the KV hold a sequence, which is a statement
/// about bytes and not about cursors. A rewind reaches a prefix only on a
/// runner whose state can follow the cursor back; a snapshot restore reaches
/// one on every architecture, which is what the prompt cache's own restore path
/// already does; and with nothing to seat on, the target is prefilled whole —
/// the same work the next request would pay for anyway.
enum KVReconstruction: Sendable, Equatable {
    /// Seat the live cursor at this position: a rewind, or nothing at all when
    /// the cursor already sits there.
    case rewind(to: Int)
    /// Restore this entry's snapshot, whose bytes are the target's first
    /// `position` tokens.
    case restore(entryID: UUID, position: Int)
    /// Reset and prefill the target whole.
    case reset(reason: ResetReason)

    /// Why no snapshot could be seated on. Both are worth telling apart in a
    /// log: the first says the chain was never built (a store that is off, or
    /// captures that failed or were refused), the second that it was broken.
    enum ResetReason: String, Sendable, Equatable {
        case noSnapshot = "no_snapshot"
        case noPrefixSnapshot = "no_prefix_snapshot"
    }

    /// A restorable snapshot as the decision sees it: where it seats the cursor,
    /// and whether the target opens with its bytes.
    struct Snapshot: Sendable, Equatable {
        let entryID: UUID
        let position: Int
        let isPrefixOfTarget: Bool
    }

    /// Where the chosen mechanism leaves the cursor — the position the target's
    /// remainder is prefilled from.
    var seatedPosition: Int {
        switch self {
        case .rewind(let position): return position
        case .restore(_, let position): return position
        case .reset: return 0
        }
    }

    /// A cursor already at the target's parting point moves for free and is
    /// taken whatever the runner supports; a rewind is the next cheapest, being
    /// a cursor move that reads nothing; then the longest snapshot the target
    /// opens with; then the target whole.
    static func plan(supportsRewind: Bool,
                     livePosition: Int,
                     rewindTo: Int,
                     snapshots: [Snapshot],
                     targetCount: Int) -> KVReconstruction {
        if livePosition == rewindTo || supportsRewind {
            return .rewind(to: rewindTo)
        }
        var best: Snapshot?
        for snapshot in snapshots
        where snapshot.isPrefixOfTarget
            && snapshot.position > 0
            && snapshot.position <= targetCount {
            if snapshot.position > (best?.position ?? 0) { best = snapshot }
        }
        if let best {
            return .restore(entryID: best.entryID, position: best.position)
        }
        return .reset(reason: snapshots.isEmpty ? .noSnapshot : .noPrefixSnapshot)
    }
}

/// What a request arriving while a rewrite is still prefilling does with it.
enum RewriteArbitration: String, Sendable, Equatable {
    /// The rewrite is this request's prefill already running: wait for it and
    /// match against the entry it leaves behind.
    case join
    /// The rewrite is producing bytes this request cannot use: stop it and
    /// take the normal match path.
    case abort

    /// Bytes only — no session identity is tested and none is needed. A
    /// follow-up turn joins by construction, because the settled sequence a
    /// completed turn is rewritten into is what that turn's next render opens
    /// with; regeneration and edits abort.
    ///
    /// A rewrite naming no target is evidence of nothing, so it aborts rather
    /// than making every request wait on the vacuous prefix.
    static func decide(target: [Int32], render: [Int32]) -> RewriteArbitration {
        guard !target.isEmpty,
              render.prefix(target.count).elementsEqual(target) else {
            return .abort
        }
        return .join
    }

    /// Decide, then wait — never wait, then decide. A request the rewrite is
    /// not prefilling for has to cancel it before it queues, or it blocks on
    /// work its own arrival invalidated.
    static func arbitrate(target: [Int32],
                          render: [Int32],
                          cancel: @Sendable () -> Void,
                          wait: @Sendable () async -> Void) async -> RewriteArbitration {
        let decision = decide(target: target, render: render)
        if decision == .abort { cancel() }
        await wait()
        return decision
    }
}

enum ServerPromptCacheMatch: Sendable, Equatable {
    case miss
    case hit(entryID: UUID, effectivePromptIDs: [Int32], cachedPromptTokens: Int)
}

struct ServerPromptCachePublication: Sendable, Equatable {
    let entry: ServerPromptCacheEntry
    let evictedEntryIDs: [UUID]
}

struct ServerPromptCache: Sendable {
    private let maximumEntries: Int
    /// Serving a partial salvage costs a state restore the runner must then seat
    /// on a shorter prefix. Where it cannot, a divergent render stays a plain
    /// miss: the entry keeps its bytes and nothing is read back to be discarded.
    private let allowsPartialSalvage: Bool
    private(set) var entries: [ServerPromptCacheEntry]

    init(maximumEntries: Int = 1,
         entries: [ServerPromptCacheEntry] = [],
         allowsPartialSalvage: Bool = true) {
        precondition(maximumEntries > 0, "maximumEntries must be positive")
        self.maximumEntries = maximumEntries
        self.allowsPartialSalvage = allowsPartialSalvage
        self.entries = Array(entries.suffix(maximumEntries))
    }

    mutating func invalidate() {
        entries.removeAll(keepingCapacity: true)
    }

    mutating func remove(entryIDs: some Sequence<UUID>) {
        let removed = Set(entryIDs)
        guard !removed.isEmpty else { return }
        entries.removeAll { removed.contains($0.id) }
    }

    @discardableResult
    mutating func publish(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        content: String,
        calls: [ParsedToolCall],
        result: RawDecodeResult,
        stopStringFiltered: Bool = false
    ) -> ServerPromptCachePublication? {
        guard result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              !stopStringFiltered,
              result.reason == .endOfTurn
                || result.reason == .eos
                || result.reason == .toolCalls
                || result.reason == .maxTokens else {
            return nil
        }
        let historicalCalls = calls.map {
            GFTokenizer.HistoricalToolCall(
                id: $0.id,
                name: $0.name,
                arguments: $0.argumentsJSON)
        }
        let assistant = GFTokenizer.Message(
            role: .assistant,
            content: calls.isEmpty ? content : nil,
            toolCalls: historicalCalls)
        let entry = ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            inputMessages: request.messages,
            tools: request.tools,
            assistantTurn: CachedAssistantTurn(
                message: assistant,
                rawStopReason: result.reason),
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs,
            kvPosition: result.kvPosition)
        var evicted = entries.filter {
            $0.domain == entry.domain
                && $0.kvBackedTokenIDs == entry.kvBackedTokenIDs
                && $0.tools == entry.tools
        }.map(\.id)
        entries.removeAll { evicted.contains($0.id) }
        entries.append(entry)
        while entries.count > maximumEntries {
            evicted.append(entries.removeFirst().id)
        }
        return ServerPromptCachePublication(
            entry: entry,
            evictedEntryIDs: evicted)
    }

    /// Follow a KV rewrite with the entry that describes it. The rewrite moved
    /// the live cursor, so the entry has to move with it or the
    /// `kvPosition == kvBackedTokenIDs.count` invariant breaks at the next match.
    @discardableResult
    mutating func rewrite(entryID: UUID,
                          kvBackedTokenIDs: [Int32]) -> ServerPromptCacheEntry? {
        guard let index = entries.firstIndex(where: { $0.id == entryID }) else {
            return nil
        }
        entries[index] = entries[index].rewritten(kvBackedTokenIDs: kvBackedTokenIDs)
        return entries[index]
    }

    mutating func match(
        domain: ServerPromptCacheDomain,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> ServerPromptCacheMatch {
        var best: (index: Int, candidate: EntryMatch)?
        for (index, entry) in entries.enumerated() {
            guard entry.domain == domain,
                  entry.tools == request.tools,
                  entry.kvPosition == entry.kvBackedTokenIDs.count,
                  entry.kvPosition > 0,
                  entry.uncommittedBoundaryTokenIDs.count == 1,
                  let candidate = match(
                    entry: entry,
                    request: request,
                    renderedPromptIDs: renderedPromptIDs,
                    tokenizer: tokenizer) else { continue }
            if candidate.cachedTokens > (best?.candidate.cachedTokens ?? -1) {
                best = (index, candidate)
            }
        }
        guard let best else { return .miss }
        let effective: [Int32]
        switch best.candidate {
        case .resume(let resumed, _):
            effective = resumed
        case .salvage(let commonPrefix):
            entries[best.index] = entries[best.index].truncated(to: commonPrefix)
            effective = renderedPromptIDs
        }
        let matched = entries.remove(at: best.index)
        entries.append(matched)
        return .hit(
            entryID: matched.id,
            effectivePromptIDs: effective,
            cachedPromptTokens: best.candidate.cachedTokens)
    }

    private enum EntryMatch {
        case resume(effective: [Int32], cachedTokens: Int)
        case salvage(commonPrefix: Int)

        var cachedTokens: Int {
            switch self {
            case .resume(_, let cachedTokens): return cachedTokens
            case .salvage(let commonPrefix): return commonPrefix
            }
        }
    }

    private func match(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        renderedPromptIDs: [Int32],
        tokenizer: GFTokenizer
    ) -> EntryMatch? {
        guard entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return nil
        }

        let comparableLength = min(renderedPromptIDs.count, entry.kvPosition)
        let commonPrefix = (0..<comparableLength).first {
            renderedPromptIDs[$0] != entry.kvBackedTokenIDs[$0]
        } ?? comparableLength
        NVMAICacheDiag.log(
            "lcp k=\(commonPrefix) kv=\(entry.kvPosition) "
                + "fraction=\(Double(commonPrefix) / Double(entry.kvPosition)) "
                + "entry=\(entry.id.uuidString.lowercased())")

        // S12: direct prefix hit. An identical-prompt replay, whose render
        // extends the entry by nothing, hits here too rather than falling
        // through to the structural paths.
        if commonPrefix == entry.kvPosition {
            return .resume(effective: renderedPromptIDs, cachedTokens: entry.kvPosition)
        }
        if renderedPromptIDs.count < entry.kvPosition {
            NVMAICacheDiag.log(
                "s12_short rendered=\(renderedPromptIDs.count) kv=\(entry.kvPosition)")
        } else {
            let lo = max(0, commonPrefix - 6)
            let hi = min(entry.kvPosition, commonPrefix + 6)
            NVMAICacheDiag.log(
                "s12_diverge at=\(commonPrefix) of kv=\(entry.kvPosition) "
                    + "window=\(lo)..<\(hi) "
                    + "rendered=\(Array(renderedPromptIDs[lo..<hi])) "
                    + "cached=\(Array(entry.kvBackedTokenIDs[lo..<hi]))")
        }

        if let structural = structuralMatch(
            entry: entry,
            request: request,
            tokenizer: tokenizer) {
            return .resume(
                effective: structural.effective,
                cachedTokens: structural.cached)
        }
        // Partial salvage runs last so it cannot truncate KV the structural path
        // would have restored whole, and only where it can win: a render the
        // entry already contains whole leaves nothing to prefill, so serving it
        // costs a restore and a truncation to save no forward pass at all.
        guard allowsPartialSalvage,
              commonPrefix > 0,
              commonPrefix < renderedPromptIDs.count else { return nil }
        return .salvage(commonPrefix: commonPrefix)
    }

    private func structuralMatch(
        entry: ServerPromptCacheEntry,
        request: ValidatedChatRequest,
        tokenizer: GFTokenizer
    ) -> (effective: [Int32], cached: Int)? {
        let inputCount = entry.inputMessages.count
        guard let assistantTurn = entry.assistantTurn,
              request.messages.count > inputCount + 1,
              request.messages.prefix(inputCount)
                .elementsEqual(entry.inputMessages),
              assistantMatches(
                request.messages[inputCount],
                assistantTurn.message) else {
            NVMAICacheDiag.log(
                "structural_reject inputCount=\(inputCount) "
                    + "reqMsgs=\(request.messages.count)")
            return nil
        }
        let continuation = Array(request.messages.dropFirst(inputCount + 1))
        NVMAICacheDiag.log(
            "structural inputCount=\(inputCount) reqMsgs=\(request.messages.count) "
                + "continuation=\(continuation.count) "
                + "entryCalls=\(assistantTurn.message.toolCalls.count) "
                + "stopReason=\(assistantTurn.rawStopReason)")

        if assistantTurn.message.toolCalls.isEmpty {
            return matchTextContinuation(
                entry: entry,
                assistantTurn: assistantTurn,
                continuation: continuation,
                tokenizer: tokenizer)
        }
        return matchToolContinuation(
            entry: entry,
            assistantTurn: assistantTurn,
            request: request,
            continuation: continuation,
            tokenizer: tokenizer)
    }

    private func assistantMatches(
        _ incoming: GFTokenizer.Message,
        _ cached: GFTokenizer.Message
    ) -> Bool {
        guard incoming.role == .assistant,
              cached.role == .assistant,
              incoming.toolCalls == cached.toolCalls,
              incoming.toolCallID == cached.toolCallID,
              incoming.name == cached.name else {
            return false
        }
        if !cached.toolCalls.isEmpty {
            return (incoming.content ?? "").isEmpty
                && (cached.content ?? "").isEmpty
        }
        return incoming.content == cached.content
    }

    private func matchTextContinuation(
        entry: ServerPromptCacheEntry,
        assistantTurn: CachedAssistantTurn,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> (effective: [Int32], cached: Int)? {
        // S13: support multi-turn continuations by matching on the trailing
        // user message. The whole tail after the cached assistant turn is
        // re-encoded with the same text-only ChatML template used to render
        // the original prompt, so the prefill reproduces the request's render
        // byte-for-byte. Kept conservative: the tail must be plain text-only
        // turns (no tool calls/results or tool ids, which the text template
        // cannot represent) and must end in a user message so the generation
        // suffix applies.
        // The bridge below is ChatML-shaped (leading "\n", <|im_end|> seam);
        // other dialects re-prefill from scratch.
        guard tokenizer.dialect == .chatml,
              let last = continuation.last,
              last.role == .user,
              continuation.allSatisfy({
                  $0.role != .tool && $0.toolCallID == nil && $0.toolCalls.isEmpty
              }),
              assistantTurn.rawStopReason == .endOfTurn
                || assistantTurn.rawStopReason == .maxTokens,
              let renderedTail = try? tokenizer.applyChatTemplate(continuation)
        else {
            return nil
        }
        // The bridge begins with the cached turn's closing <|im_end|>, then
        // the rendered tail (which includes the generation suffix).
        var bridge = [tokenizer.endOfTurnID]
            + tokenizer.encode("\n" + renderedTail, addBOS: false)
        if assistantTurn.rawStopReason == .maxTokens {
            // S14: the uncommitted boundary token (the last generated token,
            // never committed to KV) must be replayed first — but apply the
            // same first-token dedup as the endOfTurn branch so a bridge that
            // already begins with the boundary token is not doubled.
            if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
                bridge = entry.uncommittedBoundaryTokenIDs + bridge
            }
        } else if bridge.first != entry.uncommittedBoundaryTokenIDs.first {
            return nil
        }
        return (entry.kvBackedTokenIDs + bridge, entry.kvPosition)
    }

    private func matchToolContinuation(
        entry: ServerPromptCacheEntry,
        assistantTurn: CachedAssistantTurn,
        request: ValidatedChatRequest,
        continuation: [GFTokenizer.Message],
        tokenizer: GFTokenizer
    ) -> (effective: [Int32], cached: Int)? {
        let calls = assistantTurn.message.toolCalls
        guard assistantTurn.rawStopReason == .toolCalls,
              continuation.count == calls.count,
              zip(continuation, calls).allSatisfy({ message, call in
                  message.role == .tool
                    && message.toolCallID == call.id
                    && (message.name == nil || message.name == call.name)
                    && message.content != nil
                    && message.toolCalls.isEmpty
              }) else {
            return nil
        }
        guard let bridge = try? tokenizer.encodeToolResultContinuation(
            cachedMessages: entry.inputMessages,
            assistant: assistantTurn.message,
            incomingMessages: request.messages,
            tools: request.tools),
              bridge.first == entry.uncommittedBoundaryTokenIDs.first else {
            return nil
        }
        return (entry.kvBackedTokenIDs + bridge, entry.kvPosition)
    }
}

/// Opt-in cache diagnostics: set NVMAI_CACHE_DIAG=1 to have every match
/// failure say which check rejected the entry. Off by default so the hot
/// path stays quiet.
enum NVMAICacheDiag {
    static let enabled = ProcessInfo.processInfo.environment["NVMAI_CACHE_DIAG"] != nil

    static func log(_ message: String) {
        guard enabled else { return }
        FileHandle.standardError.write(Data("NVMAI prompt_cache_diag \(message)\n".utf8))
    }
}
