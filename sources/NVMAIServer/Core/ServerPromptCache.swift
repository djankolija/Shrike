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
    /// spliced onto a truncated blob would be a render nothing produced.
    func truncated(to position: Int) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: id,
            domain: domain,
            inputMessages: [],
            tools: tools,
            assistantTurn: nil,
            kvBackedTokenIDs: Array(kvBackedTokenIDs.prefix(position)),
            uncommittedBoundaryTokenIDs: uncommittedBoundaryTokenIDs,
            kvPosition: position)
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
