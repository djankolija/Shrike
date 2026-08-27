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

/// A range of KV the runner holds: the token sequence it was prefilled with and
/// the cursor that sequence ends at, keyed by the domain and tool set a render
/// has to share to be compared against it. Matching is byte comparison, so the
/// entry describes no messages, no turn, and no dialect.
struct ServerPromptCacheEntry: Codable, Sendable, Equatable {
    let id: UUID
    let domain: ServerPromptCacheDomain
    let tools: [GFTokenizer.FunctionDefinition]
    let kvBackedTokenIDs: [Int32]
    let uncommittedBoundaryTokenIDs: [Int32]
    let kvPosition: Int

    func rewritten(kvBackedTokenIDs rewritten: [Int32]) -> ServerPromptCacheEntry {
        ServerPromptCacheEntry(
            id: id,
            domain: domain,
            tools: tools,
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
    /// `dropEmission` names a sequence too — the request's own render below its
    /// generation suffix — so it is offered on the same mechanisms. A degenerate
    /// turn that declines does not merely lose its own rewrite: its blob stays
    /// below every later boundary in that conversation, and every downstream
    /// settle then fails its splice check.
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
        let reachable = supportsRewind || canRestore
        let settle: KVRewrite = reachable ? .settleLiveRegion : .none
        let degenerate: KVRewrite = reachable ? .dropEmission : .none
        switch reason {
        case .endOfTurn, .eos:
            return thoughtChannelClosed ? settle : degenerate
        case .toolCalls:
            return .none
        case .maxTokens, .stopString, .external:
            return degenerate
        }
    }

    /// Whether the table declined only for want of a mechanism — the one
    /// decline worth a line, since the turn's blob then stays in the KV under
    /// everything that follows it. Asked of the table rather than restated from
    /// it: the same inputs with the capability granted.
    static func degenerateDeclinedForCapability(
        reason: StopReason,
        thoughtChannelClosed: Bool,
        emittedToolCalls: Bool,
        stopStringFiltered: Bool,
        supportsRewind: Bool,
        canRestore: Bool
    ) -> Bool {
        guard !supportsRewind, !canRestore else { return false }
        return forCompletion(reason: reason,
                             thoughtChannelClosed: thoughtChannelClosed,
                             emittedToolCalls: emittedToolCalls,
                             stopStringFiltered: stopStringFiltered,
                             supportsRewind: true,
                             canRestore: true) == .dropEmission
    }

    /// How much of a degenerate turn's KV the drop keeps: the request's own
    /// render below its generation suffix. What the client sends back for a turn
    /// it never saw finish is its choice, so the KV keeps only the history the
    /// next render reproduces regardless.
    ///
    /// Nil unless the suffix stands byte-for-byte where the subtraction puts it.
    /// A template that renders the suffix differently from
    /// `encode(generationSuffix)` makes that arithmetic a guess, and a
    /// truncation on a guess cuts the KV at a position no render names.
    static func droppedPrefixLength(kvBackedTokenIDs: [Int32],
                                    kvPosition: Int,
                                    promptTokenCount: Int,
                                    generationSuffix: [Int32]) -> Int? {
        let target = promptTokenCount - generationSuffix.count
        guard target > 0,
              target < kvPosition,
              promptTokenCount <= kvBackedTokenIDs.count,
              kvBackedTokenIDs[target..<promptTokenCount]
                .elementsEqual(generationSuffix) else {
            return nil
        }
        return target
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

/// Why a completed turn's KV was left holding bytes the next render will not
/// reproduce.
///
/// Each of these breaks the chain for the whole conversation rather than for
/// the one turn — the unrewritten bytes sit below every later boundary, so the
/// splice check fails from here on — which is why each says so on stderr.
/// Outcomes that leave the KV live by design (a tool hop, a filtered stop
/// string, a settle whose target the KV already holds) are not declines and are
/// not here.
enum KVNormalizationDecline: String, Sendable, Equatable {
    /// `settledSequence` refused: the KV's bytes below the boundary are not the
    /// ones the settled render produces. The tell for an already-poisoned chain.
    case spliceMismatch = "splice_mismatch"
    /// The boundary or the live region would not render.
    case renderFailed = "render_failed"
    /// The settled form does not fit the context the session was built with.
    case overContext = "over_context"
    /// The runner can neither rewind nor restore, so nothing reaches the
    /// degenerate turn's target.
    case degenerateUnsupported = "degenerate_unsupported"
    /// The generation suffix is not where the drop's arithmetic puts it.
    case suffixMismatch = "suffix_mismatch"
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
        let entry = ServerPromptCacheEntry(
            id: UUID(),
            domain: domain,
            tools: request.tools,
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
        renderedPromptIDs: [Int32]
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
                    renderedPromptIDs: renderedPromptIDs) else { continue }
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
        renderedPromptIDs: [Int32]
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

        // S12: direct prefix hit, which an identical-prompt replay takes too —
        // its render extends the entry by nothing.
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

        // Salvage only where it can win: a render the entry already contains
        // whole leaves nothing to prefill, so serving it costs a restore and a
        // truncation to save no forward pass at all.
        guard allowsPartialSalvage,
              commonPrefix > 0,
              commonPrefix < renderedPromptIDs.count else { return nil }
        return .salvage(commonPrefix: commonPrefix)
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
