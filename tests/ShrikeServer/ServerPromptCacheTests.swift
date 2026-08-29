import Foundation
import Synchronization
import Testing

@testable import Shrike
@testable import ShrikeServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func aTextContinuationSalvagesTheSharedPrefixOfItsRender() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered)

        // The re-render drops the turn's reasoning, so it is not a prefix of
        // the KV and the entry keeps only what both renders share.
        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a common-prefix salvage")
            return
        }
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective == rendered)
        #expect(cached > 0)
        #expect(cached < kvBacked.count)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == Array(kvBacked.prefix(cached)))
        #expect(entry.kvPosition == cached)
    }

    @Test func kimiContinuationsHitTheRenderedPrefixWhole() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.kimiFolder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered)

        // Kimi's template is append-only (no <think> stripping, no dropped
        // analysis), so the re-render extends the cached KV byte-for-byte and
        // the whole entry is served.
        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a rendered-prefix hit")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(effective == rendered)
        #expect(rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func unsafeStopsDoNotPublishAndAChangedLineageOnlySalvagesItsPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .external] {
            let publication = cache.publish(
                domain: domain,
                request: initial,
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            // Unsafe stop reasons must not publish a cacheable entry
            // (S17). `publish` returns nil when the entry is rejected.
            #expect(publication == nil)
        }

        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: prompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered)

        // A changed first turn diverges inside the prompt, so all the entry can
        // keep is the template preamble both renders share.
        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a common-prefix salvage")
            return
        }
        #expect(effective == rendered)
        #expect(cached > 0)
        #expect(cached < prompt.count)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == Array(kvBacked.prefix(cached)))
        #expect(entry.kvPosition == cached)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        // A stop-string-filtered tail must not be published as a cacheable
        // entry: the cached KV would resume from a partial stop string (S17).
        #expect(publication == nil)
    }

    @Test func multiPrefixChoosesLongestExactPrefixAndUsesLRUEviction() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache(maximumEntries: 2)
        let shortPublication = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2],
                kvBacked: [1, 2],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let short = try #require(shortPublication)
        let longPublication = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let long = try #require(longPublication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4])
        #expect(match == .hit(
            entryID: long.entry.id,
            effectivePromptIDs: [1, 2, 3, 4],
            cachedPromptTokens: 3))

        let newestPublication = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [9],
                kvBacked: [9],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let newest = try #require(newestPublication)
        #expect(newest.evictedEntryIDs == [short.entry.id])
        #expect(cache.entries.map(\.id) == [long.entry.id, newest.entry.id])
    }

    /// An identical-prompt replay of a published entry reports the ENTIRE
    /// prompt as cached (`cachedPromptTokens == rendered.count`): at the
    /// session layer (S12) that leaves nothing to prefill, which is what
    /// "a prompt-cache hit short-circuits prefill" means. The end-to-end
    /// resume-path half of that contract lives inside `ServerModelSession`
    /// (its private init and hardcoded production-arch load make it
    /// untestable with the synthetic toy); this pins the cache layer's half.
    @Test func identicalReplayReportsEntirePromptAsCached() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt)
        #expect(match == .hit(
            entryID: entry.entry.id,
            effectivePromptIDs: prompt,
            cachedPromptTokens: prompt.count))
    }

    @Test func fullPrefixHitLeavesTheEntryWhole() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache()
        let published = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let publication = try #require(published)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4, 5])

        #expect(match == .hit(
            entryID: publication.entry.id,
            effectivePromptIDs: [1, 2, 3, 4, 5],
            cachedPromptTokens: 3))
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2, 3])
        #expect(entry.kvPosition == 3)
    }

    /// The shape the interim match order existed to protect — a render the
    /// entry outruns because the blob holds reasoning the re-render drops —
    /// and what it falls through to now: a salvage, which truncates the entry
    /// to what it served.
    ///
    /// This pins that outcome, not the order. The render diverges inside the
    /// entry, so the full-hit branch is unreachable here and a salvage-first
    /// cache would answer identically. The order is carried by
    /// `identicalReplayReportsEntirePromptAsCached` and by the closing
    /// full-prefix half of `aRunnerThatCannotRewindMissesAndKeepsTheEntry`,
    /// both of which become misses if salvage is consulted first.
    @Test func aThinkingRetainedContinuationSalvagesAndTruncatesTheEntry() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        // The blob carries the turn's reasoning and the re-render drops it, so
        // the render comes back shorter than the entry.
        let generated = tokenizer.encode(
            String(repeating: "reasoning ", count: 40) + "answer",
            addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        #expect(rendered.count < kvBacked.count)

        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a common-prefix salvage")
            return
        }
        #expect(effective == rendered)
        #expect(cached > 0)
        #expect(cached < rendered.count)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == Array(kvBacked.prefix(cached)))
        #expect(entry.kvPosition == cached)
    }

    @Test func divergentTailTruncatesTheEntryToTheCommonPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache()
        let published = cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let publication = try #require(published)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 9, 9])

        #expect(match == .hit(
            entryID: publication.entry.id,
            effectivePromptIDs: [1, 2, 9, 9],
            cachedPromptTokens: 2))
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2])
        #expect(entry.kvPosition == entry.kvBackedTokenIDs.count)
    }

    @Test func aRenderTheEntryAlreadyContainsWholeMissesRatherThanSalvaging() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // Every token of the render is cached already, so a salvage would save no
        // forward pass — and would truncate the entry to buy that nothing.
        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3])

        #expect(match == .miss)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2, 3, 4, 5])
    }

    @Test func aRunnerThatCannotRewindMissesAndKeepsTheEntry() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache(allowsPartialSalvage: false)
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // The render that salvages in divergentTailTruncatesTheEntryToTheCommon-
        // Prefix, against a cache whose runner cannot seat a shorter cursor.
        #expect(cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 9, 9]) == .miss)
        let kept = try #require(cache.entries.last)
        #expect(kept.kvBackedTokenIDs == [1, 2, 3, 4, 5])
        #expect(kept.kvPosition == 5)

        // Only the salvage is gated: a full-prefix hit needs no rewind.
        #expect(cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4, 5, 6]) == .hit(
                entryID: kept.id,
                effectivePromptIDs: [1, 2, 3, 4, 5, 6],
                cachedPromptTokens: 5))
    }

    @Test func aRenderWithNoCommonPrefixMissesAndLeavesTheEntryWhole() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [7, 8])

        #expect(match == .miss)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2, 3, 4, 5])
        #expect(entry.kvPosition == 5)
    }

    /// The post-strip view swaps only the messages and tools; every other
    /// validated field must survive, or the cached turn would silently change
    /// sampling or streaming behavior.
    @Test func replacingMessagesPreservesTheRestOfTheRequest() {
        let original = request(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [])
        let replaced = original.replacingMessages(
            [GFTokenizer.Message(role: .user, content: "stripped")],
            tools: [])

        #expect(replaced.messages.map(\.content) == ["stripped"])
        #expect(replaced.stream == original.stream)
        #expect(replaced.includeUsage == original.includeUsage)
        #expect(replaced.maximumCompletionTokens == original.maximumCompletionTokens)
        #expect(replaced.generationConfig.maxNewTokens
            == original.generationConfig.maxNewTokens)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }
}

@Suite("Completed generation KV rewrite")
struct KVRewriteTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    private static let everyStopReason: [StopReason] = [
        .endOfTurn, .eos, .toolCalls, .maxTokens, .stopString, .external,
    ]

    /// Every capability a session can have, since 6a made the trigger depend on
    /// two of them. Both rewrites ask only that *some* mechanism reaches their
    /// target, so a rewind or a store will do for either. A session with neither
    /// is offered nothing at all — its only remaining mechanism is a reset whose
    /// abort would take the cache with it.
    private static let everyCapability: [(supportsRewind: Bool, canRestore: Bool)] = [
        (true, true), (true, false), (false, true), (false, false),
    ]

    @Test func theTriggerIsTheSpecsTableOverEveryStopReasonAndCapability() throws {
        typealias Row = (closed: KVRewrite, open: KVRewrite)
        let rewinding: [StopReason: Row] = [
            .endOfTurn: (.settleLiveRegion, .dropEmission),
            .eos: (.settleLiveRegion, .dropEmission),
            .toolCalls: (.none, .none),
            .maxTokens: (.dropEmission, .dropEmission),
            .stopString: (.dropEmission, .dropEmission),
            .external: (.dropEmission, .dropEmission),
        ]
        let neither: [StopReason: Row] = [
            .endOfTurn: (.none, .none),
            .eos: (.none, .none),
            .toolCalls: (.none, .none),
            .maxTokens: (.none, .none),
            .stopString: (.none, .none),
            .external: (.none, .none),
        ]
        // A rewind reaches the target with or without a store, and 6d made the
        // drop's target a sequence a restore reaches too — so every capability
        // holding some mechanism runs one table, and only the session holding
        // none declines.
        let tables: [[StopReason: Row]] = [rewinding, rewinding, rewinding, neither]
        #expect(tables.count == Self.everyCapability.count)
        for table in tables { #expect(table.count == Self.everyStopReason.count) }
        for reason in Self.everyStopReason {
            for (index, capability) in Self.everyCapability.enumerated() {
                let row = try #require(tables[index][reason])
                #expect(KVRewrite.forCompletion(
                    reason: reason,
                    thoughtChannelClosed: true,
                    emittedToolCalls: false,
                    stopStringFiltered: false,
                    supportsRewind: capability.supportsRewind,
                    canRestore: capability.canRestore) == row.closed,
                        "\(reason) closed \(capability)")
                #expect(KVRewrite.forCompletion(
                    reason: reason,
                    thoughtChannelClosed: false,
                    emittedToolCalls: false,
                    stopStringFiltered: false,
                    supportsRewind: capability.supportsRewind,
                    canRestore: capability.canRestore) == row.open,
                        "\(reason) open \(capability)")
            }
        }
    }

    /// The loss path 6a would otherwise open: with no rewind and no store, the
    /// only mechanism left is reset-and-prefill-whole, and an abort takes the
    /// live KV with it and leaves an entry no snapshot backs. Before 6a this
    /// runner declined every rewrite; it still declines this one.
    @Test func aSessionThatCanNeitherRewindNorSnapshotDeclinesTheSettle() {
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                #expect(KVRewrite.forCompletion(
                    reason: reason,
                    thoughtChannelClosed: closed,
                    emittedToolCalls: false,
                    stopStringFiltered: false,
                    supportsRewind: false,
                    canRestore: false) == KVRewrite.none,
                        "\(reason) closed=\(closed)")
            }
        }
        // A store alone is enough to bring either rewrite back, which is what
        // separates this from a blanket capability gate.
        #expect(KVRewrite.forCompletion(
            reason: .endOfTurn,
            thoughtChannelClosed: true,
            emittedToolCalls: false,
            stopStringFiltered: false,
            supportsRewind: false,
            canRestore: true) == .settleLiveRegion)
        #expect(KVRewrite.forCompletion(
            reason: .maxTokens,
            thoughtChannelClosed: true,
            emittedToolCalls: false,
            stopStringFiltered: false,
            supportsRewind: false,
            canRestore: true) == .dropEmission)
    }

    /// The line the matrix needed and did not have: a `finish=length` turn on a
    /// runner that can neither rewind nor restore leaves its blob in the KV, and
    /// every later settle in that conversation splices onto it. Only that case
    /// is a decline — a tool hop and a filtered stop string leave the KV live by
    /// design, and would say nothing however the capability fell.
    @Test func onlyACapabilityDeclineOfADropIsWorthNaming() {
        #expect(KVRewrite.degenerateDeclinedForCapability(
            reason: .maxTokens,
            thoughtChannelClosed: true,
            emittedToolCalls: false,
            stopStringFiltered: false,
            supportsRewind: false,
            canRestore: false))
        // The settle declines on the same capability, but the drop is what the
        // measured poison came from and what this line is for.
        #expect(!KVRewrite.degenerateDeclinedForCapability(
            reason: .endOfTurn,
            thoughtChannelClosed: true,
            emittedToolCalls: false,
            stopStringFiltered: false,
            supportsRewind: false,
            canRestore: false))
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                for capability in Self.everyCapability {
                    // A capability that reaches the target declines nothing.
                    #expect(capability == (false, false)
                        || !KVRewrite.degenerateDeclinedForCapability(
                            reason: reason,
                            thoughtChannelClosed: closed,
                            emittedToolCalls: false,
                            stopStringFiltered: false,
                            supportsRewind: capability.supportsRewind,
                            canRestore: capability.canRestore),
                            "\(reason) closed=\(closed) \(capability)")
                    // A by-design-live outcome is silent whatever the runner is.
                    #expect(!KVRewrite.degenerateDeclinedForCapability(
                        reason: reason,
                        thoughtChannelClosed: closed,
                        emittedToolCalls: true,
                        stopStringFiltered: false,
                        supportsRewind: capability.supportsRewind,
                        canRestore: capability.canRestore))
                    #expect(!KVRewrite.degenerateDeclinedForCapability(
                        reason: reason,
                        thoughtChannelClosed: closed,
                        emittedToolCalls: false,
                        stopStringFiltered: true,
                        supportsRewind: capability.supportsRewind,
                        canRestore: capability.canRestore))
                }
            }
        }
    }

    /// The reasons are read out of a log by a person or a probe, so the strings
    /// are the interface and a rename is a break.
    @Test func everyDeclineNamesItselfInTheLogsVocabulary() {
        #expect(Set(
            [KVNormalizationDecline.spliceMismatch,
             .renderFailed,
             .overContext,
             .degenerateUnsupported,
             .suffixMismatch].map(\.rawValue))
            == ["splice_mismatch",
                "render_failed",
                "over_context",
                "degenerate_unsupported",
                "suffix_mismatch"])
    }

    /// The drop keeps the request's own render below its generation suffix, and
    /// verifies the suffix is there rather than trusting the subtraction — the
    /// check that caught a Jinja template rendering it differently.
    @Test func theDropKeepsTheRenderBelowAVerifiedGenerationSuffix() {
        let prompt: [Int32] = [1, 2, 3, 4, 90, 91]
        let kv = prompt + [50, 51, 52]
        #expect(KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: kv,
            kvPosition: kv.count,
            promptTokenCount: prompt.count,
            generationSuffix: [90, 91]) == 4)
        // The suffix is not what the template rendered: nothing is dropped
        // rather than cutting at a position no render names.
        #expect(KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: kv,
            kvPosition: kv.count,
            promptTokenCount: prompt.count,
            generationSuffix: [90, 99]) == nil)
        // A prompt that is nothing but its suffix leaves no history to keep.
        #expect(KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: [90, 91, 50],
            kvPosition: 3,
            promptTokenCount: 2,
            generationSuffix: [90, 91]) == nil)
        // A prompt the KV does not hold whole: the suffix cannot be verified
        // where it would have to be read past the end to look.
        #expect(KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: kv,
            kvPosition: kv.count,
            promptTokenCount: kv.count + 1,
            generationSuffix: [90, 91]) == nil)
    }

    @Test func aTurnCarryingToolCallsStaysLiveWhateverStopEndedIt() {
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                for capability in Self.everyCapability {
                    #expect(KVRewrite.forCompletion(
                        reason: reason,
                        thoughtChannelClosed: closed,
                        emittedToolCalls: true,
                        stopStringFiltered: false,
                        supportsRewind: capability.supportsRewind,
                        canRestore: capability.canRestore) == KVRewrite.none)
                }
            }
        }
    }

    /// `publish` rejects a stop-string-filtered turn, so the settle path's
    /// forward pass would be spent producing bytes nothing ever stores.
    @Test func aStopStringFilteredTurnIsNeverWorthARewrite() {
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                for capability in Self.everyCapability {
                    #expect(KVRewrite.forCompletion(
                        reason: reason,
                        thoughtChannelClosed: closed,
                        emittedToolCalls: false,
                        stopStringFiltered: true,
                        supportsRewind: capability.supportsRewind,
                        canRestore: capability.canRestore) == KVRewrite.none)
                }
            }
        }
    }

    @Test func aSettledRewriteSplicesOnlyOntoTheBytesItsBoundaryDescribes() {
        let settled = KVRewrite.settledSequence(
            kvBackedTokenIDs: [1, 2, 3, 40, 50],
            boundaryTokens: [1, 2, 3],
            liveRegionTokens: [7, 8])
        #expect(settled == [1, 2, 3, 7, 8])

        #expect(KVRewrite.settledSequence(
            kvBackedTokenIDs: [1, 2, 9, 40, 50],
            boundaryTokens: [1, 2, 3],
            liveRegionTokens: [7, 8]) == nil)
        #expect(KVRewrite.settledSequence(
            kvBackedTokenIDs: [1, 2],
            boundaryTokens: [1, 2, 3],
            liveRegionTokens: [7, 8]) == nil)
        #expect(KVRewrite.settledSequence(
            kvBackedTokenIDs: [1, 2, 3],
            boundaryTokens: [],
            liveRegionTokens: [7, 8]) == nil)

        // The no-op the caller declines on: a KV already holding the settled
        // form yields a sequence equal to it, so nothing is rewritten.
        #expect(KVRewrite.settledSequence(
            kvBackedTokenIDs: [1, 2, 3],
            boundaryTokens: [1, 2],
            liveRegionTokens: [3]) == [1, 2, 3])
    }

    /// The date the Harmony template embeds moves at midnight, so a boundary
    /// rendered now can describe a different prefix than the one the KV holds.
    @Test func aDriftedBoundaryRenderLeavesTheEntryExactlyAsPublished() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.harmonyFolder())
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        let published = cache.publish(
            domain: domain,
            request: request(messages: messages),
            result: rawResult(kvBacked: kvBacked, reason: .eos))
        let publication = try #require(published)

        let completed = messages
            + [GFTokenizer.Message(role: .assistant, content: "answer")]
        let boundary = try tokenizer.settledBoundaryTokens(
            messages: completed, tools: [])
        var drifted = kvBacked
        drifted[boundary.count - 1] = drifted[boundary.count - 1] &+ 1
        #expect(KVRewrite.settledSequence(
            kvBackedTokenIDs: drifted,
            boundaryTokens: boundary,
            liveRegionTokens: try tokenizer.settledLiveRegionTokens(
                messages: completed, tools: [])) == nil)

        let entry = try #require(cache.entries.last)
        #expect(entry == publication.entry)
        #expect(entry.kvBackedTokenIDs == kvBacked)
    }

    /// Harmony stops at `<|return|>`, which the decode loop reports as `.eos`;
    /// nothing about the dialect keeps its turns out of the cache any more.
    @Test func aHarmonyTurnPublishesLikeEveryOtherDialect() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.harmonyFolder())
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()

        let publication = cache.publish(
            domain: domain,
            request: request(messages: messages),
            result: rawResult(kvBacked: kvBacked, reason: .eos))

        let entry = try #require(publication?.entry)
        #expect(entry.kvBackedTokenIDs == kvBacked)
        #expect(entry.kvPosition == kvBacked.count)
    }

    /// The whole rewrite short of the forward pass: the settled bytes a clean
    /// turn leaves in the KV are the ones the next request's render carries.
    @Test func aSettledEntryIsAPrefixOfTheNextRequestsRender() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        let published = cache.publish(
            domain: domain,
            request: request(messages: messages),
            result: rawResult(kvBacked: kvBacked, reason: .endOfTurn))
        let publication = try #require(published)

        let completed = messages
            + [GFTokenizer.Message(role: .assistant, content: "answer")]
        let settled = try #require(KVRewrite.settledSequence(
            kvBackedTokenIDs: kvBacked,
            boundaryTokens: try tokenizer.settledBoundaryTokens(
                messages: completed, tools: []),
            liveRegionTokens: try tokenizer.settledLiveRegionTokens(
                messages: completed, tools: [])))
        #expect(settled != kvBacked)
        let settledEntry = cache.rewrite(
            entryID: publication.entry.id,
            kvBackedTokenIDs: settled)
        let rewritten = try #require(settledEntry)
        #expect(rewritten.kvPosition == rewritten.kvBackedTokenIDs.count)

        let next = request(messages: completed
            + [GFTokenizer.Message(role: .user, content: "second")])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(next.messages), addBOS: false)
        let match = cache.match(
            domain: domain,
            request: next,
            renderedPromptIDs: rendered)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected the settled entry to prefix the next render")
            return
        }
        #expect(cached == settled.count)
        #expect(effective == rendered)
        #expect(rendered.prefix(settled.count).elementsEqual(settled))
    }

    /// A rewrite publishes the cursor it reached, never the target it was
    /// computing toward.
    @Test func aRewrittenEntryAlwaysDescribesTheCursorItHolds() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        let published = cache.publish(
            domain: domain,
            request: request(messages: messages),
            result: rawResult(kvBacked: kvBacked, reason: .maxTokens))
        let publication = try #require(published)

        let suffix = tokenizer.encode(tokenizer.generationSuffix, addBOS: false)
        let preSuffix = prompt.count - suffix.count
        #expect(kvBacked[preSuffix..<prompt.count].elementsEqual(suffix))
        let truncatedEntry = cache.rewrite(
            entryID: publication.entry.id,
            kvBackedTokenIDs: Array(kvBacked.prefix(preSuffix)))
        let rewritten = try #require(truncatedEntry)

        #expect(rewritten.kvPosition == preSuffix)
        #expect(rewritten.kvBackedTokenIDs == Array(kvBacked.prefix(preSuffix)))
        let unknown = cache.rewrite(entryID: UUID(), kvBackedTokenIDs: [1])
        #expect(unknown == nil)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(kvBacked: [Int32], reason: StopReason) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: kvBacked.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: kvBacked.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [0])
    }
}

@Suite("Settle reconstruction mechanism")
struct KVReconstructionTests {
    private let shortPrefix = UUID()
    private let longPrefix = UUID()
    private let divergent = UUID()
    private let partingPoint = 80
    private let livePosition = 100
    private let targetCount = 120

    private enum Inventory: CaseIterable {
        case empty
        case nothingUsable
        case onePrefix
        case twoPrefixes
    }

    private func snapshots(_ inventory: Inventory) -> [KVReconstruction.Snapshot] {
        switch inventory {
        case .empty:
            return []
        case .nothingUsable:
            return [.init(entryID: divergent, position: 95, isPrefixOfTarget: false)]
        case .onePrefix:
            return [.init(entryID: shortPrefix, position: 40, isPrefixOfTarget: true)]
        case .twoPrefixes:
            // The longest sits last and a longer non-prefix sits between, so a
            // decision taking the first usable one — or the longest of all —
            // fails here rather than passing by accident.
            return [
                .init(entryID: shortPrefix, position: 40, isPrefixOfTarget: true),
                .init(entryID: divergent, position: 95, isPrefixOfTarget: false),
                .init(entryID: longPrefix, position: 70, isPrefixOfTarget: true),
            ]
        }
    }

    @Test func everyCapabilityAndInventoryReachesTheTargetOneWay() {
        // (the runner rewinds, the cursor already sits at the parting point,
        // what is snapshotted) -> the mechanism, spelled out rather than
        // recomputed from the rule under test.
        let expected: [(Bool, Bool, Inventory, KVReconstruction)] = [
            (true, true, .empty, .rewind(to: partingPoint)),
            (true, true, .nothingUsable, .rewind(to: partingPoint)),
            (true, true, .onePrefix, .rewind(to: partingPoint)),
            (true, true, .twoPrefixes, .rewind(to: partingPoint)),
            (true, false, .empty, .rewind(to: partingPoint)),
            (true, false, .nothingUsable, .rewind(to: partingPoint)),
            (true, false, .onePrefix, .rewind(to: partingPoint)),
            (true, false, .twoPrefixes, .rewind(to: partingPoint)),
            (false, true, .empty, .rewind(to: partingPoint)),
            (false, true, .nothingUsable, .rewind(to: partingPoint)),
            (false, true, .onePrefix, .rewind(to: partingPoint)),
            (false, true, .twoPrefixes, .rewind(to: partingPoint)),
            (false, false, .empty, .reset(reason: .noSnapshot)),
            (false, false, .nothingUsable, .reset(reason: .noPrefixSnapshot)),
            (false, false, .onePrefix, .restore(entryID: shortPrefix, position: 40)),
            (false, false, .twoPrefixes, .restore(entryID: longPrefix, position: 70)),
        ]
        #expect(expected.count == 2 * 2 * Inventory.allCases.count)
        for (supportsRewind, seated, inventory, mechanism) in expected {
            let plan = KVReconstruction.plan(
                supportsRewind: supportsRewind,
                livePosition: seated ? partingPoint : livePosition,
                rewindTo: partingPoint,
                snapshots: snapshots(inventory),
                targetCount: targetCount)
            #expect(plan == mechanism,
                    "rewind=\(supportsRewind) seated=\(seated) inventory=\(inventory)")
        }
    }

    @Test func aSnapshotReachingPastTheTargetIsNotAPrefixOfIt() {
        #expect(KVReconstruction.plan(
            supportsRewind: false,
            livePosition: livePosition,
            rewindTo: partingPoint,
            snapshots: [.init(entryID: longPrefix,
                              position: targetCount + 1,
                              isPrefixOfTarget: true)],
            targetCount: targetCount) == .reset(reason: .noPrefixSnapshot))
    }

    @Test func anEmptySnapshotSeatsNothingAndIsSkipped() {
        #expect(KVReconstruction.plan(
            supportsRewind: false,
            livePosition: livePosition,
            rewindTo: partingPoint,
            snapshots: [.init(entryID: longPrefix, position: 0, isPrefixOfTarget: true)],
            targetCount: targetCount) == .reset(reason: .noPrefixSnapshot))
    }

    /// A snapshot holding the target whole leaves an empty remainder, which is
    /// a rewrite that costs one restore and no forward pass.
    @Test func aSnapshotHoldingTheWholeTargetStillSeats() {
        #expect(KVReconstruction.plan(
            supportsRewind: false,
            livePosition: livePosition,
            rewindTo: partingPoint,
            snapshots: [.init(entryID: longPrefix,
                              position: targetCount,
                              isPrefixOfTarget: true)],
            targetCount: targetCount)
            == .restore(entryID: longPrefix, position: targetCount))
    }

    /// 6d, end to end over the pure halves: a `finish=length` turn on a runner
    /// that cannot rewind names a target — its own render below the generation
    /// suffix — and the same reconstruction that serves a settle reaches it. The
    /// previous turn's snapshot is a prefix of that target because the target is
    /// this request's render, which opens with the bytes that turn settled into.
    @Test func aDegenerateTurnOnANonRewindableRunnerNamesAReachableTarget() throws {
        let previousTurn: [Int32] = [1, 2, 3, 4, 5, 6]
        let render = previousTurn + [7, 8, 90, 91]
        let kv = render + [50, 51, 52, 53]
        let dropped = try #require(KVRewrite.droppedPrefixLength(
            kvBackedTokenIDs: kv,
            kvPosition: kv.count,
            promptTokenCount: render.count,
            generationSuffix: [90, 91]))
        let target = Array(kv.prefix(dropped))
        #expect(target == [1, 2, 3, 4, 5, 6, 7, 8])

        let previous = UUID()
        let plan = KVReconstruction.plan(
            supportsRewind: false,
            livePosition: kv.count,
            rewindTo: dropped,
            snapshots: [.init(
                entryID: previous,
                position: previousTurn.count,
                isPrefixOfTarget: target.prefix(previousTurn.count)
                    .elementsEqual(previousTurn))],
            targetCount: target.count)
        #expect(plan == .restore(entryID: previous, position: previousTurn.count))
        // The remainder is what the rewrite prefills after seating.
        #expect(Array(target[plan.seatedPosition...]) == [7, 8])

        // With nothing to seat on the drop still lands, by prefilling the target
        // whole — the work the next request would have paid for anyway.
        #expect(KVReconstruction.plan(
            supportsRewind: false,
            livePosition: kv.count,
            rewindTo: dropped,
            snapshots: [],
            targetCount: target.count) == .reset(reason: .noSnapshot))
    }

    @Test func theSeatedPositionIsWhereTheRemaindersPrefillStarts() {
        #expect(KVReconstruction.rewind(to: partingPoint).seatedPosition == partingPoint)
        #expect(KVReconstruction.restore(entryID: longPrefix, position: 70)
            .seatedPosition == 70)
        #expect(KVReconstruction.reset(reason: .noSnapshot).seatedPosition == 0)
        #expect(KVReconstruction.reset(reason: .noPrefixSnapshot).seatedPosition == 0)
    }
}

/// The property the restore mechanism rests on: a conversation's settled bytes
/// only ever grow at the end, so the snapshot `finishRewrite` captured at turn
/// N-1's settled position is a byte prefix of turn N's settled target and can
/// be seated on directly.
@Suite("Settled renders are append-only")
struct SettledAppendOnlyTests {
    private typealias Message = GFTokenizer.Message

    private static let weatherTool = GFTokenizer.FunctionDefinition(
        name: "get_weather",
        description: "Look up weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["city": .object(["type": .string("string")])]),
        ]))

    /// The sequence a settle would leave in the KV for a completed request:
    /// `KVRewrite.settledSequence`'s two halves, without the KV to validate the
    /// first against.
    private func settledTarget(_ tokenizer: GFTokenizer,
                               _ messages: [Message],
                               tools: [GFTokenizer.FunctionDefinition]) throws -> [Int32] {
        try tokenizer.settledBoundaryTokens(messages: messages, tools: tools)
            + tokenizer.settledLiveRegionTokens(messages: messages, tools: tools)
    }

    /// Every completed request in `conversation`, in order, settles into a
    /// sequence the next one opens with.
    private func expectAppendOnly(
        _ tokenizer: GFTokenizer,
        conversation: [Message],
        completedAt: [Int],
        tools: [GFTokenizer.FunctionDefinition]
    ) throws {
        var previous: [Int32]?
        for end in completedAt {
            let target = try settledTarget(tokenizer,
                                           Array(conversation[...end]),
                                           tools: tools)
            if let previous {
                #expect(previous.count < target.count,
                        "settled target did not grow at turn ending \(end)")
                #expect(target.prefix(previous.count).elementsEqual(previous),
                        "settled target diverged from its predecessor, turn \(end)")
            }
            previous = target
        }
        #expect(previous != nil)
    }

    @Test("ChatML: a plain multi-turn conversation")
    func chatMLPlainMultiTurn() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let conversation: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "Rain.", thinking: "Berlin is wet."),
            Message(role: .user, content: "And Rome?"),
            Message(role: .assistant, content: "Hot.", thinking: "Rome is hot."),
        ]
        try expectAppendOnly(tokenizer,
                             conversation: conversation,
                             completedAt: [2, 4, 6],
                             tools: [])
    }

    @Test("ChatML: a tool loop between two plain turns")
    func chatMLToolLoop() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let conversation: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ], thinking: "Berlin needs a lookup."),
            Message(role: .tool, content: "{\"temp\":12}", toolCallID: "call_1"),
            Message(role: .assistant, content: "12C.", thinking: "Cooler than Paris."),
            Message(role: .user, content: "And Rome?"),
            Message(role: .assistant, content: "Hot.", thinking: "Rome is hot."),
        ]
        try expectAppendOnly(tokenizer,
                             conversation: conversation,
                             completedAt: [2, 6, 8],
                             tools: [Self.weatherTool])
    }

    /// The hardest ChatML case: with no tools declared, the render routes on
    /// the message list, so the first turn goes through the hand-written
    /// template and the tool hop moves every later one onto the Jinja path. The
    /// chain survives only if the two agree byte-for-byte on the shared prefix.
    @Test("ChatML: a tool hop that moves the render onto the Jinja path")
    func chatMLRoutingFlipsMidConversation() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let conversation: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ], thinking: "Berlin needs a lookup."),
            Message(role: .tool, content: "{\"temp\":12}", toolCallID: "call_1"),
            Message(role: .assistant, content: "12C.", thinking: "Cooler than Paris."),
        ]
        #expect(!GFTokenizer.usesToolTemplate(messages: Array(conversation[...2]),
                                              tools: []))
        #expect(GFTokenizer.usesToolTemplate(messages: conversation, tools: []))
        try expectAppendOnly(tokenizer,
                             conversation: conversation,
                             completedAt: [2, 6],
                             tools: [])
    }

    @Test("Harmony: a plain multi-turn conversation")
    func harmonyPlainMultiTurn() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.harmonyFolder())
        let conversation: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "Rain.", thinking: "Berlin is wet."),
            Message(role: .user, content: "And Rome?"),
            Message(role: .assistant, content: "Hot.", thinking: "Rome is hot."),
        ]
        try expectAppendOnly(tokenizer,
                             conversation: conversation,
                             completedAt: [2, 4, 6],
                             tools: [])
    }

    @Test("Kimi: a plain multi-turn conversation")
    func kimiPlainMultiTurn() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.kimiFolder())
        let conversation: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "Rain.", thinking: "Berlin is wet."),
            Message(role: .user, content: "And Rome?"),
            Message(role: .assistant, content: "Hot.", thinking: "Rome is hot."),
        ]
        try expectAppendOnly(tokenizer,
                             conversation: conversation,
                             completedAt: [2, 4, 6],
                             tools: [])
    }
}

@Suite("Mid-rewrite arbitration")
struct RewriteArbitrationTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    /// Records what a stubbed rewrite was asked to do, and in which order.
    private final class RewriteStub: Sendable {
        private let events = Mutex<[String]>([])

        var order: [String] { events.withLock { $0 } }

        func cancel() { events.withLock { $0.append("cancel") } }

        func wait() async {
            events.withLock { $0.append("wait") }
            // A rewrite outlasts the request that arbitrates it; the sleep is
            // what makes "decided before the wait" an ordering claim rather
            // than a coincidence of scheduling.
            try? await Task.sleep(for: .milliseconds(20))
            events.withLock { $0.append("finished") }
        }
    }

    @Test func aTargetTheRenderOpensWithJoinsAndEverythingElseAborts() {
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: [1, 2, 3, 4, 5]) == .join)
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: [1, 2, 3]) == .join)
        // The rewrite is writing past where this render ends, so its bytes are
        // not this request's prefill however far they agree.
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: [1, 2]) == .abort)
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: [1, 9, 3, 4]) == .abort)
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: [9, 2, 3, 4]) == .abort)
        #expect(RewriteArbitration.decide(target: [1, 2, 3],
                                          render: []) == .abort)
        #expect(RewriteArbitration.decide(target: [],
                                          render: [1, 2, 3]) == .abort)
        #expect(RewriteArbitration.decide(target: [], render: []) == .abort)
    }

    @Test func aFollowUpRenderWaitsForTheRewriteItIsAlreadyBeingPrefilled() async {
        let stub = RewriteStub()
        let decision = await RewriteArbitration.arbitrate(
            target: [1, 2, 3],
            render: [1, 2, 3, 4],
            cancel: { stub.cancel() },
            wait: { await stub.wait() })
        #expect(decision == .join)
        #expect(stub.order == ["wait", "finished"])
    }

    /// The whole point of rendering before waiting: a request the rewrite is
    /// not prefilling for stops it *first*, so it never queues behind work its
    /// own arrival invalidated.
    @Test func aDivergentRenderCancelsBeforeItEverWaits() async {
        let stub = RewriteStub()
        let decision = await RewriteArbitration.arbitrate(
            target: [1, 2, 3],
            render: [1, 9, 9],
            cancel: { stub.cancel() },
            wait: { await stub.wait() })
        #expect(decision == .abort)
        #expect(stub.order == ["cancel", "wait", "finished"])
    }

    /// Against the real template rather than made-up ids: the sequence a
    /// completed turn is rewritten into is what its next turn renders, and what
    /// a regeneration of that turn does not.
    @Test func theRealSettledTargetJoinsTheNextTurnAndAbortsARegeneration() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let messages = [GFTokenizer.Message(role: .user, content: "first")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        let completed = messages
            + [GFTokenizer.Message(role: .assistant, content: "answer")]
        let target = try #require(KVRewrite.settledSequence(
            kvBackedTokenIDs: kvBacked,
            boundaryTokens: try tokenizer.settledBoundaryTokens(
                messages: completed, tools: []),
            liveRegionTokens: try tokenizer.settledLiveRegionTokens(
                messages: completed, tools: [])))

        let followUp = tokenizer.encode(
            try tokenizer.applyChatTemplate(
                completed + [GFTokenizer.Message(role: .user, content: "second")]),
            addBOS: false)
        #expect(RewriteArbitration.decide(target: target, render: followUp) == .join)

        // Asking the same turn again: the render stops where the assistant turn
        // began, so the rewrite is producing bytes past its end.
        let regenerated = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        #expect(RewriteArbitration.decide(target: target,
                                          render: regenerated) == .abort)
    }

    /// An edit aborts, and then salvages: the entry it aborted onto is the one
    /// published before the rewrite started, so the normal match path still
    /// finds the prefix the two renders share.
    @Test func anEditAbortsTheRewriteAndStillSalvagesItsSharedPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let messages = [GFTokenizer.Message(role: .user, content: "first question")]
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(messages), addBOS: false)
        let kvBacked = prompt + tokenizer.encode("answer", addBOS: false)
        let completed = messages
            + [GFTokenizer.Message(role: .assistant, content: "answer")]
        let target = try #require(KVRewrite.settledSequence(
            kvBackedTokenIDs: kvBacked,
            boundaryTokens: try tokenizer.settledBoundaryTokens(
                messages: completed, tools: []),
            liveRegionTokens: try tokenizer.settledLiveRegionTokens(
                messages: completed, tools: [])))

        let edited = request(messages: [
            GFTokenizer.Message(role: .user, content: "first question, restated"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(edited.messages), addBOS: false)
        #expect(RewriteArbitration.decide(target: target, render: rendered) == .abort)

        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: request(messages: messages),
            result: RawDecodeResult(
                prefillTokens: kvBacked.count,
                cachedPromptTokens: 0,
                computedPrefillTokens: kvBacked.count,
                prefillSeconds: 0,
                newTokens: 1,
                decodeSeconds: 0,
                reason: .endOfTurn,
                kvPosition: kvBacked.count,
                kvBackedTokenIDs: kvBacked,
                uncommittedBoundaryTokenIDs: [0]))
        let match = cache.match(
            domain: domain,
            request: edited,
            renderedPromptIDs: rendered)
        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected the pre-rewrite entry to salvage its prefix")
            return
        }
        #expect(cached > 0)
        #expect(cached < kvBacked.count)
        #expect(effective == rendered)
        let salvaged = try #require(cache.entries.first)
        #expect(salvaged.kvPosition == cached)
        #expect(salvaged.kvPosition == salvaged.kvBackedTokenIDs.count)
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }
}
