import Foundation
import Testing

@testable import NVMAI
@testable import NVMAIServerCore

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

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func harmonyEntriesNeverBridgeTextContinuations() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.harmonyFolder())
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
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: generated.last ?? 0,
                reason: .maxTokens))

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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        // The ChatML text bridge is dialect-gated, so a Harmony entry has no
        // structural path: it salvages the shared prefix, never splices a bridge.
        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a common-prefix salvage")
            return
        }
        #expect(effective == rendered)
        #expect(cached > 0)
        #expect(cached < kvBacked.count)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == Array(kvBacked.prefix(cached)))
        #expect(entry.kvPosition == cached)
    }

    @Test func kimiContinuationsHitTheRenderedPrefixWithoutABridge() async throws {
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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        // Kimi's template is append-only (no <think> stripping, no dropped
        // analysis), so the re-render extends the cached KV byte-for-byte and
        // the dialect-agnostic S12 prefix path hits — the ChatML-shaped text
        // bridge stays unused.
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
                content: "answer",
                calls: [],
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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        // A changed first turn has no structural continuation, so all the entry
        // can keep is the template preamble both renders share.
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
            content: "answer ",
            calls: [],
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
            content: "short",
            calls: [],
            result: rawResult(
                prompt: [1, 2],
                kvBacked: [1, 2],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let short = try #require(shortPublication)
        let longPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "long",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let long = try #require(longPublication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4],
            tokenizer: tokenizer)
        #expect(match == .hit(
            entryID: long.entry.id,
            effectivePromptIDs: [1, 2, 3, 4],
            cachedPromptTokens: 3))

        let newestPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "newest",
            calls: [],
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
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)
        #expect(match == .hit(
            entryID: entry.entry.id,
            effectivePromptIDs: prompt,
            cachedPromptTokens: prompt.count))
    }

    /// Regression: the cache keys on the post-strip view of a request, so a
    /// "<model>-fast" continuation re-renders its tail through CLIStrip too.
    /// Keying on the raw request instead produced a bridge that still carried
    /// the CLI's <system-reminder> scaffolding — an unstripped tail spliced
    /// onto a stripped prefix, so the cached turn silently lost the alias's
    /// strip and stopped reproducing a fresh prefill of the same request.
    @Test func strippedContinuationBridgeDropsReminderScaffolding() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let bloat = GFTokenizer.Message(role: .system, content: "you are an agent")
        let rawFirst = GFTokenizer.Message(
            role: .user,
            content: "first<system-reminder>\ncwd is /tmp\n</system-reminder>")
        let rawSecond = GFTokenizer.Message(
            role: .user,
            content: "second<system-reminder>\nfile changed\n</system-reminder>")

        // Turn 1, exactly as ServerInference composes it: strip, then key the
        // cache on the filtered view that was actually encoded.
        let firstStrip = CLIStrip.filter(messages: [bloat, rawFirst], tools: [])
        let initial = request(messages: [bloat, rawFirst])
            .replacingMessages(firstStrip.messages, tools: firstStrip.tools)
        #expect(initial.messages.map(\.content) == ["first"])

        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let kvBacked = initialPrompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // Turn 2 arrives with the bloat and both reminder blocks intact.
        let rawContinuation = [
            bloat,
            rawFirst,
            GFTokenizer.Message(role: .assistant, content: "answer"),
            rawSecond,
        ]
        let secondStrip = CLIStrip.filter(messages: rawContinuation, tools: [])
        let continuation = request(messages: rawContinuation)
            .replacingMessages(secondStrip.messages, tools: secondStrip.tools)
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit on the stripped view")
            return
        }
        // The bridge is the *stripped* user turn; the reminder block never
        // reaches the model, and the raw turn would have produced a longer one.
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked
            + tokenizer.encodeTextContinuation(userContent: "second"))
        #expect(effective != kvBacked
            + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))

        // And the shape of the defect this guards: an entry keyed on the raw
        // messages still describes a KV range prefilled from the *stripped*
        // ones, so its continuation bridge carries the reminder block — an
        // unstripped tail on a stripped prefix.
        var rawKeyed = ServerPromptCache()
        rawKeyed.publish(
            domain: domain,
            request: request(messages: [bloat, rawFirst]),
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let rawMatch = rawKeyed.match(
            domain: domain,
            request: request(messages: rawContinuation),
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)
        guard case .hit(_, let rawEffective, _) = rawMatch else {
            Issue.record("expected the raw-keyed cache to still hit")
            return
        }
        #expect(rawEffective == kvBacked
            + tokenizer.encodeTextContinuation(userContent: rawSecond.content ?? ""))
        #expect(rawEffective != effective)
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
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let publication = try #require(published)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4, 5],
            tokenizer: tokenizer)

        #expect(match == .hit(
            entryID: publication.entry.id,
            effectivePromptIDs: [1, 2, 3, 4, 5],
            cachedPromptTokens: 3))
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2, 3])
        #expect(entry.kvPosition == 3)
    }

    @Test func thinkingRetainedContinuationTakesTheStructuralPathUntruncated() async throws {
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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected the structural text-continuation hit")
            return
        }
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked
            + tokenizer.encodeTextContinuation(userContent: "second"))
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == kvBacked)
        #expect(entry.kvPosition == kvBacked.count)
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
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let publication = try #require(published)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 9, 9],
            tokenizer: tokenizer)

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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: [1, 2, 3],
            tokenizer: tokenizer)

        #expect(match == .miss)
        let entry = try #require(cache.entries.last)
        #expect(entry.kvBackedTokenIDs == [1, 2, 3, 4, 5])
        #expect(entry.assistantTurn != nil)
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
            content: "answer",
            calls: [],
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
            renderedPromptIDs: [1, 2, 9, 9],
            tokenizer: tokenizer) == .miss)
        let kept = try #require(cache.entries.last)
        #expect(kept.kvBackedTokenIDs == [1, 2, 3, 4, 5])
        #expect(kept.kvPosition == 5)
        #expect(kept.assistantTurn != nil)

        // Only the salvage is gated: a full-prefix hit needs no rewind.
        #expect(cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4, 5, 6],
            tokenizer: tokenizer) == .hit(
                entryID: kept.id,
                effectivePromptIDs: [1, 2, 3, 4, 5, 6],
                cachedPromptTokens: 5))
    }

    @Test func aTruncatedEntryCanNoLongerServeAStructuralBridge() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let kvBacked = initialPrompt + tokenizer.encode("answer", addBOS: false)
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        // A one-message request cannot continue structurally, so this salvages
        // everything but the last token and truncates the entry there.
        _ = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: Array(kvBacked.dropLast()) + [Int32.max],
            tokenizer: tokenizer)
        let truncated = try #require(cache.entries.last)
        #expect(truncated.kvBackedTokenIDs == Array(kvBacked.dropLast()))
        #expect(truncated.inputMessages.isEmpty)
        #expect(truncated.assistantTurn == nil)

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
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected a common-prefix salvage")
            return
        }
        // Short of the truncated length, so the structural path was reached and
        // refused rather than splicing a bridge onto bytes that are gone.
        #expect(cached < truncated.kvPosition)
        #expect(effective == rendered)
        #expect(effective != truncated.kvBackedTokenIDs
            + tokenizer.encodeTextContinuation(userContent: "second"))
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
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3, 4, 5],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [7, 8],
            tokenizer: tokenizer)

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
        #expect(replaced.stripCLIPrompt == original.stripCLIPrompt)
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

    @Test func theTriggerIsTheSpecsTableOverEveryStopReason() throws {
        let expected: [StopReason: (closed: KVRewrite, open: KVRewrite)] = [
            .endOfTurn: (.settleLiveRegion, .dropEmission),
            .eos: (.settleLiveRegion, .dropEmission),
            .toolCalls: (.none, .none),
            .maxTokens: (.dropEmission, .dropEmission),
            .stopString: (.dropEmission, .dropEmission),
            .external: (.dropEmission, .dropEmission),
        ]
        #expect(expected.count == Self.everyStopReason.count)
        for reason in Self.everyStopReason {
            let table = try #require(expected[reason])
            #expect(KVRewrite.forCompletion(
                reason: reason,
                thoughtChannelClosed: true,
                emittedToolCalls: false,
                supportsRewind: true) == table.closed)
            #expect(KVRewrite.forCompletion(
                reason: reason,
                thoughtChannelClosed: false,
                emittedToolCalls: false,
                supportsRewind: true) == table.open)
        }
    }

    @Test func aTurnCarryingToolCallsStaysLiveWhateverStopEndedIt() {
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                #expect(KVRewrite.forCompletion(
                    reason: reason,
                    thoughtChannelClosed: closed,
                    emittedToolCalls: true,
                    supportsRewind: true) == KVRewrite.none)
            }
        }
    }

    @Test func aRunnerThatCannotRewindDeclinesEveryRewrite() {
        for reason in Self.everyStopReason {
            for closed in [true, false] {
                #expect(KVRewrite.forCompletion(
                    reason: reason,
                    thoughtChannelClosed: closed,
                    emittedToolCalls: false,
                    supportsRewind: false) == KVRewrite.none)
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
            content: "answer",
            calls: [],
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
        #expect(entry.assistantTurn != nil)
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
            content: "answer",
            calls: [],
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
            content: "answer",
            calls: [],
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
        #expect(rewritten.inputMessages.isEmpty)
        #expect(rewritten.assistantTurn == nil)

        let next = request(messages: completed
            + [GFTokenizer.Message(role: .user, content: "second")])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(next.messages), addBOS: false)
        let match = cache.match(
            domain: domain,
            request: next,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

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
            content: "answer",
            calls: [],
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
        #expect(rewritten.inputMessages.isEmpty)
        #expect(rewritten.assistantTurn == nil)
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
