import Foundation
import Testing

@testable import Shrike
@testable import ShrikeServerCore

/// Records which model each load was for and lets loads be failed per id or
/// held open, so swap ordering is observable without a model on disk.
private final class SwapRecorder: @unchecked Sendable {
    enum StubError: Error { case loadFailed }

    private let lock = NSLock()
    private var _loads: [String] = []
    private var _contexts: [MetalContext] = []
    private var _failing: Set<String> = []

    var loads: [String] { lock.withLock { _loads } }
    var contexts: [MetalContext] { lock.withLock { _contexts } }

    func failNextLoad(of id: String) { lock.withLock { _ = _failing.insert(id) } }

    func record(id: String, context: MetalContext?) {
        lock.withLock {
            _loads.append(id)
            if let context { _contexts.append(context) }
        }
    }

    func consumeFailure(id: String) -> Bool {
        lock.withLock { _failing.remove(id) != nil }
    }
}

/// Holds every loader inside the load until opened, so tests can observe the
/// registry's state mid-load.
private actor LoadGate {
    private var isOpen = true
    private var waiters: [CheckedContinuation<Void, Never>] = []

    var waiting: Int { waiters.count }

    func close() { isOpen = false }

    func open() {
        isOpen = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }

    func pass() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

private struct StubBackend: ServerInferenceBackend {
    let maximumContext = 4_096

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1,
                               completionTokens: 1,
                               totalTokens: 2,
                               cachedTokens: 0))
    }
}

/// A backend whose generation can be held open, so a drain can be observed
/// waiting for an in-flight request.
private actor GatedGenerationBackend: ServerInferenceBackend {
    private var continuation: CheckedContinuation<Void, Never>?

    var isWaiting: Bool { continuation != nil }

    func release() {
        continuation?.resume()
        continuation = nil
    }

    func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        await withCheckedContinuation { continuation = $0 }
        onEvent(.content("ok"))
        return ServerCompletion(
            content: "ok",
            toolCalls: [],
            finishReason: "stop",
            usage: OpenAIUsage(promptTokens: 1,
                               completionTokens: 1,
                               totalTokens: 2,
                               cachedTokens: 0))
    }
}

@Suite("Model registry")
struct ModelRegistryTests {
    private func plan(_ name: String) -> ModelSessionPlan {
        ModelSessionPlan(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/\(name).gturbo"),
            maxContext: 4_096,
            promptCacheMode: .multiPrefix,
            promptCacheMaximumEntries: 1,
            promptCacheMemoryLimitBytes: 1_048_576,
            promptCacheDiskDirectory: nil,
            promptCacheDiskLimitBytes: 1_048_576,
            prefillChunkTokens: nil,
            expertCacheSlots: nil,
            mtpModelDirectory: nil,
            mtpMemoryMiB: 0)
    }

    private func makeRegistry(
        _ names: [String],
        defaultModel: String? = nil,
        idleTimeout: Duration? = nil,
        recorder: SwapRecorder,
        gate: LoadGate? = nil,
        backend: @escaping @Sendable (String) -> any ServerInferenceBackend = { _ in StubBackend() }
    ) throws -> ModelRegistry {
        let roster = try ModelRoster.resolve(
            candidates: names.map {
                RosterCandidate(bundleName: $0,
                                directory: URL(fileURLWithPath: "/nonexistent/\($0).gturbo"),
                                manifestModelID: "vendor/\($0)",
                                family: .qwen36)
            },
            overrides: defaultModel.map { [.init(dir: $0, isDefault: true)] } ?? [])
        let models = names.map { name in
            ModelRegistry.Model(
                id: name,
                plan: plan(name),
                facts: ModelSessionFacts(modelID: name,
                                         prefillChunkTokens: 4_096,
                                         promptCacheMode: .multiPrefix))
        }
        return ModelRegistry(
            models: models,
            roster: roster,
            idleTimeout: idleTimeout,
            loader: { plan, context in
                let id = plan.modelDirectory.deletingPathExtension().lastPathComponent
                recorder.record(id: id, context: context)
                if let gate { await gate.pass() }
                if recorder.consumeFailure(id: id) { throw SwapRecorder.StubError.loadFailed }
                return backend(id)
            })
    }

    private func request() -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hi")],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4, temperature: 0),
            maximumCompletionTokens: 4)
    }

    private func expectEventually(
        _ condition: () async -> Bool,
        timeout: Duration = .seconds(5)
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("condition not reached within \(timeout)")
    }

    // MARK: - Resolution

    @Test func resolvesCanonicalIDsAliasesAndTheDefault() throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha", "beta"], defaultModel: "alpha", recorder: recorder)

        #expect(registry.ids == ["alpha", "beta"])
        #expect(registry.model(for: "beta")?.id == "beta")
        #expect(registry.model(for: nil)?.id == "alpha")
        #expect(registry.model(for: "gamma") == nil)
        #expect(registry.model(for: "beta")?.maximumContext == 4_096)
        #expect(registry.model(for: "beta")?.facts.modelID == "beta")
    }

    @Test func withoutADefaultAnOmittedModelResolvesToNothing() throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder)
        #expect(registry.model(for: nil) == nil)
    }

    // MARK: - Deferral and coalescing

    @Test func doesNotLoadUntilTheFirstRequest() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)

        #expect(recorder.loads.isEmpty)
        #expect(await registry.isLoaded == false)

        let model = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(model, request()) { _ in }
        #expect(recorder.loads == ["alpha"])
        #expect(await registry.residentModelID == "alpha")
    }

    @Test func concurrentFirstRequestsLoadExactlyOnce() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let model = try #require(registry.model(for: "alpha"))

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { _ = try? await registry.generate(model, self.request()) { _ in } }
            }
        }
        #expect(recorder.loads == ["alpha"])
    }

    @Test func aFailedLoadIsRetriedRatherThanCached() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let model = try #require(registry.model(for: "alpha"))
        recorder.failNextLoad(of: "alpha")

        await #expect(throws: (any Error).self) {
            _ = try await registry.generate(model, request()) { _ in }
        }
        #expect(await registry.isLoaded == false)
        // The in-flight count must not leak when the load throws, or a later
        // drain would wait forever.
        #expect(await registry.inFlightCount == 0)

        _ = try await registry.generate(model, request()) { _ in }
        #expect(recorder.loads == ["alpha", "alpha"])
        #expect(await registry.isLoaded)
    }

    // MARK: - Swapping

    @Test func swapReleasesTheOldModelBeforeLoadingTheNew() async throws {
        let recorder = SwapRecorder()
        let gate = LoadGate()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder, gate: gate)
        let alpha = try #require(registry.model(for: "alpha"))
        let beta = try #require(registry.model(for: "beta"))

        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(await registry.residentModelID == "alpha")

        await gate.close()
        let swap = Task { _ = try await registry.generate(beta, self.request()) { _ in } }
        try await expectEventually { await gate.waiting == 1 }

        // Beta's load has begun, so alpha must already be gone: the strict
        // unload-before-load ordering is what bounds peak-of-swap to one model.
        #expect(await registry.residentModelID == nil)
        #expect(recorder.loads == ["alpha", "beta"])

        await gate.open()
        _ = try await swap.value
        #expect(await registry.residentModelID == "beta")
    }

    @Test func swapWaitsForInFlightRequestsToDrain() async throws {
        let recorder = SwapRecorder()
        let gated = GatedGenerationBackend()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder,
                                    backend: { id in
                                        id == "alpha" ? gated as any ServerInferenceBackend : StubBackend()
                                    })
        let alpha = try #require(registry.model(for: "alpha"))
        let beta = try #require(registry.model(for: "beta"))

        let generation = Task { try? await registry.generate(alpha, self.request()) { _ in } }
        try await expectEventually { await gated.isWaiting }

        let swap = Task { _ = try await registry.generate(beta, self.request()) { _ in } }
        try await Task.sleep(for: .milliseconds(100))
        // The swap must not evict alpha while its generation is in flight.
        #expect(await registry.residentModelID == "alpha")
        #expect(recorder.loads == ["alpha"])

        await gated.release()
        _ = await generation.value
        _ = try await swap.value
        #expect(await registry.residentModelID == "beta")
        #expect(recorder.loads == ["alpha", "beta"])
    }

    @Test func concurrentRequestsForTheIncomingModelCoalesceOneLoad() async throws {
        let recorder = SwapRecorder()
        let gate = LoadGate()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder, gate: gate)
        let beta = try #require(registry.model(for: "beta"))

        await gate.close()
        let first = Task { _ = try await registry.generate(beta, self.request()) { _ in } }
        try await expectEventually { await gate.waiting == 1 }
        let second = Task { _ = try await registry.generate(beta, self.request()) { _ in } }
        try await Task.sleep(for: .milliseconds(50))

        await gate.open()
        _ = try await first.value
        _ = try await second.value
        #expect(recorder.loads == ["beta"])
    }

    @Test func aForeignLoadFailureIsNotThisRequestsError() async throws {
        let recorder = SwapRecorder()
        let gate = LoadGate()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder, gate: gate)
        let alpha = try #require(registry.model(for: "alpha"))
        let beta = try #require(registry.model(for: "beta"))

        recorder.failNextLoad(of: "beta")
        await gate.close()
        let failing = Task { _ = try await registry.generate(beta, self.request()) { _ in } }
        try await expectEventually { await gate.waiting == 1 }
        let bystander = Task { _ = try await registry.generate(alpha, self.request()) { _ in } }
        try await Task.sleep(for: .milliseconds(50))
        await gate.open()

        await #expect(throws: (any Error).self) { _ = try await failing.value }
        _ = try await bystander.value
        #expect(await registry.residentModelID == "alpha")
    }

    @Test func reloadAcrossASwapReusesTheSameMetalContext() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        let beta = try #require(registry.model(for: "beta"))

        _ = try await registry.generate(alpha, request()) { _ in }
        _ = try await registry.generate(beta, request()) { _ in }

        #expect(recorder.contexts.count == 2)
        // Identity, not equality: one MTLCommandQueue and one compiled shader
        // library must survive the swap.
        #expect(recorder.contexts[0] === recorder.contexts[1])
    }

    // MARK: - Manual unload and preload

    @Test func unloadReportsTheReleasedModel() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }

        #expect(await registry.unload() == "alpha")
        #expect(await registry.isLoaded == false)
        #expect(await registry.unload() == nil)
    }

    @Test func unloadWaitsForInFlightRequestsToDrain() async throws {
        let recorder = SwapRecorder()
        let gated = GatedGenerationBackend()
        let registry = try makeRegistry(["alpha"], recorder: recorder, backend: { _ in gated })
        let alpha = try #require(registry.model(for: "alpha"))

        let generation = Task { try? await registry.generate(alpha, self.request()) { _ in } }
        try await expectEventually { await gated.isWaiting }
        #expect(await registry.inFlightCount == 1)

        let unload = Task { await registry.unload() }
        try await Task.sleep(for: .milliseconds(100))
        #expect(await registry.isLoaded)

        await gated.release()
        _ = await generation.value
        #expect(await unload.value == "alpha")
        #expect(await registry.isLoaded == false)
    }

    @Test func aCancelledUnloadHandsBackTheIdleTimer() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .seconds(600), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }
        await registry.bumpInFlightForTesting(1)

        let attempt = Task { await registry.unload() }
        try await Task.sleep(for: .milliseconds(50))
        attempt.cancel()
        let released = await attempt.value

        #expect(released == nil)
        #expect(await registry.isLoaded)
        #expect(await registry.hasReaper)

        await registry.bumpInFlightForTesting(-1)
    }

    @Test func startLoadLoadsWithoutGenerating() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))

        await registry.startLoad(alpha)
        try await expectEventually { await registry.residentModelID == "alpha" }
        #expect(recorder.loads == ["alpha"])
        #expect(await registry.inFlightCount == 0)
    }

    @Test func aFailedStartLoadSurfacesOnTheNextGenerate() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        recorder.failNextLoad(of: "alpha")

        await registry.startLoad(alpha)
        try await expectEventually { recorder.loads.count == 1 }
        #expect(await registry.isLoaded == false)

        // The next request retries the load cleanly and succeeds.
        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(recorder.loads == ["alpha", "alpha"])
        #expect(await registry.residentModelID == "alpha")
    }

    // MARK: - Health

    @Test func healthReportsResidencyAndLoading() async throws {
        let recorder = SwapRecorder()
        let gate = LoadGate()
        let registry = try makeRegistry(["alpha", "beta"], recorder: recorder, gate: gate)
        let alpha = try #require(registry.model(for: "alpha"))

        #expect(await registry.health() == .init(residentID: nil, isLoading: false, modelCount: 2))

        await gate.close()
        let generation = Task { _ = try await registry.generate(alpha, self.request()) { _ in } }
        try await expectEventually { await gate.waiting == 1 }
        #expect(await registry.health() == .init(residentID: nil, isLoading: true, modelCount: 2))

        await gate.open()
        _ = try await generation.value
        #expect(await registry.health() == .init(residentID: "alpha", isLoading: false, modelCount: 2))
    }

    // MARK: - Idle policy

    @Test func reaperWaitsWhileTheModelIsBusy() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .seconds(30), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }

        await registry.backdateActivity(by: .seconds(120))
        await registry.withInFlight {
            #expect(await registry.reaperStep() == .sleep(.seconds(30)))
        }
        #expect(await registry.reaperStep() == .stop)
    }

    @Test func reaperSleepsOnlyTheRemainingIdleTime() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .seconds(60), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }

        await registry.backdateActivity(by: .seconds(45))
        guard case .sleep(let remaining) = await registry.reaperStep() else {
            Issue.record("expected a sleep while still inside the idle window")
            return
        }
        #expect(remaining < .seconds(16))
        #expect(remaining > .seconds(14))
    }

    @Test func noIdleTimeoutMeansNoReaper() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }

        #expect(await registry.hasReaper == false)
        await registry.backdateActivity(by: .seconds(86_400))
        #expect(await registry.reaperStep() == .stop)
        #expect(await registry.isLoaded)
    }

    @Test func unloadsAfterIdleThenReloadsOnDemand() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .milliseconds(50), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(await registry.isLoaded)

        try await expectEventually { await registry.isLoaded == false }
        #expect(recorder.loads == ["alpha"])

        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(recorder.loads == ["alpha", "alpha"])
    }

    @Test func unloadCancelsTheReaper() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .seconds(600), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(await registry.hasReaper)

        #expect(await registry.unload() == "alpha")
        #expect(await registry.hasReaper == false)

        try await Task.sleep(for: .milliseconds(100))
        #expect(await registry.hasReaper == false)
        #expect(await registry.isLoaded == false)
    }

    // MARK: - Lifecycle

    @Test func shutdownCancelsTheReaperAndReleasesTheSession() async throws {
        let recorder = SwapRecorder()
        let registry = try makeRegistry(["alpha"], idleTimeout: .seconds(600), recorder: recorder)
        let alpha = try #require(registry.model(for: "alpha"))
        _ = try await registry.generate(alpha, request()) { _ in }
        #expect(await registry.hasReaper)

        await registry.shutdown()
        #expect(await registry.hasReaper == false)
        #expect(await registry.isLoaded == false)
    }
}

private extension ModelRegistry {
    /// Runs `body` with the in-flight count raised, mirroring a request that is
    /// mid-generation while the reaper wakes.
    func withInFlight(_ body: () async -> Void) async {
        bumpInFlightForTesting(1)
        await body()
        bumpInFlightForTesting(-1)
    }
}

private let fixtureManifest = """
{
  "magic": "GTURBO", "versionMajor": 1, "versionMinor": 1, "flags": {},
  "modelID": "vendor/good-4bit",
  "arch": {
    "hiddenSize": 8, "ffnIntermediate": 8, "moeIntermediateSize": 8,
    "numHeads": 2, "numKVHeads": 1, "numFullKVHeads": 1,
    "headDim": 4, "fullHeadDim": 4, "vocabSize": 16,
    "slidingWindow": 0, "finalLogitSoftcap": 0,
    "ropeTheta": 10000, "fullRopeTheta": 10000, "partialRotaryFactor": 1,
    "numLayers": 1, "numExperts": 2, "topKExperts": 1,
    "tieWordEmbeddings": false, "attentionKEqV": false,
    "hiddenActivation": "silu", "fullAttentionLayerMask": [0],
    "family": "qwen36"
  },
  "files": {},
  "expertsPerLayer": 2, "numLayers": 1, "expertStride": 16384
}
"""

@Suite struct ModelRegistryConstructionTests {
    @Test func modelsFromARosterCarryMergedArgumentsAndRosterIDs() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("registry-build-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appendingPathComponent("good.gturbo")
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data(fixtureManifest.utf8).write(to: bundle.appendingPathComponent("manifest.json"))

        let scan = try ModelRoster.scanBundles(in: root)
        let roster = try ModelRoster.resolve(
            candidates: scan.candidates,
            overrides: [.init(dir: "good.gturbo", id: "nice-name")])
        let arguments = try ServerArguments.parse(["--max-context", "32768"], environment: [:])
        let models = try ModelRegistry.models(for: roster, arguments: arguments)

        #expect(models.map(\.id) == ["nice-name"])
        #expect(models[0].facts.modelID == "nice-name")
        #expect(models[0].plan.maxContext == 32_768)
        #expect(models[0].plan.modelDirectory.resolvingSymlinksInPath().path
            == bundle.resolvingSymlinksInPath().path)
    }
}
