import Foundation
import NVMAI

/// One slot, many plans: serves every roster model through a single resident
/// session, swapping on demand. A swap is drain → unload returns → load, in
/// that strict order, so peak-of-swap is one resident model, not two.
///
/// Memory is genuinely returned on unload because every large allocation is
/// owned by a refcounted class reachable only from the session: the resident
/// weights are an `mmap` behind an `MTLBuffer` whose deallocator calls `munmap`
/// (`ResidentBuffer`), and the routed-expert slots are `posix_memalign` blocks
/// with a `free` deallocator (`PreadExpertStreamer`). Releasing the session
/// unwinds all of it.
///
/// `MetalContext` is the deliberate exception: built once, reused across every
/// swap. `MetalContext.deinit` documents that `MTLCommandQueue` has no
/// deinit-safe cleanup, and rebuilding it would recompile the shader library on
/// every model change.
public actor ModelRegistry {
    /// Builds a session. Injectable so residency and swap logic can be tested
    /// against stubs without a model on disk.
    public typealias Loader =
        @Sendable (ModelSessionPlan, MetalContext?) async throws -> any ServerInferenceBackend

    /// A servable model: its canonical API id, the plan that loads it, and the
    /// facts answerable before anything is resident.
    public struct Model: Sendable {
        public let id: String
        public let plan: ModelSessionPlan
        public let facts: ModelSessionFacts

        public var maximumContext: Int { plan.maxContext }

        public init(id: String, plan: ModelSessionPlan, facts: ModelSessionFacts) {
            self.id = id
            self.plan = plan
            self.facts = facts
        }
    }

    public struct Health: Sendable, Equatable {
        public let residentID: String?
        public let isLoading: Bool
        public let modelCount: Int

        public init(residentID: String?, isLoading: Bool, modelCount: Int) {
            self.residentID = residentID
            self.isLoading = isLoading
            self.modelCount = modelCount
        }
    }

    enum ReaperStep: Equatable {
        case sleep(Duration)
        case stop
    }

    public nonisolated let roster: ModelRoster
    /// Sorted by id; the order `GET /v1/models` reports.
    public nonisolated let models: [Model]
    private nonisolated let modelsByID: [String: Model]

    private let loader: Loader
    private let idleTimeout: Duration?

    private var resident: (id: String, backend: any ServerInferenceBackend)?
    private var loading: (id: String, task: Task<any ServerInferenceBackend, any Error>)?
    /// Requests currently inside `generate` on the resident model. Non-zero
    /// blocks any unload, manual or swap.
    private var inFlight = 0
    private var lastActivity = ContinuousClock.now
    private var reaper: Task<Void, Never>?
    private var drainWaiters: [DrainWaiter] = []
    /// Built on the first load, then reused for the process lifetime.
    private var metalContext: MetalContext?

    private struct DrainWaiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Never>
    }

    public init(models: [Model],
                roster: ModelRoster,
                idleTimeout: Duration?,
                loader: @escaping Loader = { plan, context in
                    try await plan.makeSession(reusingContext: context)
                }) {
        self.models = models.sorted { $0.id < $1.id }
        self.modelsByID = Dictionary(uniqueKeysWithValues: models.map { ($0.id, $0) })
        self.roster = roster
        self.idleTimeout = idleTimeout
        self.loader = loader
    }

    // MARK: - Resolution

    /// Resolves a request's `model` field — canonical id or bundle-name alias;
    /// `nil` resolves to the roster default. Answerable before anything is
    /// resident and with no actor hop.
    public nonisolated func model(for requested: String?) -> Model? {
        guard let requested else {
            return roster.defaultID.flatMap { modelsByID[$0] }
        }
        return roster.canonicalID(for: requested).flatMap { modelsByID[$0] }
    }

    public nonisolated var ids: [String] { models.map(\.id) }

    // MARK: - Generation

    public func generate(
        _ model: Model,
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let backend = try await acquire(model)
        defer { release() }
        return try await backend.generate(request, onEvent: onEvent)
    }

    // MARK: - Residency

    /// Returns the resident backend for `model`, loading or swapping first as
    /// needed. Every successful return has incremented `inFlight` and is
    /// balanced by exactly one `release()`.
    ///
    /// Requests for the slot's own model count themselves in flight *before*
    /// suspending on its load, so a just-loaded model cannot be drained away
    /// between load completion and first use. Requests for another model never
    /// touch the count — `inFlight` means "users of the slot", which is what a
    /// drain must wait out.
    ///
    /// The loop's awaits never spin: the load task finalizes registry state
    /// inside itself before its future completes, so an awaiter always resumes
    /// into advanced state.
    private func acquire(_ model: Model) async throws -> any ServerInferenceBackend {
        while true {
            if let resident, resident.id == model.id {
                inFlight += 1
                lastActivity = .now
                return resident.backend
            }
            if let loading {
                if loading.id == model.id {
                    inFlight += 1
                    do {
                        // Coalesce: share the in-flight load, and its failure.
                        let backend = try await loading.task.value
                        lastActivity = .now
                        return backend
                    } catch {
                        inFlight -= 1
                        noteIdle()
                        throw error
                    }
                }
                // A foreign load's failure is not this request's error.
                _ = try? await loading.task.value
                continue
            }
            if resident != nil {
                try await drainAndRelease()
                continue
            }
            return try await loadResident(model)
        }
    }

    private func release() {
        inFlight -= 1
        // Measured from when a request finished, so a long generation does not
        // count against the idle window.
        lastActivity = .now
        noteIdle()
    }

    private func loadResident(_ model: Model) async throws -> any ServerInferenceBackend {
        let context: MetalContext
        if let metalContext {
            context = metalContext
        } else {
            context = try MetalContext()
            metalContext = context
        }
        let plan = model.plan
        let loader = self.loader
        inFlight += 1
        // The task itself finalizes the registry's state before its future
        // completes, so every awaiter — this initiator included — resumes with
        // `resident`/`loading` already settled. On failure nothing becomes
        // resident and the next request retries cleanly.
        let task = Task {
            do {
                let backend = try await loader(plan, context)
                self.finishLoad(model, backend)
                return backend
            } catch {
                self.loading = nil
                throw error
            }
        }
        loading = (model.id, task)
        do {
            let backend = try await task.value
            lastActivity = .now
            return backend
        } catch {
            inFlight -= 1
            noteIdle()
            throw error
        }
    }

    private func finishLoad(_ model: Model, _ backend: any ServerInferenceBackend) {
        resident = (model.id, backend)
        loading = nil
        lastActivity = .now
        startReaper()
        ServerLog.residency("loaded \(model.id)")
    }

    /// Waits for in-flight requests to drain, then releases the resident
    /// session. Returns only after the release — the caller may map the next
    /// model immediately.
    private func drainAndRelease() async throws {
        reaper?.cancel()
        reaper = nil
        while resident != nil {
            if Task.isCancelled {
                // Giving up without unloading: the model stays resident, so
                // hand the idle timer back.
                startReaper()
                throw CancellationError()
            }
            if inFlight == 0 {
                releaseResident()
                return
            }
            await parkUntilDrained()
        }
    }

    private func releaseResident() {
        guard let resident else { return }
        self.resident = nil
        ServerLog.residency("unloaded \(resident.id)")
    }

    private func parkUntilDrained() async {
        let id = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                drainWaiters.append(DrainWaiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelDrainWaiter(id) }
        }
    }

    private func cancelDrainWaiter(_ id: UUID) {
        guard let index = drainWaiters.firstIndex(where: { $0.id == id }) else { return }
        drainWaiters.remove(at: index).continuation.resume()
    }

    /// Wakes drain waiters once the last in-flight request has finished.
    private func noteIdle() {
        guard inFlight == 0, !drainWaiters.isEmpty else { return }
        let waiters = drainWaiters
        drainWaiters.removeAll()
        for waiter in waiters {
            waiter.continuation.resume()
        }
    }

    // MARK: - Operator surface

    /// Releases the resident model on demand, waiting for in-flight requests
    /// to drain first. Returns the released model's id, or nil when nothing
    /// was resident.
    public func unload() async -> String? {
        reaper?.cancel()
        reaper = nil
        while let current = resident {
            if Task.isCancelled {
                startReaper()
                return nil
            }
            if inFlight == 0 {
                releaseResident()
                return current.id
            }
            await parkUntilDrained()
        }
        return nil
    }

    /// Loads `model` and returns once it is resident, without generating.
    public func preload(_ model: Model) async throws {
        _ = try await acquire(model)
        release()
    }

    /// Loads `model` without generating. Fire-and-forget: the load runs
    /// through the same slot serialization as any request, and a failure
    /// surfaces on the next generate exactly as a lazy-load failure does.
    public func startLoad(_ model: Model) {
        Task {
            do {
                try await self.preload(model)
            } catch {
                ServerLog.residency("load \(model.id) failed: \(error)")
            }
        }
    }

    public func health() -> Health {
        Health(residentID: resident?.id, isLoading: loading != nil, modelCount: models.count)
    }

    // MARK: - Idle policy

    func reaperStep(now: ContinuousClock.Instant = .now) -> ReaperStep {
        guard let idleTimeout, resident != nil else { return .stop }
        // A request in flight or a load in progress: the idle clock is not
        // running, so check again no sooner than a full timeout from now.
        guard inFlight == 0, loading == nil else { return .sleep(idleTimeout) }
        let idleFor = now - lastActivity
        guard idleFor >= idleTimeout else { return .sleep(idleTimeout - idleFor) }
        return .stop
    }

    private func startReaper() {
        guard idleTimeout != nil, reaper == nil else { return }
        reaper = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                switch await self.reaperStep() {
                case .sleep(let duration):
                    try? await Task.sleep(for: duration)
                case .stop:
                    await self.unloadIfIdle()
                    return
                }
            }
        }
    }

    /// Releases the session if it is still idle. Re-checks under actor
    /// isolation because the reaper's decision was made before its sleep.
    private func unloadIfIdle() {
        reaper = nil
        // A manual unload or a swap already released the session while the
        // reaper was waking; respawning here would chain reapers forever.
        guard resident != nil else { return }
        guard case .stop = reaperStep() else {
            startReaper()
            return
        }
        releaseResident()
    }

    // MARK: - Lifecycle

    public func shutdown() {
        reaper?.cancel()
        reaper = nil
        resident = nil
    }

    // MARK: - Construction from a roster

    /// One plan per roster entry from the (config-merged) arguments. Facts
    /// answer from each manifest alone, so a broken bundle fails here at
    /// launch, not on its first request.
    public static func models(for roster: ModelRoster,
                              arguments: ServerArguments) throws -> [Model] {
        try roster.entries.map { entry in
            let plan = ModelSessionPlan(
                modelDirectory: entry.directory,
                maxContext: arguments.maxContext,
                promptCacheMode: arguments.promptCacheMode,
                promptCacheMaximumEntries: arguments.promptCacheMaximumEntries,
                promptCacheMemoryLimitBytes: arguments.promptCacheMemoryMiB * 1_048_576,
                promptCacheDiskDirectory: arguments.promptCacheDiskDirectory.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                },
                promptCacheDiskLimitBytes: arguments.promptCacheDiskMiB * 1_048_576,
                prefillChunkTokens: arguments.prefillChunkTokens,
                kvCachePrecision: arguments.kvCachePrecision,
                ropeScalingMode: arguments.ropeScalingMode,
                thinkingMode: arguments.thinkingMode,
                expertCacheSlots: arguments.expertCacheSlots,
                expertCacheBudgetBytes: arguments.expertCacheBudgetBytes,
                mtpModelDirectory: arguments.mtpModel.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                },
                mtpMemoryMiB: arguments.mtpMemoryMiB)
            let facts = try plan.previewFacts(modelID: entry.id)
            return Model(id: entry.id, plan: plan, facts: facts)
        }
    }

    // MARK: - Test hooks

    var residentModelID: String? { resident?.id }
    var isLoaded: Bool { resident != nil }
    var inFlightCount: Int { inFlight }
    var hasReaper: Bool { reaper != nil }

    func backdateActivity(by duration: Duration) {
        lastActivity = .now - duration
    }

    func unloadNow() {
        resident = nil
    }

    func bumpInFlightForTesting(_ delta: Int) {
        inFlight += delta
    }
}
