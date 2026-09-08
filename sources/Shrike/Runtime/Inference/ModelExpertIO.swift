import Foundation
import Metal

public struct RoutedExpertFetchPlan: Sendable {
    public let layer: Int
    public let cachePlan: ExpertCachePlan

    public var experts: [Int] { cachePlan.experts }
    public var misses: [Int] { cachePlan.misses }
    public var hits: Int { cachePlan.hits }
    public var assignedSlots: [Int] { cachePlan.assignedSlots }
    public var adopted: [Int] { cachePlan.adopted }
    public var freedCells: [Int: Int] { cachePlan.freedCells }

    public init(layer: Int, cachePlan: ExpertCachePlan) {
        self.layer = layer
        self.cachePlan = cachePlan
    }
}

/// unchecked-invariant: the wrapped cache lease is thread-safe and immutable;
/// this forwarding owner adds no mutable state.
final class RoutedExpertLease: @unchecked Sendable {
    private let cacheLease: ExpertCacheLease

    init(cacheLease: ExpertCacheLease) {
        self.cacheLease = cacheLease
    }

    func release() { cacheLease.release() }
}

/// Model-level storage ticket. Expert views are derivable from the reserved
/// plan immediately; callers must await `completion()` before consuming miss
/// slots unless a GPU event dependency enforces the same ordering.
/// unchecked-invariant: immutable model, plan, and thread-safe storage ticket.
public final class RoutedExpertLoadOperation: @unchecked Sendable {
    public let plan: RoutedExpertFetchPlan
    public let storage: ExpertLoadOperation
    private let model: Model

    init(model: Model,
         plan: RoutedExpertFetchPlan,
         storage: ExpertLoadOperation) {
        self.model = model
        self.plan = plan
        self.storage = storage
    }

    public var state: ExpertLoadOperationState { storage.state }

    public func wait() throws -> [TensorView] {
        try storage.wait()
        return try model.routedExpertBuffers(for: plan)
    }

    public func completion() async throws -> [TensorView] {
        try await storage.completion()
        return try model.routedExpertBuffers(for: plan)
    }
}

extension Model {
    public func routedExpertStatistics() -> ExpertStreamingStatistics {
        let streamers = streamersQueue.sync { streamersBox.streamers.compactMap { $0 } }
        return streamers.reduce(.zero) { $0.adding($1.statistics()) }
    }

    public func routedExpertOffsets(layer: Int) throws -> MoEExpertOffsets {
        let expert = try packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        if config.expertsHaveAdditiveBiases {
            for role in ["gate_bias", "up_bias", "down_bias"]
            where expert.subTensors[role] == nil {
                throw ModelError.indexCorrupt(
                    detail: "layer \(layer) expert blob is missing \(role)")
            }
        }
        return MoEExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            gateBOff: offset("gate_biases"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            upBOff: offset("up_biases"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"),
            downBOff: offset("down_biases"),
            gateABOff: offset("gate_bias"),
            upABOff: offset("up_bias"),
            downABOff: offset("down_bias"))
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = [],
                                  protectedExperts: [Bool]? = nil,
                                  gpuMissedExperts: Set<Int>? = nil,
                                  leasedLandings: Set<Int> = []) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer, cachePlan: try streamer.planExpertsCached(
                experts: experts, avoidingSlots: validSlots, protectedExperts: protectedExperts,
                gpuMissedExperts: gpuMissedExperts, leasedLandings: leasedLandings))
    }

    /// The ring's cell count has to be known before the arena exists, which
    /// is the first layer's opening; a mismatch after that is a programming
    /// error, not a runtime condition.
    public func configurePrefetchCells(_ count: Int) throws {
        try streamersQueue.sync {
            if streamersBox.arena != nil, streamersBox.prefetchCellCount != count {
                throw ModelError.internalInconsistency(
                    detail: "the expert cell arena was allocated before the prefetch ring was sized")
            }
            streamersBox.prefetchCellCount = count
        }
    }

    /// The ring's cells in the arena.
    public func prefetchCells() throws -> [Int] {
        try ensureLayerOpened(firstRoutedLayer())
        return streamersQueue.sync { streamersBox.prefetchCells }
    }

    private func firstRoutedLayer() -> Int {
        packedExpertsLayout.layers.firstIndex { !$0.experts.isEmpty } ?? 0
    }

    func dropRoutedExpertLanding(layer: Int, expert: Int, cell: Int) {
        guard (try? ensureLayerOpened(layer)) != nil else { return }
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        streamer.dropLanding(expert: expert, cell: cell)
    }

    public func planRoutedExpertsIfPossible(layer: Int,
                                            experts: [Int],
                                            avoidingSlots: Set<Int> = [],
                                            protectedExperts: [Bool]? = nil) throws
        -> RoutedExpertFetchPlan? {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        guard let cachePlan = streamer.planExpertsCachedIfPossible(
            experts: experts,
            avoidingSlots: validSlots,
            protectedExperts: protectedExperts)
        else {
            return nil
        }
        return RoutedExpertFetchPlan(layer: layer, cachePlan: cachePlan)
    }

    public func abandonRoutedExpertPlan(_ plan: RoutedExpertFetchPlan) throws {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        streamer.abandonExpertCachePlan(plan.cachePlan)
    }

    /// Cache slot count is a per-model streaming property (the same for every
    /// layer), so it deliberately takes no layer argument.
    public func routedExpertCacheSlotCount() -> Int? {
        guard case .pread(let slotCount) = streamingMode else { return nil }
        return slotCount
    }

    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return Self.makeExpertViews(
            streamer.expertCachePlanBuffers(plan.cachePlan),
            layer: plan.layer,
            experts: plan.experts)
    }

    public func routedExpertResidency(layer: Int) throws -> ExpertResidencyResources {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.expertResidencyResources()
    }

    /// Returns only fully valid routed experts immediately before cache
    /// planning; never changes cache state. Used by the optional v4.3 trace
    /// probe and predictive prefetch, and by `.resident`'s hot-path sweep
    /// (v13 T5 step 2).
    public func routedExpertResidentIDs(layer: Int) throws -> [Int] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return streamer.residentExperts()
    }

    public func beginRoutedExpertPrefetch(layer: Int,
                                           experts: [Int],
                                           cells: [Int]) throws
        -> ExpertLoadOperation {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try streamer.beginPrefetch(experts: experts, cells: cells)
    }

    func pinRoutedExperts(for plan: RoutedExpertFetchPlan) throws -> RoutedExpertLease {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return RoutedExpertLease(cacheLease: try streamer.pin(plan.cachePlan))
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try await beginFetchRoutedExperts(plan: plan).completion()
    }

    public func beginFetchRoutedExperts(
        plan: RoutedExpertFetchPlan,
        eventDriven: Bool = false
    ) throws -> RoutedExpertLoadOperation {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return RoutedExpertLoadOperation(
            model: self,
            plan: plan,
            storage: try streamer.beginExpertCachePlan(
                plan.cachePlan,
                eventDriven: eventDriven))
    }

    public func fetchRoutedExperts(layer: Int, experts: [Int]) async throws -> [TensorView] {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.loadExpertsCached(experts: experts)
                    continuation.resume(returning: Self.makeExpertViews(
                        buffers,
                        layer: layer,
                        experts: experts))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func makeExpertViews(
        _ buffers: [(buffer: MTLBuffer, offset: UInt64, size: UInt64)],
        layer: Int,
        experts: [Int]
    ) -> [TensorView] {
        buffers.enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (UInt32(layer), UInt32(experts[index]), 0, 0),
                dtype: 0)
        }
    }
}
