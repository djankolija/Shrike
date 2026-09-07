import Foundation
import Testing
@testable import Shrike

@Suite struct PrefillRoutedTileSchedulerTests {
    @Test func halfCacheIssuesFirstTileWithoutPendingWork() {
        let decision = PrefillRoutedTileScheduler().decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: false,
                                            pendingAssignedSlots: [],
                                            avoidingSlotPlanAvailable: false))

        #expect(decision == .issueWithoutPending)
    }

    @Test func halfCachePrefetchesWhenAvoidingSlotPlanExists() {
        let decision = PrefillRoutedTileScheduler().decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                            pendingAssignedSlots: [3, 7, 8],
                                            avoidingSlotPlanAvailable: true))

        #expect(decision == .prefetchNext(avoidingSlots: [3, 7, 8]))
    }

    @Test func halfCacheDrainsWhenPendingTileHasNoAssignedSlots() {
        let decision = PrefillRoutedTileScheduler().decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                            pendingAssignedSlots: [],
                                            avoidingSlotPlanAvailable: true))

        #expect(decision == .drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots))
    }

    @Test func halfCacheDrainsWhenAvoidingSlotPlanIsUnavailable() {
        let decision = PrefillRoutedTileScheduler().decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                            pendingAssignedSlots: [1, 2],
                                            avoidingSlotPlanAvailable: false))

        #expect(decision == .drainBeforeIssue(reason: .avoidingSlotPlanUnavailable))
    }

    @Test func schedulerAllowsSecondLookaheadWhenDepthBudgetAllowsIt() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 4))

        let decision = scheduler.decide(PrefillRoutedTileSchedulerInput(
            hasPendingTile: true,
            pendingDepth: 2,
            pendingAssignedSlots: [1, 2, 3, 4, 5, 6, 7, 8],
            avoidingSlotPlanAvailable: true))

        #expect(decision == .prefetchNext(avoidingSlots: [1, 2, 3, 4, 5, 6, 7, 8]))
    }

    @Test func schedulerDrainsAtConfiguredPendingDepth() {
        let decision = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 4)).decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                            pendingDepth: 3,
                                            pendingAssignedSlots: [1, 2],
                                            avoidingSlotPlanAvailable: true))

        #expect(decision == .drainBeforeIssue(reason: .maxPendingDepthReached))
    }

    @Test func schedulerConfigValidatesSlotBudget() {
        let defaultConfig = PrefillRoutedTileSchedulerConfig()
        let depthTwoFourExperts = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 4)
        let depthTwoEightExperts = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 8)

        #expect(defaultConfig.fitsSlotBudget(slotCount: 16))
        #expect(depthTwoFourExperts.fitsSlotBudget(slotCount: 16))
        #expect(!depthTwoEightExperts.fitsSlotBudget(slotCount: 16))
    }

    @Test func schedulerConfigShrinksTilesForEightSlotStreamingCache() {
        let fitted = PrefillRoutedTileSchedulerConfig().fitting(slotCount: 8)

        #expect(fitted == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 1,
            tileExperts: 4))
        #expect(fitted?.fitsSlotBudget(slotCount: 8) == true)
        #expect(PrefillRoutedTileSchedulerConfig().fitting(slotCount: 1) == nil)
    }

    @Test func halfCachePreservesPendingSlotAvoidanceOrder() {
        let pendingSlots = [9, 1, 9, 3]
        let decision = PrefillRoutedTileScheduler().decide(
            PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                            pendingAssignedSlots: pendingSlots,
                                            avoidingSlotPlanAvailable: true))

        #expect(decision == .prefetchNext(avoidingSlots: pendingSlots))
    }

    @Test func batchOfOneReproducesTheSingleTileDecisions() {
        let batchOfOne = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 1))
        let unbatched = PrefillRoutedTileScheduler()
        let cases: [(PrefillRoutedTileSchedulerInput, PrefillRoutedTileSchedulerDecision)] = [
            (PrefillRoutedTileSchedulerInput(hasPendingTile: false,
                                             pendingAssignedSlots: [],
                                             avoidingSlotPlanAvailable: false),
             .issueWithoutPending),
            (PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                             pendingDepth: 2,
                                             pendingAssignedSlots: [1, 2],
                                             avoidingSlotPlanAvailable: true),
             .drainBeforeIssue(reason: .maxPendingDepthReached)),
            (PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                             pendingAssignedSlots: [],
                                             avoidingSlotPlanAvailable: true),
             .drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots)),
            (PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                             pendingAssignedSlots: [1, 2],
                                             avoidingSlotPlanAvailable: false),
             .drainBeforeIssue(reason: .avoidingSlotPlanUnavailable)),
            (PrefillRoutedTileSchedulerInput(hasPendingTile: true,
                                             pendingAssignedSlots: [3, 7, 8],
                                             avoidingSlotPlanAvailable: true),
             .prefetchNext(avoidingSlots: [3, 7, 8]))
        ]

        for (input, expected) in cases {
            #expect(batchOfOne.decide(input) == expected)
            #expect(batchOfOne.decide(input) == unbatched.decide(input))
        }
        // Width 1 commits every tile it appends and never carries one over, so
        // the open batch is empty at the top of every iteration.
        #expect(batchOfOne.batchAction(openBatchTiles: 0,
                                       openBatchSlots: [],
                                       nextTileAvoidingSlotPlanAvailable: true,
                                       isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .batchFull))
        #expect(batchOfOne.batchAction(openBatchTiles: 0,
                                       openBatchSlots: [],
                                       nextTileAvoidingSlotPlanAvailable: true,
                                       isLastTile: true)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .batchFull))
    }

    @Test func batchFillsToTheConfiguredWidth() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 4))

        #expect(scheduler.batchAction(openBatchTiles: 2,
                                      openBatchSlots: [1, 2],
                                      nextTileAvoidingSlotPlanAvailable: true,
                                      isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: nil))
        #expect(scheduler.batchAction(openBatchTiles: 3,
                                      openBatchSlots: [1, 2],
                                      nextTileAvoidingSlotPlanAvailable: true,
                                      isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .batchFull))
    }

    @Test func batchCommitsWhenTheNextTileNeedsAHeldSlot() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 4))

        #expect(scheduler.batchAction(openBatchTiles: 2,
                                      openBatchSlots: [2, 5],
                                      nextTileAvoidingSlotPlanAvailable: false,
                                      isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: true,
                                                commitAfterAppend: nil))
        #expect(scheduler.batchAction(openBatchTiles: 2,
                                      openBatchSlots: [],
                                      nextTileAvoidingSlotPlanAvailable: false,
                                      isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: nil))
    }

    @Test func lastTileAlwaysCommits() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 4))

        #expect(scheduler.batchAction(openBatchTiles: 1,
                                      openBatchSlots: [2, 5],
                                      nextTileAvoidingSlotPlanAvailable: true,
                                      isLastTile: true)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .lastTile))
        #expect(scheduler.batchAction(openBatchTiles: 0,
                                      openBatchSlots: [],
                                      nextTileAvoidingSlotPlanAvailable: true,
                                      isLastTile: true)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .lastTile))
    }

    @Test func lastTileReportsBatchFullWhenItAlsoFillsTheBatch() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 4))

        #expect(scheduler.batchAction(openBatchTiles: 3,
                                      openBatchSlots: [2, 5],
                                      nextTileAvoidingSlotPlanAvailable: true,
                                      isLastTile: true)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: false,
                                                commitAfterAppend: .batchFull))
    }

    @Test func collisionOnTheLastTileSetsBothCommitPoints() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 4))

        #expect(scheduler.batchAction(openBatchTiles: 2,
                                      openBatchSlots: [2, 5],
                                      nextTileAvoidingSlotPlanAvailable: false,
                                      isLastTile: true)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: true,
                                                commitAfterAppend: .lastTile))
        #expect(PrefillRoutedTileScheduler().batchAction(
            openBatchTiles: 1,
            openBatchSlots: [2, 5],
            nextTileAvoidingSlotPlanAvailable: false,
            isLastTile: false)
                == PrefillRoutedTileBatchAction(commitBeforeAppend: true,
                                                commitAfterAppend: .batchFull))
    }

    @Test func configClampsTheBatchWidth() {
        #expect(PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 0)
            .tilesPerCommandBuffer == 1)
        #expect(PrefillRoutedTileSchedulerConfig(tilesPerCommandBuffer: 99)
            .tilesPerCommandBuffer == 16)
    }

    @Test func slotBudgetCountsTheWholeOpenBatch() {
        let wide = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1,
                                                    tileExperts: 8,
                                                    tilesPerCommandBuffer: 4)

        #expect(wide.fitsSlotBudget(slotCount: 64))
        #expect(!wide.fitsSlotBudget(slotCount: 32))
        #expect(wide.fitting(slotCount: 64) == wide)
    }

    @Test func fittingNarrowsTheBatchWidthBeforeTheTile() {
        let wide = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1,
                                                    tileExperts: 8,
                                                    tilesPerCommandBuffer: 4)

        #expect(wide.fitting(slotCount: 32) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 1,
            tileExperts: 8,
            tilesPerCommandBuffer: 2))
        #expect(wide.fitting(slotCount: 8) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 1,
            tileExperts: 4,
            tilesPerCommandBuffer: 1))
        #expect(wide.fitting(slotCount: 32)?.fitsSlotBudget(slotCount: 32) == true)
        #expect(wide.fitting(slotCount: 8)?.fitsSlotBudget(slotCount: 8) == true)
        #expect(wide.fitting(slotCount: 1) == nil)

        let narrow = PrefillRoutedTileSchedulerConfig()
        #expect(narrow.fitting(slotCount: 16) == narrow)
    }

    @Test func fittingKeepsTheDepthAndNarrowsTheTile() {
        let config = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2)

        #expect(config.fitting(slotCount: 128) == config)
        #expect(config.fitting(slotCount: 16) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 2, tileExperts: 5, tilesPerCommandBuffer: 1))
        #expect(config.fitting(slotCount: 8) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 2, tileExperts: 2, tilesPerCommandBuffer: 1))
        #expect(config.fitting(slotCount: 2) == nil)
    }

    @Test func theSlotBudgetCeilingIsFifteenAtOneHundredTwentyEightSlots() {
        let depthFifteen = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 15)
        let depthSixteen = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 16)

        #expect(depthFifteen.fitsSlotBudget(slotCount: 128))
        #expect(!depthSixteen.fitsSlotBudget(slotCount: 128))
    }

    @Test func configFloorsTheDepth() {
        #expect(PrefillRoutedTileSchedulerConfig(maxPendingDepth: 0).maxPendingDepth == 1)
    }

    @Test func depthThreeDrainsOnlyPastThreePending() {
        let scheduler = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(maxPendingDepth: 3, tileExperts: 4))

        #expect(scheduler.decide(PrefillRoutedTileSchedulerInput(
            hasPendingTile: true,
            pendingDepth: 3,
            pendingAssignedSlots: [1, 2, 3, 4],
            avoidingSlotPlanAvailable: true))
            == .prefetchNext(avoidingSlots: [1, 2, 3, 4]))
        #expect(scheduler.decide(PrefillRoutedTileSchedulerInput(
            hasPendingTile: true,
            pendingDepth: 4,
            pendingAssignedSlots: [1, 2, 3, 4],
            avoidingSlotPlanAvailable: true))
            == .drainBeforeIssue(reason: .maxPendingDepthReached))
    }

    @Test func prefillDescriptionReportsTheRouterBitsTheResidencyFailureAndTheTrace() {
        #expect(RealForwardRunner.prefillDescription(
            routerBits: 8, poolResidencyUnavailableReason: nil, prefetchTrace: false)
            == "prefill_router_bits=8")
        #expect(RealForwardRunner.prefillDescription(
            routerBits: 4, poolResidencyUnavailableReason: "boom", prefetchTrace: false)
            == "prefill_router_bits=4 prefill_pool_residency=unavailable reason=boom")
        #expect(RealForwardRunner.prefillDescription(
            routerBits: 8, poolResidencyUnavailableReason: nil, prefetchTrace: true)
            == "prefill_router_bits=8 prefetch_trace=on")
    }

    @Test func prefetchTraceOpensFailClosed() throws {
        #expect(try RealForwardRunner.openPrefetchTrace(nil) == -1)
        #expect(throws: RuntimeConfigurationError.self) {
            try RealForwardRunner.openPrefetchTrace("/nonexistent-\(UUID().uuidString)/prefetch.jsonl")
        }
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefetch-\(UUID().uuidString).jsonl").path
        let descriptor = try RealForwardRunner.openPrefetchTrace(path)
        #expect(descriptor >= 0)
        close(descriptor)
        try FileManager.default.removeItem(atPath: path)
    }

    @Test func chunkExpertProtectionClearsThePlannedTileInPlace() {
        var protection = PrefillChunkExpertProtection(routedExperts: [1, 2, 3, 4, 5], expertsPerLayer: 10)
        #expect(protection.remaining == [false, true, true, true, true, true, false, false, false, false])

        protection.planning([1, 2])
        #expect(protection.remaining == [false, false, false, true, true, true, false, false, false, false])

        protection.planning([3])
        #expect(protection.remaining == [false, false, false, false, true, true, false, false, false, false])

        protection.planning([4, 5])
        #expect(protection.remaining.allSatisfy { !$0 })
    }

    @Test func chunkExpertProtectionIgnoresAnExpertNeverRouted() {
        var protection = PrefillChunkExpertProtection(routedExperts: [1, 2], expertsPerLayer: 10)

        protection.planning([9])

        #expect(protection.remaining[1] == true)
        #expect(protection.remaining[2] == true)
        #expect(protection.remaining[9] == false)
    }

    @Test func chunkExpertProtectionBuildsDirectlyFromRoutedGroups() {
        let groups = [
            PrefillMoEGroup(expert: 2, pairStart: 0, pairCount: 1),
            PrefillMoEGroup(expert: 7, pairStart: 1, pairCount: 2),
        ]

        let protection = PrefillChunkExpertProtection(routedGroups: groups, expertsPerLayer: 10)

        #expect(protection.remaining == [false, false, true, false, false, false, false, true, false, false])
    }

    @Test func slotLifetimeRejectsReuseInsideAnOpenBatch() throws {
        var lifetime = PrefillStreamedTileSlotLifetime()
        try lifetime.begin(tileIndex: 0, plannedSlots: [1, 2])

        #expect(throws: PrefillStreamedTileLifetimeError.self) {
            try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3])
        }

        try lifetime.complete(tileIndex: 0)
        try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3])
    }

    @Test func slotLifetimeRejectsReuseWhileTheLookaheadTileIsInFlight() throws {
        var lifetime = PrefillStreamedTileSlotLifetime()
        try lifetime.begin(tileIndex: 0, plannedSlots: [1, 2])
        try lifetime.begin(tileIndex: 1, plannedSlots: [3, 4])

        #expect(throws: PrefillStreamedTileLifetimeError.self) {
            try lifetime.begin(tileIndex: 2, plannedSlots: [2, 4])
        }

        try lifetime.complete(tileIndex: 0)
        #expect(throws: PrefillStreamedTileLifetimeError.self) {
            try lifetime.begin(tileIndex: 2, plannedSlots: [4, 5])
        }

        try lifetime.complete(tileIndex: 1)
        try lifetime.begin(tileIndex: 2, plannedSlots: [4, 5])
    }

    @Test func theLookaheadCountsOneMoreTileInTheSlotBudget() {
        let config = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, tileExperts: 8, fetchLookahead: 1)

        #expect(config.fitsSlotBudget(slotCount: 128))
        #expect(!config.fitsSlotBudget(slotCount: 31))

        let ceiling = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 14, tileExperts: 8, fetchLookahead: 1)
        let overCeiling = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 15, tileExperts: 8, fetchLookahead: 1)
        #expect(ceiling.fitsSlotBudget(slotCount: 128))
        #expect(!overCeiling.fitsSlotBudget(slotCount: 128))
    }

    @Test func fittingKeepsTheLookaheadAndNarrowsTheTile() {
        let config = PrefillRoutedTileSchedulerConfig(maxPendingDepth: 2, fetchLookahead: 1)

        #expect(config.fitting(slotCount: 128) == config)
        #expect(config.fitting(slotCount: 16) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 2, tileExperts: 4, tilesPerCommandBuffer: 1, fetchLookahead: 1))
        #expect(config.fitting(slotCount: 8) == PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: 2, tileExperts: 2, tilesPerCommandBuffer: 1, fetchLookahead: 1))
        #expect(config.fitting(slotCount: 3) == nil)
    }

    @Test func shouldBeginLookaheadRequiresALookaheadAPlanAndASuccessor() {
        let withLookahead = PrefillRoutedTileScheduler(
            config: PrefillRoutedTileSchedulerConfig(fetchLookahead: 1))
        let withoutLookahead = PrefillRoutedTileScheduler()

        #expect(withLookahead.shouldBeginLookahead(
            afterTileIndex: 0, tileCount: 3, avoidingSlotPlanAvailable: true))
        #expect(!withoutLookahead.shouldBeginLookahead(
            afterTileIndex: 0, tileCount: 3, avoidingSlotPlanAvailable: true))
        #expect(!withLookahead.shouldBeginLookahead(
            afterTileIndex: 0, tileCount: 3, avoidingSlotPlanAvailable: false))
        #expect(!withLookahead.shouldBeginLookahead(
            afterTileIndex: 2, tileCount: 3, avoidingSlotPlanAvailable: true))
    }

    @Test func matrixPathAcceptsHonoursALoweredMinimum() {
        let params = PrefillAttentionParams(
            startPosition: 0, queryCount: 21, headDim: 256, numQHeads: 16, numKVHeads: 2,
            kvValidCount: 21, slidingWindow: 0, kvTokenStrideElements: 261,
            qTokenStrideElements: 259, oTokenStrideElements: 263, scale: 0.0625)
        #expect(!PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 0, hasSinks: false))
        #expect(PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 0, hasSinks: false, minimumQueries: 16))
        #expect(!PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 0, hasSinks: false, minimumQueries: 22))

        var window = params
        window.slidingWindow = 10
        #expect(!PrefillAttention.matrixPathAccepts(window, kvRingCapacity: 0, hasSinks: false, minimumQueries: 16))
        var wide = params
        wide.headDim = 512
        #expect(!PrefillAttention.matrixPathAccepts(wide, kvRingCapacity: 0, hasSinks: false, minimumQueries: 16))
        #expect(!PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 4096, hasSinks: false, minimumQueries: 16))
        #expect(!PrefillAttention.matrixPathAccepts(params, kvRingCapacity: 0, hasSinks: true, minimumQueries: 16))
    }

    @Test func projectionDispatchPolicyHonoursALoweredMinimum() {
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .kv, chunkTokens: 21) == .repeatedGEMV)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .kv, chunkTokens: 21, minimumRows: 16) == .qmm)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .o, chunkTokens: 21, minimumRows: 16) == .qmm)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .q, chunkTokens: 21, minimumRows: 16) == .repeatedGEMV)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .kv, chunkTokens: 15, minimumRows: 16) == .repeatedGEMV)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .o, chunkTokens: 15, minimumRows: 16) == .repeatedGEMV)
        #expect(PrefillProjectionDispatchPolicy.selectedDispatch(
            for: .q, chunkTokens: 15, minimumRows: 16) == .repeatedGEMV)
    }

    @Test func routeTraceLineFormatsPrefillAndDecode() {
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 42, layer: 3, experts: [1, 2, 3, 4, 5, 6, 7, 8])
            == "42 3 1 2 3 4 5 6 7 8\n")
        #expect(RealForwardRunner.formatRouteTraceLine(position: 42, layer: 3, experts: [])
            == "42 3\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11])
            == "p 100 5 2 9 10 11\n")
        #expect(RealForwardRunner.formatRouteTraceLine(position: 100, layer: 5, tile: 0, experts: [])
            == "p 100 5 0\n")
        #expect(RealForwardRunner.formatRouteTraceLine(cachedTokens: 21, promptTokens: 2366)
            == "r 21 2366\n")
        #expect(RealForwardRunner.formatRouteTraceLine(cachedTokens: 0, promptTokens: 300)
            == "r 0 300\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11], rowCounts: [4, 12, 1])
            == "p 100 5 2 9 10 11 | 4 12 1\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11], rowCounts: nil)
            == "p 100 5 2 9 10 11\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11], rowCounts: [])
            == "p 100 5 2 9 10 11\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11],
            rowCounts: [4, 12, 1], lastRows: [7, 20, 2])
            == "p 100 5 2 9 10 11 | 4:7 12:20 1:2\n")
        #expect(RealForwardRunner.formatRouteTraceLine(
            position: 100, layer: 5, tile: 2, experts: [9, 10, 11],
            rowCounts: [4, 12, 1], lastRows: nil)
            == "p 100 5 2 9 10 11 | 4 12 1\n")
    }

}
