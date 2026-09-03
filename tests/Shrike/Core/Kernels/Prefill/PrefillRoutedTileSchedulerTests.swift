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

    @Test func parsePrefillTileDepthClampsToOneThroughEight() {
        #expect(RealForwardRunner.parsePrefillTileDepth(nil) == 2)
        #expect(RealForwardRunner.parsePrefillTileDepth("") == 2)
        #expect(RealForwardRunner.parsePrefillTileDepth("not-a-number") == 2)
        #expect(RealForwardRunner.parsePrefillTileDepth("0") == 1)
        #expect(RealForwardRunner.parsePrefillTileDepth("-3") == 1)
        #expect(RealForwardRunner.parsePrefillTileDepth("1") == 1)
        #expect(RealForwardRunner.parsePrefillTileDepth(" 4 ") == 4)
        #expect(RealForwardRunner.parsePrefillTileDepth("8") == 8)
        #expect(RealForwardRunner.parsePrefillTileDepth("9") == 8)
        #expect(RealForwardRunner.parsePrefillTileDepth("100") == 8)
    }

    @Test func prefillTileDepthDescriptionReportsTheRequestedValueOnly() {
        #expect(RealForwardRunner.prefillTileDepthDescription(
            PrefillRoutedTileSchedulerConfig(maxPendingDepth: 1)) == "depth=1")
        #expect(RealForwardRunner.prefillTileDepthDescription(
            PrefillRoutedTileSchedulerConfig(maxPendingDepth: 4, tileExperts: 2, tilesPerCommandBuffer: 3))
            == "depth=4")
    }

    @Test func prefillGapLeversDescriptionReportsResidencyAllocationsAndCacheLayout() {
        #expect(RealForwardRunner.prefillGapLeversDescription(
            overlap: true, residencyAllocationCount: 24, poolResidencyUnavailableReason: nil,
            sweepMode: .fixed, cacheLayout: .pool)
            == "overlap=on residency=set allocations=24 sweep=fixed cache_layout=pool")
        #expect(RealForwardRunner.prefillGapLeversDescription(
            overlap: false, residencyAllocationCount: 0, poolResidencyUnavailableReason: nil,
            sweepMode: .alternate, cacheLayout: .perSlot)
            == "overlap=off residency=set allocations=0 sweep=alternate cache_layout=per-slot")
        #expect(RealForwardRunner.prefillGapLeversDescription(
            overlap: true, residencyAllocationCount: nil, poolResidencyUnavailableReason: "boom",
            sweepMode: .alternate, cacheLayout: .pool)
            == "overlap=on residency=unavailable reason=boom sweep=alternate cache_layout=pool")
        #expect(RealForwardRunner.prefillGapLeversDescription(
            overlap: true, residencyAllocationCount: nil, poolResidencyUnavailableReason: nil,
            sweepMode: .fixed, cacheLayout: .pool)
            == "overlap=on residency=none sweep=fixed cache_layout=pool")
    }

    @Test func sweepModeParsesItsThreeValues() {
        #expect(RealForwardRunner.parsePrefillSweepMode("alternate") == .alternate)
        #expect(RealForwardRunner.parsePrefillSweepMode("fixed") == .fixed)
        #expect(RealForwardRunner.parsePrefillSweepMode("carry") == .carry)
        #expect(RealForwardRunner.parsePrefillSweepMode(nil) == .carry)
        #expect(RealForwardRunner.parsePrefillSweepMode("") == .carry)
        #expect(RealForwardRunner.parsePrefillSweepMode("bogus") == .carry)
    }

    @Test func prefillGapLeversDescriptionReportsTheSweepMode() {
        #expect(RealForwardRunner.prefillGapLeversDescription(
            overlap: true, residencyAllocationCount: 24, poolResidencyUnavailableReason: nil,
            sweepMode: .carry, cacheLayout: .pool)
            == "overlap=on residency=set allocations=24 sweep=carry cache_layout=pool")
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

}
