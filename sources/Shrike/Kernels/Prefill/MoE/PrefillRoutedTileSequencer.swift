/// The routed tile loop's I/O as `PrefillRoutedTileSequencer` drives it; the
/// runner's conformer owns the real plans, begun fetches and command buffers.
protocol PrefillRoutedTileDriver: AnyObject {
    var openBatchTiles: Int { get }
    var openBatchSlots: [Int] { get }
    var pendingDepth: Int { get }
    var pendingAssignedSlots: [Int] { get }

    /// `false` when no plan avoiding the batches' slots (and the begun tile
    /// `inFlight`'s) fits the cache, in which case no plan is kept.
    func plan(tile: Int, avoidingInFlight inFlight: Int?) throws -> Bool
    /// Plans `tile` afresh against the batches' slots when no plan is kept,
    /// throwing when the cache has no room.
    func begin(tile: Int) throws
    func abandonPlan(tile: Int) throws
    func encode(tile: Int) async throws
    /// A no-op without an open batch.
    func commitOpenBatch()
    func drainOldestBatch() throws
    /// The error path: waits out every begun, unencoded fetch and releases
    /// every kept plan.
    func abandonBegunFetches()
}

/// One loop for the routed tiles at every fetch depth: a tile is awaited and
/// encoded only once its successor's fetch is in flight, so the drive's next
/// reads run under the host's encode.
struct PrefillRoutedTileSequencer {
    let scheduler: PrefillRoutedTileScheduler

    func run(tileCount: Int, driver: some PrefillRoutedTileDriver) async throws {
        do {
            try await sequence(tileCount: tileCount, driver: driver)
        } catch {
            driver.abandonBegunFetches()
            throw error
        }
    }

    private func sequence(tileCount: Int, driver: some PrefillRoutedTileDriver) async throws {
        var carried: Int?
        for tile in 0..<tileCount {
            let isLastTile = tile == tileCount - 1
            let carriedFetch = carried == tile
            carried = nil
            let planAvailable: Bool
            if carriedFetch {
                planAvailable = true
            } else {
                planAvailable = try driver.plan(tile: tile, avoidingInFlight: nil)
            }
            let batchAction = scheduler.batchAction(
                openBatchTiles: driver.openBatchTiles,
                openBatchSlots: driver.openBatchSlots,
                nextTileAvoidingSlotPlanAvailable: planAvailable,
                isLastTile: isLastTile)
            if batchAction.commitBeforeAppend {
                driver.commitOpenBatch()
            }
            if !carriedFetch {
                try issue(tile: tile, planAvailable: planAvailable, driver: driver)
            }
            if scheduler.plansLookahead(afterTileIndex: tile, tileCount: tileCount) {
                let successor = tile + 1
                let successorPlanAvailable = try driver.plan(tile: successor, avoidingInFlight: tile)
                if scheduler.shouldBeginLookahead(afterTileIndex: tile,
                                                  tileCount: tileCount,
                                                  avoidingSlotPlanAvailable: successorPlanAvailable) {
                    try driver.begin(tile: successor)
                    carried = successor
                }
            }
            try await driver.encode(tile: tile)
            if batchAction.commitAfterAppend != nil {
                driver.commitOpenBatch()
            }
            // A still-open batch holds the tile just encoded and counts
            // against the depth budget with the committed ones.
            while driver.pendingDepth + (driver.openBatchTiles > 0 ? 1 : 0)
                > scheduler.config.maxPendingDepth {
                try driver.drainOldestBatch()
            }
        }
        driver.commitOpenBatch()
        while driver.pendingDepth > 0 {
            try driver.drainOldestBatch()
        }
    }

    private func issue(tile: Int, planAvailable: Bool, driver: some PrefillRoutedTileDriver) throws {
        let decision = scheduler.decide(PrefillRoutedTileSchedulerInput(
            hasPendingTile: driver.pendingDepth > 0,
            pendingDepth: driver.pendingDepth,
            pendingAssignedSlots: driver.pendingAssignedSlots,
            avoidingSlotPlanAvailable: planAvailable))
        switch decision {
        case .issueWithoutPending, .prefetchNext:
            break
        case .drainBeforeIssue:
            if planAvailable {
                try driver.abandonPlan(tile: tile)
            }
            try driver.drainOldestBatch()
        }
        try driver.begin(tile: tile)
    }
}
