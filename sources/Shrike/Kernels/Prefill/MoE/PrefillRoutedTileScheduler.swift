enum PrefillRoutedTileSchedulerDrainReason: Sendable, Equatable {
    case pendingTileHasNoAssignedSlots
    case avoidingSlotPlanUnavailable
    case maxPendingDepthReached
}

enum PrefillRoutedTileBatchCloseReason: Sendable, Equatable {
    case batchFull
    case lastTile
}

struct PrefillRoutedTileBatchAction: Sendable, Equatable {
    /// The tile's plan collides with a slot the open batch still holds.
    let commitBeforeAppend: Bool
    /// `nil` leaves the batch open for the next tile.
    let commitAfterAppend: PrefillRoutedTileBatchCloseReason?
}

enum PrefillRoutedTileSchedulerDecision: Sendable, Equatable {
    case issueWithoutPending
    case prefetchNext(avoidingSlots: [Int])
    case drainBeforeIssue(reason: PrefillRoutedTileSchedulerDrainReason)
}

struct PrefillRoutedTileSchedulerInput: Sendable, Equatable {
    let hasPendingTile: Bool
    let pendingDepth: Int
    let pendingAssignedSlots: [Int]
    let avoidingSlotPlanAvailable: Bool

    init(hasPendingTile: Bool,
         pendingDepth: Int? = nil,
         pendingAssignedSlots: [Int],
         avoidingSlotPlanAvailable: Bool) {
        self.hasPendingTile = hasPendingTile
        self.pendingDepth = max(0, pendingDepth ?? (hasPendingTile ? 1 : 0))
        self.pendingAssignedSlots = pendingAssignedSlots
        self.avoidingSlotPlanAvailable = avoidingSlotPlanAvailable
    }
}

struct PrefillRoutedTileSchedulerConfig: Sendable, Equatable {
    let maxPendingDepth: Int
    let tileExperts: Int
    let tilesPerCommandBuffer: Int
    /// A tile fetched one ahead of the batch pipeline, held from plan time
    /// until its own GPU work completes; 0 begins each tile's fetch at its
    /// own turn.
    let fetchLookahead: Int

    init(maxPendingDepth: Int = 1, tileExperts: Int = 8, tilesPerCommandBuffer: Int = 1,
         fetchLookahead: Int = 0) {
        self.maxPendingDepth = max(1, maxPendingDepth)
        self.tileExperts = max(1, min(16, tileExperts))
        self.tilesPerCommandBuffer = max(1, min(16, tilesPerCommandBuffer))
        self.fetchLookahead = max(0, min(1, fetchLookahead))
    }

    /// Every tile of the open batch and of each pending batch holds its slots
    /// until that batch completes, so the budget multiplies by the batch
    /// width; the lookahead tile is one further tile held the same way.
    func fitsSlotBudget(slotCount: Int, reservedHits: Int = 0) -> Bool {
        guard slotCount > 0, reservedHits >= 0 else { return false }
        return maxInFlightTiles * tileExperts + reservedHits <= slotCount
    }

    /// Narrows the batch width first and only then the I/O tile, so a
    /// deliberately small streamed-expert cache costs submission batching
    /// before it costs the streamer the tile size it was tuned for.
    func fitting(slotCount: Int, reservedHits: Int = 0) -> Self? {
        guard slotCount > reservedHits, reservedHits >= 0 else { return nil }
        let available = slotCount - reservedHits
        let narrowedWidth = Self(
            maxPendingDepth: maxPendingDepth,
            tileExperts: tileExperts,
            tilesPerCommandBuffer: max(1, min(tilesPerCommandBuffer,
                                              (available - fetchLookahead * tileExperts)
                                                  / (inFlightBatches * tileExperts))),
            fetchLookahead: fetchLookahead)
        if narrowedWidth.fitsSlotBudget(slotCount: slotCount, reservedHits: reservedHits) {
            return narrowedWidth
        }
        let availablePerTile = available / (inFlightBatches + fetchLookahead)
        guard availablePerTile > 0 else { return nil }
        return Self(maxPendingDepth: maxPendingDepth,
                    tileExperts: min(tileExperts, availablePerTile),
                    tilesPerCommandBuffer: 1,
                    fetchLookahead: fetchLookahead)
    }

    private var inFlightBatches: Int { maxPendingDepth + 1 }
    private var maxInFlightTiles: Int { inFlightBatches * tilesPerCommandBuffer + fetchLookahead }
}

struct PrefillRoutedTileScheduler: Sendable, Equatable {
    let config: PrefillRoutedTileSchedulerConfig

    init(config: PrefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()) {
        self.config = config
    }

    func decide(_ input: PrefillRoutedTileSchedulerInput) -> PrefillRoutedTileSchedulerDecision {
        guard input.hasPendingTile else {
            return .issueWithoutPending
        }
        // `PrefillRoutedTileSequencer`'s post-append drain loop already holds
        // pendingDepth ≤ maxPendingDepth before the next `decide`, so this
        // guard only fires if a caller violates that invariant.
        guard input.pendingDepth <= config.maxPendingDepth else {
            return .drainBeforeIssue(reason: .maxPendingDepthReached)
        }
        guard !input.pendingAssignedSlots.isEmpty else {
            return .drainBeforeIssue(reason: .pendingTileHasNoAssignedSlots)
        }
        guard input.avoidingSlotPlanAvailable else {
            return .drainBeforeIssue(reason: .avoidingSlotPlanUnavailable)
        }
        return .prefetchNext(avoidingSlots: input.pendingAssignedSlots)
    }

    /// Whether the successor tile is planned ahead at all: only while the
    /// config asks for a lookahead and a successor tile remains.
    func plansLookahead(afterTileIndex index: Int, tileCount: Int) -> Bool {
        config.fetchLookahead > 0 && index + 1 < tileCount
    }

    /// Whether to begin the next tile's fetch before awaiting the current
    /// one's: only while the successor is planned ahead and its
    /// avoiding-slots plan actually resolved.
    func shouldBeginLookahead(afterTileIndex index: Int,
                              tileCount: Int,
                              avoidingSlotPlanAvailable: Bool) -> Bool {
        plansLookahead(afterTileIndex: index, tileCount: tileCount) && avoidingSlotPlanAvailable
    }

    func batchAction(openBatchTiles: Int,
                     openBatchSlots: [Int],
                     nextTileAvoidingSlotPlanAvailable: Bool,
                     isLastTile: Bool) -> PrefillRoutedTileBatchAction {
        let commitBeforeAppend = !openBatchSlots.isEmpty && !nextTileAvoidingSlotPlanAvailable
        let tilesAfterAppend = (commitBeforeAppend ? 0 : max(0, openBatchTiles)) + 1
        let closeReason: PrefillRoutedTileBatchCloseReason?
        if tilesAfterAppend >= config.tilesPerCommandBuffer {
            closeReason = .batchFull
        } else if isLastTile {
            closeReason = .lastTile
        } else {
            closeReason = nil
        }
        return PrefillRoutedTileBatchAction(commitBeforeAppend: commitBeforeAppend,
                                            commitAfterAppend: closeReason)
    }
}
