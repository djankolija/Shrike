import Foundation
import Metal
import Darwin

enum PrefillGroupedRoutedMoEBufferIndex {
    static let hidden = 0
    static let sortedPairs = 1
    static let routePartials = 5
    static let gateUpActScratch = 7
    static let downScratch = 8
    static let expertArgumentState = 9
    static let params = 10
}

enum PrefillRoutedRowBlockBufferIndex {
    static let source = 0
    static let sortedPairs = 1
    static let destination = 2
    static let params = 3
    static let blocks = 4
    static let rowTileBlock = 5
}

struct PrefillRoutedRowBlockParams: Equatable, Sendable {
    var pairStart: UInt32
    var rows: UInt32
    var d: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
}

struct PrefillRoutedGroupedParams: Equatable, Sendable {
    var paddedRows: UInt32
    var d: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var rowTile: UInt32
}

/// One expert's `[pairStart, pairStart + pairCount)` window in `sortedPairs`.
@frozen
public struct PrefillExpertPairRange: Equatable, Sendable {
    public var expert: UInt32
    public var pairStart: Int
    public var pairCount: Int

    public init(expert: UInt32, pairStart: Int, pairCount: Int) {
        self.expert = expert
        self.pairStart = pairStart
        self.pairCount = pairCount
    }

    /// The tile's experts in `sortedPairs` order, which is the order
    /// `PrefillStreamedTileBinding.expertIDs` and `views` are bound in.
    public static func ranges(forTile tile: PrefillMoETile,
                              routes: PrefillMoEGroupedRoutes) throws -> [PrefillExpertPairRange] {
        let start = Int(tile.groupStart)
        let count = Int(tile.groupCount)
        guard count > 0, count <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile has \(count) live experts; expected 1...16")
        }
        guard start >= 0, start + count <= routes.groups.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile group range \(start)..<\(start + count) exceeds \(routes.groups.count)")
        }
        let end = start + count
        return routes.groups[start..<end].map { group in
            PrefillExpertPairRange(expert: group.expert,
                                   pairStart: Int(group.pairStart),
                                   pairCount: Int(group.pairCount))
        }
    }
}

/// Mirrored field-for-field by `MPPGroupedBlockMSL` and
/// `PrefillRoutedGroupedBlockMSL`; `rowTileStart` lets a threadgroup find its
/// row origin from its grid position without a search.
struct PrefillRoutedExpertBlock: Equatable, Sendable {
    var slot: UInt32
    var pairStart: UInt32
    var rows: UInt32
    var stagingRow: UInt32
    var rowTileStart: UInt32
}

/// `tailRows` is the 32-row tail region at the end of the wave (0 without the
/// tail tile); the body region `[0, paddedRows - tailRows)` is 64-row tiles.
/// A body block's `rowTileStart` is its 64-row tile; a tail block's is its
/// 32-row tile counted from the tail region's start.
struct PrefillRoutedExpertWave: Equatable, Sendable {
    var blocks: [PrefillRoutedExpertBlock]
    var paddedRows: Int
    var tailRows: Int = 0

    var bodyPaddedRows: Int { paddedRows - tailRows }
}

struct PrefillRoutedWaveTables: Equatable, Sendable {
    var gather: [UInt32]
    var body: [UInt32]
    var tail: [UInt32]
}

extension PrefillGroupedRoutedMoE {
    static let groupedRowTile = MPPPrefillInt4QMM.tileM
    static let groupedTailRowTile = groupedRowTile / 2

    /// An expert longer than a wave is split at row-tile boundaries; a 1-pair
    /// expert takes one `rowTile`-row tile, the rest padded — unless `tailTile`
    /// packs remainders body-first (`planExpertWavesWithTail`).
    static func planExpertWaves(ranges: [PrefillExpertPairRange],
                                binding: PrefillStreamedTileBinding,
                                stagingRows: Int,
                                rowTile: Int = groupedRowTile,
                                tailTile: Int = 0) throws -> [PrefillRoutedExpertWave] {
        if tailTile != 0 {
            guard rowTile == groupedRowTile else {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "a tail tile packs \(groupedRowTile)-row bodies; rowTile \(rowTile) cannot combine with it")
            }
            return try planExpertWavesWithTail(ranges: ranges, binding: binding,
                                               stagingRows: stagingRows, tailTile: tailTile)
        }
        let tile = rowTile
        guard stagingRows >= tile, stagingRows.isMultiple(of: tile) else {
            throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                "\(stagingRows) staging rows is not a positive multiple of \(tile)")
        }
        var waves: [PrefillRoutedExpertWave] = []
        var blocks: [PrefillRoutedExpertBlock] = []
        var cursor = 0
        func closeWave() {
            guard !blocks.isEmpty else { return }
            waves.append(PrefillRoutedExpertWave(blocks: blocks, paddedRows: cursor))
            blocks.removeAll(keepingCapacity: true)
            cursor = 0
        }
        for range in ranges {
            guard range.pairCount > 0 else { continue }
            guard let slot = binding.localSlot(for: range.expert) else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert \(range.expert) is not bound in the tile")
            }
            var consumed = 0
            while consumed < range.pairCount {
                if cursor == stagingRows { closeWave() }
                let rows = min(range.pairCount - consumed, stagingRows - cursor)
                blocks.append(PrefillRoutedExpertBlock(
                    slot: UInt32(slot),
                    pairStart: UInt32(range.pairStart + consumed),
                    rows: UInt32(rows),
                    stagingRow: UInt32(cursor),
                    rowTileStart: UInt32(cursor / tile)))
                cursor += (rows + tile - 1) / tile * tile
                consumed += rows
            }
        }
        closeWave()
        return waves
    }

    /// Body-first packing: a range piece of `r` rows is `64·⌊r/64⌋` body rows
    /// plus a tail block of `r mod 64` rows when that remainder is ≤ 32 — a
    /// larger remainder stays on a padded 64-row tile, since two 32-row tiles
    /// cost more than one 64-row tile. Body blocks are laid out first, tail
    /// blocks after `bodyPaddedRows`; a piece that does not fit the wave's
    /// padded capacity is cut at the largest `64a + 32b` that does — `used` is
    /// always a multiple of the tail tile, so that is the wave's remaining
    /// capacity.
    private static func planExpertWavesWithTail(ranges: [PrefillExpertPairRange],
                                                binding: PrefillStreamedTileBinding,
                                                stagingRows: Int,
                                                tailTile: Int) throws -> [PrefillRoutedExpertWave] {
        let body = groupedRowTile
        guard tailTile == groupedTailRowTile else {
            throw MPPPrefillInt4QMMError.invalidArguments(
                "tail tile \(tailTile) is not \(groupedTailRowTile)")
        }
        guard stagingRows >= body, stagingRows.isMultiple(of: body) else {
            throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                "\(stagingRows) staging rows is not a positive multiple of \(body)")
        }
        struct Piece { var slot: UInt32; var pairStart: Int; var rows: Int }
        var waves: [PrefillRoutedExpertWave] = []
        var bodyPieces: [Piece] = []
        var tailPieces: [Piece] = []
        var used = 0
        func closeWave() {
            guard !bodyPieces.isEmpty || !tailPieces.isEmpty else { return }
            var blocks: [PrefillRoutedExpertBlock] = []
            var cursor = 0
            for piece in bodyPieces {
                blocks.append(PrefillRoutedExpertBlock(
                    slot: piece.slot, pairStart: UInt32(piece.pairStart), rows: UInt32(piece.rows),
                    stagingRow: UInt32(cursor), rowTileStart: UInt32(cursor / body)))
                cursor += (piece.rows + body - 1) / body * body
            }
            let bodyPadded = cursor
            for (index, piece) in tailPieces.enumerated() {
                blocks.append(PrefillRoutedExpertBlock(
                    slot: piece.slot, pairStart: UInt32(piece.pairStart), rows: UInt32(piece.rows),
                    stagingRow: UInt32(cursor), rowTileStart: UInt32(index)))
                cursor += tailTile
            }
            waves.append(PrefillRoutedExpertWave(blocks: blocks, paddedRows: cursor,
                                                 tailRows: cursor - bodyPadded))
            bodyPieces.removeAll(keepingCapacity: true)
            tailPieces.removeAll(keepingCapacity: true)
            used = 0
        }
        func paddedCost(_ rows: Int) -> Int {
            let remainder = rows % body
            return rows - remainder + (remainder == 0 ? 0 : remainder <= tailTile ? tailTile : body)
        }
        func place(_ slot: UInt32, _ pairStart: Int, _ rows: Int) {
            let remainder = rows % body
            let bodyRows = remainder > tailTile ? rows : rows - remainder
            if bodyRows > 0 { bodyPieces.append(Piece(slot: slot, pairStart: pairStart, rows: bodyRows)) }
            if remainder > 0, remainder <= tailTile {
                tailPieces.append(Piece(slot: slot, pairStart: pairStart + bodyRows, rows: remainder))
            }
            used += paddedCost(rows)
        }
        for range in ranges {
            guard range.pairCount > 0 else { continue }
            guard let slot = binding.localSlot(for: range.expert) else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert \(range.expert) is not bound in the tile")
            }
            var consumed = 0
            while consumed < range.pairCount {
                let remaining = range.pairCount - consumed
                let available = stagingRows - used
                if used + paddedCost(remaining) <= stagingRows {
                    place(UInt32(slot), range.pairStart + consumed, remaining)
                    consumed += remaining
                    continue
                }
                let cut = available / body * body + (available % body >= tailTile ? tailTile : 0)
                if cut > 0 {
                    place(UInt32(slot), range.pairStart + consumed, cut)
                    consumed += cut
                }
                closeWave()
            }
        }
        closeWave()
        return waves
    }

    /// The three tables of a body/tail wave: `gather` at 32-row granularity
    /// over every padded row (the gather and scatter kernels), `body` per
    /// 64-row tile of the body region and `tail` per 32-row tile of the tail
    /// region (the two GEMM dispatches), each indexing `wave.blocks`.
    static func rowTileTables(for wave: PrefillRoutedExpertWave) -> PrefillRoutedWaveTables {
        let body = groupedRowTile
        let tail = groupedTailRowTile
        var gather = [UInt32](repeating: 0, count: wave.paddedRows / tail)
        var bodyTable = [UInt32](repeating: 0, count: wave.bodyPaddedRows / body)
        var tailTable = [UInt32](repeating: 0, count: wave.tailRows / tail)
        for (index, block) in wave.blocks.enumerated() {
            let start = Int(block.stagingRow)
            let isTail = start >= wave.bodyPaddedRows
            let padded = isTail ? tail : (Int(block.rows) + body - 1) / body * body
            for row in stride(from: start, to: start + padded, by: tail) {
                gather[row / tail] = UInt32(index)
            }
            if isTail {
                tailTable[Int(block.rowTileStart)] = UInt32(index)
            } else {
                for tileIndex in (start / body)..<((start + padded) / body) {
                    bodyTable[tileIndex] = UInt32(index)
                }
            }
        }
        return PrefillRoutedWaveTables(gather: gather, body: bodyTable, tail: tailTable)
    }

    /// `rowTileBlock[t]` is the index of the block that owns row tile `t`.
    static func rowTileTable(for wave: PrefillRoutedExpertWave,
                             rowTile: Int = groupedRowTile) -> [UInt32] {
        var table = [UInt32](repeating: 0, count: wave.paddedRows / rowTile)
        for (index, block) in wave.blocks.enumerated() {
            let first = Int(block.rowTileStart)
            let count = (Int(block.rows) + rowTile - 1) / rowTile
            for tile in first..<(first + count) {
                table[tile] = UInt32(index)
            }
        }
        return table
    }
}

/// Contiguous fp16 blocks the matrix path stages one expert's rows through:
/// gathered hidden rows, the gate and up GEMM outputs, and the down GEMM
/// output before it is scattered back into the per-pair partial rows.
struct PrefillExpertStaging {
    let hidden: MTLBuffer
    let gate: MTLBuffer
    let up: MTLBuffer
    let down: MTLBuffer
    let rowBlock: Int
    let hiddenSize: Int
    let intermediate: Int

    static func allocate(device: MTLDevice,
                         rowBlock: Int,
                         hiddenSize: Int,
                         intermediate: Int) throws -> PrefillExpertStaging {
        func buffer(_ elements: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed(label)
            }
            buffer.label = label
            return buffer
        }
        return PrefillExpertStaging(
            hidden: try buffer(rowBlock * hiddenSize, "prefill.routedExpertHiddenStaging"),
            gate: try buffer(rowBlock * intermediate, "prefill.routedExpertGateStaging"),
            up: try buffer(rowBlock * intermediate, "prefill.routedExpertUpStaging"),
            down: try buffer(rowBlock * hiddenSize, "prefill.routedExpertDownStaging"),
            rowBlock: rowBlock,
            hiddenSize: hiddenSize,
            intermediate: intermediate)
    }
}

struct PrefillGroupedRoutedMoEStreamedMetadataBuffers {
    let sortedPairs: MTLBuffer
}

struct PrefillStreamedTileArgumentBuffer {
    let buffer: MTLBuffer
}

public struct PrefillStreamedTileFetchResult {
    public let expertIDs: [Int]
    public let binding: PrefillStreamedTileBinding
    public let usedPlannedFetch: Bool
    public let plannedHits: Int
    public let plannedMissIndices: [Int]
    public let plannedAssignedSlots: [Int]
    public let plannedMissSlots: [Int]

    public init(expertIDs: [Int],
                binding: PrefillStreamedTileBinding,
                usedPlannedFetch: Bool,
                plannedHits: Int,
                plannedMissIndices: [Int],
                plannedAssignedSlots: [Int],
                plannedMissSlots: [Int]) {
        self.expertIDs = expertIDs
        self.binding = binding
        self.usedPlannedFetch = usedPlannedFetch
        self.plannedHits = plannedHits
        self.plannedMissIndices = plannedMissIndices
        self.plannedAssignedSlots = plannedAssignedSlots
        self.plannedMissSlots = plannedMissSlots
    }
}

/// A tile's fetch, begun but not yet awaited; `plan.assignedSlots` is valid
/// immediately, before `operation` completes.
public struct PrefillStreamedTileFetchBegin {
    public let expertIDs: [Int]
    public let plan: RoutedExpertFetchPlan
    public let operation: RoutedExpertLoadOperation

    public init(expertIDs: [Int], plan: RoutedExpertFetchPlan, operation: RoutedExpertLoadOperation) {
        self.expertIDs = expertIDs
        self.plan = plan
        self.operation = operation
    }
}

enum PrefillStreamedTileLifetimeError: Error, Equatable, CustomStringConvertible {
    case duplicateSlots(tileIndex: Int, slots: [Int])
    case slotReuseBeforeCompletion(tileIndex: Int, conflictingTileIndex: Int, slots: [Int])
    case completeWithoutInFlightTile(tileIndex: Int)

    public var description: String {
        switch self {
        case .duplicateSlots(let tileIndex, let slots):
            return "prefill streamed tile \(tileIndex) has duplicate planned slots \(slots)"
        case .slotReuseBeforeCompletion(let tileIndex, let conflictingTileIndex, let slots):
            return "prefill streamed tile \(tileIndex) would reuse planned slots \(slots) while tile \(conflictingTileIndex) is in flight"
        case .completeWithoutInFlightTile(let tileIndex):
            return "prefill streamed tile \(tileIndex) completed without a matching in-flight tile"
        }
    }
}

struct PrefillStreamedTileSlotLifetime: Sendable, Equatable {
    private var inFlightSlotsByTile: [Int: Set<Int>] = [:]

    init() {}

    mutating func begin(tileIndex: Int, plannedSlots: [Int]) throws {
        let slots = try normalizedSlots(tileIndex: tileIndex, plannedSlots: plannedSlots)
        for (otherTile, otherSlots) in inFlightSlotsByTile {
            let overlap = slots.intersection(otherSlots)
            if !overlap.isEmpty {
                throw PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
                    tileIndex: tileIndex,
                    conflictingTileIndex: otherTile,
                    slots: overlap.sorted())
            }
        }
        inFlightSlotsByTile[tileIndex] = slots
    }

    mutating func complete(tileIndex: Int) throws {
        guard inFlightSlotsByTile.removeValue(forKey: tileIndex) != nil else {
            throw PrefillStreamedTileLifetimeError.completeWithoutInFlightTile(tileIndex: tileIndex)
        }
    }

    private func normalizedSlots(tileIndex: Int, plannedSlots: [Int]) throws -> Set<Int> {
        var slots = Set<Int>()
        for slot in plannedSlots {
            guard slots.insert(slot).inserted else {
                throw PrefillStreamedTileLifetimeError.duplicateSlots(
                    tileIndex: tileIndex,
                    slots: plannedSlots.sorted())
            }
        }
        return slots
    }
}

struct PrefillGroupedRoutedMoEStreamedParams: Equatable, Sendable {
    var pairStart: UInt32
    var pairCount: UInt32
    var d: UInt32
    var routedIntermediate: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var liveExpertCount: UInt32
    var localExpert0: UInt32
    var localExpert1: UInt32
    var localExpert2: UInt32
    var localExpert3: UInt32
    var localExpert4: UInt32
    var localExpert5: UInt32
    var localExpert6: UInt32
    var localExpert7: UInt32
    var localExpert8: UInt32
    var localExpert9: UInt32
    var localExpert10: UInt32
    var localExpert11: UInt32
    var localExpert12: UInt32
    var localExpert13: UInt32
    var localExpert14: UInt32
    var localExpert15: UInt32
    var gateWOff: UInt32
    var gateSOff: UInt32
    var gateBOff: UInt32
    var upWOff: UInt32
    var upSOff: UInt32
    var upBOff: UInt32
    var downWOff: UInt32
    var downSOff: UInt32
    var downBOff: UInt32
    var gateABOff: UInt32
    var upABOff: UInt32
    var downABOff: UInt32

    init(pairStart: UInt32,
                pairCount: UInt32,
                d: UInt32,
                routedIntermediate: UInt32,
                topK: UInt32,
                hiddenStrideElements: UInt32,
                binding: PrefillStreamedTileBinding,
                offsets: MoEExpertOffsets) {
        var ids = Array(repeating: UInt32.max, count: 16)
        for (index, expert) in binding.expertIDs.enumerated() {
            ids[index] = UInt32(expert)
        }
        self.pairStart = pairStart
        self.pairCount = pairCount
        self.d = d
        self.routedIntermediate = routedIntermediate
        self.topK = topK
        self.hiddenStrideElements = hiddenStrideElements
        self.liveExpertCount = UInt32(binding.expertIDs.count)
        self.localExpert0 = ids[0]
        self.localExpert1 = ids[1]
        self.localExpert2 = ids[2]
        self.localExpert3 = ids[3]
        self.localExpert4 = ids[4]
        self.localExpert5 = ids[5]
        self.localExpert6 = ids[6]
        self.localExpert7 = ids[7]
        self.localExpert8 = ids[8]
        self.localExpert9 = ids[9]
        self.localExpert10 = ids[10]
        self.localExpert11 = ids[11]
        self.localExpert12 = ids[12]
        self.localExpert13 = ids[13]
        self.localExpert14 = ids[14]
        self.localExpert15 = ids[15]
        self.gateWOff = offsets.gateWOff
        self.gateSOff = offsets.gateSOff
        self.gateBOff = offsets.gateBOff
        self.upWOff = offsets.upWOff
        self.upSOff = offsets.upSOff
        self.upBOff = offsets.upBOff
        self.downWOff = offsets.downWOff
        self.downSOff = offsets.downSOff
        self.downBOff = offsets.downBOff
        self.gateABOff = offsets.gateABOff
        self.upABOff = offsets.upABOff
        self.downABOff = offsets.downABOff
    }
}

public struct PrefillStreamedTileBinding: Sendable, Equatable {
    public let expertIDs: [Int]
    public let views: [TensorView]

    public init(expertIDs: [Int], views: [TensorView]) throws {
        guard !expertIDs.isEmpty else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding("tile binding must include at least one expert")
        }
        guard expertIDs.count <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile binding has \(expertIDs.count) experts; maximum is 16")
        }
        guard expertIDs.count == views.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "expertIDs.count \(expertIDs.count) != views.count \(views.count)")
        }
        var seen = Set<Int>()
        for expert in expertIDs {
            guard expert >= 0 else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "expert id \(expert) must be non-negative")
            }
            guard seen.insert(expert).inserted else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "duplicate expert id \(expert) in tile binding")
            }
        }
        self.expertIDs = expertIDs
        self.views = views
    }

    public func localSlot(for expert: UInt32) -> Int? {
        expertIDs.firstIndex(of: Int(expert))
    }

    public static func expertIDs(forTile tileIndex: Int,
                                 routes: PrefillMoEGroupedRoutes) throws -> [Int] {
        guard routes.tiles.indices.contains(tileIndex) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile index \(tileIndex) is out of range")
        }
        let tile = routes.tiles[tileIndex]
        let groupStart = Int(tile.groupStart)
        let groupCount = Int(tile.groupCount)
        guard groupCount > 0, groupCount <= 16 else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile has \(groupCount) live experts; expected 1...16")
        }
        guard groupStart >= 0, groupStart + groupCount <= routes.groups.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "tile group range \(groupStart)..<\(groupStart + groupCount) exceeds \(routes.groups.count)")
        }
        return routes.groups[groupStart..<(groupStart + groupCount)].map { Int($0.expert) }
    }

    /// The unsplit plan-fetch-bind reference the tests exercise; production
    /// runs `beginFetchForTile` and `bindingForCompletedFetch` instead.
    public static func fetchBindingForTile(model: Model,
                                           layer: Int,
                                           tileIndex: Int,
                                           routes: PrefillMoEGroupedRoutes,
                                           plannedFetch: RoutedExpertFetchPlan? = nil,
                                           avoidingSlots: Set<Int> = [],
                                           protectedExperts: [Bool]? = nil) async throws
        -> PrefillStreamedTileFetchResult {
        let expertIDs = try expertIDs(forTile: tileIndex, routes: routes)
        let plan = try plannedFetch ?? model.planRoutedExperts(layer: layer,
                                                               experts: expertIDs,
                                                               avoidingSlots: avoidingSlots,
                                                               protectedExperts: protectedExperts)
        let views: [TensorView]
        let usedPlannedFetch: Bool
        let plannedHits: Int
        let plannedMissIndices: [Int]
        let plannedAssignedSlots: [Int]
        let plannedMissSlots: [Int]
        if let plan {
            guard plan.layer == layer, plan.experts == expertIDs else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "preplanned fetch does not match tile \(tileIndex)")
            }
            views = try await model.fetchRoutedExperts(plan: plan)
            usedPlannedFetch = true
            plannedHits = plan.hits
            plannedMissIndices = plan.misses
            plannedAssignedSlots = plan.assignedSlots
            plannedMissSlots = plan.misses.map { plan.assignedSlots[$0] }
        } else {
            views = try await model.fetchRoutedExperts(layer: layer, experts: expertIDs)
            usedPlannedFetch = false
            plannedHits = 0
            plannedMissIndices = []
            plannedAssignedSlots = []
            plannedMissSlots = []
        }
        let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
        return PrefillStreamedTileFetchResult(expertIDs: expertIDs,
                                             binding: binding,
                                             usedPlannedFetch: usedPlannedFetch,
                                             plannedHits: plannedHits,
                                             plannedMissIndices: plannedMissIndices,
                                             plannedAssignedSlots: plannedAssignedSlots,
                                             plannedMissSlots: plannedMissSlots)
    }

    /// The plan-and-begin half of `fetchBindingForTile`, split out so a
    /// caller can begin tile N+1's fetch before awaiting tile N's: the plan
    /// (and its `assignedSlots`) is available synchronously, before the I/O
    /// this starts has completed.
    public static func beginFetchForTile(model: Model,
                                         layer: Int,
                                         tileIndex: Int,
                                         routes: PrefillMoEGroupedRoutes,
                                         plannedFetch: RoutedExpertFetchPlan? = nil,
                                         avoidingSlots: Set<Int> = [],
                                         protectedExperts: [Bool]? = nil) throws
        -> PrefillStreamedTileFetchBegin {
        let expertIDs = try expertIDs(forTile: tileIndex, routes: routes)
        guard let plan = try plannedFetch ?? model.planRoutedExperts(layer: layer,
                                                                     experts: expertIDs,
                                                                     avoidingSlots: avoidingSlots,
                                                                     protectedExperts: protectedExperts) else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "no routed expert plan available for tile \(tileIndex)")
        }
        guard plan.layer == layer, plan.experts == expertIDs else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "preplanned fetch does not match tile \(tileIndex)")
        }
        let operation = try model.beginFetchRoutedExperts(plan: plan)
        return PrefillStreamedTileFetchBegin(expertIDs: expertIDs, plan: plan, operation: operation)
    }

    /// The binding-from-completed-views half of `fetchBindingForTile`: turns
    /// a begun fetch's awaited views into the same result the unsplit call
    /// produces.
    public static func bindingForCompletedFetch(begin: PrefillStreamedTileFetchBegin,
                                                views: [TensorView]) throws
        -> PrefillStreamedTileFetchResult {
        let binding = try PrefillStreamedTileBinding(expertIDs: begin.expertIDs, views: views)
        return PrefillStreamedTileFetchResult(
            expertIDs: begin.expertIDs,
            binding: binding,
            usedPlannedFetch: true,
            plannedHits: begin.plan.hits,
            plannedMissIndices: begin.plan.misses,
            plannedAssignedSlots: begin.plan.assignedSlots,
            plannedMissSlots: begin.plan.misses.map { begin.plan.assignedSlots[$0] })
    }

    public func validateCoversPairs(_ pairs: [PrefillTokenExpertPair],
                                    pairStart: Int,
                                    pairCount: Int) throws {
        guard pairStart >= 0, pairCount >= 0, pairStart + pairCount <= pairs.count else {
            throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                "pair range \(pairStart)..<\(pairStart + pairCount) exceeds \(pairs.count)")
        }
        for pair in pairs[pairStart..<(pairStart + pairCount)] {
            guard localSlot(for: pair.expert) != nil else {
                throw PrefillGroupedRoutedMoEError.invalidStreamedTileBinding(
                    "route expert \(pair.expert) is not bound in tile")
            }
        }
    }

    public static func == (lhs: PrefillStreamedTileBinding,
                           rhs: PrefillStreamedTileBinding) -> Bool {
        guard lhs.expertIDs == rhs.expertIDs, lhs.views.count == rhs.views.count else {
            return false
        }
        for index in lhs.views.indices {
            let l = lhs.views[index]
            let r = rhs.views[index]
            guard l.buffer === r.buffer,
                  l.offset == r.offset,
                  l.length == r.length,
                  l.scaleOffset == r.scaleOffset,
                  l.scaleLength == r.scaleLength,
                  l.biasOffset == r.biasOffset,
                  l.biasLength == r.biasLength,
                  l.shape == r.shape,
                  l.dtype == r.dtype else {
                return false
            }
        }
        return true
    }
}

enum PrefillGroupedRoutedMoEError: Error, Equatable, CustomStringConvertible {
    case invalidStreamedTileBinding(String)
    case allocationFailed(String)
    case stagingTooSmall(String)

    public var description: String {
        switch self {
        case .invalidStreamedTileBinding(let reason):
            return "invalid streamed tile binding: \(reason)"
        case .allocationFailed(let label):
            return "failed to allocate \(label)"
        case .stagingTooSmall(let detail):
            return "prefill routed expert staging is too small: \(detail)"
        }
    }
}

final class PrefillGroupedRoutedMoE {
    private let batchedPhase1PSO: MTLComputePipelineState
    private let batchedDownPSO: MTLComputePipelineState
    private let gatherRowsPSO: MTLComputePipelineState
    private let scatterRowsPSO: MTLComputePipelineState
    private let groupedGatherRowsPSO: MTLComputePipelineState
    private let groupedScatterRowsPSO: MTLComputePipelineState
    private let activationPSO: MTLComputePipelineState
    private let streamedArgEncoder: MTLArgumentEncoder
    private let weightBits: Int
    private let supportsMatrixPath: Bool

    /// Below this an expert's pairs do not fill one GEMM tile, so the scalar
    /// microbatch path still wins.
    static let matrixPathMinimumRows = 32

    /// The MPP instance `encodeExpertGEMMs` may use, or nil when this runtime
    /// belongs on the scalar path: no MPP object, a pipeline that failed to
    /// compile, a bit width the GEMM cannot read, an epilogue the GEMM path
    /// cannot reproduce (gpt-oss additive biases, clamped SwiGLU), or a
    /// reduction length the GEMM would reject mid-tile — `encode` requires a
    /// `k` that is a whole number of quantization groups, and the gate/up and
    /// down GEMMs reduce over `d` and `intermediate` respectively.
    func matrixPath(for mpp: MPPPrefillInt4QMM?,
                    d: Int,
                    intermediate: Int) -> MPPPrefillInt4QMM? {
        guard supportsMatrixPath,
              let mpp,
              mpp.isAvailable,
              mpp.weightBits == weightBits,
              d > 0,
              intermediate > 0,
              d.isMultiple(of: MPPPrefillInt4QMM.tileK),
              intermediate.isMultiple(of: MPPPrefillInt4QMM.tileK) else { return nil }
        return mpp
    }

    /// The grouped kernel reads the tile argument buffer this module encodes,
    /// so the two argument layouts have to agree byte for byte.
    func groupedPathAvailable(for mpp: MPPPrefillInt4QMM) -> Bool {
        mpp.groupedAvailable
            && mpp.groupedArgumentEncodedLength == streamedArgEncoder.encodedLength
    }

    func makeStreamedArgumentBuffer(device: MTLDevice,
                                           binding: PrefillStreamedTileBinding) throws -> PrefillStreamedTileArgumentBuffer {
        guard let buffer = device.makeBuffer(length: streamedArgEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill streamed expert argument buffer")
        }
        buffer.label = "prefill.groupedMoe.streamedArgumentBuffer"

        streamedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            streamedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }

        return PrefillStreamedTileArgumentBuffer(buffer: buffer)
    }

    init(context: MetalContext,
         siluActivation: Bool = false,
         weightBits: Int = 4,
         expertAdditiveBiases: Bool = false,
         clampedSwiGLU: Bool = false) throws {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.supportsMatrixPath = weightBits == 4 && !expertAdditiveBiases && !clampedSwiGLU
        let biasConstants: [MetalFunctionConstant] = expertAdditiveBiases
            ? [MetalFunctionConstant(index: 120, value: .bool(true))]
            : []
        var activationConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 78,
                                  value: .uint32(UInt32(weightBits)))
        ] + biasConstants
        if siluActivation {
            activationConstants.append(
                MetalFunctionConstant(index: 77, value: .bool(true)))
        }
        if clampedSwiGLU {
            activationConstants.append(
                MetalFunctionConstant(index: 121, value: .bool(true)))
        }
        self.batchedPhase1PSO = try context.pipeline(
            "prefill_grouped_routed_moe_batched_phase1",
            constants: activationConstants)
        self.batchedDownPSO = try context.pipeline(
            "prefill_grouped_routed_moe_batched_down",
            constants: [MetalFunctionConstant(index: 78,
                                               value: .uint32(UInt32(weightBits)))]
                + biasConstants)
        self.gatherRowsPSO = try context.pipeline("prefill_routed_gather_rows")
        self.scatterRowsPSO = try context.pipeline("prefill_routed_scatter_rows")
        self.groupedGatherRowsPSO = try context.pipeline("prefill_routed_gather_rows_grouped")
        self.groupedScatterRowsPSO = try context.pipeline("prefill_routed_scatter_rows_grouped")
        self.activationPSO = try context.pipeline(
            siluActivation ? "silu_mul_fp16" : "gelu_mul_fp16")
        guard let streamedFn = context.library.makeFunction(name: "prefill_grouped_routed_moe_batched_phase1") else {
            throw MetalError.missingFunction("prefill_grouped_routed_moe_batched_phase1")
        }
        self.streamedArgEncoder = streamedFn.makeArgumentEncoder(
            bufferIndex: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
    }

    func makeStreamedMetadataBuffers(
        device: MTLDevice,
        routes: PrefillMoEGroupedRoutes
    ) throws -> PrefillGroupedRoutedMoEStreamedMetadataBuffers {
        let bytes = routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride
        // K13: sortedPairs can be empty (zero routed pairs for the chunk).
        // withUnsafeBufferPointer on an empty array yields a nil baseAddress,
        // so guard the empty case and hand out a zero-length buffer instead of
        // force-unwrapping. The batched kernels guard on pair_count == 0.
        guard !routes.sortedPairs.isEmpty else {
            guard let empty = device.makeBuffer(length: 0, options: .storageModeShared) else {
                throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
            }
            return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: empty)
        }
        guard let sortedPairs = routes.sortedPairs.withUnsafeBufferPointer({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: bytes,
                              options: .storageModeShared)
        }) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed("prefill sorted route pairs")
        }
        return PrefillGroupedRoutedMoEStreamedMetadataBuffers(sortedPairs: sortedPairs)
    }

    @discardableResult
    func encodeStreamedBatched(commandBuffer: MTLCommandBuffer,
                                      hidden: MTLBuffer,
                                      hiddenOffset: Int = 0,
                                      sortedPairs: MTLBuffer,
                                      sortedPairsOffset: Int = 0,
                                      routePartials: MTLBuffer,
                                      routePartialsOffset: Int = 0,
                                      gateUpActScratch: MTLBuffer,
                                      gateUpActScratchOffset: Int = 0,
                                      downScratch: MTLBuffer,
                                      downScratchOffset: Int = 0,
                                      argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                      binding: PrefillStreamedTileBinding,
                                      params: PrefillGroupedRoutedMoEStreamedParams,
                                      pairMicrobatchRows: Int = 32) throws -> Int {
        guard params.pairCount > 0,
              params.liveExpertCount == UInt32(binding.views.count),
              pairMicrobatchRows > 0 else { return 0 }
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            guard let enc1 = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc1.setComputePipelineState(batchedPhase1PSO)
            enc1.setBuffer(hidden, offset: hiddenOffset, index: PrefillGroupedRoutedMoEBufferIndex.hidden)
            enc1.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc1.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                          index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
            enc1.setBuffer(argumentBuffer.buffer, offset: 0,
                          index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc1.setBytes(&p,
                         length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                         index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                enc1.useResource(view.buffer, usage: .read)
            }
            enc1.dispatchThreads(MTLSize(width: Int(p.routedIntermediate),
                                        height: Int(p.pairCount),
                                        depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc1.endEncoding()

            guard let enc2 = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalError.commandEncoderFailed
            }
            enc2.setComputePipelineState(batchedDownPSO)
            enc2.setBuffer(sortedPairs, offset: sortedPairsOffset, index: PrefillGroupedRoutedMoEBufferIndex.sortedPairs)
            enc2.setBuffer(routePartials, offset: routePartialsOffset,
                          index: PrefillGroupedRoutedMoEBufferIndex.routePartials)
            enc2.setBuffer(gateUpActScratch, offset: gateUpActScratchOffset,
                          index: PrefillGroupedRoutedMoEBufferIndex.gateUpActScratch)
            enc2.setBuffer(downScratch, offset: downScratchOffset,
                          index: PrefillGroupedRoutedMoEBufferIndex.downScratch)
            enc2.setBuffer(argumentBuffer.buffer, offset: 0,
                          index: PrefillGroupedRoutedMoEBufferIndex.expertArgumentState)
            enc2.setBytes(&p,
                         length: MemoryLayout<PrefillGroupedRoutedMoEStreamedParams>.stride,
                         index: PrefillGroupedRoutedMoEBufferIndex.params)
            for view in binding.views {
                enc2.useResource(view.buffer, usage: .read)
            }
            enc2.dispatchThreads(MTLSize(width: Int(p.d),
                                        height: Int(p.pairCount),
                                        depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc2.endEncoding()

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
    }

    /// Per expert in the tile: gather its pair rows into contiguous staging,
    /// run gate / up / down as GEMMs over the packed expert slot, and scatter
    /// the result into the same per-pair `routePartials` rows the scalar path
    /// writes, unweighted — the router weight stays in the reduce. Experts
    /// with fewer than `minimumRows` pairs come back untouched for the scalar
    /// path.
    func encodeExpertGEMMs(commandBuffer: MTLCommandBuffer,
                           mpp: MPPPrefillInt4QMM,
                           hidden: MTLBuffer,
                           hiddenOffset: Int = 0,
                           sortedPairs: MTLBuffer,
                           sortedPairsOffset: Int = 0,
                           routePartials: MTLBuffer,
                           routePartialsOffset: Int = 0,
                           binding: PrefillStreamedTileBinding,
                           ranges: [PrefillExpertPairRange],
                           staging: PrefillExpertStaging,
                           params: PrefillGroupedRoutedMoEStreamedParams,
                           minimumRows: Int = PrefillGroupedRoutedMoE.matrixPathMinimumRows) throws
        -> [PrefillExpertPairRange] {
        guard staging.rowBlock > 0,
              staging.hiddenSize >= Int(params.d),
              staging.intermediate >= Int(params.routedIntermediate) else {
            throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                "\(staging.rowBlock) rows of \(staging.hiddenSize)/\(staging.intermediate)"
                    + " cannot hold \(params.d)/\(params.routedIntermediate)")
        }
        var leftovers: [PrefillExpertPairRange] = []
        for range in ranges {
            guard range.pairCount >= minimumRows,
                  let slot = binding.localSlot(for: range.expert) else {
                leftovers.append(range)
                continue
            }
            var consumed = 0
            while consumed < range.pairCount {
                let rows = min(staging.rowBlock, range.pairCount - consumed)
                try encodeExpertRowBlock(
                    commandBuffer: commandBuffer,
                    mpp: mpp,
                    hidden: hidden,
                    hiddenOffset: hiddenOffset,
                    sortedPairs: sortedPairs,
                    sortedPairsOffset: sortedPairsOffset,
                    routePartials: routePartials,
                    routePartialsOffset: routePartialsOffset,
                    view: binding.views[slot],
                    staging: staging,
                    params: params,
                    pairStart: range.pairStart + consumed,
                    rows: rows)
                consumed += rows
            }
        }
        return leftovers
    }

    private func encodeExpertRowBlock(commandBuffer: MTLCommandBuffer,
                                      mpp: MPPPrefillInt4QMM,
                                      hidden: MTLBuffer,
                                      hiddenOffset: Int,
                                      sortedPairs: MTLBuffer,
                                      sortedPairsOffset: Int,
                                      routePartials: MTLBuffer,
                                      routePartialsOffset: Int,
                                      view: TensorView,
                                      staging: PrefillExpertStaging,
                                      params: PrefillGroupedRoutedMoEStreamedParams,
                                      pairStart: Int,
                                      rows: Int) throws {
        let d = Int(params.d)
        let f = Int(params.routedIntermediate)
        var block = PrefillRoutedRowBlockParams(pairStart: UInt32(pairStart),
                                                rows: UInt32(rows),
                                                d: UInt32(d),
                                                topK: params.topK,
                                                hiddenStrideElements: params.hiddenStrideElements)
        try encodeRowBlockCopy(pso: gatherRowsPSO,
                               commandBuffer: commandBuffer,
                               source: hidden,
                               sourceOffset: hiddenOffset,
                               sortedPairs: sortedPairs,
                               sortedPairsOffset: sortedPairsOffset,
                               destination: staging.hidden,
                               destinationOffset: 0,
                               params: &block,
                               width: d)
        try project(mpp: mpp, on: commandBuffer, view: view,
                    weightsOffset: Int(params.gateWOff),
                    scalesOffset: Int(params.gateSOff),
                    biasesOffset: Int(params.gateBOff),
                    x: staging.hidden, y: staging.gate, m: rows, n: f, k: d)
        try project(mpp: mpp, on: commandBuffer, view: view,
                    weightsOffset: Int(params.upWOff),
                    scalesOffset: Int(params.upSOff),
                    biasesOffset: Int(params.upBOff),
                    x: staging.hidden, y: staging.up, m: rows, n: f, k: d)
        try encodeActivation(commandBuffer: commandBuffer,
                             gate: staging.gate,
                             up: staging.up,
                             count: rows * f)
        try project(mpp: mpp, on: commandBuffer, view: view,
                    weightsOffset: Int(params.downWOff),
                    scalesOffset: Int(params.downSOff),
                    biasesOffset: Int(params.downBOff),
                    x: staging.gate, y: staging.down, m: rows, n: d, k: f)
        try encodeRowBlockCopy(pso: scatterRowsPSO,
                               commandBuffer: commandBuffer,
                               source: staging.down,
                               sourceOffset: 0,
                               sortedPairs: sortedPairs,
                               sortedPairsOffset: sortedPairsOffset,
                               destination: routePartials,
                               destinationOffset: routePartialsOffset,
                               params: &block,
                               width: d)
    }

    /// Six dispatches per wave whatever its expert count.
    func encodeGroupedExpertGEMMs(commandBuffer: MTLCommandBuffer,
                                  mpp: MPPPrefillInt4QMM,
                                  hidden: MTLBuffer,
                                  hiddenOffset: Int = 0,
                                  sortedPairs: MTLBuffer,
                                  sortedPairsOffset: Int = 0,
                                  routePartials: MTLBuffer,
                                  routePartialsOffset: Int = 0,
                                  binding: PrefillStreamedTileBinding,
                                  argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                  waves: [PrefillRoutedExpertWave],
                                  staging: PrefillExpertStaging,
                                  params: PrefillGroupedRoutedMoEStreamedParams,
                                  rowTile: MPPPrefillInt4QMM.GroupedRowTile = .m64,
                                  tailTile: Int = 0) throws {
        if tailTile != 0 {
            guard rowTile == .m64 else {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "a tail tile packs 64-row bodies; rowTile \(rowTile.rawValue) cannot combine with it")
            }
            try encodeBodyTailExpertGEMMs(commandBuffer: commandBuffer, mpp: mpp,
                                          hidden: hidden, hiddenOffset: hiddenOffset,
                                          sortedPairs: sortedPairs, sortedPairsOffset: sortedPairsOffset,
                                          routePartials: routePartials, routePartialsOffset: routePartialsOffset,
                                          binding: binding, argumentBuffer: argumentBuffer,
                                          waves: waves, staging: staging, params: params)
            return
        }
        guard groupedPathAvailable(for: mpp) else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(
                reason: "grouped MPP path unavailable or its argument layout differs")
        }
        guard staging.rowBlock > 0,
              staging.hiddenSize >= Int(params.d),
              staging.intermediate >= Int(params.routedIntermediate) else {
            throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                "\(staging.rowBlock) rows of \(staging.hiddenSize)/\(staging.intermediate)"
                    + " cannot hold \(params.d)/\(params.routedIntermediate)")
        }
        let d = Int(params.d)
        let f = Int(params.routedIntermediate)
        for wave in waves {
            guard wave.paddedRows > 0, wave.paddedRows <= staging.rowBlock else {
                throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                    "wave of \(wave.paddedRows) padded rows exceeds \(staging.rowBlock) staging rows")
            }
            let rowTileBlock = Self.rowTileTable(for: wave, rowTile: rowTile.rawValue)
            var groupedParams = PrefillRoutedGroupedParams(
                paddedRows: UInt32(wave.paddedRows),
                d: UInt32(d),
                topK: params.topK,
                hiddenStrideElements: params.hiddenStrideElements,
                rowTile: UInt32(rowTile.rawValue))
            try encodeGroupedRowCopy(pso: groupedGatherRowsPSO,
                                     commandBuffer: commandBuffer,
                                     source: hidden,
                                     sourceOffset: hiddenOffset,
                                     sortedPairs: sortedPairs,
                                     sortedPairsOffset: sortedPairsOffset,
                                     destination: staging.hidden,
                                     destinationOffset: 0,
                                     params: &groupedParams,
                                     blocks: wave.blocks,
                                     rowTileBlock: rowTileBlock)
            try mpp.encodeGrouped(commandBuffer: commandBuffer,
                                  experts: argumentBuffer.buffer,
                                  expertViews: binding.views,
                                  blocks: wave.blocks,
                                  rowTileBlock: rowTileBlock,
                                  weightsOffset: Int(params.gateWOff),
                                  scalesOffset: Int(params.gateSOff),
                                  biasesOffset: Int(params.gateBOff),
                                  x: staging.hidden, y: staging.gate,
                                  paddedRows: wave.paddedRows, n: f, k: d, rowTile: rowTile)
            try mpp.encodeGrouped(commandBuffer: commandBuffer,
                                  experts: argumentBuffer.buffer,
                                  expertViews: binding.views,
                                  blocks: wave.blocks,
                                  rowTileBlock: rowTileBlock,
                                  weightsOffset: Int(params.upWOff),
                                  scalesOffset: Int(params.upSOff),
                                  biasesOffset: Int(params.upBOff),
                                  x: staging.hidden, y: staging.up,
                                  paddedRows: wave.paddedRows, n: f, k: d, rowTile: rowTile)
            try encodeActivation(commandBuffer: commandBuffer,
                                 gate: staging.gate,
                                 up: staging.up,
                                 count: wave.paddedRows * f)
            try mpp.encodeGrouped(commandBuffer: commandBuffer,
                                  experts: argumentBuffer.buffer,
                                  expertViews: binding.views,
                                  blocks: wave.blocks,
                                  rowTileBlock: rowTileBlock,
                                  weightsOffset: Int(params.downWOff),
                                  scalesOffset: Int(params.downSOff),
                                  biasesOffset: Int(params.downBOff),
                                  x: staging.gate, y: staging.down,
                                  paddedRows: wave.paddedRows, n: d, k: f, rowTile: rowTile)
            try encodeGroupedRowCopy(pso: groupedScatterRowsPSO,
                                     commandBuffer: commandBuffer,
                                     source: staging.down,
                                     sourceOffset: 0,
                                     sortedPairs: sortedPairs,
                                     sortedPairsOffset: sortedPairsOffset,
                                     destination: routePartials,
                                     destinationOffset: routePartialsOffset,
                                     params: &groupedParams,
                                     blocks: wave.blocks,
                                     rowTileBlock: rowTileBlock)
        }
    }

    /// The tail-tile path: the gather and scatter walk the wave at 32-row
    /// granularity; each GEMM is one 64-row dispatch over the body region and
    /// one 32-row dispatch over the tail region, either skipped when empty.
    private func encodeBodyTailExpertGEMMs(commandBuffer: MTLCommandBuffer,
                                           mpp: MPPPrefillInt4QMM,
                                           hidden: MTLBuffer,
                                           hiddenOffset: Int,
                                           sortedPairs: MTLBuffer,
                                           sortedPairsOffset: Int,
                                           routePartials: MTLBuffer,
                                           routePartialsOffset: Int,
                                           binding: PrefillStreamedTileBinding,
                                           argumentBuffer: PrefillStreamedTileArgumentBuffer,
                                           waves: [PrefillRoutedExpertWave],
                                           staging: PrefillExpertStaging,
                                           params: PrefillGroupedRoutedMoEStreamedParams) throws {
        guard groupedPathAvailable(for: mpp),
              mpp.groupedRowTile32Available(forK: Int(params.d)),
              mpp.groupedRowTile32Available(forK: Int(params.routedIntermediate)) else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(
                reason: "grouped MPP path or its 32-row instantiation unavailable for d \(params.d) / f \(params.routedIntermediate)")
        }
        guard staging.rowBlock > 0,
              staging.hiddenSize >= Int(params.d),
              staging.intermediate >= Int(params.routedIntermediate) else {
            throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                "\(staging.rowBlock) rows of \(staging.hiddenSize)/\(staging.intermediate)"
                    + " cannot hold \(params.d)/\(params.routedIntermediate)")
        }
        let d = Int(params.d)
        let f = Int(params.routedIntermediate)
        for wave in waves {
            guard wave.paddedRows > 0, wave.paddedRows <= staging.rowBlock else {
                throw PrefillGroupedRoutedMoEError.stagingTooSmall(
                    "wave of \(wave.paddedRows) padded rows exceeds \(staging.rowBlock) staging rows")
            }
            let tables = Self.rowTileTables(for: wave)
            let tailBlocks = Array(wave.blocks.suffix(wave.tailRows / Self.groupedTailRowTile))
            let bodyBlocks = Array(wave.blocks.prefix(wave.blocks.count - tailBlocks.count))
            let tailTable = tables.tail.map { $0 - UInt32(bodyBlocks.count) }
            var groupedParams = PrefillRoutedGroupedParams(
                paddedRows: UInt32(wave.paddedRows),
                d: UInt32(d),
                topK: params.topK,
                hiddenStrideElements: params.hiddenStrideElements,
                rowTile: UInt32(Self.groupedTailRowTile))
            try encodeGroupedRowCopy(pso: groupedGatherRowsPSO, commandBuffer: commandBuffer,
                                     source: hidden, sourceOffset: hiddenOffset,
                                     sortedPairs: sortedPairs, sortedPairsOffset: sortedPairsOffset,
                                     destination: staging.hidden, destinationOffset: 0,
                                     params: &groupedParams, blocks: wave.blocks, rowTileBlock: tables.gather)
            // Body and tail write disjoint rows, so both dispatches share one
            // encoder: a tail wave issues the same six encoders as a plain one.
            func gemm(_ wOff: UInt32, _ sOff: UInt32, _ bOff: UInt32,
                      x: MTLBuffer, y: MTLBuffer, n: Int, k: Int) throws {
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw MetalError.commandEncoderFailed
                }
                defer { encoder.endEncoding() }
                if !bodyBlocks.isEmpty {
                    try mpp.encodeGrouped(commandBuffer: commandBuffer, experts: argumentBuffer.buffer,
                                          expertViews: binding.views, blocks: bodyBlocks,
                                          rowTileBlock: tables.body,
                                          weightsOffset: Int(wOff), scalesOffset: Int(sOff), biasesOffset: Int(bOff),
                                          x: x, y: y, paddedRows: wave.paddedRows, n: n, k: k,
                                          rowTile: .m64, regionOrigin: 0, regionRows: wave.bodyPaddedRows,
                                          encoder: encoder)
                }
                if !tailBlocks.isEmpty {
                    try mpp.encodeGrouped(commandBuffer: commandBuffer, experts: argumentBuffer.buffer,
                                          expertViews: binding.views, blocks: tailBlocks,
                                          rowTileBlock: tailTable,
                                          weightsOffset: Int(wOff), scalesOffset: Int(sOff), biasesOffset: Int(bOff),
                                          x: x, y: y, paddedRows: wave.paddedRows, n: n, k: k,
                                          rowTile: .m32, regionOrigin: wave.bodyPaddedRows, regionRows: wave.tailRows,
                                          encoder: encoder)
                }
            }
            try gemm(params.gateWOff, params.gateSOff, params.gateBOff, x: staging.hidden, y: staging.gate, n: f, k: d)
            try gemm(params.upWOff, params.upSOff, params.upBOff, x: staging.hidden, y: staging.up, n: f, k: d)
            try encodeActivation(commandBuffer: commandBuffer, gate: staging.gate, up: staging.up,
                                 count: wave.paddedRows * f)
            try gemm(params.downWOff, params.downSOff, params.downBOff, x: staging.gate, y: staging.down, n: d, k: f)
            try encodeGroupedRowCopy(pso: groupedScatterRowsPSO, commandBuffer: commandBuffer,
                                     source: staging.down, sourceOffset: 0,
                                     sortedPairs: sortedPairs, sortedPairsOffset: sortedPairsOffset,
                                     destination: routePartials, destinationOffset: routePartialsOffset,
                                     params: &groupedParams, blocks: wave.blocks, rowTileBlock: tables.gather)
        }
    }

    private func encodeGroupedRowCopy(pso: MTLComputePipelineState,
                                      commandBuffer: MTLCommandBuffer,
                                      source: MTLBuffer,
                                      sourceOffset: Int,
                                      sortedPairs: MTLBuffer,
                                      sortedPairsOffset: Int,
                                      destination: MTLBuffer,
                                      destinationOffset: Int,
                                      params: inout PrefillRoutedGroupedParams,
                                      blocks: [PrefillRoutedExpertBlock],
                                      rowTileBlock: [UInt32]) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(source, offset: sourceOffset,
                          index: PrefillRoutedRowBlockBufferIndex.source)
        encoder.setBuffer(sortedPairs, offset: sortedPairsOffset,
                          index: PrefillRoutedRowBlockBufferIndex.sortedPairs)
        encoder.setBuffer(destination, offset: destinationOffset,
                          index: PrefillRoutedRowBlockBufferIndex.destination)
        encoder.setBytes(&params,
                         length: MemoryLayout<PrefillRoutedGroupedParams>.stride,
                         index: PrefillRoutedRowBlockBufferIndex.params)
        blocks.withUnsafeBufferPointer { table in
            encoder.setBytes(table.baseAddress!,
                             length: table.count * MemoryLayout<PrefillRoutedExpertBlock>.stride,
                             index: PrefillRoutedRowBlockBufferIndex.blocks)
        }
        rowTileBlock.withUnsafeBufferPointer { table in
            encoder.setBytes(table.baseAddress!,
                             length: table.count * MemoryLayout<UInt32>.stride,
                             index: PrefillRoutedRowBlockBufferIndex.rowTileBlock)
        }
        encoder.dispatchThreads(MTLSize(width: Int(params.d), height: Int(params.paddedRows), depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        encoder.endEncoding()
    }

    private func project(mpp: MPPPrefillInt4QMM,
                         on commandBuffer: MTLCommandBuffer,
                         view: TensorView,
                         weightsOffset: Int,
                         scalesOffset: Int,
                         biasesOffset: Int,
                         x: MTLBuffer,
                         y: MTLBuffer,
                         m: Int,
                         n: Int,
                         k: Int) throws {
        let base = Int(view.offset)
        try mpp.encode(commandBuffer: commandBuffer,
                       weights: view.buffer, weightsOffset: base + weightsOffset,
                       scales: view.buffer, scalesOffset: base + scalesOffset,
                       biases: view.buffer, biasesOffset: base + biasesOffset,
                       x: x,
                       y: y,
                       m: m,
                       n: n,
                       k: k,
                       required: true)
    }

    /// The activation folds back into the gate block, as the shared expert's
    /// chunk path does: the kernel reads and writes only its own element.
    private func encodeActivation(commandBuffer: MTLCommandBuffer,
                                  gate: MTLBuffer,
                                  up: MTLBuffer,
                                  count: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(activationPSO)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        encoder.setBuffer(gate, offset: 0, index: 2)
        var elements = UInt32(count)
        encoder.setBytes(&elements, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(activationPSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeRowBlockCopy(pso: MTLComputePipelineState,
                                    commandBuffer: MTLCommandBuffer,
                                    source: MTLBuffer,
                                    sourceOffset: Int,
                                    sortedPairs: MTLBuffer,
                                    sortedPairsOffset: Int,
                                    destination: MTLBuffer,
                                    destinationOffset: Int,
                                    params: inout PrefillRoutedRowBlockParams,
                                    width: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(source, offset: sourceOffset,
                          index: PrefillRoutedRowBlockBufferIndex.source)
        encoder.setBuffer(sortedPairs, offset: sortedPairsOffset,
                          index: PrefillRoutedRowBlockBufferIndex.sortedPairs)
        encoder.setBuffer(destination, offset: destinationOffset,
                          index: PrefillRoutedRowBlockBufferIndex.destination)
        encoder.setBytes(&params,
                         length: MemoryLayout<PrefillRoutedRowBlockParams>.stride,
                         index: PrefillRoutedRowBlockBufferIndex.params)
        encoder.dispatchThreads(MTLSize(width: width, height: Int(params.rows), depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        encoder.endEncoding()
    }
}
