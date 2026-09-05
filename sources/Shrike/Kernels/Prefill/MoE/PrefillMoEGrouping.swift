import Foundation

@frozen
public struct PrefillMoEGroup: Equatable, Sendable {
    public var expert: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(expert: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.expert = expert
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

@frozen
public struct PrefillMoETile: Equatable, Sendable {
    public var groupStart: UInt32
    public var groupCount: UInt32
    public var pairStart: UInt32
    public var pairCount: UInt32

    public init(groupStart: UInt32, groupCount: UInt32, pairStart: UInt32, pairCount: UInt32) {
        self.groupStart = groupStart
        self.groupCount = groupCount
        self.pairStart = pairStart
        self.pairCount = pairCount
    }
}

public struct PrefillMoEGroupedRoutes: Equatable, Sendable {
    public let sortedPairs: [PrefillTokenExpertPair]
    public let perExpertOffsets: [UInt32]
    public let perExpertCounts: [UInt32]
    public let groups: [PrefillMoEGroup]
    public let tiles: [PrefillMoETile]
    public let queryCount: Int

    public var maxPairsPerExpert: Int {
        groups.map { Int($0.pairCount) }.max() ?? 0
    }

    public var maxPairsPerTile: Int {
        tiles.map { Int($0.pairCount) }.max() ?? 0
    }

    public var maxLiveExpertsPerTile: Int {
        tiles.map { Int($0.groupCount) }.max() ?? 0
    }
}

enum PrefillMoEGroupingError: Error, Equatable, CustomStringConvertible {
    case invalidQueryCount(Int)
    case invalidTopK(Int)
    case invalidNumExperts(Int)
    case invalidTileExpertCount(Int)
    case pairCountMismatch(expected: Int, actual: Int)
    case tokenOutOfRange(UInt32)
    case rankOutOfRange(UInt32)
    case expertOutOfRange(UInt32)
    case duplicateTokenRank(token: UInt32, rank: UInt32)
    case expertSortKeyCountMismatch(expected: Int, actual: Int)
    case expertTileCountsMismatch(expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .invalidQueryCount(let value):
            return "queryCount must be non-negative, got \(value)"
        case .invalidTopK(let value):
            return "topK must be positive, got \(value)"
        case .invalidNumExperts(let value):
            return "numExperts must be positive, got \(value)"
        case .invalidTileExpertCount(let value):
            return "tileExpertCount must be in 1...16, got \(value)"
        case .pairCountMismatch(let expected, let actual):
            return "expected \(expected) route pairs, got \(actual)"
        case .tokenOutOfRange(let value):
            return "route token \(value) is out of range"
        case .rankOutOfRange(let value):
            return "route rank \(value) is out of range"
        case .expertOutOfRange(let value):
            return "route expert \(value) is out of range"
        case .duplicateTokenRank(let token, let rank):
            return "duplicate route pair for token \(token), rank \(rank)"
        case .expertSortKeyCountMismatch(let expected, let actual):
            return "expected \(expected) expert sort keys, got \(actual)"
        case .expertTileCountsMismatch(let expected, let actual):
            return "expert tile counts summed to \(actual), expected \(expected)"
        }
    }
}

/// One `recencyBalanced` result: a flat expert order and the per-tile
/// expert counts consecutive runs of it are sliced into.
struct PrefillSweepBalancedOrder: Equatable {
    let order: [UInt32]
    let tileExpertCounts: [Int]
}

/// `SHRIKE_PREFILL_SWEEP=recency`'s expert order.
enum PrefillSweepOrder {
    /// Ascending by last row in the chunk, ties by expert id ascending.
    static func recency(lastRowByExpert: [UInt32: Int]) -> [UInt32] {
        lastRowByExpert
            .sorted { lhs, rhs in
                if lhs.value != rhs.value { return lhs.value < rhs.value }
                return lhs.key < rhs.key
            }
            .map(\.key)
    }

    /// Splits the chunk's ranked experts into a head and a tail of the last
    /// `tail` (the decode-adjacent set), then packs each group's own tiles
    /// of at most `tileWidth` experts by total row weight, heaviest first
    /// into the lightest open tile, ties to the lower tile index.
    static func recencyBalanced(rowsByExpert: [UInt32: Int],
                                lastRowByExpert: [UInt32: Int],
                                tail: Int,
                                tileWidth: Int) -> PrefillSweepBalancedOrder {
        let ranked = recency(lastRowByExpert: lastRowByExpert)
        let tailCount = min(tail, ranked.count)
        let headCount = ranked.count - tailCount
        let head = packByRows(Array(ranked[0..<headCount]), rowsByExpert: rowsByExpert,
                              tileWidth: tileWidth)
        let tailGroup = packByRows(Array(ranked[headCount...]), rowsByExpert: rowsByExpert,
                                   tileWidth: tileWidth)
        return PrefillSweepBalancedOrder(order: head.order + tailGroup.order,
                                         tileExpertCounts: head.tileExpertCounts + tailGroup.tileExpertCounts)
    }

    /// `order`'s rank per expert id, `numExperts`-sized so it takes
    /// `groupTokenExpertPairs`'s existing `expertSortKeys` branch (one array
    /// read per comparison side, no dictionary): an expert not in `order`
    /// (never routed this chunk) ranks past every real entry.
    static func expertSortKeys(forOrder order: [UInt32], numExperts: Int) -> [UInt64] {
        var keys = [UInt64](repeating: UInt64(order.count), count: numExperts)
        for (rank, expert) in order.enumerated() where expert < numExperts {
            keys[Int(expert)] = UInt64(rank)
        }
        return keys
    }

    private static func packByRows(_ experts: [UInt32], rowsByExpert: [UInt32: Int],
                                   tileWidth: Int) -> (order: [UInt32], tileExpertCounts: [Int]) {
        guard !experts.isEmpty else { return ([], []) }
        let tileCount = (experts.count + tileWidth - 1) / tileWidth
        var bins: [(total: Int, experts: [UInt32])] = Array(repeating: (0, []), count: tileCount)
        let heaviestFirst = experts.sorted { lhs, rhs in
            let lhsRows = rowsByExpert[lhs] ?? 0
            let rhsRows = rowsByExpert[rhs] ?? 0
            if lhsRows != rhsRows { return lhsRows > rhsRows }
            return lhs < rhs
        }
        for expert in heaviestFirst {
            var bestIndex = 0
            var bestTotal = Int.max
            for (index, bin) in bins.enumerated() where bin.experts.count < tileWidth {
                if bin.total < bestTotal {
                    bestTotal = bin.total
                    bestIndex = index
                }
            }
            bins[bestIndex].experts.append(expert)
            bins[bestIndex].total += rowsByExpert[expert] ?? 0
        }
        return (bins.flatMap(\.experts), bins.map { $0.experts.count })
    }
}

enum PrefillMoEGrouping {
    static func groupTokenExpertPairs(
        _ pairs: [PrefillTokenExpertPair],
        queryCount: Int,
        topK: Int,
        numExperts: Int,
        tileExpertCount: Int = 16,
        expertSortKeys: [UInt64]? = nil,
        descending: Bool = false,
        expertTileCounts: [Int]? = nil
    ) throws -> PrefillMoEGroupedRoutes {
        guard queryCount >= 0 else {
            throw PrefillMoEGroupingError.invalidQueryCount(queryCount)
        }
        guard topK > 0 else {
            throw PrefillMoEGroupingError.invalidTopK(topK)
        }
        guard numExperts > 0 else {
            throw PrefillMoEGroupingError.invalidNumExperts(numExperts)
        }
        guard (1...16).contains(tileExpertCount) else {
            throw PrefillMoEGroupingError.invalidTileExpertCount(tileExpertCount)
        }
        if let expertSortKeys, expertSortKeys.count != numExperts {
            throw PrefillMoEGroupingError.expertSortKeyCountMismatch(expected: numExperts,
                                                                    actual: expertSortKeys.count)
        }
        let expectedPairs = queryCount * topK
        guard pairs.count == expectedPairs else {
            throw PrefillMoEGroupingError.pairCountMismatch(expected: expectedPairs,
                                                           actual: pairs.count)
        }

        var seenTokenRanks: Set<UInt64> = []
        seenTokenRanks.reserveCapacity(pairs.count)
        for pair in pairs {
            guard pair.token < UInt32(queryCount) else {
                throw PrefillMoEGroupingError.tokenOutOfRange(pair.token)
            }
            guard pair.rank < UInt32(topK) else {
                throw PrefillMoEGroupingError.rankOutOfRange(pair.rank)
            }
            guard pair.expert < UInt32(numExperts) else {
                throw PrefillMoEGroupingError.expertOutOfRange(pair.expert)
            }
            let key = UInt64(pair.token) << 32 | UInt64(pair.rank)
            guard seenTokenRanks.insert(key).inserted else {
                throw PrefillMoEGroupingError.duplicateTokenRank(token: pair.token,
                                                                rank: pair.rank)
            }
        }

        let sortedPairs = pairs.sorted {
            if let expertSortKeys {
                let lhsKey = expertSortKeys[Int($0.expert)]
                let rhsKey = expertSortKeys[Int($1.expert)]
                if lhsKey != rhsKey { return descending ? lhsKey > rhsKey : lhsKey < rhsKey }
            }
            if $0.expert != $1.expert { return descending ? $0.expert > $1.expert : $0.expert < $1.expert }
            if $0.token != $1.token { return $0.token < $1.token }
            return $0.rank < $1.rank
        }

        var offsets = Array(repeating: UInt32.max, count: numExperts)
        var counts = Array(repeating: UInt32(0), count: numExperts)
        var groups: [PrefillMoEGroup] = []
        groups.reserveCapacity(min(numExperts, sortedPairs.count))

        var i = 0
        while i < sortedPairs.count {
            let expert = sortedPairs[i].expert
            let start = i
            while i < sortedPairs.count, sortedPairs[i].expert == expert {
                i += 1
            }
            let count = i - start
            offsets[Int(expert)] = UInt32(start)
            counts[Int(expert)] = UInt32(count)
            groups.append(PrefillMoEGroup(expert: expert,
                                          pairStart: UInt32(start),
                                          pairCount: UInt32(count)))
        }

        if let expertTileCounts {
            let suppliedTotal = expertTileCounts.reduce(0, +)
            guard suppliedTotal == groups.count else {
                throw PrefillMoEGroupingError.expertTileCountsMismatch(expected: groups.count,
                                                                       actual: suppliedTotal)
            }
        }

        var tiles: [PrefillMoETile] = []
        if let expertTileCounts {
            var groupStart = 0
            for count in expertTileCounts where count > 0 {
                let groupEnd = groupStart + count
                let first = groups[groupStart]
                let last = groups[groupEnd - 1]
                tiles.append(PrefillMoETile(groupStart: UInt32(groupStart),
                                            groupCount: UInt32(count),
                                            pairStart: first.pairStart,
                                            pairCount: last.pairStart + last.pairCount - first.pairStart))
                groupStart = groupEnd
            }
        } else {
            var groupStart = 0
            while groupStart < groups.count {
                let groupEnd = min(groups.count, groupStart + tileExpertCount)
                let first = groups[groupStart]
                let last = groups[groupEnd - 1]
                let pairStart = first.pairStart
                let pairEnd = last.pairStart + last.pairCount
                tiles.append(PrefillMoETile(groupStart: UInt32(groupStart),
                                            groupCount: UInt32(groupEnd - groupStart),
                                            pairStart: pairStart,
                                            pairCount: pairEnd - pairStart))
                groupStart = groupEnd
            }
        }

        return PrefillMoEGroupedRoutes(sortedPairs: sortedPairs,
                                       perExpertOffsets: offsets,
                                       perExpertCounts: counts,
                                       groups: groups,
                                       tiles: tiles,
                                       queryCount: queryCount)
    }
}
