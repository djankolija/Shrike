import Testing
@testable import Shrike

@Suite struct PrefillMoEGroupingTests {
    @Test func groupingSortsPairsAndBuildsOffsetsCountsAndTiles() throws {
        let pairs = [
            Self.pair(token: 0, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 0, expert: 1, rank: 1, weightBits: 11),
            Self.pair(token: 0, expert: 4, rank: 2, weightBits: 12),
            Self.pair(token: 1, expert: 1, rank: 0, weightBits: 13),
            Self.pair(token: 1, expert: 3, rank: 1, weightBits: 14),
            Self.pair(token: 1, expert: 2, rank: 2, weightBits: 15),
            Self.pair(token: 2, expert: 4, rank: 0, weightBits: 16),
            Self.pair(token: 2, expert: 3, rank: 1, weightBits: 17),
            Self.pair(token: 2, expert: 1, rank: 2, weightBits: 18),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 19),
            Self.pair(token: 3, expert: 1, rank: 1, weightBits: 20),
            Self.pair(token: 3, expert: 4, rank: 2, weightBits: 21),
        ].reversed()

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            Array(pairs),
            queryCount: 4,
            topK: 3,
            numExperts: 6,
            tileExpertCount: 2)

        #expect(grouped.sortedPairs.map { $0.expert } == [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5])
        #expect(grouped.sortedPairs.map { [$0.token, $0.rank] } == [
            [0, 1], [1, 0], [2, 2], [3, 1],
            [1, 2],
            [0, 0], [1, 1], [2, 1],
            [0, 2], [2, 0], [3, 2],
            [3, 0],
        ])
        #expect(grouped.perExpertOffsets == [UInt32.max, 0, 4, 5, 8, 11])
        #expect(grouped.perExpertCounts == [0, 4, 1, 3, 3, 1])
        #expect(grouped.groups == [
            PrefillMoEGroup(expert: 1, pairStart: 0, pairCount: 4),
            PrefillMoEGroup(expert: 2, pairStart: 4, pairCount: 1),
            PrefillMoEGroup(expert: 3, pairStart: 5, pairCount: 3),
            PrefillMoEGroup(expert: 4, pairStart: 8, pairCount: 3),
            PrefillMoEGroup(expert: 5, pairStart: 11, pairCount: 1),
        ])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 2, pairStart: 0, pairCount: 5),
            PrefillMoETile(groupStart: 2, groupCount: 2, pairStart: 5, pairCount: 6),
            PrefillMoETile(groupStart: 4, groupCount: 1, pairStart: 11, pairCount: 1),
        ])
        #expect(grouped.maxPairsPerExpert == 4)
        #expect(grouped.maxPairsPerTile == 6)
        #expect(grouped.maxLiveExpertsPerTile == 2)
    }

    @Test func groupingIsDeterministicAcrossInputOrderAndPreservesWeightBits() throws {
        let tokenMajor = [
            Self.pair(token: 0, expert: 2, rank: 0, weightBits: 101),
            Self.pair(token: 0, expert: 1, rank: 1, weightBits: 102),
            Self.pair(token: 1, expert: 2, rank: 0, weightBits: 103),
            Self.pair(token: 1, expert: 0, rank: 1, weightBits: 104),
            Self.pair(token: 2, expert: 1, rank: 0, weightBits: 105),
            Self.pair(token: 2, expert: 2, rank: 1, weightBits: 106),
        ]
        let shuffled = [tokenMajor[5], tokenMajor[1], tokenMajor[3],
                        tokenMajor[0], tokenMajor[4], tokenMajor[2]]

        let a = try PrefillMoEGrouping.groupTokenExpertPairs(
            tokenMajor,
            queryCount: 3,
            topK: 2,
            numExperts: 3,
            tileExpertCount: 2)
        let b = try PrefillMoEGrouping.groupTokenExpertPairs(
            shuffled,
            queryCount: 3,
            topK: 2,
            numExperts: 3,
            tileExpertCount: 2)

        #expect(a == b)
        #expect(a.sortedPairs.map { $0.weightBitsAndReserved } == [104, 102, 105, 101, 103, 106])
    }

    @Test func groupingCanOrderTilesByExpertSortKeysWhileKeepingPairRangesContiguous() throws {
        let pairs = [
            Self.pair(token: 0, expert: 1, rank: 0, weightBits: 10),
            Self.pair(token: 0, expert: 2, rank: 1, weightBits: 20),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 30),
            Self.pair(token: 1, expert: 1, rank: 1, weightBits: 11),
            Self.pair(token: 2, expert: 2, rank: 0, weightBits: 21),
            Self.pair(token: 2, expert: 3, rank: 1, weightBits: 31),
        ]

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 4,
            tileExpertCount: 2,
            expertSortKeys: [0, 30, 10, 20])

        #expect(grouped.groups.map(\.expert) == [2, 3, 1])
        #expect(grouped.sortedPairs.map(\.expert) == [2, 2, 3, 3, 1, 1])
        #expect(grouped.perExpertOffsets == [UInt32.max, 4, 0, 2])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 2, pairStart: 0, pairCount: 4),
            PrefillMoETile(groupStart: 2, groupCount: 1, pairStart: 4, pairCount: 2),
        ])
        for tile in grouped.tiles {
            let slice = grouped.sortedPairs[Int(tile.pairStart)..<Int(tile.pairStart + tile.pairCount)]
            let tileExperts = grouped.groups[Int(tile.groupStart)..<Int(tile.groupStart + tile.groupCount)]
                .map(\.expert)
            #expect(Set(slice.map(\.expert)) == Set(tileExperts))
        }
    }

    @Test func groupingRejectsInvalidMetadataBeforeKernelUse() throws {
        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 1,
                tileExpertCount: 17)
        } throws: { error in
            if case PrefillMoEGroupingError.invalidTileExpertCount(17) = error { return true }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 2, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 2,
                tileExpertCount: 1)
        } throws: { error in
            if case PrefillMoEGroupingError.expertOutOfRange(2) = error { return true }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [
                    Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1),
                    Self.pair(token: 0, expert: 1, rank: 0, weightBits: 2),
                ],
                queryCount: 1,
                topK: 2,
                numExperts: 2,
                tileExpertCount: 1)
        } throws: { error in
            if case PrefillMoEGroupingError.duplicateTokenRank(token: 0, rank: 0) = error {
                return true
            }
            return false
        }

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                [Self.pair(token: 0, expert: 0, rank: 0, weightBits: 1)],
                queryCount: 1,
                topK: 1,
                numExperts: 2,
                tileExpertCount: 1,
                expertSortKeys: [0])
        } throws: { error in
            if case PrefillMoEGroupingError.expertSortKeyCountMismatch(expected: 2, actual: 1) = error {
                return true
            }
            return false
        }
    }

    @Test func groupingToleratesRealTracePairPressureAtT32() throws {
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(256)
        for token in 0..<32 {
            for rank in 0..<8 {
                let expert = UInt32((token + rank * 7) % 74)
                pairs.append(Self.pair(token: UInt32(token),
                                       expert: expert,
                                       rank: UInt32(rank),
                                       weightBits: UInt32(token * 8 + rank)))
            }
        }

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 32,
            topK: 8,
            numExperts: 128,
            tileExpertCount: 16)

        #expect(grouped.sortedPairs.count == 256)
        #expect(grouped.groups.count <= 74)
        #expect(grouped.maxLiveExpertsPerTile <= 16)
        #expect(grouped.maxPairsPerTile <= 256)
        for tile in grouped.tiles {
            #expect(tile.groupCount <= 16)
            #expect(tile.pairCount > 0)
        }
    }

    @Test func groupingCanOrderTilesByASweepOrderWhileKeepingPairRangesContiguous() throws {
        let pairs = [
            Self.pair(token: 0, expert: 8, rank: 0, weightBits: 80),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 2, expert: 8, rank: 0, weightBits: 81),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 50),
        ]
        let order: [UInt32] = [3, 8, 5]
        let sortKeys = PrefillSweepOrder.expertSortKeys(forOrder: order, numExperts: 9)

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 4,
            topK: 1,
            numExperts: 9,
            tileExpertCount: 2,
            expertSortKeys: sortKeys)

        #expect(grouped.groups.map(\.expert) == [3, 8, 5])
        #expect(grouped.sortedPairs.map(\.expert) == [3, 8, 8, 5])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 2, pairStart: 0, pairCount: 3),
            PrefillMoETile(groupStart: 2, groupCount: 1, pairStart: 3, pairCount: 1),
        ])
    }

    @Test func expertSortKeysRanksTheOrderAscendingAndPutsAbsenteesPastTheEnd() throws {
        let keys = PrefillSweepOrder.expertSortKeys(forOrder: [3, 8, 5], numExperts: 9)
        #expect(keys[3] == 0)
        #expect(keys[8] == 1)
        #expect(keys[5] == 2)
        for expert in [0, 1, 2, 4, 6, 7] {
            #expect(keys[expert] == 3)
        }
    }

    @Test func expertSortKeysSkipsAnOutOfRangeExpertRatherThanTrapping() throws {
        let keys = PrefillSweepOrder.expertSortKeys(forOrder: [3, 99, 5], numExperts: 9)
        #expect(keys.count == 9)
        #expect(keys[3] == 0)
        #expect(keys[5] == 2)
    }

    @Test func residentFirstBalancedSpreadsAbsentUniformlyPreservesRecencyAndBreaksResidentTiesToTheLowerTileIndex() throws {
        var rowsByExpert: [UInt32: Int] = [90: 9, 91: 7, 92: 5, 93: 3]
        var lastRowByExpert: [UInt32: Int] = [90: 290, 91: 291, 92: 292, 93: 293]
        for id in 1...20 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = id
        }
        var resident = [Bool](repeating: false, count: 94)
        for expert in [90, 91, 92, 93] { resident[expert] = true }

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: resident, slots: 50, tileWidth: 8)

        let expected: [UInt32] = [1, 2, 3, 4, 5, 6, 90, 93,
                                  7, 8, 9, 10, 11, 12, 13, 91,
                                  14, 15, 16, 17, 18, 19, 20, 92]
        #expect(order == expected)
        #expect(order.count == 24)
    }

    @Test func residentFirstBalancedGrowsAResidentHeadWhenResidentsNearlyFillThePool() throws {
        var rowsByExpert: [UInt32: Int] = [:]
        var lastRowByExpert: [UInt32: Int] = [:]
        for id in 1...9 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = 9 + id
        }
        for id in 100...102 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = id - 100
        }
        var resident = [Bool](repeating: false, count: 103)
        for id in 1...9 { resident[id] = true }

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: resident, slots: 20, tileWidth: 8)

        #expect(order == [1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 9])
    }

    @Test func residentFirstBalancedGuardsAnExhaustedHeadSearchAgainstDroppingTheAbsentGroup() throws {
        var rowsByExpert: [UInt32: Int] = [999: 1]
        var lastRowByExpert: [UInt32: Int] = [999: 0]
        for id in 1...100 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = id
        }
        var resident = [Bool](repeating: false, count: 1000)
        resident[999] = true

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: resident, slots: 1, tileWidth: 8)

        #expect(Set(order) == Set(rowsByExpert.keys))
        #expect(order.count == 101)
    }

    @Test func residentFirstBalancedOmitsAResidentExpertNotRoutedThisChunk() throws {
        let rowsByExpert: [UInt32: Int] = [3: 1, 5: 1]
        let lastRowByExpert: [UInt32: Int] = [3: 0, 5: 1]
        var resident = [Bool](repeating: false, count: 10)
        resident[3] = true
        resident[7] = true

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: resident, slots: 128, tileWidth: 8)

        #expect(Set(order) == Set([3, 5] as [UInt32]))
        #expect(!order.contains(7))
        #expect(order == [5, 3])
    }

    @Test func residentFirstBalancedHandlesAResidentArrayShorterThanNumExpertsWithoutTrapping() throws {
        let rowsByExpert: [UInt32: Int] = [3: 1, 99: 1]
        let lastRowByExpert: [UInt32: Int] = [3: 0, 99: 1]
        let shortResident: [Bool] = [false, false, false, true]

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: shortResident, slots: 128, tileWidth: 8)

        #expect(order == [99, 3])
    }

    @Test func residentFirstBalancedHandlesAResidentArrayLongerThanNumExpertsWithoutTrapping() throws {
        let rowsByExpert: [UInt32: Int] = [2: 3, 5: 1]
        let lastRowByExpert: [UInt32: Int] = [2: 4, 5: 1]
        var longResident = [Bool](repeating: false, count: 200)
        longResident[5] = true

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: longResident, slots: 128, tileWidth: 8)

        #expect(order == [2, 5])
    }

    @Test func groupingReproducesResidentFirstBalancedFlatTilesEndToEnd() throws {
        var rowsByExpert: [UInt32: Int] = [:]
        var lastRowByExpert: [UInt32: Int] = [:]
        var pairs: [PrefillTokenExpertPair] = []
        for id in 1...9 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = 9 + id
            pairs.append(Self.pair(token: UInt32(id - 1), expert: UInt32(id), rank: 0,
                                   weightBits: UInt32(id)))
        }
        for id in 100...102 {
            rowsByExpert[UInt32(id)] = 1
            lastRowByExpert[UInt32(id)] = id - 100
            pairs.append(Self.pair(token: UInt32(id - 91), expert: UInt32(id), rank: 0,
                                   weightBits: UInt32(id)))
        }
        var resident = [Bool](repeating: false, count: 103)
        for id in 1...9 { resident[id] = true }

        let order = PrefillSweepOrder.residentFirstBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert,
            resident: resident, slots: 20, tileWidth: 8)
        #expect(order == [1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102, 9])

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 12,
            topK: 1,
            numExperts: 103,
            tileExpertCount: 8,
            expertSortKeys: PrefillSweepOrder.expertSortKeys(forOrder: order, numExperts: 103))

        #expect(grouped.groups.map(\.expert) == order)
        #expect(grouped.tiles.count == 2)
        #expect(grouped.tiles[0] == PrefillMoETile(groupStart: 0, groupCount: 8, pairStart: 0, pairCount: 8))
        #expect(grouped.tiles[1] == PrefillMoETile(groupStart: 8, groupCount: 4, pairStart: 8, pairCount: 4))
        let headResidents = Set<UInt32>([1, 2, 3, 4, 5, 6, 7, 8])
        let firstTileExperts = Set(grouped.groups[0..<Int(grouped.tiles[0].groupCount)].map(\.expert))
        #expect(headResidents == firstTileExperts)
    }

    private static func pair(token: UInt32,
                             expert: UInt32,
                             rank: UInt32,
                             weightBits: UInt32) -> PrefillTokenExpertPair {
        PrefillTokenExpertPair(token: token,
                               expert: expert,
                               rank: rank,
                               weightBitsAndReserved: weightBits)
    }
}
