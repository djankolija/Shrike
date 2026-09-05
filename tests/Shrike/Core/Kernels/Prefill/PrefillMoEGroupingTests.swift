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

    @Test func groupingCanOrderTilesByDescendingExpertSortKeysWhileKeepingPairRangesContiguous() throws {
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
            expertSortKeys: [0, 30, 10, 20],
            descending: true)

        #expect(grouped.groups.map(\.expert) == [1, 3, 2])
        #expect(grouped.sortedPairs.map(\.expert) == [1, 1, 3, 3, 2, 2])
        #expect(grouped.perExpertOffsets == [UInt32.max, 0, 4, 2])
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
        for group in grouped.groups {
            let slice = grouped.sortedPairs[Int(group.pairStart)..<Int(group.pairStart + group.pairCount)]
            var previous: (token: UInt32, rank: UInt32)?
            for pair in slice {
                if let previous {
                    #expect(previous.token < pair.token
                        || (previous.token == pair.token && previous.rank < pair.rank))
                }
                previous = (pair.token, pair.rank)
            }
        }
    }

    @Test func groupingWithDescendingFalseMatchesDefaultBehavior() throws {
        let pairs = [
            Self.pair(token: 0, expert: 1, rank: 0, weightBits: 10),
            Self.pair(token: 0, expert: 2, rank: 1, weightBits: 20),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 30),
            Self.pair(token: 1, expert: 1, rank: 1, weightBits: 11),
            Self.pair(token: 2, expert: 2, rank: 0, weightBits: 21),
            Self.pair(token: 2, expert: 3, rank: 1, weightBits: 31),
        ]

        let defaulted = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 4,
            tileExpertCount: 2,
            expertSortKeys: [0, 30, 10, 20])
        let explicit = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 3,
            topK: 2,
            numExperts: 4,
            tileExpertCount: 2,
            expertSortKeys: [0, 30, 10, 20],
            descending: false)

        #expect(defaulted == explicit)
    }

    @Test func chunkSweepParityAlternatesByChunkIndex() throws {
        let chunkTokens = 4_096
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 0, chunkTokens: chunkTokens) == false)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 4_095, chunkTokens: chunkTokens) == false)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 4_096, chunkTokens: chunkTokens) == true)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 8_191, chunkTokens: chunkTokens) == true)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 8_192, chunkTokens: chunkTokens) == false)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 2_048, chunkTokens: 2_048) == true)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            startPosition: 4_096, chunkTokens: 2_048) == false)
    }

    @Test func carriedSweepStartsOppositeThePreviousRequestsLastChunk() throws {
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            mode: .carry, carried: true, startPosition: 0, chunkTokens: 4_096) == false)
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            mode: .carry, carried: false, startPosition: 0, chunkTokens: 4_096) == true)
    }

    @Test func carriedSweepAlternatesFromTheCarriedStart() throws {
        // The runner reads back what it just wrote, so each chunk's result
        // must feed forward as the next chunk's `carried`, not a fixed carry
        // stepped across positions.
        let chunkTokens = 4_096
        let initialCarries: [Bool?] = [nil, true, false]
        for initial in initialCarries {
            let d0 = RealForwardRunner.prefillChunkSweepIsDescending(
                mode: .carry, carried: initial, startPosition: 0, chunkTokens: chunkTokens)
            let d1 = RealForwardRunner.prefillChunkSweepIsDescending(
                mode: .carry, carried: d0, startPosition: chunkTokens, chunkTokens: chunkTokens)
            #expect(d1 == !d0)
            let d2 = RealForwardRunner.prefillChunkSweepIsDescending(
                mode: .carry, carried: d1, startPosition: 2 * chunkTokens, chunkTokens: chunkTokens)
            #expect(d2 == !d1)
            #expect(d2 == d0)
        }
    }

    @Test func nonParticipatingCallsAlwaysComputeAlternateBehaviour() throws {
        let chunkTokens = 4_096
        let modes: [PrefillSweepMode] = [.alternate, .fixed, .carry]
        let carriedValues: [Bool?] = [nil, true, false]
        for mode in modes {
            for carried in carriedValues {
                for startPosition in stride(from: 0, through: 3 * chunkTokens, by: chunkTokens) {
                    #expect(RealForwardRunner.prefillChunkSweepIsDescending(
                        mode: mode, carried: carried, startPosition: startPosition,
                        chunkTokens: chunkTokens, participatesInCarry: false)
                        == RealForwardRunner.prefillChunkSweepIsDescending(
                            startPosition: startPosition, chunkTokens: chunkTokens))
                }
            }
        }
    }

    @Test func carriedSweepWithNoHistoryIsAscending() throws {
        #expect(RealForwardRunner.prefillChunkSweepIsDescending(
            mode: .carry, carried: nil, startPosition: 0, chunkTokens: 4_096) == false)
    }

    @Test func alternateModeIgnoresTheCarriedDirection() throws {
        let chunkTokens = 4_096
        let carriedValues: [Bool?] = [nil, true, false]
        for carried in carriedValues {
            for startPosition in stride(from: 0, through: 3 * chunkTokens, by: chunkTokens) {
                #expect(RealForwardRunner.prefillChunkSweepIsDescending(
                    mode: .alternate, carried: carried,
                    startPosition: startPosition, chunkTokens: chunkTokens)
                    == RealForwardRunner.prefillChunkSweepIsDescending(
                        startPosition: startPosition, chunkTokens: chunkTokens))
            }
        }
    }

    @Test func fixedModeIsAscendingAtEveryChunkAndCarry() throws {
        let chunkTokens = 4_096
        let carriedValues: [Bool?] = [nil, true, false]
        for carried in carriedValues {
            for startPosition in stride(from: 0, through: 3 * chunkTokens, by: chunkTokens) {
                #expect(RealForwardRunner.prefillChunkSweepIsDescending(
                    mode: .fixed, carried: carried,
                    startPosition: startPosition, chunkTokens: chunkTokens) == false)
            }
        }
    }

    @Test func recencyBalanceRequiresBothRecencyModeAndCarryParticipation() throws {
        #expect(RealForwardRunner.prefillChunkUsesRecencyBalance(
            mode: .recency, participatesInCarry: true) == true)
        #expect(RealForwardRunner.prefillChunkUsesRecencyBalance(
            mode: .recency, participatesInCarry: false) == false)
        #expect(RealForwardRunner.prefillChunkUsesRecencyBalance(
            mode: .carry, participatesInCarry: true) == false)
        #expect(RealForwardRunner.prefillChunkUsesRecencyBalance(
            mode: .carry, participatesInCarry: false) == false)
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

    @Test func recencyOrdersByLastRowAscending() throws {
        #expect(PrefillSweepOrder.recency(lastRowByExpert: [8: 2, 3: 1, 5: 3]) == [3, 8, 5])
    }

    @Test func recencyBreaksTiesByExpertIdAscending() throws {
        #expect(PrefillSweepOrder.recency(lastRowByExpert: [5: 4, 2: 4, 9: 1]) == [9, 2, 5])
    }

    @Test func recencyHandlesASingleExpertChunk() throws {
        #expect(PrefillSweepOrder.recency(lastRowByExpert: [42: 7]) == [42])
    }

    @Test func groupingCanOrderTilesByRecencyWhileKeepingPairRangesContiguous() throws {
        let pairs = [
            Self.pair(token: 0, expert: 8, rank: 0, weightBits: 80),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 2, expert: 8, rank: 0, weightBits: 81),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 50),
        ]
        let order = PrefillSweepOrder.recency(lastRowByExpert: [8: 2, 3: 1, 5: 3])
        #expect(order == [3, 8, 5])
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

    @Test func recencyBalancedSplitsHeadAndTailByTail() throws {
        let lastRowByExpert: [UInt32: Int] = [1: 0, 2: 1, 3: 2, 4: 3, 5: 4]
        let rowsByExpert: [UInt32: Int] = [1: 10, 2: 10, 3: 10, 4: 10, 5: 10]

        let balanced = PrefillSweepOrder.recencyBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert, tail: 2, tileWidth: 8)

        #expect(balanced.order == [1, 2, 3, 4, 5])
        #expect(balanced.tileExpertCounts == [3, 2])
    }

    @Test func recencyBalancedHeadIsEmptyWhenChunkHasAtMostTailExperts() throws {
        let lastRowByExpert: [UInt32: Int] = [1: 0, 2: 1, 3: 2]
        let rowsByExpert: [UInt32: Int] = [1: 5, 2: 5, 3: 5]

        let balanced = PrefillSweepOrder.recencyBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert, tail: 96, tileWidth: 8)

        #expect(balanced.order == [1, 2, 3])
        #expect(balanced.tileExpertCounts == [3])
    }

    @Test func recencyBalancedPacksTilesCloseToEvenByRowWeight() throws {
        var lastRowByExpert: [UInt32: Int] = [:]
        var rowsByExpert: [UInt32: Int] = [:]
        for id in UInt32(1)...20 {
            lastRowByExpert[id] = Int(id)
            rowsByExpert[id] = Int(21 - id)
        }

        let balanced = PrefillSweepOrder.recencyBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert, tail: 20, tileWidth: 8)

        #expect(balanced.order.count == 20)
        #expect(Set(balanced.order) == Set((1...20).map { UInt32($0) }))
        #expect(balanced.tileExpertCounts.count == 3)
        #expect(balanced.tileExpertCounts.reduce(0, +) == 20)
        #expect(balanced.tileExpertCounts.allSatisfy { $0 <= 8 })

        var totals: [Int] = []
        var cursor = 0
        for count in balanced.tileExpertCounts {
            let tileExperts = balanced.order[cursor..<(cursor + count)]
            totals.append(tileExperts.reduce(0) { $0 + (rowsByExpert[$1] ?? 0) })
            cursor += count
        }
        let heaviestWeight = rowsByExpert.values.max() ?? 0
        let maxTotal = totals.max() ?? 0
        let minTotal = totals.min() ?? 0
        #expect((maxTotal - minTotal) <= heaviestWeight)

        let firstTileExperts = balanced.order[0..<balanced.tileExpertCounts[0]]
        #expect(firstTileExperts.contains(1))
    }

    @Test func groupingHonoursExplicitExpertTileCounts() throws {
        let pairs = [
            Self.pair(token: 0, expert: 8, rank: 0, weightBits: 80),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 2, expert: 8, rank: 0, weightBits: 81),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 50),
        ]

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 4,
            topK: 1,
            numExperts: 9,
            tileExpertCount: 2,
            expertSortKeys: PrefillSweepOrder.expertSortKeys(forOrder: [3, 8, 5], numExperts: 9),
            expertTileCounts: [1, 2])

        #expect(grouped.groups.map(\.expert) == [3, 8, 5])
        #expect(grouped.tiles == [
            PrefillMoETile(groupStart: 0, groupCount: 1, pairStart: 0, pairCount: 1),
            PrefillMoETile(groupStart: 1, groupCount: 2, pairStart: 1, pairCount: 3),
        ])
    }

    @Test func groupingThrowsOnExpertTileCountsMismatch() throws {
        let pairs = [
            Self.pair(token: 0, expert: 8, rank: 0, weightBits: 80),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 10),
        ]

        #expect {
            _ = try PrefillMoEGrouping.groupTokenExpertPairs(
                pairs,
                queryCount: 2,
                topK: 1,
                numExperts: 9,
                expertTileCounts: [1])
        } throws: { error in
            if case PrefillMoEGroupingError.expertTileCountsMismatch(expected: 2, actual: 1) = error {
                return true
            }
            return false
        }
    }

    @Test func groupingReproducesRecencyBalancedTilesEndToEnd() throws {
        let pairs = [
            Self.pair(token: 0, expert: 8, rank: 0, weightBits: 80),
            Self.pair(token: 1, expert: 3, rank: 0, weightBits: 10),
            Self.pair(token: 2, expert: 8, rank: 0, weightBits: 81),
            Self.pair(token: 3, expert: 5, rank: 0, weightBits: 50),
        ]
        let lastRowByExpert: [UInt32: Int] = [8: 2, 3: 1, 5: 3]
        let rowsByExpert: [UInt32: Int] = [8: 2, 3: 1, 5: 1]

        let balanced = PrefillSweepOrder.recencyBalanced(
            rowsByExpert: rowsByExpert, lastRowByExpert: lastRowByExpert, tail: 96, tileWidth: 2)

        let grouped = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: 4,
            topK: 1,
            numExperts: 9,
            tileExpertCount: 2,
            expertSortKeys: PrefillSweepOrder.expertSortKeys(forOrder: balanced.order, numExperts: 9),
            expertTileCounts: balanced.tileExpertCounts)

        #expect(grouped.tiles.map { Int($0.groupCount) } == balanced.tileExpertCounts)
        #expect(grouped.tiles.reduce(0) { $0 + Int($1.pairCount) } == pairs.count)
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
