import Testing
@testable import Shrike

@Suite struct RouterHostReadbackTests {
    private static let topK = 4

    private static func words(tag: UInt32,
                              hitCount: UInt32 = 3, missCount: UInt32 = 1,
                              ids: [UInt32] = [5, 9, 200, 3],
                              weightBits: [UInt32] = [0x3c00, 0x3800, 0x3400, 0x3000],
                              hitPositions: [UInt32] = [0, 2, 3, 0],
                              missPositions: [UInt32] = [1, 0, 0, 0],
                              predicted: [UInt32] = [1, 2, 3, 4]) -> [UInt32] {
        ([hitCount, missCount] + ids + weightBits + hitPositions + missPositions + predicted)
            .map { RouterHostReadback.word(tag: tag, value: $0) }
    }

    @Test func wordCountCoversTheCountsAndFiveTopKRuns() {
        #expect(RouterHostReadback.wordCount(topK: 4) == 22)
        #expect(RouterHostReadback.wordCount(topK: 8) == 42)
    }

    @Test func decodeReadsCountsIdsWeightsAndPositions() {
        let words = Self.words(tag: 7)
        let readback = words.withUnsafeBufferPointer {
            RouterHostReadback.decode(words: $0.baseAddress!, topK: Self.topK, tag: 7)
        }
        #expect(readback == RouterHostReadback(
            hitCount: 3, missCount: 1,
            expertIDs: [5, 9, 200, 3],
            weightBits: [0x3c00, 0x3800, 0x3400, 0x3000],
            hitPositions: [0, 2, 3],
            missPositions: [1],
            predictedIDs: [1, 2, 3, 4]))
    }

    @Test func decodeReturnsNilUntilEveryWordCarriesTheTag() {
        var words = Self.words(tag: 7)
        words[RouterHostReadback.wordCount(topK: Self.topK) - 1] =
            RouterHostReadback.word(tag: 6, value: 4)
        let stale = words.withUnsafeBufferPointer {
            (RouterHostReadback.isComplete(words: $0.baseAddress!, topK: Self.topK, tag: 7),
             RouterHostReadback.decode(words: $0.baseAddress!, topK: Self.topK, tag: 7))
        }
        #expect(stale.0 == false)
        #expect(stale.1 == nil)
        let zeroed = [UInt32](repeating: 0, count: RouterHostReadback.wordCount(topK: Self.topK))
        let empty = zeroed.withUnsafeBufferPointer {
            RouterHostReadback.isComplete(words: $0.baseAddress!, topK: Self.topK, tag: 1)
        }
        #expect(empty == false)
    }

    @Test func decodeReturnsNilWhenAMiddleWordIsStale() {
        var words = Self.words(tag: 7)
        words[RouterHostReadback.fixedWords + Self.topK + 1] =
            RouterHostReadback.word(tag: 6, value: 0x3800)
        let stale = words.withUnsafeBufferPointer {
            RouterHostReadback.decode(words: $0.baseAddress!, topK: Self.topK, tag: 7)
        }
        #expect(stale == nil)
    }

    @Test func decodeReturnsNilWhenOnlyTheCountsAreTagged() {
        var words = [UInt32](repeating: 0, count: RouterHostReadback.wordCount(topK: Self.topK))
        words[0] = RouterHostReadback.word(tag: 7, value: 3)
        words[1] = RouterHostReadback.word(tag: 7, value: 1)
        let partial = words.withUnsafeBufferPointer {
            (RouterHostReadback.isComplete(words: $0.baseAddress!, topK: Self.topK, tag: 7),
             RouterHostReadback.decode(words: $0.baseAddress!, topK: Self.topK, tag: 7))
        }
        #expect(partial.0 == false)
        #expect(partial.1 == nil)
    }

    @Test func decodeClampsCountsToTopK() {
        let words = Self.words(tag: 3, hitCount: 9, missCount: 9)
        let readback = words.withUnsafeBufferPointer {
            RouterHostReadback.decode(words: $0.baseAddress!, topK: Self.topK, tag: 3)
        }
        #expect(readback?.hitCount == 4)
        #expect(readback?.missCount == 4)
        #expect(readback?.hitPositions.count == 4)
    }

    @Test func nextTagSkipsZeroAndWrapsAt16Bits() {
        #expect(RouterHostReadback.nextTag(after: 0) == 1)
        #expect(RouterHostReadback.nextTag(after: 5) == 6)
        #expect(RouterHostReadback.nextTag(after: 0xffff) == 1)
    }
}
