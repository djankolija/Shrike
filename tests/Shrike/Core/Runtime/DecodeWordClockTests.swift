import Testing

@testable import Shrike

@Suite struct DecodeWordClockTests {
    @Test func aTokensWordsBecomePerLayerWallsFromTheCommit() {
        var clock = DecodeWordClock(layers: 3)
        clock.beginToken(at: 1_000)
        clock.word(layer: 0, at: 1_400)
        clock.word(layer: 1, at: 2_400)
        clock.word(layer: 2, at: 3_000)
        clock.boundary(at: 3_250)

        #expect(clock.tokens == 1)
        #expect(clock.firstNanos == 400)
        #expect(clock.layerNanos == [0, 1_000, 600])
        #expect(clock.boundaryNanos == 250)
    }

    @Test func aSecondTokenAccumulatesAndTheMeansDivideByTokens() {
        var clock = DecodeWordClock(layers: 2)
        clock.beginToken(at: 0)
        clock.word(layer: 0, at: 100)
        clock.word(layer: 1, at: 300)
        clock.boundary(at: 400)
        clock.beginToken(at: 10_000)
        clock.word(layer: 0, at: 10_300)
        clock.word(layer: 1, at: 10_500)
        clock.boundary(at: 10_600)

        #expect(clock.tokens == 2)
        #expect(clock.layerNanos == [0, 400])
        #expect(clock.meanMillis(clock.firstNanos) == 0.0002)
        #expect(clock.meanMillis(clock.layerNanos[1]) == 0.0002)
        #expect(clock.line().hasPrefix("word_clock tokens=2 first_ms=0.0002 layer_ms=0.0000,0.0002 boundary_ms=0.0001"))
    }

    @Test func aTokenWithoutABoundaryWordCountsItsLayersOnly() {
        var clock = DecodeWordClock(layers: 2)
        clock.beginToken(at: 0)
        clock.word(layer: 0, at: 50)
        clock.word(layer: 1, at: 80)
        clock.endToken()

        #expect(clock.tokens == 1)
        #expect(clock.layerNanos == [0, 30])
        #expect(clock.boundaryNanos == 0)
    }
}
