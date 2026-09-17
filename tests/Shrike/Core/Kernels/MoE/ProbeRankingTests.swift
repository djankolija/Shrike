import Testing
@testable import Shrike

struct ProbeRankingTests {
    @Test func rankingFollowsTheSelectKernelsOrder() {
        let logits: [Float] = [0.5, 2.0, 2.0, -1.0, 3.0]
        let bias: [UInt16] = [0, 0, 0, 0x4080, 0]
        logits.withUnsafeBufferPointer { scores in
            bias.withUnsafeBufferPointer { shifts in
                let ranking = RealForwardRunner.probeRanking(
                    logits: scores.baseAddress!, bias: shifts.baseAddress!,
                    sigmoid: false, count: 5, width: 4)
                #expect(ranking == [3, 4, 1, 2])
            }
            let plain = RealForwardRunner.probeRanking(
                logits: scores.baseAddress!, bias: nil, sigmoid: false, count: 5, width: 8)
            #expect(plain == [4, 1, 2, 0, 3])
        }
    }

    @Test func sigmoidScoringKeepsTheOrderOfMonotoneLogits() {
        let logits: [Float] = [-2.0, 4.0, 0.0]
        logits.withUnsafeBufferPointer { scores in
            let ranking = RealForwardRunner.probeRanking(
                logits: scores.baseAddress!, bias: nil, sigmoid: true, count: 3, width: 3)
            #expect(ranking == [1, 2, 0])
        }
    }

    @Test func traceLineFormats() {
        #expect(RealForwardRunner.formatRouteTraceTokenLine(position: 289, id: 1234) == "t 289 1234\n")
        #expect(RealForwardRunner.formatRouteTracePrefillRowLine(position: 7, layer: 3, experts: [5, 1, 9])
            == "q 7 3 5 1 9\n")
    }
}
