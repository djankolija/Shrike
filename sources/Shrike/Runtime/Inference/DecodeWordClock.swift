import Foundation

/// The per-layer clock of a token under one command per token (v20 T3.2), in
/// place of the per-layer GPU rows the fold retires: a layer's wall is the gap
/// between consecutive words, the first routed layer's from the commit, the
/// boundary's from the last word to the token word. The word lands about 45 µs
/// after its classifier (v18's mid-command visibility probe), so the clock is
/// good to that.
struct DecodeWordClock {
    private(set) var tokens = 0
    private(set) var firstNanos: UInt64 = 0
    private(set) var layerNanos: [UInt64]
    private(set) var boundaryNanos: UInt64 = 0
    private var lastNanos: UInt64 = 0
    private var wordsThisToken = 0

    init(layers: Int) {
        layerNanos = Array(repeating: 0, count: layers)
    }

    mutating func beginToken(at nanos: UInt64) {
        lastNanos = nanos
        wordsThisToken = 0
    }

    mutating func word(layer: Int, at nanos: UInt64) {
        guard layer >= 0, layer < layerNanos.count else { return }
        let delta = nanos &- lastNanos
        if wordsThisToken == 0 {
            firstNanos &+= delta
        } else {
            layerNanos[layer] &+= delta
        }
        lastNanos = nanos
        wordsThisToken += 1
    }

    mutating func boundary(at nanos: UInt64) {
        boundaryNanos &+= nanos &- lastNanos
        lastNanos = nanos
        endToken()
    }

    mutating func endToken() {
        tokens += 1
        wordsThisToken = 0
    }

    func meanMillis(_ nanos: UInt64) -> Double {
        tokens > 0 ? Double(nanos) / Double(tokens) / 1_000_000 : 0
    }

    func line() -> String {
        let layers = layerNanos.map { String(format: "%.4f", meanMillis($0)) }
            .joined(separator: ",")
        return String(format: "word_clock tokens=%d first_ms=%.4f layer_ms=%@ boundary_ms=%.4f",
                      tokens, meanMillis(firstNanos), layers, meanMillis(boundaryNanos))
    }
}
