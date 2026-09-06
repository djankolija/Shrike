import Foundation
import ShrikeKernelsC

/// The residency classifier's copy of what the host reads after a router, each
/// word tagged with the layer's sequence in its high 16 bits: a host polling
/// the copy before the command is marked complete can trust only words that
/// validate themselves (on the M1 a payload trails a flag by up to 8.5 µs).
struct RouterHostReadback: Equatable {
    static let tagBits: UInt32 = 16
    static let valueMask: UInt32 = 0xffff
    static let fixedWords = 2

    let hitCount: Int
    let missCount: Int
    let expertIDs: [UInt32]
    let weightBits: [UInt16]
    let hitPositions: [UInt32]
    let missPositions: [UInt32]
    let predictedIDs: [UInt32]

    /// The counts, then five `topK` runs: ids, weight bits, hit positions,
    /// miss positions, predicted ids (`moe_publish_router_readback` writes them).
    static func wordCount(topK: Int) -> Int { fixedWords + 5 * topK }

    static func word(tag: UInt32, value: UInt32) -> UInt32 {
        (tag << tagBits) | (value & valueMask)
    }

    /// Zero is never a tag, so a zeroed buffer never reads as complete.
    static func nextTag(after tag: UInt32) -> UInt32 {
        tag >= valueMask ? 1 : tag + 1
    }

    static func isComplete(words: UnsafePointer<UInt32>, topK: Int, tag: UInt32) -> Bool {
        for index in 0..<wordCount(topK: topK)
        where shrike_load_acquire_u32(words + index) >> tagBits != tag {
            return false
        }
        return true
    }

    static func decode(words: UnsafePointer<UInt32>, topK: Int, tag: UInt32) -> RouterHostReadback? {
        let count = wordCount(topK: topK)
        var values = [UInt32](repeating: 0, count: count)
        for index in 0..<count {
            let word = shrike_load_acquire_u32(words + index)
            guard word >> tagBits == tag else { return nil }
            values[index] = word & valueMask
        }
        func run(_ offset: Int, _ length: Int) -> [UInt32] {
            Array(values[offset..<offset + length])
        }
        let hits = min(Int(values[0]), topK)
        let misses = min(Int(values[1]), topK)
        return RouterHostReadback(
            hitCount: hits,
            missCount: misses,
            expertIDs: run(fixedWords, topK),
            weightBits: run(fixedWords + topK, topK).map { UInt16($0) },
            hitPositions: run(fixedWords + 2 * topK, hits),
            missPositions: run(fixedWords + 3 * topK, misses),
            predictedIDs: run(fixedWords + 4 * topK, topK))
    }
}
