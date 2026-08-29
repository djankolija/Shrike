import Testing
import Foundation
import Metal
@testable import Shrike

/// Tests `KVCacheManager` FP16 shape, growth, separate K/V storage, ring,
/// and reset semantics against the Qwen 3.6 config. Qwen has no
/// sliding-window layers (mask values are 1 = full and 2 = linear), so the
/// FP16 ring never engages and linear layers carry no per-token K/V rows.
@Suite struct KVCacheManagerTests {

    private let config = ArchConfig.qwen36_35B_A3B

    private func makeManager(maxContext: Int,
                             fp16RingEnabled: Bool = false) throws -> (MetalContext, KVCacheManager) {
        let ctx = try MetalContext()
        let kv = try KVCacheManager(device: ctx.device,
                                    config: config,
                                    maxContext: maxContext,
                                    fp16RingEnabled: fp16RingEnabled,
                                    slidingWindow: config.slidingWindow,
                                    maxPrefillChunkTokens: 128)
        return (ctx, kv)
    }

    @Test func strideAndBufferSizes_matchConfig() throws {
        let (_, kv) = try makeManager(maxContext: 128)

        // Full: numFullKVHeads(2) * fullHeadDim(256) * 2 = 1024 B/token.
        // Linear layers carry no per-token K/V storage at all.
        #expect(kv.kRange(layer: 3, start: 0, count: 1).stride == 2 * 256 * 2)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 128 * 1024)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.stride(layer: 0) == 0)
        #expect(kv.capacity(layer: 0) == 0)
    }

    @Test func linearGrowth_tracksAdvance() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.position == 0)
        for n in 1...100 {
            kv.advance()
            #expect(kv.position == n)
        }
    }

    /// Full layers run k_norm + RoPE on K while V runs the no-scale norm
    /// without RoPE, so they require separate cache slots.
    @Test func fullLayer_separatesKAndVBuffers() throws {
        let (_, kv) = try makeManager(maxContext: 16)
        let k = kv.keyBuffer(layer: 3, validTokenCount: 0)
        let v = kv.valueBuffer(layer: 3, validTokenCount: 0)
        #expect(k !== v, "full-layer K and V must NOT alias")
        let ks = kv.kSlot(layer: 3, position: 3)
        let vs = kv.vSlot(layer: 3, position: 3)
        #expect(ks.buffer !== vs.buffer, "full-layer K/V slots must NOT alias")
        // Offsets are still per-position-strided in both buffers.
        #expect(ks.offset == vs.offset)
    }

    @Test func slotOffsets_areLinear() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 3).offset == 3 * 1024)
        #expect(kv.vSlot(layer: 3, position: 7).offset == 7 * 1024)
    }

    @Test func fp16Ring_neverEngagesWithoutSWALayers() throws {
        let (_, kv) = try makeManager(maxContext: 4096,
                                      fp16RingEnabled: true)

        #expect(kv.fp16RingEnabled)
        // Full layers stay linear; no SWA layer exists to cap.
        #expect(kv.capacity(layer: 3) == 4096)
        #expect(kv.ringCapacity(layer: 3) == 0)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0).length == 4096 * 1024)
        // Linear layers keep no KV rows even with the ring enabled.
        #expect(kv.capacity(layer: 0) == 0)
        #expect(kv.ringCapacity(layer: 0) == 0)
    }

    @Test func fp16Ring_slotOffsetsNeverWrap() throws {
        let (_, kv) = try makeManager(maxContext: 128,
                                      fp16RingEnabled: true)

        // No SWA layer wraps; full-layer slots stay linear within maxContext.
        #expect(kv.kSlot(layer: 3, position: 0).offset == 0)
        #expect(kv.kSlot(layer: 3, position: 127).offset == 127 * 1024)
        #expect(kv.vSlot(layer: 3, position: 35).offset == 35 * 1024)
    }

    @Test func rangeSlotsHaveLinearOffsets() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        let fullStride = kv.kRange(layer: 3, start: 0, count: 1).stride

        let k = kv.kRange(layer: 3, start: 7, count: 3)
        let v = kv.vRange(layer: 7, start: 11, count: 5)

        #expect(k.offset == 7 * fullStride)
        #expect(k.stride == fullStride)
        #expect(v.offset == 11 * fullStride)
        #expect(v.stride == fullStride)
        #expect(k.buffer === kv.keyBuffer(layer: 3, validTokenCount: 0))
        #expect(v.buffer === kv.valueBuffer(layer: 7, validTokenCount: 0))
    }

    @Test func advanceByCountTracksCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 31)
        #expect(kv.position == 31)
        kv.advance(by: 0)
        #expect(kv.position == 31)
        kv.advance()
        #expect(kv.position == 32)
    }

    @Test func reset_clearsPosition() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        for _ in 0..<100 { kv.advance() }
        #expect(kv.position == 100)
        kv.reset()
        #expect(kv.position == 0)
        // Cursor reusable after reset.
        kv.advance()
        #expect(kv.position == 1)
    }

    @Test func speculativeRewindMovesOnlyTheLogicalCursor() throws {
        let (_, kv) = try makeManager(maxContext: 128)
        kv.advance(by: 17)
        let slotBefore = kv.kSlot(layer: 3, position: 12)
        try kv.rewind(to: 12)
        #expect(kv.position == 12)
        let slotAfter = kv.kSlot(layer: 3, position: 12)
        #expect(slotBefore.buffer === slotAfter.buffer)
        #expect(slotBefore.offset == slotAfter.offset)
        #expect(throws: InferenceStateSnapshotError.self) {
            try kv.rewind(to: 13)
        }
    }

    // MARK: - Kimi MLA layers (mask value 3)

    private func makeKimiManager(maxContext: Int,
                                 precision: KVCachePrecision = .fp16) throws
        -> KVCacheManager {
        let ctx = try MetalContext()
        return try KVCacheManager(device: ctx.device,
                                  config: .kimiLinear48bA3b,
                                  maxContext: maxContext,
                                  precision: precision,
                                  maxPrefillChunkTokens: 128)
    }

    /// MLA rows are one fused [latent 512 | k_pe 64] FP16 row per token even
    /// under a quantized manager precision, and V aliases K.
    @Test func mlaLayers_fuseKVIntoOneFP16Buffer() throws {
        let kv = try makeKimiManager(maxContext: 128, precision: .int8)
        let mlaLayer = 3   // 1-indexed layer 4
        #expect(kv.layerKind(mlaLayer) == .mla)
        #expect(kv.layerKind(0) == .linear)
        #expect(kv.stride(layer: mlaLayer) == 576 * 2)
        let view = kv.keyView(layer: mlaLayer, validTokenCount: 0)
        #expect(view.precision == .fp16)
        #expect(view.valueBytes == 576 * 2)
        #expect(kv.keyBuffer(layer: mlaLayer, validTokenCount: 0) ===
                kv.valueBuffer(layer: mlaLayer, validTokenCount: 0),
                "MLA V must alias K")
        #expect(kv.kSlot(layer: mlaLayer, position: 5).offset == 5 * 1152)
    }

    @Test func mlaSnapshot_emitsZeroLengthVSegmentsAndRoundtrips() throws {
        let kv = try makeKimiManager(maxContext: 16)
        kv.advance(by: 3)
        let lengths = try kv.snapshotSegmentLengths(at: 3)
        // 7 MLA layers, each a (K, V) pair with the V segment empty.
        let mlaPairs = lengths.enumerated().filter { $0.offset % 2 == 1 }
        #expect(lengths.count == 14)
        #expect(mlaPairs.allSatisfy { $0.element == 0 })
        #expect(lengths.enumerated()
            .filter { $0.offset % 2 == 0 }
            .allSatisfy { $0.element == 3 * 1152 })

        let slot = kv.kSlot(layer: 3, position: 1)
        let marker = slot.buffer.contents().advanced(by: slot.offset)
            .assumingMemoryBound(to: Float16.self)
        for i in 0..<576 { marker[i] = Float16(Float(i % 13) * 0.25) }

        var payload = Data()
        try kv.appendSnapshotPayload(to: &payload, segmentLengths: lengths)

        let restored = try makeKimiManager(maxContext: 16)
        try payload.withUnsafeBytes { bytes in
            var offset = 0
            try restored.restoreSnapshot(position: 3, segmentLengths: lengths,
                                         bytes: bytes, offset: &offset)
            #expect(offset == payload.count)
        }
        #expect(restored.position == 3)
        let restoredSlot = restored.kSlot(layer: 3, position: 1)
        let restoredRow = restoredSlot.buffer.contents()
            .advanced(by: restoredSlot.offset)
            .assumingMemoryBound(to: Float16.self)
        for i in 0..<576 {
            #expect(restoredRow[i] == marker[i], "row element \(i) diverged")
        }
    }

    @Test func mlaGrowth_preservesAliasAndContent() throws {
        let kv = try makeKimiManager(maxContext: 32_768)
        kv.advance(by: 1)
        let slot = kv.kSlot(layer: 3, position: 0)
        let row = slot.buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<576 { row[i] = Float16(Float(i % 7)) }

        try kv.reserve(tokens: KVCacheManager.initialCapacityTokens + 1)
        #expect(kv.capacity(layer: 3) == 2 * KVCacheManager.initialCapacityTokens)
        #expect(kv.keyBuffer(layer: 3, validTokenCount: 0) ===
                kv.valueBuffer(layer: 3, validTokenCount: 0),
                "growth must preserve the MLA K/V alias")
        let grown = kv.kSlot(layer: 3, position: 0)
        let grownRow = grown.buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<576 {
            #expect(grownRow[i] == Float16(Float(i % 7)), "element \(i) lost in growth")
        }
    }

}
