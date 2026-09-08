import Darwin
import Foundation
import Metal
import Testing

@testable import Shrike

extension PreadExpertStreamerTests {
  @Test func preadRoundTrip_matchesTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    for e in 0..<Self.numExperts {
      let r = try streamer.loadExpertsCached(experts: [e])[0]
      #expect(r.size == UInt64(Self.expertStride))
      let got = Self.bytes(of: r.buffer, offset: r.offset, count: Self.expertStride)
      #expect(
        got.allSatisfy { $0 == Self.tagByte(e) },
        "expert \(e) slot not uniformly tagged")
    }
  }

  @Test func slotReuse_evictionOverwrites() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 2)

    // With slotCount=2, experts 0,1,2,3 land in slots 0,1,0,1: expert 2
    // reuses slot 0's storage, which must then hold expert 2's tag. Slot
    // identity is the (buffer, offset) pair — pool layout shares one slab.
    let r0 = try streamer.loadExpertsCached(experts: [0])[0]
    let r1 = try streamer.loadExpertsCached(experts: [1])[0]
    let r2 = try streamer.loadExpertsCached(experts: [2])[0]
    let r3 = try streamer.loadExpertsCached(experts: [3])[0]

    #expect(r0.buffer === r2.buffer && r0.offset == r2.offset,
            "expert 0 and 2 should share slot 0")
    #expect(r1.buffer === r3.buffer && r1.offset == r3.offset,
            "expert 1 and 3 should share slot 1")
    #expect(r0.buffer !== r1.buffer || r0.offset != r1.offset,
            "slots 0 and 1 must be distinct storage")

    // r0 was overwritten by r2; reading slot 0 now yields expert 2's tag.
    let slot0 = Self.bytes(of: r2.buffer, offset: r2.offset, count: Self.expertStride)
    #expect(slot0.allSatisfy { $0 == Self.tagByte(2) })
    let slot1 = Self.bytes(of: r3.buffer, offset: r3.offset, count: Self.expertStride)
    #expect(slot1.allSatisfy { $0 == Self.tagByte(3) })
  }

}
