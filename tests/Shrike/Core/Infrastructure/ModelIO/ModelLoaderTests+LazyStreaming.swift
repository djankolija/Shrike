import Foundation
import Metal
import Testing

@testable import Shrike

extension ModelLoaderTests {
  @Test func touchingOneLayerOpensExactlyOneStreamer() async throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .qwenToy())
    #expect(model.openLayerFileCount() == 0)
    _ = try await model.fetchRoutedExperts(layer: 0, experts: [3])
    #expect(model.openLayerFileCount() == 1)
    // Touch layer 0 again — no new open.
    _ = try await model.fetchRoutedExperts(layer: 0, experts: [5])
    #expect(model.openLayerFileCount() == 1)
    // Touch layer 1 — second open.
    _ = try await model.fetchRoutedExperts(layer: 1, experts: [0])
    #expect(model.openLayerFileCount() == 2)
  }

  @Test func routedExpertBytesRoundTrip() async throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .qwenToy())
    let view = try await model.fetchRoutedExperts(layer: 1, experts: [4])[0]
    let bufContents = view.buffer.contents()
    let b0 = bufContents.load(fromByteOffset: Int(view.offset), as: UInt8.self)
    let b1 = bufContents.load(fromByteOffset: Int(view.offset) + 1, as: UInt8.self)
    let b2 = bufContents.load(fromByteOffset: Int(view.offset) + 2, as: UInt8.self)
    let b3 = bufContents.load(fromByteOffset: Int(view.offset) + 3, as: UInt8.self)
    #expect(b0 == 1)  // layer 1
    #expect(b1 == 4)  // expert 4
    #expect(b2 == 0xC1)
    #expect(b3 == 0xC2)
  }

  @Test func tamperedLayerFileFailsOnFirstTouch() async throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let model = try Model.load(
      directoryURL: dir, device: device,
      expecting: .qwenToy())

    // Flip one byte inside layer_01.bin AFTER the manifest is written.
    let url = dir.appendingPathComponent("packed_experts/layer_01.bin")
    var data = try Data(contentsOf: url)
    data[100] ^= 0xFF
    try data.write(to: url)

    // Layer 0 still loads.
    _ = try await model.fetchRoutedExperts(layer: 0, experts: [0])
    // Layer 1 first touch fails with checksumMismatch.
    await #expect {
      _ = try await model.fetchRoutedExperts(layer: 1, experts: [0])
    } throws: { error in
      if case ModelError.checksumMismatch(let f) = error {
        return f == "packed_experts/layer_01.bin"
      }
      return false
    }
  }
}
