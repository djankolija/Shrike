import Testing
import Foundation
import Metal
@testable import Shrike

/// `Model.load(streamingMode:)` integration for the bounded pread cache.
@Suite struct ModelLoaderPreadTests {

    private static func readBytes(_ view: TensorView) -> [UInt8] {
        let base = view.buffer.contents().advanced(by: Int(view.offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: Int(view.length)))
    }

    @Test func loadsUnderPread_routedExpertBytesAndLazyOpen() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwenToy(),
                                   streamingMode: .pread(slotCount: 2))

        #expect(model.openLayerFileCount() == 0)
        let view = try await model.fetchRoutedExperts(layer: 1, experts: [4])[0]
        #expect(model.openLayerFileCount() == 1)

        // The view is a cache cell of the shared arena, not the expert's file offset:
        // layer 1's first slot is its first cell, one layer's worth of cells in.
        let residency = try model.routedExpertResidency(layer: 1)
        #expect(residency.poolChunks.contains { $0 === view.buffer })
        #expect(view.offset == UInt64(2) * residency.poolSlotStride)

        // Same tagged-byte contract as ModelLoaderTests.routedExpertBytesRoundTrip.
        let b = Self.readBytes(view)
        #expect(b[0] == 1)       // layer 1
        #expect(b[1] == 4)       // expert 4
        #expect(b[2] == 0xC1)
        #expect(b[3] == 0xC2)
    }

    @Test func routedExpertCacheSlotCountDoesNotOpenLayerFile() throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwenToy(),
                                   streamingMode: .pread(slotCount: 2))

        #expect(model.openLayerFileCount() == 0)
        #expect(model.routedExpertCacheSlotCount() == 2)
        #expect(model.openLayerFileCount() == 0)
    }

    @Test func expertCacheDescriptionNamesTheTableAndThePolicy() {
        #expect(RealForwardRunner.expertCacheDescription(.pread(slotCount: 128))
            == "expert_slots=uniform:128 policy=aging-lfu")
        #expect(RealForwardRunner.expertCacheDescription(
            .pread(slotCount: 128, perLayer: [0, 240, 103, 169], policy: .slru(protectedShare: 0.5)))
            == "expert_slots=103..240 policy=slru:0.5")
    }

    @Test func slruEvictsProbationBeforeProtectedWhereAgingLFUKeepsTheCounts() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let sequence = [0, 1, 2, 0, 1, 3, 4]
        let slru = try Model.load(directoryURL: dir, device: device, expecting: .qwenToy(),
                                  streamingMode: .pread(slotCount: 3,
                                                        policy: .slru(protectedShare: 0.34)))
        for expert in sequence {
            _ = try await slru.fetchRoutedExperts(layer: 1, experts: [expert])
        }
        #expect(Set(try slru.routedExpertResidentIDs(layer: 1)) == [1, 3, 4])
        let lfu = try Model.load(directoryURL: dir, device: device, expecting: .qwenToy(),
                                 streamingMode: .pread(slotCount: 3))
        for expert in sequence {
            _ = try await lfu.fetchRoutedExperts(layer: 1, experts: [expert])
        }
        #expect(Set(try lfu.routedExpertResidentIDs(layer: 1)) == [0, 1, 4])
    }

    @Test func perLayerSlotTablePlacesEachLayersCellsAfterThePreviousLayers() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let probe = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwenToy(),
                                   streamingMode: .pread(slotCount: 2))
        var table = Array(repeating: 2, count: probe.config.numLayers)
        table[0] = 3
        table[1] = 1
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwenToy(),
                                   streamingMode: .pread(slotCount: 2, perLayer: table))
        #expect(model.routedExpertCacheSlotCount() == 2)
        #expect(model.routedExpertCacheSlotCount(layer: 0) == 3)
        #expect(model.routedExpertCacheSlotCount(layer: 1) == 1)

        let view = try await model.fetchRoutedExperts(layer: 1, experts: [4])[0]
        let residency = try model.routedExpertResidency(layer: 1)
        #expect(residency.poolChunks.contains { $0 === view.buffer })
        #expect(view.offset == UInt64(3) * residency.poolSlotStride)
        let first = try await model.fetchRoutedExperts(layer: 0, experts: [4])[0]
        #expect(first.offset < UInt64(3) * residency.poolSlotStride)

        let wrong = Array(repeating: 2, count: probe.config.numLayers + 1)
        #expect(throws: ModelError.self) {
            _ = try Model.load(directoryURL: dir, device: device,
                               expecting: .qwenToy(),
                               streamingMode: .pread(slotCount: 2, perLayer: wrong))
        }
    }

    @Test func beginOpeningRoutedExpertStreamerIsCompatibleWithLazyFetch() async throws {
        let dir = try ModelLoaderTests.writeToySynthetic()
        defer { try? FileManager.default.removeItem(at: dir) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: dir, device: device,
                                   expecting: .qwenToy(),
                                   streamingMode: .pread(slotCount: 2))

        model.beginOpeningRoutedExpertStreamer(layer: 1)
        let view = try await model.fetchRoutedExperts(layer: 1, experts: [4])[0]

        #expect(model.openLayerFileCount() == 1)
        let b = Self.readBytes(view)
        #expect(b[0] == 1)
        #expect(b[1] == 4)
    }

}
