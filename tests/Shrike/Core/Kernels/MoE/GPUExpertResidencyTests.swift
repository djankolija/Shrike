import Metal
import Testing

@testable import Shrike

@Suite struct GPUExpertResidencyTests {
    private struct Classification {
        let hits: [UInt32]
        let misses: [UInt32]
        let missExperts: [UInt32]
        let slots: [UInt32]
        let generations: [UInt64]
        var specArgs: [UInt32] = []
    }

    private static let phase1FullGrid = MTLSize(width: 256, height: 1, depth: 1)
    private static let phase2FullGrid = MTLSize(width: 13, height: 7, depth: 1)
    private static let zeroGrids: [UInt32] = [0, 1, 1, 0, 1, 1]
    private static let fullGrids: [UInt32] = [256, 1, 1, 13, 7, 1]

    @Test func loadingResidentAndEvictedEntriesClassifyCorrectly() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path),
            device: context.device,
            slotCount: 2)
        let moe = try MoE(context: context,
                          siluActivation: true,
                          specializedD: 2048,
                          specializedF: 512,
                          specializedNumExperts: 4)

        let loading = try streamer.planExpertsCached(experts: [0])
        #expect(streamer.residencyEntry(expert: 0).state
                == ExpertResidencyEntry.loading)
        var result = try classify([0, 1, 2, 3], streamer: streamer,
                                  moe: moe, context: context)
        #expect(result.hits.isEmpty)
        #expect(result.misses == [0, 1, 2, 3])

        _ = try streamer.executeExpertCachePlan(loading)
        _ = try streamer.loadExpertsCached(experts: [2])
        result = try classify([0, 1, 2, 3], streamer: streamer,
                              moe: moe, context: context)
        #expect(result.hits == [0, 2])
        #expect(result.misses == [1, 3])
        #expect(result.missExperts == [1, 3])
        #expect(result.slots[0] != ExpertResidencyEntry.notResidentSlot)
        #expect(result.slots[2] != ExpertResidencyEntry.notResidentSlot)
        #expect(result.generations[0] > 0)

        _ = try streamer.loadExpertsCached(experts: [1, 3])
        result = try classify([0, 1, 2, 3], streamer: streamer,
                              moe: moe, context: context)
        #expect(result.hits == [1, 3])
        #expect(result.misses == [0, 2])
        #expect(streamer.residencyEntry(expert: 0).state
                == ExpertResidencyEntry.empty)
    }

    @Test func speculativeDispatchArgumentsFollowResidency() throws {
        let url = try PreadExpertStreamerTests.writeSyntheticLayer()
        defer { try? FileManager.default.removeItem(at: url) }
        let context = try MetalContext()
        let streamer = try PreadExpertStreamer(
            layout: PreadExpertStreamerTests.makeLayout(path: url.path),
            device: context.device,
            slotCount: 2)
        let moe = try MoE(context: context,
                          siluActivation: true,
                          specializedD: 2048,
                          specializedF: 512,
                          specializedNumExperts: 4)

        var result = try classify([0, 1, 2, 3], streamer: streamer,
                                  moe: moe, context: context, speculative: true)
        #expect(result.misses == [0, 1, 2, 3])
        #expect(result.specArgs == Self.zeroGrids)

        _ = try streamer.executeExpertCachePlan(
            try streamer.planExpertsCached(experts: [1]))
        _ = try streamer.loadExpertsCached(experts: [3])
        result = try classify([0, 1, 2, 3], streamer: streamer,
                              moe: moe, context: context, speculative: true)
        #expect(result.hits == [1, 3])
        #expect(result.specArgs == Self.zeroGrids)

        result = try classify([1, 3], streamer: streamer,
                              moe: moe, context: context, speculative: true)
        #expect(result.misses.isEmpty)
        #expect(result.specArgs == Self.fullGrids)

        let base = try classify([1, 3], streamer: streamer,
                                moe: moe, context: context)
        #expect(base.hits == result.hits)
        #expect(base.slots == result.slots)
        #expect(base.generations == result.generations)
    }

    private func classify(_ experts: [UInt32],
                          streamer: PreadExpertStreamer,
                          moe: MoE,
                          context: MetalContext,
                          speculative: Bool = false) throws -> Classification {
        func buffer<T>(_ values: [T]) -> MTLBuffer {
            values.withUnsafeBytes { bytes in
                context.device.makeBuffer(
                    bytes: bytes.baseAddress!,
                    length: max(1, bytes.count),
                    options: .storageModeShared)!
            }
        }
        let topK = buffer(experts)
        let hitCount = buffer([UInt32(0)])
        let hitPositions = buffer([UInt32](repeating: 0, count: experts.count))
        let missCount = buffer([UInt32(0)])
        let missPositions = buffer([UInt32](repeating: 0, count: experts.count))
        let missExperts = buffer([UInt32](repeating: 0, count: experts.count))
        let slots = buffer([UInt32](repeating: 0, count: experts.count))
        let generations = buffer([UInt64](repeating: 0, count: experts.count))
        let resources = streamer.expertResidencyResources()
        let specArgsBuffer = speculative
            ? context.device.makeBuffer(length: MoE.specDispatchArgsLength,
                                        options: .storageModeShared)!
            : nil
        let commandBuffer = context.queue.makeCommandBuffer()!
        try moe.encodeResidencyClassification(
            commandBuffer: commandBuffer,
            topKIndices: topK,
            residencyTable: resources.table,
            hitCount: hitCount,
            hitPositions: hitPositions,
            missCount: missCount,
            missPositions: missPositions,
            missExperts: missExperts,
            resolvedSlots: slots,
            resolvedGenerations: generations,
            topK: UInt32(experts.count),
            numExperts: UInt32(resources.expertCount),
            speculative: specArgsBuffer.map {
                MoE.SpeculativeDispatchArguments(
                    arguments: $0,
                    phase1Threadgroups: Self.phase1FullGrid,
                    phase2Threadgroups: Self.phase2FullGrid)
            })
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }

        func values<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
            Array(UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: T.self, capacity: count),
                count: count))
        }
        let hitN = Int(hitCount.contents().load(as: UInt32.self))
        let missN = Int(missCount.contents().load(as: UInt32.self))
        return Classification(
            hits: values(hitPositions, count: hitN, as: UInt32.self),
            misses: values(missPositions, count: missN, as: UInt32.self),
            missExperts: values(missExperts, count: missN, as: UInt32.self),
            slots: values(slots, count: experts.count, as: UInt32.self),
            generations: values(generations, count: experts.count, as: UInt64.self),
            specArgs: specArgsBuffer.map {
                values($0, count: 6, as: UInt32.self)
            } ?? [])
    }
}
