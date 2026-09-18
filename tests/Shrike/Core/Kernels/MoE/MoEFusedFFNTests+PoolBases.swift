import Metal
import Testing
@testable import Shrike

extension MoEFusedFFNTests {
    /// The arena's `PoolBases` for a pool that is one buffer: every base the
    /// same address, the cells per chunk past any cell the tests name.
    static func poolBases(_ device: MTLDevice, _ pool: MTLBuffer) -> MTLBuffer {
        let bases = device.makeBuffer(length: ExpertCellArena.maxChunks * 8 + 8,
                                      options: .storageModeShared)!
        let addresses = bases.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<ExpertCellArena.maxChunks { addresses[i] = pool.gpuAddress }
        bases.contents().storeBytes(of: UInt32.max, toByteOffset: ExpertCellArena.maxChunks * 8,
                                    as: UInt32.self)
        return bases
    }
}
