import Darwin
import Foundation
import Metal
import Testing

@testable import Shrike

@Suite struct ExpertCellArenaTests {
    @Test func cellsArePageAlignedRegionsOfOneBufferAtTheStride() throws {
        let device = try MetalContext().device
        let pageSize = Int(getpagesize())
        let stride = 2 * pageSize
        let arena = try ExpertCellArena(device: device, cellCount: 5, stride: stride)
        #expect(arena.cellCount == 5)
        #expect(arena.stride == stride)
        #expect(arena.chunkBuffers.count == 1)
        #expect(arena.cellsPerChunk == 5)
        #expect(arena.chunkBuffers[0].length == 5 * stride)
        for cell in 0..<5 {
            #expect(arena.offset(cell: cell) == UInt64(cell * stride))
            #expect(arena.bufferOffset(cell: cell) == UInt64(cell * stride))
            #expect(arena.cell(atOffset: arena.offset(cell: cell)) == cell)
            #expect(arena.buffer(cell: cell) === arena.chunkBuffers[0])
            #expect(arena.pointer(cell: cell) == arena.chunkBuffers[0].contents().advanced(by: cell * stride))
            #expect(Int(bitPattern: arena.pointer(cell: cell)) % pageSize == 0)
        }
    }

    @Test func cellsSpanChunksWhenOneBufferCannotHoldThemAndKeepTheirGlobalNames() throws {
        let device = try MetalContext().device
        let stride = 2 * Int(getpagesize())
        let arena = try ExpertCellArena(device: device, cellCount: 5, stride: stride, chunkBytes: 2 * stride)
        #expect(arena.cellsPerChunk == 2)
        #expect(arena.chunkBuffers.map(\.length) == [2 * stride, 2 * stride, stride])
        for cell in 0..<5 {
            let chunk = cell / 2
            #expect(arena.buffer(cell: cell) === arena.chunkBuffers[chunk])
            #expect(arena.bufferOffset(cell: cell) == UInt64((cell % 2) * stride))
            #expect(arena.offset(cell: cell) == UInt64(cell * stride))
            #expect(arena.cell(atOffset: arena.offset(cell: cell)) == cell)
            #expect(arena.pointer(cell: cell)
                    == arena.chunkBuffers[chunk].contents().advanced(by: (cell % 2) * stride))
        }
        let addresses = arena.bases.contents().assumingMemoryBound(to: UInt64.self)
        #expect(addresses[0] == arena.chunkBuffers[0].gpuAddress)
        #expect(addresses[2] == arena.chunkBuffers[2].gpuAddress)
        #expect(addresses[7] == arena.chunkBuffers[2].gpuAddress)
        #expect(arena.bases.contents().load(fromByteOffset: 64, as: UInt32.self) == 2)
    }

    @Test func moreChunksThanTheKernelsAddressAreRefused() throws {
        let device = try MetalContext().device
        let stride = Int(getpagesize())
        #expect(throws: (any Error).self) {
            try ExpertCellArena(device: device, cellCount: 9, stride: stride, chunkBytes: stride)
        }
    }

    @Test func everyCellStartsAtZeroAndEveryBumpDrawsAGreaterValueFromOneClock() throws {
        let device = try MetalContext().device
        let arena = try ExpertCellArena(device: device, cellCount: 3, stride: 2 * Int(getpagesize()))
        #expect((0..<3).map { arena.cellGeneration($0) } == [0, 0, 0])
        let first = arena.bumpCellGeneration(1)
        let second = arena.bumpCellGeneration(2)
        let third = arena.bumpCellGeneration(1)
        #expect(first > 0)
        #expect(second > first)
        #expect(third > second)
        #expect(arena.cellGeneration(0) == 0)
        #expect(arena.cellGeneration(1) == third)
        #expect(arena.cellGeneration(2) == second)
    }

    @Test func anEmptyOrUnalignedGeometryIsRefused() throws {
        let device = try MetalContext().device
        let pageSize = Int(getpagesize())
        #expect(throws: (any Error).self) {
            try ExpertCellArena(device: device, cellCount: 0, stride: pageSize)
        }
        #expect(throws: (any Error).self) {
            try ExpertCellArena(device: device, cellCount: 1, stride: pageSize + 1)
        }
    }
}
