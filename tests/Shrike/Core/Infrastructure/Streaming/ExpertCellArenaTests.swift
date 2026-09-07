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
        #expect(arena.buffer.length == 5 * stride)
        for cell in 0..<5 {
            #expect(arena.offset(cell: cell) == UInt64(cell * stride))
            #expect(arena.cell(atOffset: arena.offset(cell: cell)) == cell)
            #expect(arena.pointer(cell: cell) == arena.buffer.contents().advanced(by: cell * stride))
            #expect(Int(bitPattern: arena.pointer(cell: cell)) % pageSize == 0)
        }
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
