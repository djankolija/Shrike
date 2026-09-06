import Foundation
import Metal
import Testing

@testable import Shrike

@Suite struct PrefetchAdoptionTransferTests {
    private static let stride = 16_384

    private func makeBuffer(_ device: MTLDevice, length: Int, fill: UInt8) throws -> MTLBuffer {
        let buffer = try #require(device.makeBuffer(length: length, options: .storageModeShared))
        memset(buffer.contents(), Int32(fill), length)
        return buffer
    }

    @Test func encodeCopyLandsEachSourceInItsSlabRegion() throws {
        let context = try MetalContext()
        let first = try makeBuffer(context.device, length: Self.stride, fill: 0xA1)
        let second = try makeBuffer(context.device, length: Self.stride, fill: 0xB2)
        let slab = try makeBuffer(context.device, length: 4 * Self.stride, fill: 0)
        var released = 0
        let transfer = PrefetchAdoptionTransfer(
            sources: [first, second], destinations: [slab, slab],
            destinationOffsets: [Self.stride, 3 * Self.stride], byteCount: Self.stride) { _ in released += 1 }

        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        try transfer.encodeCopy(commandBuffer: commandBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let bytes = UnsafeRawBufferPointer(start: slab.contents(), count: slab.length)
        #expect(bytes[0] == 0 && bytes[Self.stride - 1] == 0)
        #expect(bytes[Self.stride] == 0xA1 && bytes[2 * Self.stride - 1] == 0xA1)
        #expect(bytes[2 * Self.stride] == 0)
        #expect(bytes[3 * Self.stride] == 0xB2 && bytes[4 * Self.stride - 1] == 0xB2)

        transfer.release()
        transfer.release()
        #expect(released == 1)
    }

    @Test func encodeCopyRefusesARangeBeyondTheSlab() throws {
        let context = try MetalContext()
        let source = try makeBuffer(context.device, length: Self.stride, fill: 0xC3)
        let slab = try makeBuffer(context.device, length: Self.stride, fill: 0)
        let transfer = PrefetchAdoptionTransfer(
            sources: [source], destinations: [slab], destinationOffsets: [Self.stride / 2],
            byteCount: Self.stride) { _ in }
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        #expect(throws: ModelError.self) { try transfer.encodeCopy(commandBuffer: commandBuffer) }
    }
}
