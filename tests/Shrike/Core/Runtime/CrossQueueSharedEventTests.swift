import Foundation
import Metal
import Testing
@testable import Shrike

/// Pins the cross-queue shared-event ordering assumptions the expert-IO
/// event synchronization relies on (v10 T5's default, the only path since v17:
/// the miss-fixup CB waits on the reader queue's completion signal).
@Suite struct CrossQueueSharedEventTests {
    private static let timeout: DispatchTimeInterval = .seconds(20)

    @Test func sameQueueCommitOrderMakesWritesVisible() throws {
        let context = try MetalContext()
        let device = context.device
        let staging = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        let out = try #require(device.makeBuffer(length: 4, options: .storageModeShared))

        let first = try #require(context.queue.makeCommandBuffer())
        let fill = try #require(first.makeBlitCommandEncoder())
        fill.fill(buffer: staging, range: 0..<4, value: 7)
        fill.endEncoding()

        let second = try #require(context.queue.makeCommandBuffer())
        let copy = try #require(second.makeBlitCommandEncoder())
        copy.copy(from: staging, sourceOffset: 0, to: out, destinationOffset: 0, size: 4)
        copy.endEncoding()

        first.commit()
        second.commit()
        second.waitUntilCompleted()
        #expect(first.status == .completed)
        #expect(second.status == .completed)
        #expect(out.contents().load(as: UInt8.self) == 7)
    }

    @Test func crossQueueSignalReleasesWaiterCommittedFirst() throws {
        let context = try MetalContext()
        let device = context.device
        let event = try #require(device.makeSharedEvent())
        let fixupQueue = try #require(device.makeCommandQueue())
        let staging = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        let out = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        defer { event.signaledValue = .max }

        let waiter = try #require(context.queue.makeCommandBuffer())
        waiter.encodeWaitForEvent(event, value: 1)
        let consume = try #require(waiter.makeBlitCommandEncoder())
        consume.copy(from: staging, sourceOffset: 0, to: out, destinationOffset: 0, size: 4)
        consume.endEncoding()
        let done = DispatchSemaphore(value: 0)
        waiter.addCompletedHandler { _ in done.signal() }
        waiter.commit()

        let signaler = try #require(fixupQueue.makeCommandBuffer())
        let produce = try #require(signaler.makeBlitCommandEncoder())
        produce.fill(buffer: staging, range: 0..<4, value: 42)
        produce.endEncoding()
        signaler.encodeSignalEvent(event, value: 1)
        signaler.commit()

        #expect(done.wait(timeout: .now() + Self.timeout) == .success,
                "cross-queue signal never released the waiting command buffer")
        #expect(waiter.status == .completed)
        #expect(out.contents().load(as: UInt8.self) == 42)
    }

    @Test func hostSignalReleasesWaitingCommandBuffer() throws {
        let context = try MetalContext()
        let device = context.device
        let event = try #require(device.makeSharedEvent())
        let out = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        defer { event.signaledValue = .max }

        let waiter = try #require(context.queue.makeCommandBuffer())
        waiter.encodeWaitForEvent(event, value: 5)
        let fill = try #require(waiter.makeBlitCommandEncoder())
        fill.fill(buffer: out, range: 0..<4, value: 9)
        fill.endEncoding()
        let done = DispatchSemaphore(value: 0)
        waiter.addCompletedHandler { _ in done.signal() }
        waiter.commit()

        event.signaledValue = 5

        #expect(done.wait(timeout: .now() + Self.timeout) == .success,
                "host signal never released the waiting command buffer")
        #expect(waiter.status == .completed)
        #expect(out.contents().load(as: UInt8.self) == 9)
    }

    @Test func alternatingHostAndCommandBufferSignalersAdvanceTheTimeline() throws {
        let context = try MetalContext()
        let device = context.device
        let event = try #require(device.makeSharedEvent())
        let fixupQueue = try #require(device.makeCommandQueue())
        let out = try #require(device.makeBuffer(length: 4, options: .storageModeShared))
        defer { event.signaledValue = .max }

        let waiter = try #require(context.queue.makeCommandBuffer())
        waiter.encodeWaitForEvent(event, value: 3)
        let fill = try #require(waiter.makeBlitCommandEncoder())
        fill.fill(buffer: out, range: 0..<4, value: 11)
        fill.endEncoding()
        let done = DispatchSemaphore(value: 0)
        waiter.addCompletedHandler { _ in done.signal() }
        waiter.commit()

        let first = try #require(fixupQueue.makeCommandBuffer())
        first.encodeSignalEvent(event, value: 1)
        first.commit()
        event.signaledValue = 2
        let third = try #require(fixupQueue.makeCommandBuffer())
        third.encodeSignalEvent(event, value: 3)
        third.commit()

        #expect(done.wait(timeout: .now() + Self.timeout) == .success,
                "alternating signalers never advanced the timeline to the waited value")
        #expect(waiter.status == .completed)
        #expect(event.signaledValue >= 3)
        #expect(out.contents().load(as: UInt8.self) == 11)
    }
}
