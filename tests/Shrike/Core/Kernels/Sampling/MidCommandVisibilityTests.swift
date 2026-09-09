import Testing
import Foundation
import Metal
@testable import Shrike

/// The merge behind the classifier (v18 Task 3) needs the host to see a word
/// written mid-command before the command ends.
@Suite struct MidCommandVisibilityTests {
  @Test func hostSeesAMidCommandWriteBeforeTheCommandEnds() throws {
    let ctx = try MetalContext()
    let sampler = try Sample(context: ctx)
    let v = 256
    let big = 256 << 20
    guard let probs = ctx.device.makeBuffer(length: v * MemoryLayout<Float16>.size,
                                            options: .storageModeShared),
          let word = ctx.device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                           options: .storageModeShared),
          let source = ctx.device.makeBuffer(length: big, options: .storageModeShared),
          let sink = ctx.device.makeBuffer(length: big, options: .storageModeShared)
    else { throw ModelError.residentBufferWrapFailed }
    let p = probs.contents().bindMemory(to: Float16.self, capacity: v)
    for i in 0..<v { p[i] = 0 }
    p[7] = 1
    word.contents().storeBytes(of: 0xFFFF_FFFF, as: UInt32.self)

    guard let cb = ctx.queue.makeCommandBuffer() else { throw ModelError.residentBufferWrapFailed }
    try sampler.encode(commandBuffer: cb, probs: probs, outToken: word,
                       v: UInt32(v), temperature: 0, seed: 1)
    for _ in 0..<6 {
      guard let blit = cb.makeBlitCommandEncoder() else { throw MetalError.commandEncoderFailed }
      blit.copy(from: source, sourceOffset: 0, to: sink, destinationOffset: 0, size: big)
      blit.endEncoding()
    }
    cb.commit()
    let deadline = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) + 5_000_000_000
    var seen: UInt64 = 0
    while clock_gettime_nsec_np(CLOCK_UPTIME_RAW) < deadline {
      if word.contents().load(as: UInt32.self) != 0xFFFF_FFFF {
        seen = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        break
      }
    }
    cb.waitUntilCompleted()
    let gpuStart = UInt64(max(0, cb.gpuStartTime) * 1e9)
    let gpuEnd = UInt64(max(0, cb.gpuEndTime) * 1e9)
    let token = word.contents().load(as: UInt32.self)
    #expect(token == 7)
    #expect(seen != 0)
    let afterStart = seen >= gpuStart ? Double(seen - gpuStart) / 1000 : -Double(gpuStart - seen) / 1000
    let beforeEnd = seen <= gpuEnd ? Double(gpuEnd - seen) / 1000 : -Double(seen - gpuEnd) / 1000
    print("mid-command word: seen \(afterStart) us after the command's GPU start, "
          + "\(beforeEnd) us before its end, the command's span \(Double(gpuEnd - gpuStart) / 1000) us")
    #expect(seen < gpuEnd, "the word became visible only after the command ended")
    #expect(gpuEnd > seen && gpuEnd - seen > 1_000_000,
            "the word was seen less than a millisecond before the command's end")
  }
}
