import Foundation
import Metal
import Shrike

enum Timing {
    static func medianGPUSeconds(context: MetalContext, warmup: Int, repeats: Int,
                                 encode: (MTLCommandBuffer) throws -> Void) throws -> Double {
        var times: [Double] = []
        for i in 0..<(warmup + repeats) {
            guard let cb = context.queue.makeCommandBuffer() else { throw BenchError.commandBuffer }
            try encode(cb)
            cb.commit()
            cb.waitUntilCompleted()
            guard cb.status == .completed else {
                throw BenchError.gpu(cb.error.map { "\($0)" } ?? "status \(cb.status.rawValue)")
            }
            if i >= warmup { times.append(cb.gpuEndTime - cb.gpuStartTime) }
        }
        return times.sorted()[times.count / 2]
    }

    static func fnv1a(_ buffer: MTLBuffer, bytes: Int) -> UInt64 {
        let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for i in 0..<bytes {
            hash ^= UInt64(p[i])
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    static func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { acc, pair in
            let d = abs(pair.0 - pair.1)
            return d.isNaN ? .nan : max(acc, d)
        }
    }
}
