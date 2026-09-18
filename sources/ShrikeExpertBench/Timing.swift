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
}
