import Foundation
import Metal
import Shrike

extension ShrikeBench {
    /// `router_block`: `PrefillRouter.encodeBlock` at the ornith prefill shape,
    /// scored against the MPS fp16 ceiling at the same (m, k, n) from this run.
    static func runRouterBlock(iterations: Int, context: MetalContext) throws {
        var ceiling = 0.0
        if let shape = gemmShapes.first(where: { $0.label == "router_chunk4096" }) {
            ceiling = try runGEMMShape(shape, iterationCeiling: iterations,
                                       device: context.device, queue: context.queue)
        }
        let result = try PrefillRouterBenchmark.run(context: context, iterations: iterations)
        var line = "kernel=prefill_router_block T=\(result.queryCount) D=\(result.d) "
            + "experts=\(result.numExperts) top_k=\(result.topK) "
            + "per_launch_ms=\(String(format: "%.4f", result.millisPerLaunch)) "
            + "gflop=\(String(format: "%.3f", result.gflop)) "
            + "achieved_tflops=\(String(format: "%.3f", result.tflops)) "
            + "weight_bytes_per_threadgroup=\(result.weightBytesPerThreadgroup) "
            + "threadgroups=\(result.threadgroups) "
            + "threadgroup_width=\(result.threadgroupWidth)"
        if ceiling > 0 {
            line += " ceiling_tflops=\(String(format: "%.3f", ceiling)) "
                + "ceiling_ms=\(String(format: "%.4f", result.gflop / ceiling)) "
                + "ceiling_share=\(String(format: "%.3f", result.tflops / ceiling))"
        } else {
            line += " ceiling=none"
        }
        print(line)
    }
}
