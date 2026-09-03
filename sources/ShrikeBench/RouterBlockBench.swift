import Foundation
import Metal
import Shrike

extension ShrikeBench {
    /// `router_block`: `PrefillRouter.encodeBlock` at the ornith prefill shape for
    /// both kernels, scored against the MPS fp16 ceiling at the same (m, k, n)
    /// from this run; `SHRIKE_PREFILL_ROUTER_TOKENS` picks the tiled token block.
    static func runRouterBlock(iterations: Int, context: MetalContext) throws {
        var ceiling = 0.0
        if let shape = gemmShapes.first(where: { $0.label == "router_chunk4096" }) {
            ceiling = try runGEMMShape(shape, iterationCeiling: iterations,
                                       device: context.device, queue: context.queue)
        }
        let tokenBlock = PrefillRouterBenchmark.environmentTokenBlock()
        for kind in ["block", "tiled"] {
            let result = try PrefillRouterBenchmark.run(context: context, iterations: iterations,
                                                        kind: kind, tokenBlock: tokenBlock)
            let symbol = result.kind == "tiled" ? "prefill_router_block_tiled" : "prefill_router_block"
            var line = "kernel=\(symbol) kind=\(result.kind) T=\(result.queryCount) D=\(result.d) "
                + "experts=\(result.numExperts) top_k=\(result.topK) tokens=\(result.tokenBlock) "
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
}
