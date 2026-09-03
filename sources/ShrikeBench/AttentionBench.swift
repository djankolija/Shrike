import Foundation
import Metal
import Shrike

extension ShrikeBench {
    static func runPrefillMatrixMode(kernelName: String, iterations: Int, context: MetalContext) throws -> Bool {
        switch kernelName {
        case "routed_gemm": try runRoutedGEMM(iterations: iterations, context: context)
        case "attn": try runAttention(iterations: iterations, context: context)
        default: return false
        }
        return true
    }

    /// The 12k prompt's three 4,096-query chunks per tile: `core_ms_per_token_12k`
    /// is the ten attention layers' kernel time per prompt token, the part of
    /// `prefill_attn_router` this kernel owns.
    static func runAttention(iterations: Int, context: MetalContext) throws {
        let tiles = CommandLine.arguments.count > 3
            ? Array(CommandLine.arguments[3...])
            : ["g2k256d", "g4k128d", "f4k128", "f4k64", "f8k128"]
        let chunks = [(start: 0, chunk: 4096), (start: 4096, chunk: 4096), (start: 8192, chunk: 4096)]
        let promptTokens = 12_285.0
        let attentionLayers = 10.0
        for tile in tiles {
            var totalMillis = 0.0
            var totalGflop = 0.0
            var lines: [String] = []
            do {
                for shape in chunks {
                    let result = try PrefillAttentionBenchmark.run(context: context, iterations: iterations,
                                                                   tile: tile, start: shape.start,
                                                                   chunk: shape.chunk)
                    totalMillis += result.millisPerChunk
                    totalGflop += result.gflop
                    lines.append("kernel=attn tile=\(tile) start=\(result.start) chunk=\(result.chunk) kv=\(result.kvValid) "
                        + "ms=\(String(format: "%.3f", result.millisPerChunk)) "
                        + "tflops=\(String(format: "%.3f", result.tflops))")
                }
            } catch {
                print("kernel=attn tile=\(tile) unavailable=\(error)")
                continue
            }
            lines.forEach { print($0) }
            print("kernel=attn tile=\(tile) prompt_12k_ms=\(String(format: "%.1f", totalMillis)) "
                + "tflops=\(String(format: "%.3f", totalGflop / totalMillis)) "
                + "core_ms_per_token_12k=\(String(format: "%.3f", totalMillis * attentionLayers / promptTokens))")
        }
    }
}
