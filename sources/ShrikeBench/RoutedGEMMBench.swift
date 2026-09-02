import Foundation
import Metal
import Shrike

extension ShrikeBench {
    static func runRoutedGEMM(iterations: Int, context: MetalContext) throws {
        for stagingRows in [1024, 2048] {
            let result = try PrefillRoutedGEMMBenchmark.run(context: context,
                                                            iterations: iterations,
                                                            stagingRows: stagingRows)
            print("kernel=routed_gemm_per_expert experts=\(result.experts) rows=\(result.rowsPerExpert) "
                + "d=\(result.d) f=\(result.f) gflop_per_tile=\(String(format: "%.3f", result.gflopPerTile)) "
                + "per_tile_ms=\(String(format: "%.4f", result.perExpertMillisPerTile)) "
                + "achieved_tflops=\(String(format: "%.3f", result.perExpertTFLOPS))")
            print("kernel=routed_gemm_grouped staging_rows=\(result.stagingRows) waves=\(result.groupedWaves) "
                + "per_tile_ms=\(String(format: "%.4f", result.groupedMillisPerTile)) "
                + "achieved_tflops=\(String(format: "%.3f", result.groupedTFLOPS)) "
                + "speedup=\(String(format: "%.2f", result.perExpertMillisPerTile / result.groupedMillisPerTile))x")
        }
        try runGEMM(kernelName: "gemm_expert_gateup_128rows", iterations: iterations, context: context)
        try runGEMM(kernelName: "gemm_expert_down_128rows", iterations: iterations, context: context)
    }
}
