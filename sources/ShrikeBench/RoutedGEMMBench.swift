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
            print("kernel=routed_gemm_grouped variant=\(result.variant) loads=\(result.weightLoads) staging_rows=\(result.stagingRows) waves=\(result.groupedWaves) "
                + "per_tile_ms=\(String(format: "%.4f", result.groupedMillisPerTile)) "
                + "achieved_tflops=\(String(format: "%.3f", result.groupedTFLOPS)) "
                + "speedup=\(String(format: "%.2f", result.perExpertMillisPerTile / result.groupedMillisPerTile))x")
        }
        // 128/97/65 rows-per-expert plan the same 16 tiles at stagingRows=1024, so a flat per_tile_ms means a padded row costs a real row.
        for rowsPerExpert in [128, 97, 65] {
            let padded = try PrefillRoutedGEMMBenchmark.run(context: context,
                                                            iterations: iterations,
                                                            experts: 8,
                                                            rowsPerExpert: rowsPerExpert,
                                                            stagingRows: 1024)
            print("kernel=routed_gemm_padding_sweep rows_per_expert=\(padded.rowsPerExpert) "
                + "real_rows=\(padded.experts * padded.rowsPerExpert) "
                + "waves=\(padded.groupedWaves) padded_rows=\(padded.groupedWaves * padded.stagingRows) "
                + "per_tile_ms=\(String(format: "%.4f", padded.groupedMillisPerTile))")
        }
        try runRoutedGEMMRowTile(iterations: iterations, context: context)
        try runGEMM(kernelName: "gemm_expert_gateup_128rows", iterations: iterations, context: context)
        try runGEMM(kernelName: "gemm_expert_down_128rows", iterations: iterations, context: context)
    }

    /// The same 1,024 rows as 32 tiles of 32 against 16 tiles of 64: alpha is a
    /// 32-row tile's cost relative to a 64-row tile's, the whole tail-tile lever.
    static func runRoutedGEMMRowTile(iterations: Int, context: MetalContext) throws {
        let wide = try PrefillRoutedGEMMBenchmark.run(context: context, iterations: iterations,
                                                       experts: 8, rowsPerExpert: 128, stagingRows: 1024)
        do {
            let narrow = try PrefillRoutedGEMMBenchmark.run(context: context, iterations: iterations,
                                                             experts: 8, rowsPerExpert: 128,
                                                             stagingRows: 1024, rowTile: 32)
            let alpha = narrow.groupedMillisPerTile / (2 * wide.groupedMillisPerTile)
            print("kernel=routed_gemm_row_tile row_tile=\(narrow.rowTile) per_tile_ms=\(String(format: "%.4f", narrow.groupedMillisPerTile)) "
                + "row_tile_64_ms=\(String(format: "%.4f", wide.groupedMillisPerTile)) "
                + "alpha=\(String(format: "%.3f", alpha))")
        } catch {
            print("kernel=routed_gemm_row_tile row_tile=32 unavailable=\(error)")
        }
        // 128 → no remainder, 97 → a 33-row remainder (stays on 64-row tiles), 65 → a 1-row tail per block.
        for rowsPerExpert in [128, 97, 65] {
            do {
                let plain = try PrefillRoutedGEMMBenchmark.run(context: context, iterations: iterations,
                                                                experts: 8, rowsPerExpert: rowsPerExpert,
                                                                stagingRows: 1024)
                let tailed = try PrefillRoutedGEMMBenchmark.run(context: context, iterations: iterations,
                                                                 experts: 8, rowsPerExpert: rowsPerExpert,
                                                                 stagingRows: 1024, tailTile: 32)
                print("kernel=routed_gemm_tail_tile rows_per_expert=\(rowsPerExpert) tail_tile=\(tailed.tailTile) "
                    + "per_tile_ms=\(String(format: "%.4f", tailed.groupedMillisPerTile)) "
                    + "off_ms=\(String(format: "%.4f", plain.groupedMillisPerTile)) "
                    + "ratio=\(String(format: "%.3f", tailed.groupedMillisPerTile / plain.groupedMillisPerTile))")
            } catch {
                print("kernel=routed_gemm_tail_tile rows_per_expert=\(rowsPerExpert) tail_tile=32 unavailable=\(error)")
            }
        }
    }
}
