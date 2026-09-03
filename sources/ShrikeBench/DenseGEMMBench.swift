import Foundation
import Metal
import Shrike

extension ShrikeBench {
    /// A production dense prefill shape and the `gemmShapes` label whose MPS
    /// ceiling is measured in the same run; the a/b pair has no MPS twin.
    private struct DenseShape {
        let role: String
        let m: Int
        let k: Int
        let n: Int
        let ceiling: String?
    }

    private static let denseShapes: [DenseShape] = [
        DenseShape(role: "gdn_qkv_and_attn_q", m: 4096, k: 2048, n: 8192,
                   ceiling: "qproj_chunk4096"),
        DenseShape(role: "gdn_z", m: 4096, k: 2048, n: 4096,
                   ceiling: "gdn_zproj_chunk4096"),
        DenseShape(role: "gdn_out", m: 4096, k: 4096, n: 2048,
                   ceiling: "oproj_chunk4096"),
        DenseShape(role: "gdn_ab", m: 4096, k: 2048, n: 32, ceiling: nil),
    ]

    private static let denseVariants = ["n32b1", "n32k128b1", "n32k256b1"]
    private static let denseWeightLoads = ["byte", "vector"]

    /// `dense_gemm`: `MPPPrefillInt4QMM.encode` at the four m = 4,096 dense
    /// prefill shapes, every tile variant against every load body, each arm
    /// scored as a share of the MPS ceiling measured in this same process.
    static func runDenseGEMM(iterations: Int, context: MetalContext) throws {
        var ceilings: [String: Double] = [:]
        for label in denseShapes.compactMap(\.ceiling) {
            guard let shape = gemmShapes.first(where: { $0.label == label }) else {
                fatalError("unknown ceiling shape \(label)")
            }
            ceilings[label] = try runGEMMShape(shape, iterationCeiling: iterations,
                                               device: context.device, queue: context.queue)
        }
        for shape in denseShapes {
            let ceiling = shape.ceiling.flatMap { ceilings[$0] }
            for variant in denseVariants {
                for loads in denseWeightLoads {
                    let result = try MPPPrefillDenseBenchmark.run(
                        context: context, iterations: iterations,
                        m: shape.m, k: shape.k, n: shape.n,
                        variant: variant, weightLoads: loads)
                    printDenseArm(shape, result, ceiling: ceiling)
                }
            }
        }
    }

    private static func printDenseArm(_ shape: DenseShape,
                                      _ result: MPPPrefillDenseBenchmark.Result,
                                      ceiling: Double?) {
        var line = "kernel=dense_gemm role=\(shape.role) m=\(result.m) k=\(result.k) "
            + "n=\(result.n) variant=\(result.variant) loads=\(result.weightLoads) "
            + "launches=\(result.launches) gflop=\(String(format: "%.3f", result.gflop)) "
            + "per_launch_ms=\(String(format: "%.4f", result.millisPerLaunch)) "
            + "achieved_tflops=\(String(format: "%.3f", result.tflops))"
        if let ceiling, ceiling > 0 {
            let ceilingMillis = result.gflop / ceiling
            line += " ceiling=\(shape.ceiling ?? "") "
                + "ceiling_tflops=\(String(format: "%.3f", ceiling)) "
                + "ceiling_share=\(String(format: "%.3f", result.tflops / ceiling)) "
                + "headroom_ms=\(String(format: "%.4f", result.millisPerLaunch - ceilingMillis))"
        } else {
            line += " ceiling=none"
        }
        print(line)
    }
}
