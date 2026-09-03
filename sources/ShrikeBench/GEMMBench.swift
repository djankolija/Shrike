import Foundation
import Metal
import MetalPerformanceShaders
import Shrike

/// Achievable fp16 matmul throughput through Apple's own MPS kernel, at the
/// ornith prefill shapes: the "real sticker" prefill kernels are judged against.
///
/// Usage: ShrikeBench gemm [iterations] | gemm_<label> [iterations]
/// `iterations` is a ceiling; each shape runs enough launches to accumulate ~3 TFLOP.
extension ShrikeBench {
    struct GEMMShape {
        let label: String
        let m: Int
        let k: Int
        let n: Int

        var flopsPerLaunch: Double { 2.0 * Double(m) * Double(n) * Double(k) }
    }

    static let gemmShapes: [GEMMShape] = [
        GEMMShape(label: "square4096", m: 4096, k: 4096, n: 4096),
        GEMMShape(label: "square2048", m: 2048, k: 2048, n: 2048),
        GEMMShape(label: "qproj_chunk4096", m: 4096, k: 2048, n: 8192),
        GEMMShape(label: "gdn_zproj_chunk4096", m: 4096, k: 2048, n: 4096),
        GEMMShape(label: "oproj_chunk4096", m: 4096, k: 4096, n: 2048),
        GEMMShape(label: "gdn_inproj_chunk4096", m: 4096, k: 2048, n: 12288),
        GEMMShape(label: "router_chunk4096", m: 4096, k: 2048, n: 256),
        GEMMShape(label: "expert_gateup_128rows", m: 128, k: 2048, n: 1024),
        GEMMShape(label: "expert_down_128rows", m: 128, k: 512, n: 2048),
        GEMMShape(label: "expert_gateup_512rows", m: 512, k: 2048, n: 1024),
        GEMMShape(label: "expert_gateup_32rows", m: 32, k: 2048, n: 1024),
    ]

    static func runGEMM(kernelName: String, iterations: Int, context: MetalContext) throws {
        let device = context.device
        let selected: [GEMMShape]
        if kernelName == "gemm" {
            selected = gemmShapes
        } else {
            let label = String(kernelName.dropFirst("gemm_".count))
            selected = gemmShapes.filter { $0.label == label }
            guard !selected.isEmpty else {
                fatalError("unknown gemm shape \(label); known: \(gemmShapes.map(\.label))")
            }
        }

        for shape in selected {
            try runGEMMShape(shape, iterationCeiling: iterations, device: device, queue: context.queue)
        }
    }

    @discardableResult
    static func runGEMMShape(_ shape: GEMMShape,
                             iterationCeiling: Int,
                             device: MTLDevice,
                             queue: MTLCommandQueue) throws -> Double {
        func matrix(rows: Int, columns: Int, fill: UInt8) -> MPSMatrix {
            let rowBytes = columns * MemoryLayout<Float16>.size
            guard let buffer = device.makeBuffer(length: rows * rowBytes,
                                                 options: .storageModeShared) else {
                fatalError("could not allocate \(rows)x\(columns) fp16 matrix")
            }
            memset(buffer.contents(), Int32(fill), rows * rowBytes)
            let descriptor = MPSMatrixDescriptor(rows: rows, columns: columns,
                                                 rowBytes: rowBytes, dataType: .float16)
            return MPSMatrix(buffer: buffer, descriptor: descriptor)
        }

        // 0x3C3C is fp16 1.06: products stay far from overflow at k = 4096.
        let left = matrix(rows: shape.m, columns: shape.k, fill: 0x3C)
        let right = matrix(rows: shape.k, columns: shape.n, fill: 0x3C)
        let result = matrix(rows: shape.m, columns: shape.n, fill: 0)

        let multiply = MPSMatrixMultiplication(device: device,
                                               transposeLeft: false,
                                               transposeRight: false,
                                               resultRows: shape.m,
                                               resultColumns: shape.n,
                                               interiorColumns: shape.k,
                                               alpha: 1.0,
                                               beta: 0.0)

        let targetFlop = 3.0e12
        let launches = max(10, min(iterationCeiling, Int(targetFlop / shape.flopsPerLaunch)))

        guard let warm = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        multiply.encode(commandBuffer: warm, leftMatrix: left, rightMatrix: right, resultMatrix: result)
        warm.commit()
        warm.waitUntilCompleted()

        guard let timed = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        for _ in 0..<launches {
            multiply.encode(commandBuffer: timed, leftMatrix: left, rightMatrix: right, resultMatrix: result)
        }
        timed.commit()
        timed.waitUntilCompleted()
        if let error = timed.error {
            print("COMMAND BUFFER ERROR: \(error)")
            return 0
        }

        let seconds = timed.gpuEndTime - timed.gpuStartTime
        let perLaunch = seconds / Double(launches)
        let tflops = shape.flopsPerLaunch / perLaunch / 1.0e12
        print("kernel=gemm_\(shape.label) m=\(shape.m) k=\(shape.k) n=\(shape.n) "
            + "launches=\(launches) per_launch_ms=\(String(format: "%.4f", perLaunch * 1000)) "
            + "achieved_tflops=\(String(format: "%.3f", tflops))")
        return tflops
    }
}
