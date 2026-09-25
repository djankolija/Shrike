import Foundation
import Metal
import Shrike

final class BenchRunner {
    private let args: ExpertBenchCommand
    private let context: MetalContext
    private let kernels: ExpertKernels
    private let stride: Int
    private let expertCount: Int
    private let blobs: [MTLBuffer]
    private let offsets: PlainOffsets
    private let arguments: MTLBuffer
    private let phase1Bytes: Int
    private let x: MTLBuffer
    private let acts: MTLBuffer

    init(args: ExpertBenchCommand) throws {
        self.args = args
        self.context = try MetalContext()
        self.kernels = try ExpertKernels(context: context)
        let loaded = try Experts.load(model: args.model, layer: args.layer, count: args.experts)
        self.stride = loaded.stride
        self.expertCount = loaded.experts.count
        let device = context.device

        func buffer(_ bytes: [UInt8], label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared) else {
                throw BenchError.allocation(label)
            }
            b.label = label
            return b
        }

        self.blobs = try loaded.experts.map { try buffer($0.bytes, label: "expert \($0.index)") }
        let first = loaded.experts[0]
        func off(_ name: String) throws -> UInt32 { UInt32(try first.tensor(name).offset) }
        self.offsets = PlainOffsets(
            gateW: try off("gate"), gateS: try off("gate_scales"), gateB: try off("gate_biases"),
            upW: try off("up"), upS: try off("up_scales"), upB: try off("up_biases"),
            downW: try off("down"), downS: try off("down_scales"), downB: try off("down_biases"))
        self.arguments = try ExpertKernels.argumentBuffer(device: device, blobs: blobs)
        self.phase1Bytes = try ["gate", "gate_scales", "gate_biases", "up", "up_scales", "up_biases"]
            .map { try first.tensor($0).size }.reduce(0, +)

        let d = Int(ExpertKernels.hidden)
        var state = args.seed.value
        var halves = [UInt16](repeating: 0, count: d)
        for i in 0..<d {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(Double(state >> 11) / Double(1 << 53))
            halves[i] = Float16(unit * 2 - 1).bitPattern
        }
        self.x = try buffer(halves.withUnsafeBufferPointer { Array(UnsafeRawBufferPointer($0)) }, label: "x")
        let actCount = Int(ExpertKernels.topK * ExpertKernels.intermediate)
        guard let acts = device.makeBuffer(length: actCount * 2, options: .storageModeShared) else {
            throw BenchError.allocation("acts")
        }
        self.acts = acts
    }

    func run() throws {
        print("device \(context.device.name); layer \(args.layer), \(expertCount) experts of \(stride) bytes; "
              + "repeats \(args.repeats), warmup \(args.warmup), \(args.batch) dispatches per command buffer")
        print(String(format: "%14@ %10@ %8@", "phase1 B/expert", "gpu_us", "GB/s"))
        _ = try Timing.medianGPUSeconds(context: context, warmup: 0, repeats: 20) { cb in
            try self.encode(cb)
        }
        let batch = args.batch
        let seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                  repeats: args.repeats) { cb in
            for _ in 0..<batch { try self.encode(cb) }
        } / Double(batch)
        print(String(format: "%14d %10.1f %8.2f", phase1Bytes, seconds * 1e6,
                     Double(phase1Bytes * expertCount) / seconds / 1e9))
    }

    private func encode(_ cb: MTLCommandBuffer) throws {
        try kernels.encodeProduction(cb, arguments: arguments, blobs: blobs,
                                     offsets: offsets, x: x, acts: acts)
    }
}
