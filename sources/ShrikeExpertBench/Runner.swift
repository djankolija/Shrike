import Foundation
import Metal
import Shrike

final class BenchRunner {
    private let args: ExpertBenchCommand
    private let context: MetalContext
    private let kernels: ExpertKernels
    private let stride: Int
    private let experts: [PlainExpert]
    private let code: PrefixCode
    private let plainBlobs: [MTLBuffer]
    private let plainOffsets: PlainOffsets
    private let plainArguments: MTLBuffer
    private let plainPhase1Bytes: Int
    private let coded: [CodedExpert]
    private let codedOffsets: CodedOffsets
    private let codedStride: Int
    private let codedBlobs: [MTLBuffer]
    private let codedArguments: MTLBuffer
    private let codedAux: [CodedExpert]
    private let codedAuxOffsets: CodedOffsets
    private let codedAuxStride: Int
    private let codedAuxBlobs: [MTLBuffer]
    private let codedAuxArguments: MTLBuffer
    private let table: MTLBuffer
    private let auxTables: MTLBuffer
    private let auxEntries: Int
    private let x: MTLBuffer
    private let acts: MTLBuffer
    private let reference: MTLBuffer

    init(args: ExpertBenchCommand) throws {
        self.args = args
        self.context = try MetalContext()
        self.kernels = try ExpertKernels(context: context)
        let loaded = try Experts.load(model: args.model, layer: args.layer, count: args.experts)
        self.stride = loaded.stride
        self.experts = loaded.experts
        let device = context.device

        func buffer(_ bytes: [UInt8], label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared) else {
                throw BenchError.allocation(label)
            }
            b.label = label
            return b
        }

        self.plainBlobs = try experts.map { try buffer($0.bytes, label: "plain \($0.index)") }
        let first = experts[0]
        func off(_ name: String) throws -> UInt32 { UInt32(try first.tensor(name).offset) }
        self.plainOffsets = PlainOffsets(
            gateW: try off("gate"), gateS: try off("gate_scales"), gateB: try off("gate_biases"),
            upW: try off("up"), upS: try off("up_scales"), upB: try off("up_biases"),
            downW: try off("down"), downS: try off("down_scales"), downB: try off("down_biases"))
        self.plainArguments = try ExpertKernels.argumentBuffer(device: device, blobs: plainBlobs)
        self.plainPhase1Bytes = try ["gate", "gate_scales", "gate_biases", "up", "up_scales", "up_biases"]
            .map { try first.tensor($0).size }.reduce(0, +)

        let code = try PrefixCode.build(frequencies: try Coder.nibbleHistogram(experts: experts,
                                                                                tensors: ["gate", "up"]))
        self.code = code
        let tables = Coder.AuxTables(
            gateS: try AuxTable(experts: experts, tensor: "gate_scales"),
            gateB: try AuxTable(experts: experts, tensor: "gate_biases"),
            upS: try AuxTable(experts: experts, tensor: "up_scales"),
            upB: try AuxTable(experts: experts, tensor: "up_biases"))
        let plainLayout = try Coder.codeExperts(experts, code: code, aux: nil)
        let auxLayout = try Coder.codeExperts(experts, code: code, aux: tables)
        let coded = plainLayout.coded
        let codedAux = auxLayout.coded
        self.coded = coded
        self.codedAux = codedAux
        self.codedOffsets = plainLayout.offsets
        self.codedAuxOffsets = auxLayout.offsets
        self.codedStride = plainLayout.phase1Bytes
        self.codedAuxStride = auxLayout.phase1Bytes
        let codedBlobs = try coded.enumerated().map { try buffer($0.element.bytes, label: "coded \($0.offset)") }
        let codedAuxBlobs = try codedAux.enumerated().map {
            try buffer($0.element.bytes, label: "coded+aux \($0.offset)")
        }
        self.codedBlobs = codedBlobs
        self.codedAuxBlobs = codedAuxBlobs
        self.codedArguments = try ExpertKernels.argumentBuffer(device: device, blobs: codedBlobs)
        self.codedAuxArguments = try ExpertKernels.argumentBuffer(device: device, blobs: codedAuxBlobs)
        self.table = try buffer(code.table, label: "prefix table")
        let concatenated = tables.concatenated
        self.auxEntries = concatenated.count
        self.auxTables = try buffer(concatenated.withUnsafeBufferPointer { Array(UnsafeRawBufferPointer($0)) },
                                    label: "aux tables")

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
        guard let acts = device.makeBuffer(length: actCount * 2, options: .storageModeShared),
              let reference = device.makeBuffer(length: actCount * 2, options: .storageModeShared) else {
            throw BenchError.allocation("acts")
        }
        self.acts = acts
        self.reference = reference
    }

    func run() throws {
        let k = experts.count
        let symbols = 2 * 512 * 2048 * k
        let streamBits = Double(coded.reduce(0) { $0 + $1.streamBytes } * 8)
        print("device \(context.device.name); layer \(args.layer), \(k) experts of \(stride) bytes; "
              + "repeats \(args.repeats), warmup \(args.warmup), \(args.batch) dispatches per command buffer")
        print("code lengths " + code.lengths.map(String.init).joined(separator: " ")
              + String(format: "; %.3f bits per index in the streams, %d aux patterns", streamBits / Double(symbols), auxEntries))
        print(String(format: "phase-1 stride per expert: plain %d, coded %d (%.4f), coded+aux %d (%.4f); "
                     + "the largest expert's sections set it",
                     plainPhase1Bytes, codedStride, Double(codedStride) / Double(plainPhase1Bytes),
                     codedAuxStride, Double(codedAuxStride) / Double(plainPhase1Bytes)))
        print(String(format: "%-11@ %14@ %8@ %10@ %8@ %8@ %10@ %12@",
                     "arm", "phase1 B/expert", "ratio", "gpu_us", "vs_plain", "GB/s", "mismatch", "max_abs_diff"))
        try spinUp()
        try encodeOnce { cb in try self.encodePlain(cb, into: self.reference) }
        var plainSeconds: Double?
        for arm in args.arms.names {
            let bytesPerExpert: Int
            let encode: (MTLCommandBuffer) throws -> Void
            switch arm {
            case "plain":
                bytesPerExpert = plainPhase1Bytes
                encode = { cb in try self.encodePlain(cb, into: self.acts) }
            case "coded":
                bytesPerExpert = coded.reduce(0) { $0 + $1.phase1Bytes } / k
                encode = { cb in try self.encodeCoded(cb, aux: false, into: self.acts) }
            case "coded+aux":
                bytesPerExpert = codedAux.reduce(0) { $0 + $1.phase1Bytes } / k
                encode = { cb in try self.encodeCoded(cb, aux: true, into: self.acts) }
            default:
                throw BenchError.usage("unknown arm \(arm)")
            }
            let batch = args.batch
            let seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                      repeats: args.repeats) { cb in
                for _ in 0..<batch { try encode(cb) }
            } / Double(batch)
            if arm == "plain" { plainSeconds = seconds }
            let (mismatches, maxDiff) = compare()
            let bytes = Double(bytesPerExpert * k)
            print(String(format: "%-11@ %14d %8.4f %10.1f %8@ %8.2f %10d %12.3e",
                         arm, bytesPerExpert, Double(bytesPerExpert) / Double(plainPhase1Bytes),
                         seconds * 1e6,
                         plainSeconds.map { String(format: "%.3f", seconds / $0) } ?? "-",
                         bytes / seconds / 1e9, mismatches, maxDiff))
        }
    }

    private func spinUp() throws {
        _ = try Timing.medianGPUSeconds(context: context, warmup: 0, repeats: 20) { cb in
            try self.encodePlain(cb, into: self.acts)
        }
    }

    private func encodeOnce(_ encode: (MTLCommandBuffer) throws -> Void) throws {
        _ = try Timing.medianGPUSeconds(context: context, warmup: 0, repeats: 1, encode: encode)
    }

    private func encodePlain(_ cb: MTLCommandBuffer, into out: MTLBuffer) throws {
        try kernels.encodeProduction(cb, arguments: plainArguments, blobs: plainBlobs,
                                     offsets: plainOffsets, x: x, acts: out)
    }

    private func encodeCoded(_ cb: MTLCommandBuffer, aux: Bool, into out: MTLBuffer) throws {
        try kernels.encodeCoded(cb, aux: aux,
                                arguments: aux ? codedAuxArguments : codedArguments,
                                blobs: aux ? codedAuxBlobs : codedBlobs,
                                offsets: aux ? codedAuxOffsets : codedOffsets,
                                table: table, auxTables: auxTables, x: x, acts: out)
    }

    private func compare() -> (Int, Float) {
        let count = Int(ExpertKernels.topK * ExpertKernels.intermediate)
        let a = acts.contents().assumingMemoryBound(to: UInt16.self)
        let b = reference.contents().assumingMemoryBound(to: UInt16.self)
        var mismatches = 0
        var maxDiff: Float = 0
        for i in 0..<count where a[i] != b[i] {
            mismatches += 1
            let diff = abs(Float(Float16(bitPattern: a[i])) - Float(Float16(bitPattern: b[i])))
            maxDiff = diff.isNaN ? .nan : max(maxDiff, diff)
        }
        return (mismatches, maxDiff)
    }
}
