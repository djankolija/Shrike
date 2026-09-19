import Foundation
import Metal
import Shrike
import ShrikeValidationSupport

final class BenchRunner {
    private let args: AttnBenchCommand
    private let arms: [Arm]
    private let context: MetalContext
    private let rows: SyntheticRows
    private let ladder: LadderKernel
    private let stream: StreamKernel
    private let production: Attention
    private let productionPlain: Attention
    private let productionStream: Attention
    private var productionOut: [Int: [Float]] = [:]

    init(args: AttnBenchCommand) throws {
        self.args = args
        self.arms = try args.arms.names.map(Arm.parse)
        self.context = try MetalContext()
        let maxSeq = args.positions.counts.max() ?? 1024
        self.rows = try SyntheticRows(context: context, maxSeq: maxSeq, seed: args.seed.value)
        self.ladder = try LadderKernel(device: context.device, numQHeads: rows.numQHeads,
                                       headDim: rows.headDim)
        self.stream = try StreamKernel(device: context.device, scratch: ladder)
        self.production = try Attention(context: context, partialLoopVariant: .kvShared)
        self.productionPlain = try Attention(context: context, partialLoopVariant: .kvShared,
                                             specializesKVShared: false)
        self.productionStream = try Attention(context: context, partialLoopVariant: .stream)
    }

    func run() throws {
        print("device \(context.device.name); rows \(rows.maxSeq) positions, "
              + "\(rows.bytesPerPosition) bytes each; repeats \(args.repeats), warmup \(args.warmup)")
        print(String(format: "%-20@ %9@ %10@ %9@ %8@ %8@ %18@ %10@",
                     "arm", "positions", "gpu_us", "us/pos", "ns/KB", "GB/s", "partials", "maxd_prod"))
        try spinUp()
        for arm in arms {
            for seqLen in args.positions.counts {
                try runOne(arm, seqLen: seqLen)
            }
        }
    }

    /// The first timed arm of a process ran 3× slow on the M4 Pro before the GPU ramped.
    private func spinUp() throws {
        _ = try Timing.medianGPUSeconds(context: context, warmup: 0, repeats: 15) { cb in
            try ladder.encode(commandBuffer: cb, rows: rows, seqLen: rows.maxSeq,
                              switches: LadderSwitches())
        }
    }

    private func runOne(_ arm: Arm, seqLen: Int) throws {
        let seconds: Double
        var hash = "-"
        var maxDiff = "-"
        switch arm.kind {
        case .production, .productionStream:
            let attention: Attention
            if case .production(let specialized) = arm.kind {
                attention = specialized ? production : productionPlain
            } else {
                attention = productionStream
            }
            seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                  repeats: args.repeats) { cb in
                try encodeProduction(attention, cb, seqLen: seqLen)
            }
            let out = Fp16Buffer.read(rows.outBuf, count: rows.numQHeads * rows.headDim)
            if let reference = productionOut[seqLen] {
                maxDiff = String(format: "%.3e", Timing.maxAbsDifference(out, reference))
            } else {
                productionOut[seqLen] = out
            }
        case .ladder(let sw):
            seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                  repeats: args.repeats) { cb in
                try ladder.encode(commandBuffer: cb, rows: rows, seqLen: seqLen, switches: sw)
            }
            if sw.loadOnly {
                let floats = (sw.fullRow ? 1 : rows.numKVHeads) * LadderKernel.numChunks
                    * LadderKernel.threadsPerGroup
                hash = String(format: "%016llx", Timing.fnv1a(ladder.oBuf, bytes: floats * 4))
            } else {
                hash = String(format: "%016llx", partialsHash())
                if let reference = productionOut[seqLen], !sw.noSoftmax, !sw.noV {
                    let combined = ladder.combineOnCPU(numQHeads: rows.numQHeads,
                                                       headDim: rows.headDim)
                    maxDiff = String(format: "%.3e", Timing.maxAbsDifference(combined, reference))
                }
            }
        case .stream(let sw):
            seconds = try Timing.medianGPUSeconds(context: context, warmup: args.warmup,
                                                  repeats: args.repeats) { cb in
                try stream.encode(commandBuffer: cb, rows: rows, seqLen: seqLen, switches: sw)
            }
            hash = String(format: "%016llx", partialsHash())
            if let reference = productionOut[seqLen], !sw.noLoad {
                let combined = ladder.combineOnCPU(numQHeads: rows.numQHeads,
                                                   headDim: rows.headDim)
                maxDiff = String(format: "%.3e", Timing.maxAbsDifference(combined, reference))
            }
        }
        let bytes = Double(seqLen * rows.bytesPerPosition)
        print(String(format: "%-20@ %9d %10.1f %9.4f %8.2f %8.2f %18@ %10@",
                     arm.name, seqLen, seconds * 1e6, seconds * 1e6 / Double(seqLen),
                     seconds * 1e9 / (bytes / 1024), bytes / seconds / 1e9, hash, maxDiff))
    }

    private func encodeProduction(_ attention: Attention, _ cb: MTLCommandBuffer,
                                  seqLen: Int) throws {
        try attention.encodeFull(commandBuffer: cb,
                                  q: rows.qBuf,
                                  k: rows.keyView.buffer, kOffset: rows.keyView.offset,
                                  v: rows.valueView.buffer, vOffset: rows.valueView.offset,
                                  out: rows.outBuf,
                                  headDim: UInt32(rows.headDim),
                                  numQHeads: UInt32(rows.numQHeads),
                                  numKVHeads: UInt32(rows.numKVHeads),
                                  seqLen: UInt32(seqLen),
                                  kvFormat: rows.keyView)
    }

    private func partialsHash() -> UInt64 {
        let m = Timing.fnv1a(ladder.mBuf, bytes: ladder.partialCount * 4)
        let d = Timing.fnv1a(ladder.dBuf, bytes: ladder.partialCount * 4)
        let o = Timing.fnv1a(ladder.oBuf, bytes: ladder.oFloats * 4)
        return m ^ (d &* 31) ^ (o &* 131)
    }
}
