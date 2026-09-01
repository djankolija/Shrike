import Testing
import Metal
@testable import Shrike
import ShrikeValidationSupport

/// v11 V3a: the depth-tax discriminator. Times the production split-KV decode
/// attention (partial + combine, production binding via `encodeFull`) over a
/// seqLen ladder, one axis per arm: KV precision (int8 dequant cost) and KV
/// storage mode (shared-vs-private read cost). Not a correctness suite — it
/// prints; run with SHRIKE_V3A_BENCH=1 and read the slopes.
@Suite struct AttentionDepthBenchTests {
    static let enabled = ProcessInfo.processInfo.environment["SHRIKE_V3A_BENCH"] != nil

    @Test(.enabled(if: AttentionDepthBenchTests.enabled))
    func depthLadder() throws {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let maxSeq = 4096
        let ladder = [256, 1024, 2048, 4096]
        let iterations = 15
        let rowElements = numKVHeads * headDim
        let qCount = numQHeads * headDim

        let context = try MetalContext()
        let attention = try Attention(context: context)
        print("V3A partial maxTotalThreadsPerThreadgroup=" +
              "\(attention.partialPipelineMaxThreadsForBench)")

        var rng = SeedTree(0xBE7C).key("v3a")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let kShared = Fp16Buffer.make(context.device, halves: k),
              let vShared = Fp16Buffer.make(context.device, halves: v),
              let outBuf = Fp16Buffer.make(context.device, count: qCount) else {
            Issue.record("fp16 buffer allocation failed"); return
        }

        func privateCopy(_ source: MTLBuffer) throws -> MTLBuffer {
            guard let dst = context.device.makeBuffer(length: source.length,
                                                      options: .storageModePrivate),
                  let cb = context.queue.makeCommandBuffer(),
                  let blit = cb.makeBlitCommandEncoder() else {
                throw MetalError.bufferAllocationFailed("v3a private copy")
            }
            blit.copy(from: source, sourceOffset: 0, to: dst,
                      destinationOffset: 0, size: source.length)
            blit.endEncoding()
            cb.commit(); cb.waitUntilCompleted()
            return dst
        }

        let cache = try KVCacheManager(device: context.device, config: config,
                                       maxContext: maxSeq, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        let keyView = cache.keyView(layer: 3, validTokenCount: maxSeq)
        let valueView = cache.valueView(layer: 3, validTokenCount: maxSeq)
        guard let quantCB = context.queue.makeCommandBuffer() else {
            Issue.record("quantize CB failed"); return
        }
        try quantizer.encode(commandBuffer: quantCB, source: kShared,
                             sourceTokenStrideElements: rowElements,
                             destination: keyView, tokenCount: maxSeq,
                             elementCount: rowElements)
        try quantizer.encode(commandBuffer: quantCB, source: vShared,
                             sourceTokenStrideElements: rowElements,
                             destination: valueView, tokenCount: maxSeq,
                             elementCount: rowElements)
        quantCB.commit(); quantCB.waitUntilCompleted()

        let kPrivate = try privateCopy(kShared)
        let vPrivate = try privateCopy(vShared)
        let kQuantPrivate = try privateCopy(keyView.buffer)
        let vQuantPrivate = try privateCopy(valueView.buffer)

        struct Arm {
            let name: String
            let k: MTLBuffer
            let v: MTLBuffer
            let format: KVView?
        }
        let arms = [
            Arm(name: "fp16-shared ", k: kShared, v: vShared, format: nil),
            Arm(name: "fp16-private", k: kPrivate, v: vPrivate, format: nil),
            Arm(name: "int8-shared ", k: keyView.buffer, v: valueView.buffer,
                format: keyView),
            Arm(name: "int8-private", k: kQuantPrivate, v: vQuantPrivate,
                format: keyView),
        ]

        for arm in arms {
            for seqLen in ladder {
                var total = 0.0
                var best = Double.greatestFiniteMagnitude
                for _ in 0..<iterations {
                    guard let cb = context.queue.makeCommandBuffer() else {
                        Issue.record("bench CB failed"); return
                    }
                    try attention.encodeFull(commandBuffer: cb,
                                             q: qBuf, k: arm.k, v: arm.v,
                                             out: outBuf,
                                             headDim: UInt32(headDim),
                                             numQHeads: UInt32(numQHeads),
                                             numKVHeads: UInt32(numKVHeads),
                                             seqLen: UInt32(seqLen),
                                             kvFormat: arm.format)
                    cb.commit(); cb.waitUntilCompleted()
                    let t = cb.gpuEndTime - cb.gpuStartTime
                    total += t
                    best = min(best, t)
                }
                let meanUs = total / Double(iterations) * 1e6
                let bestUs = best * 1e6
                print(String(format: "V3A %@ T=%4d  mean %8.1f us  best %8.1f us",
                             arm.name, seqLen, meanUs, bestUs))
            }
        }
    }

    @Test(.enabled(if: AttentionDepthBenchTests.enabled))
    func gqaLadder() throws {
        let headDim = 256
        let numQHeads = 16
        let maxSeq = 4096
        let iterations = 15
        let qCount = numQHeads * headDim

        let context = try MetalContext()
        let attention = try Attention(context: context)
        var rng = SeedTree(0x69A4).key("v3a-gqa")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let outBuf = Fp16Buffer.make(context.device, count: qCount) else {
            Issue.record("q/out allocation failed"); return
        }

        var buffers: [Int: (k: MTLBuffer, v: MTLBuffer)] = [:]
        for numKVHeads in [2, 4, 8, 16] {
            let rowElements = numKVHeads * headDim
            let kv = (0..<(maxSeq * rowElements)).map { _ in
                Float16(rng.uniform(-0.5, 0.5))
            }
            guard let kvShared = Fp16Buffer.make(context.device, halves: kv),
                  let kBuf = context.device.makeBuffer(length: kvShared.length,
                                                       options: .storageModePrivate),
                  let vBuf = context.device.makeBuffer(length: kvShared.length,
                                                       options: .storageModePrivate),
                  let copyCB = context.queue.makeCommandBuffer(),
                  let blit = copyCB.makeBlitCommandEncoder() else {
                Issue.record("kv allocation failed"); return
            }
            blit.copy(from: kvShared, sourceOffset: 0, to: kBuf,
                      destinationOffset: 0, size: kvShared.length)
            blit.copy(from: kvShared, sourceOffset: 0, to: vBuf,
                      destinationOffset: 0, size: kvShared.length)
            blit.endEncoding()
            copyCB.commit(); copyCB.waitUntilCompleted()
            buffers[numKVHeads] = (kBuf, vBuf)
        }

        func timeOne(_ numKVHeads: Int, _ seqLen: Int) throws -> Double {
            guard let bufs = buffers[numKVHeads],
                  let cb = context.queue.makeCommandBuffer() else { return .nan }
            try attention.encodeFull(commandBuffer: cb,
                                     q: qBuf, k: bufs.k, v: bufs.v,
                                     out: outBuf,
                                     headDim: UInt32(headDim),
                                     numQHeads: UInt32(numQHeads),
                                     numKVHeads: UInt32(numKVHeads),
                                     seqLen: UInt32(seqLen))
            cb.commit(); cb.waitUntilCompleted()
            return cb.gpuEndTime - cb.gpuStartTime
        }

        for numKVHeads in [2, 4, 8, 16] {
            _ = try timeOne(numKVHeads, maxSeq)
        }

        var best: [String: Double] = [:]
        for _ in 0..<3 {
            for numKVHeads in [2, 4, 8, 16] {
                for seqLen in [1024, 4096] {
                    for _ in 0..<iterations {
                        let t = try timeOne(numKVHeads, seqLen)
                        let key = "\(numKVHeads)-\(seqLen)"
                        best[key] = min(best[key] ?? .greatestFiniteMagnitude, t)
                    }
                }
            }
        }
        for numKVHeads in [2, 4, 8, 16] {
            for seqLen in [1024, 4096] {
                let t = best["\(numKVHeads)-\(seqLen)"] ?? .nan
                print(String(format: "V3A gqa NKV=%2d T=%4d  best %8.1f us",
                             numKVHeads, seqLen, t * 1e6))
            }
        }
    }
}
