import Testing
import Foundation
import Metal
@testable import NVMAI
import NVMAIValidationSupport

/// Validates the Kimi MLA kernels: the batched absorbed-form embed/unembed
/// projections against CPU dequant matmuls, the strided-rows latent RMSNorm,
/// and decode/prefill attention over fused [latent | k_pe] rows against
/// `MLAAttentionRef` and each other.
@Suite struct MLAKernelTests {

    // Structurally faithful mini shape and the real Kimi-Linear shape.
    private static let miniCfg = MLAConfig(latentDim: 128, qkNopeDim: 64,
                                           qkRopeDim: 32, valueHeadDim: 64)
    private static let kimiCfg = MLAConfig(latentDim: 512, qkNopeDim: 128,
                                           qkRopeDim: 64, valueHeadDim: 128)

    // MARK: - Helpers

    private static func bf16(_ x: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: x.bitPattern >> 16)
    }

    private static func bf16Value(_ x: Float) -> Float {
        Float(bitPattern: UInt32(bf16(x)) << 16)
    }

    private static func makeBF16Buffer(_ device: MTLDevice,
                                       values: [Float]) -> MTLBuffer? {
        let bits = values.map { bf16($0) }
        return device.makeBuffer(bytes: bits, length: bits.count * 2,
                                 options: .storageModeShared)
    }

    /// One affine matrix packed resident-style ([pad | weights | scales |
    /// biases], 2-byte but not 4-byte aligned) plus the exact dequantized
    /// rows the kernel should reproduce.
    private struct PackedMatrix {
        let view: TensorView
        let dequantRows: [[Float]]

        init?(device: MTLDevice, rows: Int, n: Int, bits: Int, weightPad: Int,
              rng: inout SplitMix64) {
            precondition(bits == 4 || bits == 8)
            let bytesPerRow = bits == 4 ? n / 2 : n
            let groups = n / Quantization.groupSize
            var weights = [UInt8](repeating: 0, count: rows * bytesPerRow)
            var scales = [UInt16](repeating: 0, count: rows * groups)
            var biases = [UInt16](repeating: 0, count: rows * groups)
            var dequant: [[Float]] = []
            dequant.reserveCapacity(rows)
            for row in 0..<rows {
                let values = (0..<n).map { _ in rng.uniform(-0.5, 0.5) }
                if bits == 4 {
                    let q = Quantization.quantizeInt4Affine(values)
                    for i in 0..<bytesPerRow { weights[row * bytesPerRow + i] = q.packed[i] }
                    for i in 0..<groups {
                        scales[row * groups + i] = q.scales[i]
                        biases[row * groups + i] = q.biases[i]
                    }
                    dequant.append(Quantization.dequantizeInt4Affine(q, n: n))
                } else {
                    let q = Quantization.quantizeInt8Affine(values)
                    for i in 0..<bytesPerRow { weights[row * bytesPerRow + i] = q.packed[i] }
                    for i in 0..<groups {
                        scales[row * groups + i] = q.scales[i]
                        biases[row * groups + i] = q.biases[i]
                    }
                    dequant.append(Quantization.dequantizeInt8Affine(q, n: n))
                }
            }
            let weightsOffset = weightPad
            let scaleOffset = weightsOffset + weights.count
            let biasOffset = scaleOffset + scales.count * 2
            let total = biasOffset + biases.count * 2
            var bytes = [UInt8](repeating: 0, count: total)
            bytes.replaceSubrange(weightsOffset..<(weightsOffset + weights.count),
                                  with: weights)
            scales.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(scaleOffset..<(scaleOffset + raw.count), with: raw)
            }
            biases.withUnsafeBufferPointer { src in
                let raw = UnsafeRawBufferPointer(src)
                bytes.replaceSubrange(biasOffset..<(biasOffset + raw.count), with: raw)
            }
            guard let buffer = device.makeBuffer(bytes: bytes, length: total,
                                                 options: .storageModeShared) else {
                return nil
            }
            self.view = TensorView(buffer: buffer,
                                   offset: UInt64(weightsOffset),
                                   length: UInt64(weights.count),
                                   scaleOffset: UInt64(scaleOffset),
                                   scaleLength: UInt64(scales.count * 2),
                                   biasOffset: UInt64(biasOffset),
                                   biasLength: UInt64(biases.count * 2),
                                   shape: (UInt32(rows), UInt32(n), 1, 1),
                                   dtype: 0)
            self.dequantRows = dequant
        }
    }

    // MARK: - Absorbed projections

    @Test func embedQMatchesCPUDequantMatmul() throws {
        let cfg = Self.miniCfg
        let H = 4, T = 3
        var rng = SplitMix64(seed: 0x31A1)
        let ctx = try MetalContext()
        let mla = try MLA(context: ctx, config: cfg, numHeads: H)
        guard let embed = PackedMatrix(device: ctx.device,
                                       rows: H * cfg.latentDim,
                                       n: cfg.qkNopeDim, bits: 8, weightPad: 2,
                                       rng: &rng) else {
            Issue.record("Failed to pack embed matrix"); return
        }
        let qStride = cfg.qkNopeDim + cfg.qkRopeDim
        let qRawDim = H * qStride
        let qRaw = (0..<(T * qRawDim)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let rowsPerHead = cfg.latentDim + cfg.qkRopeDim
        let outCount = T * H * rowsPerHead
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qRaw),
              let yBuf = Fp16Buffer.make(ctx.device, count: outCount),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        try mla.encodeEmbedQ(commandBuffer: cb, embedQ: embed.view,
                         qRaw: qBuf, y: yBuf, tokens: T)
        cb.commit()
        cb.waitUntilCompleted()

        var want = [Float](repeating: 0, count: outCount)
        for t in 0..<T {
            for h in 0..<H {
                let xBase = t * qRawDim + h * qStride
                let outBase = (t * H + h) * rowsPerHead
                for r in 0..<cfg.latentDim {
                    let w = embed.dequantRows[h * cfg.latentDim + r]
                    var acc: Float = 0
                    for i in 0..<cfg.qkNopeDim { acc += w[i] * Float(qRaw[xBase + i]) }
                    want[outBase + r] = acc
                }
                for i in 0..<cfg.qkRopeDim {
                    want[outBase + cfg.latentDim + i] =
                        Float(qRaw[xBase + cfg.qkNopeDim + i])
                }
            }
        }
        let got = Fp16Buffer.read(yBuf, count: outCount)
        let rel = RelError.compute(actual: got, reference: want)
        #expect(rel < Tolerance.fp16ChainedReduction, "embed_q rel=\(rel)")
    }

    @Test func unembedMatchesCPUDequantMatmul() throws {
        let cfg = Self.miniCfg
        let H = 4, T = 3
        var rng = SplitMix64(seed: 0x31A2)
        let ctx = try MetalContext()
        let mla = try MLA(context: ctx, config: cfg, numHeads: H)
        guard let unembed = PackedMatrix(device: ctx.device,
                                         rows: H * cfg.valueHeadDim,
                                         n: cfg.latentDim, bits: 4, weightPad: 2,
                                         rng: &rng) else {
            Issue.record("Failed to pack unembed matrix"); return
        }
        let attn = (0..<(T * H * cfg.latentDim)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let outCount = T * H * cfg.valueHeadDim
        guard let attnBuf = Fp16Buffer.make(ctx.device, halves: attn),
              let yBuf = Fp16Buffer.make(ctx.device, count: outCount),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        try mla.encodeUnembed(commandBuffer: cb, unembedOut: unembed.view,
                          attn: attnBuf, y: yBuf, tokens: T)
        cb.commit()
        cb.waitUntilCompleted()

        var want = [Float](repeating: 0, count: outCount)
        for t in 0..<T {
            for h in 0..<H {
                let xBase = (t * H + h) * cfg.latentDim
                for r in 0..<cfg.valueHeadDim {
                    let w = unembed.dequantRows[h * cfg.valueHeadDim + r]
                    var acc: Float = 0
                    for i in 0..<cfg.latentDim { acc += w[i] * Float(attn[xBase + i]) }
                    want[(t * H + h) * cfg.valueHeadDim + r] = acc
                }
            }
        }
        let got = Fp16Buffer.read(yBuf, count: outCount)
        let rel = RelError.compute(actual: got, reference: want)
        #expect(rel < Tolerance.fp16ChainedReduction, "unembed rel=\(rel)")
    }

    // MARK: - Latent rows norm

    @Test func latentRowsNormMatchesCPUAndLeavesTail() throws {
        let cfg = Self.miniCfg
        let rows = 5
        let stride = cfg.latentDim + cfg.qkRopeDim
        var rng = SeedTree(0x31A3).key("mla-rows-norm")
        let x = (0..<(rows * stride)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let weight = (0..<cfg.latentDim).map { _ in
            Self.bf16Value(rng.uniform(0.5, 1.5))
        }
        let eps: Float = 1e-5
        let ctx = try MetalContext()
        let rms = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let wBuf = Self.makeBF16Buffer(ctx.device, values: weight),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        try rms.encodeBF16WRows(commandBuffer: cb, x: xBuf, weight: wBuf,
                            out: xBuf, d: UInt32(cfg.latentDim), rows: rows,
                            rowStrideElements: UInt32(stride), eps: eps)
        cb.commit()
        cb.waitUntilCompleted()

        let got = Fp16Buffer.read(xBuf, count: rows * stride)
        for row in 0..<rows {
            let base = row * stride
            var sumsq: Float = 0
            for i in 0..<cfg.latentDim {
                let v = Float(x[base + i])
                sumsq += v * v
            }
            let inv = 1 / sqrtf(sumsq / Float(cfg.latentDim) + eps)
            for i in 0..<cfg.latentDim {
                let want = Float(x[base + i]) * inv * weight[i]
                #expect(abs(got[base + i] - want) <= max(1e-2, abs(want) * 2e-2),
                        "row \(row) latent \(i): got \(got[base + i]), want \(want)")
            }
            for i in cfg.latentDim..<stride {
                #expect(got[base + i] == Float(x[base + i]),
                        "row \(row) k_pe \(i) was touched by the latent norm")
            }
        }
    }

    // MARK: - Attention

    private static func expectDecodeMatchesReference(
        cfg: MLAConfig, numQHeads: Int, seqLen: Int, seed: UInt64) throws {
        let qkDim = cfg.latentDim + cfg.qkRopeDim
        var rng = SeedTree(seed).key("mla-attn-\(numQHeads)-\(seqLen)")
        let q = (0..<(numQHeads * qkDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let rows = (0..<(seqLen * qkDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let scale = 1 / Float(cfg.qkNopeDim + cfg.latentDim).squareRoot()

        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx,
                                   maxQHeads: max(16, numQHeads),
                                   maxHeadDim: qkDim,
                                   supportsMLA: true)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let kvBuf = Fp16Buffer.make(ctx.device, halves: rows),
              let outBuf = Fp16Buffer.make(ctx.device,
                                           count: numQHeads * cfg.latentDim),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        try kernel.encodeMLA(commandBuffer: cb,
                         q: qBuf, kv: kvBuf, out: outBuf,
                         qkDim: UInt32(qkDim), vDim: UInt32(cfg.latentDim),
                         numQHeads: UInt32(numQHeads),
                         seqLen: UInt32(seqLen), scale: scale)
        cb.commit()
        cb.waitUntilCompleted()

        let ref = MLAAttentionRef.apply(
            q: q.map { Float($0) },
            rows: rows.map { Float($0) },
            qkDim: qkDim, vDim: cfg.latentDim,
            numQHeads: numQHeads, seqLen: seqLen, scale: scale)
        let got = Fp16Buffer.read(outBuf, count: numQHeads * cfg.latentDim)
        let rel = RelError.compute(actual: got, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "mla decode rel=\(rel)")
    }

    @Test func decodeAttentionMatchesReference() throws {
        try Self.expectDecodeMatchesReference(cfg: Self.miniCfg, numQHeads: 4,
                                              seqLen: 37, seed: 0x31A4)
    }

    @Test func decodeAttentionMatchesReference_kimiShape() throws {
        try Self.expectDecodeMatchesReference(cfg: Self.kimiCfg, numQHeads: 32,
                                              seqLen: 19, seed: 0x31A5)
    }

    @Test func prefillCausalMatchesSequentialDecode() throws {
        let cfg = Self.miniCfg
        let H = 4
        let qkDim = cfg.latentDim + cfg.qkRopeDim
        let history = 2, chunk = 6
        let total = history + chunk
        var rng = SeedTree(0x31A6).key("mla-prefill")
        let rows = (0..<(total * qkDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let qChunk = (0..<(chunk * H * qkDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let scale: Float = 0.11

        let ctx = try MetalContext()
        let decodeKernel = try Attention(context: ctx,
                                         maxQHeads: max(16, H),
                                         maxHeadDim: qkDim,
                                         supportsMLA: true)
        let prefillKernel = try PrefillAttention(context: ctx, supportsMLA: true)
        guard let rowsBuf = Fp16Buffer.make(ctx.device, halves: rows),
              let qBuf = Fp16Buffer.make(ctx.device, halves: qChunk),
              let prefillOut = Fp16Buffer.make(ctx.device,
                                               count: chunk * H * cfg.latentDim),
              let decodeOut = Fp16Buffer.make(ctx.device,
                                              count: H * cfg.latentDim),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate buffers"); return
        }
        let params = PrefillAttentionParams(
            startPosition: UInt32(history),
            queryCount: UInt32(chunk),
            headDim: UInt32(qkDim),
            numQHeads: UInt32(H),
            numKVHeads: 1,
            kvValidCount: UInt32(total),
            slidingWindow: 0,
            kvTokenStrideElements: UInt32(qkDim),
            qTokenStrideElements: UInt32(H * qkDim),
            oTokenStrideElements: UInt32(H * cfg.latentDim),
            scale: scale)
        try prefillKernel.encodeMLACausal(commandBuffer: cb,
                                      q: qBuf, kv: rowsBuf, out: prefillOut,
                                      params: params,
                                      vDim: UInt32(cfg.latentDim))
        cb.commit()
        cb.waitUntilCompleted()
        let prefill = Fp16Buffer.read(prefillOut, count: chunk * H * cfg.latentDim)

        for t in 0..<chunk {
            guard let stepCB = ctx.queue.makeCommandBuffer() else {
                Issue.record("no command buffer"); return
            }
            try decodeKernel.encodeMLA(commandBuffer: stepCB,
                                   q: qBuf,
                                   qOffset: t * H * qkDim * MemoryLayout<Float16>.stride,
                                   kv: rowsBuf, out: decodeOut,
                                   qkDim: UInt32(qkDim),
                                   vDim: UInt32(cfg.latentDim),
                                   numQHeads: UInt32(H),
                                   seqLen: UInt32(history + t + 1),
                                   scale: scale)
            stepCB.commit()
            stepCB.waitUntilCompleted()
            let want = Fp16Buffer.read(decodeOut, count: H * cfg.latentDim)
            for i in 0..<(H * cfg.latentDim) {
                let got = prefill[t * H * cfg.latentDim + i]
                #expect(abs(got - want[i]) <= max(1e-2, abs(want[i]) * 2e-2),
                        "row \(t) element \(i): prefill \(got), decode \(want[i])")
            }
        }
    }
}
