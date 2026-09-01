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

    @Test(.enabled(if: AttentionDepthBenchTests.enabled))
    func kvSharedSlope() throws {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let maxSeq = 4096
        let rowElements = numKVHeads * headDim
        let qCount = numQHeads * headDim

        let context = try MetalContext()
        let base = try Attention(context: context)
        let shared = try Attention(context: context,
                                   partialLoopVariant: .kvShared)

        var rng = SeedTree(0x54AF).key("v4")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<(maxSeq * rowElements)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let kBuf = Fp16Buffer.make(context.device, halves: k),
              let vBuf = Fp16Buffer.make(context.device, halves: v),
              let outBuf = Fp16Buffer.make(context.device, count: qCount) else {
            Issue.record("allocation failed"); return
        }

        let cache = try KVCacheManager(device: context.device, config: config,
                                       maxContext: maxSeq, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        let keyView = cache.keyView(layer: 3, validTokenCount: maxSeq)
        let valueView = cache.valueView(layer: 3, validTokenCount: maxSeq)
        guard let quantCB = context.queue.makeCommandBuffer() else {
            Issue.record("quantize CB failed"); return
        }
        try quantizer.encode(commandBuffer: quantCB, source: kBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: keyView, tokenCount: maxSeq,
                             elementCount: rowElements)
        try quantizer.encode(commandBuffer: quantCB, source: vBuf,
                             sourceTokenStrideElements: rowElements,
                             destination: valueView, tokenCount: maxSeq,
                             elementCount: rowElements)
        quantCB.commit(); quantCB.waitUntilCompleted()

        func timeOne(_ attention: Attention, _ seqLen: Int,
                     quantized: Bool) throws -> Double {
            guard let cb = context.queue.makeCommandBuffer() else { return .nan }
            try attention.encodeFull(commandBuffer: cb,
                                     q: qBuf,
                                     k: quantized ? keyView.buffer : kBuf,
                                     v: quantized ? valueView.buffer : vBuf,
                                     out: outBuf,
                                     headDim: UInt32(headDim),
                                     numQHeads: UInt32(numQHeads),
                                     numKVHeads: UInt32(numKVHeads),
                                     seqLen: UInt32(seqLen),
                                     kvFormat: quantized ? keyView : nil)
            cb.commit(); cb.waitUntilCompleted()
            return cb.gpuEndTime - cb.gpuStartTime
        }

        _ = try timeOne(base, maxSeq, quantized: true)
        _ = try timeOne(shared, maxSeq, quantized: true)

        var best: [String: Double] = [:]
        for _ in 0..<3 {
            for (name, attn) in [("base", base), ("kvsh", shared)] {
                for quantized in [true, false] {
                    for seqLen in [1024, 4096] {
                        for _ in 0..<10 {
                            let t = try timeOne(attn, seqLen, quantized: quantized)
                            let key = "\(name)-\(quantized)-\(seqLen)"
                            best[key] = min(best[key] ?? .greatestFiniteMagnitude, t)
                        }
                    }
                }
            }
        }
        for (name, _) in [("base", base), ("kvsh", shared)] {
            for quantized in [true, false] {
                let kind = quantized ? "int8" : "fp16"
                let a = best["\(name)-\(quantized)-1024"] ?? .nan
                let b = best["\(name)-\(quantized)-4096"] ?? .nan
                let slope = (b - a) / 3072 * 1e9
                print(String(format:
                    "V4 %@ %@  T=1024 %7.1f us  T=4096 %7.1f us  slope %6.3f us/pos",
                    name, kind, a * 1e6, b * 1e6, slope / 1000))
            }
        }
    }

    // MARK: - v11 V5 cost ledger

    /// Bench-only copy of `attention_decode_partial_shared`, hardcoded to the
    /// ornith gated shape (256/16/2, int8 g64), with one subtractive toggle
    /// per function constant. The all-defaults build is the fidelity anchor:
    /// it must time within ~10 % of the production PSO or the ledger is void
    /// (the ShrikeBench-MoE harness lesson).
    private static let v5KernelSource = """
    #include <metal_stdlib>
    using namespace metal;
    constant bool V5_PLANAR [[function_constant(0)]];
    constant uint V5_POS_BLOCK [[function_constant(1)]];
    constant bool V5_DOT_ONLY [[function_constant(2)]];
    constant bool V5_VEC4 [[function_constant(3)]];
    constant constexpr uint HD = 256;
    constant constexpr uint QPKV = 8;
    constant constexpr uint ROW_STRIDE = 544;
    constant constexpr uint VALUES = 512;
    constant constexpr uint PLANE_STRIDE = 272;
    constant constexpr uint PLANE_VALUES = 256;

    static inline float v5_load(device const uchar* cache, uint max_seq,
                                uint kv_head, uint pos, uint i) {
        if (V5_PLANAR) {
            device const uchar* row = cache
                + (uint(kv_head) * max_seq + pos) * PLANE_STRIDE;
            device const half* scales =
                reinterpret_cast<device const half*>(row + PLANE_VALUES);
            device const half* biases = scales + 4;
            const uint group = i >> 6;
            return float(row[i]) * float(scales[group]) + float(biases[group]);
        }
        device const uchar* row = cache + pos * ROW_STRIDE;
        const uint flat = kv_head * HD + i;
        device const half* scales =
            reinterpret_cast<device const half*>(row + VALUES);
        device const half* biases = scales + 8;
        const uint group = flat >> 6;
        return float(row[flat]) * float(scales[group]) + float(biases[group]);
    }

    static inline void v5_load4(device const uchar* cache, uint max_seq,
                                uint kv_head, uint pos, uint i4,
                                threadgroup float* dst) {
        device const uchar* row;
        device const half* scales;
        uint base;
        if (V5_PLANAR) {
            row = cache + (uint(kv_head) * max_seq + pos) * PLANE_STRIDE;
            scales = reinterpret_cast<device const half*>(row + PLANE_VALUES);
            base = i4 * 4u;
        } else {
            row = cache + pos * ROW_STRIDE + kv_head * HD;
            scales = reinterpret_cast<device const half*>(cache + pos * ROW_STRIDE + VALUES);
            base = i4 * 4u;
        }
        const uint group = V5_PLANAR ? (base >> 6) : ((kv_head * HD + base) >> 6);
        const uint groups = V5_PLANAR ? 4u : 8u;
        const float s = float(scales[group]);
        const float b = float(scales[groups + group]);
        const uchar4 raw = *reinterpret_cast<device const uchar4*>(row + base);
        dst[base + 0u] = float(raw.x) * s + b;
        dst[base + 1u] = float(raw.y) * s + b;
        dst[base + 2u] = float(raw.z) * s + b;
        dst[base + 3u] = float(raw.w) * s + b;
    }

    [[kernel, max_total_threads_per_threadgroup(256)]]
    void v5_partial_shared(
        device const half*  Q      [[buffer(0)]],
        device const uchar* K      [[buffer(1)]],
        device const uchar* V      [[buffer(2)]],
        device float* m_out        [[buffer(3)]],
        device float* d_out        [[buffer(4)]],
        device float* o_out        [[buffer(5)]],
        constant uint& seq_len     [[buffer(6)]],
        constant uint& chunk_len   [[buffer(7)]],
        constant uint& num_chunks  [[buffer(8)]],
        constant float& scale      [[buffer(9)]],
        constant uint& max_seq     [[buffer(10)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint lid [[thread_position_in_threadgroup]],
        uint lsize [[threads_per_threadgroup]],
        uint simd_lane_id [[thread_index_in_simdgroup]],
        uint simd_group_id [[simdgroup_index_in_threadgroup]]
    ) {
        threadgroup float q_smem[QPKV * HD];
        threadgroup float k_smem[8 * HD];
        threadgroup float v_smem[8 * HD];
        const uint kv_head = tg_id / num_chunks;
        const uint chunk = tg_id % num_chunks;
        const uint p_start = chunk * chunk_len;
        uint p_end = p_start + chunk_len;
        if (p_end > seq_len) { p_end = seq_len; }

        for (uint i = lid; i < QPKV * HD; i += lsize) {
            const uint head = i / HD;
            q_smem[i] = float(Q[(kv_head * QPKV + head) * HD + (i - head * HD)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o_local[HD / 32];
        for (uint s = 0; s < HD / 32; ++s) { o_local[s] = 0.0f; }
        float m_run = -INFINITY;
        float d_run = 0.0f;
        const bool liveHead = simd_group_id < QPKV;
        threadgroup const float* q_mine = q_smem + simd_group_id * HD;

        for (uint pb = p_start; pb < p_end; pb += V5_POS_BLOCK) {
            const uint blockCount = min(V5_POS_BLOCK, p_end - pb);
            if (V5_VEC4) {
                for (uint e4 = lid; e4 < blockCount * (HD / 4u); e4 += lsize) {
                    const uint j = e4 / (HD / 4u);
                    const uint i4 = e4 - j * (HD / 4u);
                    v5_load4(K, max_seq, kv_head, pb + j, i4, k_smem + j * HD);
                    v5_load4(V, max_seq, kv_head, pb + j, i4, v_smem + j * HD);
                }
            } else {
                for (uint e = lid; e < blockCount * HD; e += lsize) {
                    const uint j = e / HD;
                    const uint i = e - j * HD;
                    k_smem[e] = v5_load(K, max_seq, kv_head, pb + j, i);
                    v_smem[e] = v5_load(V, max_seq, kv_head, pb + j, i);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (liveHead) {
                for (uint j = 0; j < blockCount; ++j) {
                    threadgroup const float* k_row = k_smem + j * HD;
                    threadgroup const float* v_row = v_smem + j * HD;
                    float partial = 0.0f;
                    for (uint i = simd_lane_id; i < HD; i += 32u) {
                        partial = fma(q_mine[i], k_row[i], partial);
                    }
                    const float s = simd_sum(partial) * scale;
                    if (V5_DOT_ONLY) {
                        d_run += s;
                        uint slot = 0;
                        for (uint i = simd_lane_id; i < HD; i += 32u) {
                            o_local[slot] = fma(s, v_row[i], o_local[slot]);
                            slot += 1;
                        }
                    } else {
                        const float m_new = max(m_run, s);
                        const float alpha = fast::exp(m_run - m_new);
                        const float p_exp = fast::exp(s - m_new);
                        d_run = d_run * alpha + p_exp;
                        uint slot = 0;
                        for (uint i = simd_lane_id; i < HD; i += 32u) {
                            o_local[slot] = o_local[slot] * alpha + p_exp * v_row[i];
                            slot += 1;
                        }
                        m_run = m_new;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (liveHead) {
            const uint q_head = kv_head * QPKV + simd_group_id;
            const uint base = q_head * num_chunks + chunk;
            if (simd_lane_id == 0) { m_out[base] = m_run; d_out[base] = d_run; }
            device float* o_row = o_out + base * HD;
            uint slot = 0;
            for (uint i = simd_lane_id; i < HD; i += 32u) {
                o_row[i] = o_local[slot];
                slot += 1;
            }
        }
    }
    """

    private struct V5Arm {
        let name: String
        let pso: MTLComputePipelineState?
        let planar: Bool
    }

    private static func v5Pipeline(_ device: MTLDevice,
                                   planar: Bool, block: UInt32,
                                   dotOnly: Bool,
                                   vec4: Bool = false) throws -> MTLComputePipelineState {
        let options = MTLCompileOptions()
        let library = try device.makeLibrary(source: v5KernelSource, options: options)
        let values = MTLFunctionConstantValues()
        var p = planar, b = block, d = dotOnly, v4 = vec4
        values.setConstantValue(&p, type: .bool, index: 0)
        values.setConstantValue(&b, type: .uint, index: 1)
        values.setConstantValue(&d, type: .bool, index: 2)
        values.setConstantValue(&v4, type: .bool, index: 3)
        let function = try library.makeFunction(name: "v5_partial_shared",
                                                constantValues: values)
        return try device.makeComputePipelineState(function: function)
    }

    /// Repack the production interleaved cache rows ([pos][2 heads][values |
    /// scales | biases], 544 B) into dense per-head planes ([head][pos][256
    /// values | 4 scales | 4 biases], 272 B) — dequant-identical bytes.
    private static func v5PlanarCopy(_ device: MTLDevice, source: MTLBuffer,
                                     maxSeq: Int) -> MTLBuffer? {
        guard let planar = device.makeBuffer(length: 2 * maxSeq * 272,
                                             options: .storageModeShared) else {
            return nil
        }
        let src = source.contents().assumingMemoryBound(to: UInt8.self)
        let dst = planar.contents().assumingMemoryBound(to: UInt8.self)
        for pos in 0..<maxSeq {
            let row = pos * 544
            for head in 0..<2 {
                let out = (head * maxSeq + pos) * 272
                memcpy(dst + out, src + row + head * 256, 256)
                memcpy(dst + out + 256, src + row + 512 + head * 8, 8)
                memcpy(dst + out + 264, src + row + 528 + head * 8, 8)
            }
        }
        return planar
    }

    @Test(.enabled(if: AttentionDepthBenchTests.enabled))
    func v5CostLedger() throws {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let maxSeq = 4096
        let numChunks = 64
        let qCount = numQHeads * headDim

        let context = try MetalContext()
        let attention = try Attention(context: context, partialLoopVariant: .kvShared)
        var rng = SeedTree(0x55ED).key("v5")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kv = (0..<(maxSeq * numKVHeads * headDim)).map { _ in
            Float16(rng.uniform(-0.5, 0.5))
        }
        let cache = try KVCacheManager(device: context.device, config: config,
                                       maxContext: maxSeq, precision: .int8)
        let quantizer = try KVCacheQuantizer(context: context)
        let keyView = cache.keyView(layer: 3, validTokenCount: maxSeq)
        let valueView = cache.valueView(layer: 3, validTokenCount: maxSeq)
        guard let qBuf = Fp16Buffer.make(context.device, halves: q),
              let kvBuf = Fp16Buffer.make(context.device, halves: kv),
              let mBuf = context.device.makeBuffer(length: qCount * numChunks * 4),
              let dBuf = context.device.makeBuffer(length: qCount * numChunks * 4),
              let oBuf = context.device.makeBuffer(
                length: numQHeads * numChunks * headDim * 4),
              let quantCB = context.queue.makeCommandBuffer() else {
            Issue.record("v5 allocation failed"); return
        }
        try quantizer.encode(commandBuffer: quantCB, source: kvBuf,
                             sourceTokenStrideElements: numKVHeads * headDim,
                             destination: keyView, tokenCount: maxSeq,
                             elementCount: numKVHeads * headDim)
        try quantizer.encode(commandBuffer: quantCB, source: kvBuf,
                             sourceTokenStrideElements: numKVHeads * headDim,
                             destination: valueView, tokenCount: maxSeq,
                             elementCount: numKVHeads * headDim)
        quantCB.commit(); quantCB.waitUntilCompleted()
        guard let kPlanar = Self.v5PlanarCopy(context.device,
                                              source: keyView.buffer,
                                              maxSeq: maxSeq),
              let vPlanar = Self.v5PlanarCopy(context.device,
                                              source: valueView.buffer,
                                              maxSeq: maxSeq) else {
            Issue.record("planar repack failed"); return
        }

        let prodPSO = attention.partialPipeline(headDim: UInt32(headDim),
                                                numQHeads: UInt32(numQHeads),
                                                numKVHeads: UInt32(numKVHeads),
                                                numChunks: numChunks,
                                                useGQAPartial: false,
                                                kvFormat: keyView)
        let arms: [V5Arm] = try [
            V5Arm(name: "prod-pso ", pso: prodPSO, planar: false),
            V5Arm(name: "copy-base", pso: Self.v5Pipeline(context.device,
                  planar: false, block: 4, dotOnly: false), planar: false),
            V5Arm(name: "planar   ", pso: Self.v5Pipeline(context.device,
                  planar: true, block: 4, dotOnly: false), planar: true),
            V5Arm(name: "block2   ", pso: Self.v5Pipeline(context.device,
                  planar: false, block: 2, dotOnly: false), planar: false),
            V5Arm(name: "block8   ", pso: Self.v5Pipeline(context.device,
                  planar: false, block: 8, dotOnly: false), planar: false),
            V5Arm(name: "dot-only ", pso: Self.v5Pipeline(context.device,
                  planar: false, block: 4, dotOnly: true), planar: false),
            V5Arm(name: "plan+dot ", pso: Self.v5Pipeline(context.device,
                  planar: true, block: 4, dotOnly: true), planar: true),
            V5Arm(name: "vec4     ", pso: Self.v5Pipeline(context.device,
                  planar: false, block: 4, dotOnly: false, vec4: true),
                  planar: false),
            V5Arm(name: "vec4+plan", pso: Self.v5Pipeline(context.device,
                  planar: true, block: 4, dotOnly: false, vec4: true),
                  planar: true),
        ]

        var best: [String: Double] = [:]
        for _ in 0..<3 {
            for arm in arms {
                guard let pso = arm.pso else { continue }
                for seqLen in [1024, 4096] {
                    for _ in 0..<10 {
                        let t = try v5Time(context: context, pso: pso,
                                           arm: arm, attention: attention,
                                           qBuf: qBuf, keyView: keyView,
                                           valueView: valueView,
                                           kPlanar: kPlanar, vPlanar: vPlanar,
                                           mBuf: mBuf, dBuf: dBuf, oBuf: oBuf,
                                           headDim: headDim, seqLen: seqLen,
                                           numChunks: numChunks, maxSeq: maxSeq)
                        let key = "\(arm.name)-\(seqLen)"
                        best[key] = min(best[key] ?? .greatestFiniteMagnitude, t)
                    }
                }
            }
        }
        for arm in arms {
            let a = best["\(arm.name)-1024"] ?? .nan
            let b = best["\(arm.name)-4096"] ?? .nan
            let slope = (b - a) / 3072 * 1e9 / 1000
            print(String(format: "V5 %@ T=1024 %7.1f us  T=4096 %7.1f us  slope %6.3f us/pos",
                         arm.name, a * 1e6, b * 1e6, slope))
        }
    }

    private func v5Time(context: MetalContext, pso: MTLComputePipelineState,
                        arm: V5Arm, attention: Attention,
                        qBuf: MTLBuffer, keyView: KVView, valueView: KVView,
                        kPlanar: MTLBuffer, vPlanar: MTLBuffer,
                        mBuf: MTLBuffer, dBuf: MTLBuffer, oBuf: MTLBuffer,
                        headDim: Int, seqLen: Int, numChunks: Int,
                        maxSeq: Int) throws -> Double {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return .nan }
        let chunkLen = (seqLen + numChunks - 1) / numChunks
        enc.setComputePipelineState(pso)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(arm.planar ? kPlanar : keyView.buffer, offset: 0, index: 1)
        enc.setBuffer(arm.planar ? vPlanar : valueView.buffer, offset: 0, index: 2)
        enc.setBuffer(mBuf, offset: 0, index: 3)
        enc.setBuffer(dBuf, offset: 0, index: 4)
        enc.setBuffer(oBuf, offset: 0, index: 5)
        var params = [UInt32(headDim), UInt32(16), UInt32(2), UInt32(seqLen),
                      UInt32(0), UInt32(chunkLen), UInt32(numChunks)]
        var scale = Attention.defaultScale(headDim: UInt32(headDim))
        if arm.name.hasPrefix("prod") {
            enc.setBytes(&params[0], length: 4, index: 6)
            enc.setBytes(&params[1], length: 4, index: 7)
            enc.setBytes(&params[2], length: 4, index: 8)
            enc.setBytes(&params[3], length: 4, index: 9)
            enc.setBytes(&params[4], length: 4, index: 10)
            enc.setBytes(&params[5], length: 4, index: 11)
            enc.setBytes(&params[6], length: 4, index: 12)
            enc.setBytes(&scale, length: 4, index: 13)
            var bits = UInt32(keyView.precision.rawValue)
            var stride = UInt32(keyView.stride)
            var valueBytes = UInt32(keyView.valueBytes)
            var groupSize = UInt32(keyView.groupSize)
            enc.setBytes(&bits, length: 4, index: 14)
            enc.setBytes(&stride, length: 4, index: 15)
            enc.setBytes(&valueBytes, length: 4, index: 16)
            enc.setBytes(&groupSize, length: 4, index: 17)
        } else {
            var maxSeqU = UInt32(maxSeq)
            enc.setBytes(&params[3], length: 4, index: 6)
            enc.setBytes(&params[5], length: 4, index: 7)
            enc.setBytes(&params[6], length: 4, index: 8)
            enc.setBytes(&scale, length: 4, index: 9)
            enc.setBytes(&maxSeqU, length: 4, index: 10)
        }
        let width = min(256, pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: 2 * numChunks, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width,
                                                                height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        return cb.gpuEndTime - cb.gpuStartTime
    }
}
