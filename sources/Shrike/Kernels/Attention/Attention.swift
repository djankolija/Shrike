import Foundation
import Metal


struct AttentionSplitGeometry: Sendable, Equatable {
    let effectiveLength: Int
    let numChunks: Int
    let chunkLength: Int
    let partialThreadgroups: Int
    let useSWAGroupedPartial: Bool
}


/// Swift wrapper for sliding-window and full-causal decode attention.
///
/// The kernels assume a single decoded query token (`M_q = 1`) and a
/// contiguous KV cache of length `seqLen`. The MPP-tensor-core prefill path
/// (`M_q > 1`) is separate.
///
/// Buffer contracts (FP16 throughout):
///   - `q`   : `[numQHeads, headDim]`
///   - `k`   : `[seqLen, numKVHeads, headDim]`
///   - `v`   : same shape as `k`. Full-layer K and V must remain distinct after
///             their separate per-head normalization and RoPE paths.
///   - `out` : `[numQHeads, headDim]`
public final class Attention {
    private let ctx: MetalContext
    private let psoPartial: MTLComputePipelineState
    let psoPartialShared: MTLComputePipelineState
    private let partialLoopVariant: PartialLoopVariant
    /// v11 V4.1: bake the shape and KV storage format into the shared
    /// partial's pipeline so its per-element group indexing strength-reduces.
    /// Off exists for the bitwise arm only — the outputs must match exactly.
    private let specializesKVShared: Bool

    private struct KVSharedShapeKey: Hashable {
        let headDim: UInt32
        let numQHeads: UInt32
        let numKVHeads: UInt32
        let kvBits: UInt32
        let kvStride: UInt32
        let kvValueBytes: UInt32
        let kvGroupSize: UInt32
    }
    private struct SpecializedPartialKey: Hashable {
        let name: String
        let shape: KVSharedShapeKey
    }
    private var specializedPartialPSOs: [SpecializedPartialKey: MTLComputePipelineState] = [:]
    /// Mirrors `kAttnStreamCount`: the partial count stays `maxChunks`, the dispatch is a quarter of it.
    static let streamCount = 4

    /// Which full-attention partial a shape takes; one function decides it for the
    /// encode's geometry and for the pipeline lookup, so the two cannot disagree.
    enum SharedPartialChoice: Equatable { case stream, shared, generic }
    private(set) var lastSplitChoice: SharedPartialChoice?

    /// v11 V4 applicability: full-attention path only, head slice fits the
    /// kernel's threadgroup staging, and the GQA fan-in fits the simdgroups.
    private func kvSharedApplicable(headDim: UInt32,
                                    numQHeads: UInt32,
                                    numKVHeads: UInt32,
                                    useGQAPartial: Bool,
                                    ringCapacity: UInt32) -> Bool {
        (partialLoopVariant == .kvShared || partialLoopVariant == .stream)
            && !useGQAPartial
            && ringCapacity == 0
            && headDim <= 256
            && numQHeads % numKVHeads == 0
            && numQHeads / numKVHeads <= 8
    }
    var partialPipelineMaxThreadsForBench: Int {
        psoPartial.maxTotalThreadsPerThreadgroup
    }

    /// The shape the streaming scan is written for (v19 Task 3): the runner refuses
    /// a model outside it rather than fall back silently; the wrapper itself still
    /// falls back, for the tests and the bench.
    public static func streamServesShape(headDim: Int, numQHeads: Int, numKVHeads: Int) -> Bool {
        headDim == 256 && numKVHeads > 0 && numQHeads == 8 * numKVHeads
    }

    public static func streamServes(headDim: Int, numQHeads: Int, numKVHeads: Int,
                                    precision: KVCachePrecision) -> Bool {
        streamServesShape(headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads)
            && precision == .int8
    }

    private func streamApplicable(headDim: UInt32,
                                  numQHeads: UInt32,
                                  numKVHeads: UInt32,
                                  useGQAPartial: Bool,
                                  ringCapacity: UInt32,
                                  kvFormat: KVView?) -> Bool {
        guard partialLoopVariant == .stream, !useGQAPartial, ringCapacity == 0,
              let kvFormat,
              Self.streamServes(headDim: Int(headDim), numQHeads: Int(numQHeads),
                                numKVHeads: Int(numKVHeads), precision: kvFormat.precision),
              kvFormat.groupSize == KVCacheManager.quantizationGroupSize else {
            return false
        }
        return true
    }

    func sharedPartialChoice(headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             useGQAPartial: Bool, ringCapacity: UInt32,
                             kvFormat: KVView?) -> SharedPartialChoice {
        if streamApplicable(headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                            useGQAPartial: useGQAPartial, ringCapacity: ringCapacity,
                            kvFormat: kvFormat) {
            return .stream
        }
        if kvSharedApplicable(headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                              useGQAPartial: useGQAPartial, ringCapacity: ringCapacity) {
            return .shared
        }
        return .generic
    }
    private let psoGQAPartial: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    private let psoPartialSWA: MTLComputePipelineState
    private let psoPartialFull: MTLComputePipelineState
    private let psoGQAPartialSWA: MTLComputePipelineState
    private let psoGQAPartialSWAChunks16: MTLComputePipelineState
    private let psoPartialFullChunks16: MTLComputePipelineState
    private let psoCombineSWA: MTLComputePipelineState
    private let psoCombineFull: MTLComputePipelineState
    private let psoCombineSWAChunks16: MTLComputePipelineState
    private let psoCombineFullChunks16: MTLComputePipelineState
    private let psoCombineSinks: MTLComputePipelineState?
    private let psoMLAPartial: MTLComputePipelineState?

    /// Mirrors `kAttnThreads` in `attention.metal`. The kernel was authored
    /// with a hardcoded 256-thread group so its threadgroup-memory scratch
    /// (q_smem[512] + reduce[8] + bcast) sizes are correct.
    static let threadsPerGroup: Int = 256

    /// `kAttnMaxHeadDim` in attention.metal — the kernel's threadgroup scratch
    /// ceiling, independent of the instance split-KV scratch limits below.
    static let kernelMaxHeadDim = 512
    /// `kAttnMLAMaxQKDim` — the MLA partial's q-smem ceiling (576-wide rows);
    /// its o partials stay within `kernelMaxHeadDim`.
    static let kernelMaxMLAQKDim = 576
    static let maxChunks = 64

    /// Split-KV partial-scratch limits, sized from the architecture at init
    /// (defaults describe the Qwen baseline: 16 Q heads · 64 chunks · 512
    /// head dim ≈ 2 MB of FP32 o-scratch).
    let maxQHeads: Int
    let maxHeadDim: Int
    /// Full attention uses 16 base chunks.
    private static let defaultFullChunks = min(maxChunks, 16)
    private static let defaultGQASWAChunks = 8

    // Partial state written by pass 1, read by pass 2. One shared allocation:
    // attention runs once per layer, serially, and pass 2 hazard-tracks pass 1
    // within the same command buffer — no race (mirrors MoE.routerLogits).
    let mPartial: MTLBuffer
    let dPartial: MTLBuffer
    let oPartial: MTLBuffer

    /// Which inner loop the full-attention decode partial runs: `.stream` (v19)
    /// where its shape gate allows, `.kvShared` (v11) elsewhere.
    public enum PartialLoopVariant: Sendable { case blockReduce, kvShared, stream }

    public init(context: MetalContext,
                maxQHeads: Int = 16,
                maxHeadDim: Int = 512,
                supportsSinks: Bool = false,
                supportsMLA: Bool = false,
                partialLoopVariant: PartialLoopVariant = .blockReduce,
                specializesKVShared: Bool = true) throws {
        self.partialLoopVariant = partialLoopVariant
        self.specializesKVShared = specializesKVShared
        // maxHeadDim may exceed kernelMaxHeadDim (Kimi's 576-wide MLA rows
        // size the o-scratch); the per-encode paths enforce their own kernel
        // ceilings.
        precondition(maxQHeads > 0 && maxHeadDim > 0
                     && maxHeadDim <= Self.kernelMaxMLAQKDim,
                     "split-KV scratch limits must be positive and fit a kernel scratch")
        self.ctx = context
        self.maxQHeads = maxQHeads
        self.maxHeadDim = maxHeadDim
        self.psoCombineSinks = supportsSinks
            ? try context.pipeline("attention_decode_combine",
                                   constants: [MetalFunctionConstant(index: 66, value: .bool(true))])
            : nil
        self.psoMLAPartial = supportsMLA
            ? try context.pipeline("attention_decode_mla_partial")
            : nil
        self.psoPartial = try context.pipeline("attention_decode_partial")
        self.psoPartialShared = try context.pipeline("attention_decode_partial_shared")
        self.psoGQAPartial = try context.pipeline("attention_decode_gqa_swa_partial")
        self.psoCombine = try context.pipeline("attention_decode_combine")
        self.psoPartialSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_partial",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoPartialFull = try Self.specializedPipeline(context,
                                                           "attention_decode_partial",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoGQAPartialSWA = try Self.specializedPipeline(context,
                                                             "attention_decode_gqa_swa_partial",
                                                             headDim: 256,
                                                             numQHeads: 16,
                                                             numKVHeads: 8)
        self.psoGQAPartialSWAChunks16 = try Self.specializedPipeline(context,
                                                                     "attention_decode_gqa_swa_partial",
                                                                     headDim: 256,
                                                                     numQHeads: 16,
                                                                     numKVHeads: 8,
                                                                     numChunks: 16)
        self.psoPartialFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_partial",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        self.psoCombineSWA = try Self.specializedPipeline(context,
                                                          "attention_decode_combine",
                                                          headDim: 256,
                                                          numQHeads: 16,
                                                          numKVHeads: 8)
        self.psoCombineFull = try Self.specializedPipeline(context,
                                                           "attention_decode_combine",
                                                           headDim: 512,
                                                           numQHeads: 16,
                                                           numKVHeads: 2)
        self.psoCombineSWAChunks16 = try Self.specializedPipeline(context,
                                                                  "attention_decode_combine",
                                                                  headDim: 256,
                                                                  numQHeads: 16,
                                                                  numKVHeads: 8,
                                                                  numChunks: 16)
        self.psoCombineFullChunks16 = try Self.specializedPipeline(context,
                                                                   "attention_decode_combine",
                                                                   headDim: 512,
                                                                   numQHeads: 16,
                                                                   numKVHeads: 2,
                                                                   numChunks: 16)
        let md = maxQHeads * Self.maxChunks
        guard let m = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                                options: .storageModeShared),
	              let d = context.device.makeBuffer(length: md * MemoryLayout<Float>.size,
	                                                options: .storageModeShared),
	              let o = context.device.makeBuffer(length: md * maxHeadDim * MemoryLayout<Float>.size,
	                                                options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("attention split-KV scratch")
        }
        self.mPartial = m; self.dPartial = d; self.oPartial = o
        self.splitStateLock = NSLock()
        self.splitInFlight = false
    }

    // K15: the split-KV partial scratch (mPartial/dPartial/oPartial) is shared
    // instance state written by pass 1 and read by pass 2. The runtime encodes
    // attention strictly serially — one layer at a time, one command buffer —
    // so no two encodeSplit calls may be in flight. Guard that contract at
    // runtime: a reentrant encodeSplit would corrupt pass-1 state and is a
    // programming error, so it throws loudly instead of silently corrupting.
    private let splitStateLock: NSLock
    private var splitInFlight: Bool

    /// Number of K/V chunks for a range of `effLen` positions — the split
    /// factor used by the production split path.
    static func chunkCount(effLen: Int, preferGQASWA: Bool = false) -> Int {
        let eff = max(1, effLen)
        let defaultChunks = preferGQASWA ? defaultGQASWAChunks : defaultFullChunks
        return max(1, min(defaultChunks, min(maxChunks, eff)))
    }

    static func splitGeometry(numQHeads: UInt32,
                                     numKVHeads: UInt32,
                                     seqLen: UInt32,
                                     kvStart: UInt32,
                                     preferGQASWA: Bool) -> AttentionSplitGeometry {
        let qPerKV = Int(numQHeads / numKVHeads)
        let useSWAGQAPartial = preferGQASWA && qPerKV <= 2
        let effectiveLength = Int(seqLen) - Int(kvStart)
        let baseChunks = Self.chunkCount(effLen: effectiveLength,
                                         preferGQASWA: useSWAGQAPartial)
        let numChunks = useSWAGQAPartial
            ? max(baseChunks, min(Self.maxChunks, baseChunks * qPerKV))
            : baseChunks
        let chunkLength = (max(1, effectiveLength) + numChunks - 1) / numChunks
        let partialHeadGroups = useSWAGQAPartial ? Int(numKVHeads) : Int(numQHeads)
        return AttentionSplitGeometry(effectiveLength: effectiveLength,
                                      numChunks: numChunks,
                                      chunkLength: chunkLength,
                                      partialThreadgroups: partialHeadGroups * numChunks,
                                      useSWAGroupedPartial: useSWAGQAPartial)
    }


    /// Sliding-window attention. `window` caps the K/V positions to the most
    /// recent `window` entries (`[max(0, seqLen-window), seqLen)`).
    /// `scale` defaults to `rsqrt(head_dim)` for generic callers;
    /// callers with a configured attention scale pass it explicitly.
    func encodeSWA(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          v: MTLBuffer, vOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          headDim: UInt32,
                          numQHeads: UInt32,
                          numKVHeads: UInt32,
                          seqLen: UInt32,
                          window: UInt32,
                          scale: Float? = nil,
                          ringCapacity: UInt32 = 0,
                          sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                          kvFormat: KVView? = nil) throws {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        let sc = scale ?? Self.defaultScale(headDim: headDim)
        let kvStart = seqLen > window ? seqLen - window : 0

        try encodeSplit(commandBuffer: commandBuffer,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: kvStart, scale: sc,
                    preferGQASWA: true,
                    ringCapacity: ringCapacity,
                    sinks: sinks, sinksOffset: sinksOffset,
                    kvFormat: kvFormat)
    }

    /// Full attention. Separate normalization and RoPE make the cache streams
    /// distinct here. `scale` mirrors `encodeSWA`.
    public func encodeFull(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer, qOffset: Int = 0,
                           k: MTLBuffer, kOffset: Int = 0,
                           v: MTLBuffer, vOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           seqLen: UInt32,
                           scale: Float? = nil,
                           sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                           kvFormat: KVView? = nil) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { encoder.endEncoding() }
        try encodeFull(encoder: encoder,
                       q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                       v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                       headDim: headDim, numQHeads: numQHeads,
                       numKVHeads: numKVHeads, seqLen: seqLen, scale: scale,
                       sinks: sinks, sinksOffset: sinksOffset,
                       kvFormat: kvFormat)
    }

    func encodeFull(encoder: MTLComputeCommandEncoder,
                           q: MTLBuffer, qOffset: Int = 0,
                           k: MTLBuffer, kOffset: Int = 0,
                           v: MTLBuffer, vOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           headDim: UInt32,
                           numQHeads: UInt32,
                           numKVHeads: UInt32,
                           seqLen: UInt32,
                           scale: Float? = nil,
                           sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                           kvFormat: KVView? = nil) throws {
        precondition(numQHeads % numKVHeads == 0,
                     "numQHeads must be a multiple of numKVHeads for GQA")
        precondition(headDim <= 512,
                     "head_dim must be <= 512 (kernel scratch is sized for the full-attn case)")
        precondition(seqLen > 0, "full attention requires at least one KV position")
        let sc = scale ?? Self.defaultScale(headDim: headDim)

        try encodeSplit(encoder: encoder,
                    q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                    v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                    headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
                    seqLen: seqLen, kvStart: 0, scale: sc,
                    preferGQASWA: false,
                    sinks: sinks, sinksOffset: sinksOffset,
                    kvFormat: kvFormat)
    }


    /// MLA (Kimi-Linear) decode attention: MQA over fused FP16 cache rows of
    /// `qkDim` elements where V is each row's `vDim`-prefix. Same two-pass
    /// split-KV shape as `encodeFull`; the combine runs at `head_dim = vDim`.
    func encodeMLA(commandBuffer: MTLCommandBuffer,
                   q: MTLBuffer, qOffset: Int = 0,
                   kv: MTLBuffer, kvOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   qkDim: UInt32,
                   vDim: UInt32,
                   numQHeads: UInt32,
                   seqLen: UInt32,
                   scale: Float) throws {
        guard let partialPSO = psoMLAPartial else {
            throw MetalError.invalidState(
                "encodeMLA on an Attention built without supportsMLA")
        }
        precondition(qkDim <= UInt32(Self.kernelMaxMLAQKDim),
                     "MLA qkDim exceeds the kernel's q scratch")
        precondition(vDim <= qkDim && Int(vDim) <= min(maxHeadDim, Self.kernelMaxHeadDim),
                     "MLA vDim must fit the o scratch and the K row prefix")
        precondition(Int(numQHeads) <= maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(maxQHeads))")
        precondition(seqLen > 0, "MLA attention requires at least one KV position")
        splitStateLock.lock()
        let reentered = splitInFlight
        splitInFlight = true
        splitStateLock.unlock()
        defer {
            splitStateLock.lock()
            splitInFlight = false
            splitStateLock.unlock()
        }
        guard !reentered else {
            throw MetalError.invalidState(
                "encodeMLA re-entered while a split-KV pass was in flight; attention must encode serially per layer")
        }
        let geometry = Self.splitGeometry(numQHeads: numQHeads,
                                          numKVHeads: 1,
                                          seqLen: seqLen,
                                          kvStart: 0,
                                          preferGQASWA: false)
        let nChunks = geometry.numChunks
        let tgWidth = min(Self.threadsPerGroup, Int(partialPSO.maxTotalThreadsPerThreadgroup))

        guard let p1 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(kv, offset: kvOffset, index: 1)
        p1.setBuffer(mPartial, offset: 0, index: 2)
        p1.setBuffer(dPartial, offset: 0, index: 3)
        p1.setBuffer(oPartial, offset: 0, index: 4)
        var qk = qkDim, vd = vDim, nq = numQHeads, sl = seqLen
        var cl = UInt32(geometry.chunkLength), nc = UInt32(nChunks), sc = scale
        p1.setBytes(&qk, length: MemoryLayout<UInt32>.size, index: 5)
        p1.setBytes(&vd, length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq, length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&cl, length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&sc, length: MemoryLayout<Float>.size,  index: 11)
        p1.dispatchThreadgroups(MTLSize(width: geometry.partialThreadgroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        p1.endEncoding()

        guard let p2 = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        p2.setComputePipelineState(psoCombine)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = vDim, nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(psoCombine.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
        p2.endEncoding()
    }

    /// Two-pass split-KV (Flash-Decoding) dispatch shared by SWA and full
    /// attention — they differ only by `kvStart`. Pass 1 fans the head's
    /// `[kvStart, seqLen)` range across `chunkCount` threadgroups per head;
    /// pass 2 merges the partials. Both encoders go on the same command buffer
    /// so pass 2 hazard-tracks the partial scratch written by pass 1.
    private func encodeSplit(commandBuffer: MTLCommandBuffer,
                             q: MTLBuffer, qOffset: Int,
                             k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             seqLen: UInt32, kvStart: UInt32, scale: Float,
                             preferGQASWA: Bool,
                             ringCapacity: UInt32 = 0,
                             sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                             kvFormat: KVView? = nil) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        defer { encoder.endEncoding() }
        try encodeSplit(encoder: encoder,
                        q: q, qOffset: qOffset, k: k, kOffset: kOffset,
                        v: v, vOffset: vOffset, out: out, outOffset: outOffset,
                        headDim: headDim, numQHeads: numQHeads,
                        numKVHeads: numKVHeads, seqLen: seqLen, kvStart: kvStart,
                        scale: scale, preferGQASWA: preferGQASWA,
                        ringCapacity: ringCapacity,
                        sinks: sinks, sinksOffset: sinksOffset,
                        kvFormat: kvFormat)
    }

    private func encodeSplit(encoder: MTLComputeCommandEncoder,
                             q: MTLBuffer, qOffset: Int,
                             k: MTLBuffer, kOffset: Int,
                             v: MTLBuffer, vOffset: Int,
                             out: MTLBuffer, outOffset: Int,
                             headDim: UInt32, numQHeads: UInt32, numKVHeads: UInt32,
                             seqLen: UInt32, kvStart: UInt32, scale: Float,
                             preferGQASWA: Bool,
                             ringCapacity: UInt32 = 0,
                             sinks: MTLBuffer? = nil, sinksOffset: Int = 0,
                             kvFormat: KVView? = nil) throws {
        precondition(Int(numQHeads) <= maxQHeads,
                     "numQHeads \(numQHeads) exceeds split-KV scratch (max \(maxQHeads))")
        precondition(Int(headDim) <= maxHeadDim,
                     "head_dim \(headDim) exceeds split-KV scratch (max \(maxHeadDim))")
        precondition(ringCapacity == 0 || preferGQASWA,
                     "KV ring is only valid for SWA attention")
        splitStateLock.lock()
        let reentered = splitInFlight
        splitInFlight = true
        splitStateLock.unlock()
        defer {
            splitStateLock.lock()
            splitInFlight = false
            splitStateLock.unlock()
        }
        guard !reentered else {
            throw MetalError.invalidState(
                "encodeSplit re-entered while the previous split-KV pass was in flight; attention must encode serially per layer")
        }
        let geometry = Self.splitGeometry(numQHeads: numQHeads,
                                          numKVHeads: numKVHeads,
                                          seqLen: seqLen,
                                          kvStart: kvStart,
                                          preferGQASWA: preferGQASWA)
        let useSWAGQAPartial = geometry.useSWAGroupedPartial
        var nChunks = geometry.numChunks
        var chunkLen = geometry.chunkLength
        let choice = sharedPartialChoice(headDim: headDim, numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         useGQAPartial: useSWAGQAPartial,
                                         ringCapacity: ringCapacity, kvFormat: kvFormat)
        let effective = max(1, Int(seqLen) - Int(kvStart))
        switch choice {
        case .shared:
            // The shared partial has numKVHeads-fold fewer TGs per chunk, so it
            // takes the full chunk budget to keep the machine occupied.
            nChunks = max(1, min(Self.maxChunks, effective))
            chunkLen = (effective + nChunks - 1) / nChunks
        case .stream:
            // Every (chunk, stream) slot writes a partial, empty ones as
            // (-inf, 0, 0), so the partial count stays maxChunks at any length.
            nChunks = Self.maxChunks
            let dispatched = Self.maxChunks / Self.streamCount
            chunkLen = (effective + dispatched - 1) / dispatched
        case .generic:
            break
        }
        lastSplitChoice = choice
        let partialPSO = partialPipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks,
                                         useGQAPartial: useSWAGQAPartial,
                                         ringCapacity: ringCapacity,
                                         kvFormat: kvFormat)
        let tgWidth = min(Self.threadsPerGroup, Int(partialPSO.maxTotalThreadsPerThreadgroup))

        let p1 = encoder
        p1.setComputePipelineState(partialPSO)
        p1.setBuffer(q, offset: qOffset, index: 0)
        p1.setBuffer(k, offset: kOffset, index: 1)
        p1.setBuffer(v, offset: vOffset, index: 2)
        p1.setBuffer(mPartial, offset: 0, index: 3)
        p1.setBuffer(dPartial, offset: 0, index: 4)
        p1.setBuffer(oPartial, offset: 0, index: 5)
        var hd = headDim, nq = numQHeads, nkv = numKVHeads, sl = seqLen, ks = kvStart
        var cl = UInt32(chunkLen), nc = UInt32(nChunks), sc = scale
        p1.setBytes(&hd,  length: MemoryLayout<UInt32>.size, index: 6)
        p1.setBytes(&nq,  length: MemoryLayout<UInt32>.size, index: 7)
        p1.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        p1.setBytes(&sl,  length: MemoryLayout<UInt32>.size, index: 9)
        p1.setBytes(&ks,  length: MemoryLayout<UInt32>.size, index: 10)
        p1.setBytes(&cl,  length: MemoryLayout<UInt32>.size, index: 11)
        p1.setBytes(&nc,  length: MemoryLayout<UInt32>.size, index: 12)
        p1.setBytes(&sc,  length: MemoryLayout<Float>.size,  index: 13)
        var kvBits = UInt32(kvFormat?.precision.rawValue ?? 16)
        var kvStride = UInt32(kvFormat?.stride ?? 0)
        var kvValueBytes = UInt32(kvFormat?.valueBytes ?? 0)
        var kvGroupSize = UInt32(kvFormat?.groupSize ?? KVCacheManager.quantizationGroupSize)
        p1.setBytes(&kvBits, length: MemoryLayout<UInt32>.size, index: 14)
        p1.setBytes(&kvStride, length: MemoryLayout<UInt32>.size, index: 15)
        p1.setBytes(&kvValueBytes, length: MemoryLayout<UInt32>.size, index: 16)
        p1.setBytes(&kvGroupSize, length: MemoryLayout<UInt32>.size, index: 17)
        let partialGroups: Int
        switch choice {
        case .stream: partialGroups = Int(numKVHeads) * (nChunks / Self.streamCount)
        case .shared: partialGroups = Int(numKVHeads) * nChunks
        case .generic: partialGroups = geometry.partialThreadgroups
        }
        p1.dispatchThreadgroups(MTLSize(width: partialGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))

        let p2 = encoder
        let combinePSO: MTLComputePipelineState
        if sinks != nil {
            guard let sinkPSO = psoCombineSinks else {
                throw MetalError.invalidState(
                    "sinks bound to an Attention built without supportsSinks")
            }
            combinePSO = sinkPSO
        } else {
            combinePSO = combinePipeline(headDim: headDim,
                                         numQHeads: numQHeads,
                                         numKVHeads: numKVHeads,
                                         numChunks: nChunks)
        }
        p2.setComputePipelineState(combinePSO)
        p2.setBuffer(mPartial, offset: 0, index: 0)
        p2.setBuffer(dPartial, offset: 0, index: 1)
        p2.setBuffer(oPartial, offset: 0, index: 2)
        p2.setBuffer(out, offset: outOffset, index: 3)
        var hd2 = headDim, nc2 = UInt32(nChunks)
        p2.setBytes(&hd2, length: MemoryLayout<UInt32>.size, index: 4)
        p2.setBytes(&nc2, length: MemoryLayout<UInt32>.size, index: 5)
        if let sinks { p2.setBuffer(sinks, offset: sinksOffset, index: 6) }
        let combineTGWidth = min(Self.threadsPerGroup,
                                 Int(combinePSO.maxTotalThreadsPerThreadgroup))
        p2.dispatchThreadgroups(MTLSize(width: Int(numQHeads), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: combineTGWidth, height: 1, depth: 1))
    }

    /// `1 / sqrt(head_dim)` — the classic transformer scaling. Used as the
    /// default for callers without a configured attention scale (and for the
    /// existing tests that pre-date the runtime scale arg).
    static func defaultScale(headDim: UInt32) -> Float {
        Float(1.0) / Float(headDim).squareRoot()
    }

    private static func specializedPipeline(_ context: MetalContext,
                                            _ name: String,
                                            headDim: UInt32,
                                            numQHeads: UInt32,
                                            numKVHeads: UInt32,
                                            numChunks: UInt32? = nil,
                                            ringCapacity: UInt32? = nil,
                                            kvShapeKey: KVSharedShapeKey? = nil) throws -> MTLComputePipelineState {
        var constants = [
            MetalFunctionConstant(index: 60, value: .uint32(headDim)),
            MetalFunctionConstant(index: 61, value: .uint32(numQHeads)),
            MetalFunctionConstant(index: 62, value: .uint32(numKVHeads)),
            MetalFunctionConstant(index: 63, value: .bool(true)),
        ]
        if let numChunks {
            constants.append(MetalFunctionConstant(index: 65, value: .uint32(numChunks)))
        }
        if let ringCapacity {
            constants.append(MetalFunctionConstant(index: 69, value: .uint32(ringCapacity)))
        }
        if let kvShapeKey {
            constants.append(contentsOf: [
                MetalFunctionConstant(index: 96, value: .uint32(kvShapeKey.kvBits)),
                MetalFunctionConstant(index: 97, value: .uint32(kvShapeKey.kvStride)),
                MetalFunctionConstant(index: 98, value: .uint32(kvShapeKey.kvValueBytes)),
                MetalFunctionConstant(index: 99, value: .uint32(kvShapeKey.kvGroupSize)),
            ])
        }
        return try context.pipeline(name, constants: constants)
    }

    private func specializedPartialPipeline(named name: String,
                                            numChunks: UInt32?,
                                            headDim: UInt32,
                                            numQHeads: UInt32,
                                            numKVHeads: UInt32,
                                            kvFormat: KVView?) -> MTLComputePipelineState {
        let shape = KVSharedShapeKey(
            headDim: headDim,
            numQHeads: numQHeads,
            numKVHeads: numKVHeads,
            kvBits: UInt32(kvFormat?.precision.rawValue ?? 16),
            kvStride: UInt32(kvFormat?.stride ?? 0),
            kvValueBytes: UInt32(kvFormat?.valueBytes ?? 0),
            kvGroupSize: UInt32(kvFormat?.groupSize ?? KVCacheManager.quantizationGroupSize))
        let key = SpecializedPartialKey(name: name, shape: shape)
        if let cached = specializedPartialPSOs[key] { return cached }
        do {
            let pso = try Self.specializedPipeline(ctx, name,
                                                   headDim: headDim,
                                                   numQHeads: numQHeads,
                                                   numKVHeads: numKVHeads,
                                                   numChunks: numChunks,
                                                   kvShapeKey: shape)
            specializedPartialPSOs[key] = pso
            return pso
        } catch {
            preconditionFailure("failed to build the \(name) pipeline: \(error)")
        }
    }

    func partialPipeline(headDim: UInt32,
                         numQHeads: UInt32,
                         numKVHeads: UInt32,
                         numChunks: Int,
                         useGQAPartial: Bool,
                         ringCapacity: UInt32 = 0,
                         kvFormat: KVView? = nil) -> MTLComputePipelineState {
        if ringCapacity > 0 {
            let name = useGQAPartial ? "attention_decode_gqa_swa_partial" : "attention_decode_partial"
            let specializedChunks = numChunks == 16 ? Optional(UInt32(numChunks)) : nil
            do {
                return try Self.specializedPipeline(ctx,
                                                    name,
                                                    headDim: headDim,
                                                    numQHeads: numQHeads,
                                                    numKVHeads: numKVHeads,
                                                    numChunks: specializedChunks,
                                                    ringCapacity: ringCapacity)
            } catch {
                preconditionFailure("failed to build KV ring attention pipeline: \(error)")
            }
        }
        switch sharedPartialChoice(headDim: headDim, numQHeads: numQHeads,
                                   numKVHeads: numKVHeads, useGQAPartial: useGQAPartial,
                                   ringCapacity: ringCapacity, kvFormat: kvFormat) {
        case .stream:
            return specializedPartialPipeline(named: "attention_decode_partial_stream",
                                              numChunks: UInt32(Self.maxChunks),
                                              headDim: headDim, numQHeads: numQHeads,
                                              numKVHeads: numKVHeads, kvFormat: kvFormat)
        case .shared:
            guard specializesKVShared else { return psoPartialShared }
            return specializedPartialPipeline(named: "attention_decode_partial_shared",
                                              numChunks: nil,
                                              headDim: headDim, numQHeads: numQHeads,
                                              numKVHeads: numKVHeads, kvFormat: kvFormat)
        case .generic:
            break
        }
        if useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            if numChunks == 16 {
                return psoGQAPartialSWAChunks16
            }
            return psoGQAPartialSWA
        }
        if !useGQAPartial && headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return psoPartialSWA
        }
        if !useGQAPartial && headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            if numChunks == 16 {
                return psoPartialFullChunks16
            }
            return psoPartialFull
        }
        return useGQAPartial ? psoGQAPartial : psoPartial
    }

    private func combinePipeline(headDim: UInt32,
                                 numQHeads: UInt32,
                                 numKVHeads: UInt32,
                                 numChunks: Int) -> MTLComputePipelineState {
        if headDim == 256 && numQHeads == 16 && numKVHeads == 8 {
            return numChunks == 16 ? psoCombineSWAChunks16 : psoCombineSWA
        }
        if headDim == 512 && numQHeads == 16 && numKVHeads == 2 {
            return numChunks == 16 ? psoCombineFullChunks16 : psoCombineFull
        }
        return psoCombine
    }
}
