import Foundation
import Metal
import Shrike

extension ShrikeBench {
    /// The ornith GDN shape at one 4,096-row prefill chunk.
    private enum PreScanShape {
        static let kHeads = 16
        static let vHeads = 32
        static let headDim = 128
        static let rows = 4096
        static let hidden = 2048
        static let taps = 4
        static let normThreads = 128
        static let rmsEps: Float = 1e-6
        static var qkvDim: Int { 2 * kHeads * headDim + vHeads * headDim }
        static var valueDim: Int { vHeads * headDim }
    }

    private struct PreScanRNG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return state
        }
        mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
            lo + (hi - lo) * Float(next() >> 40) / Float(1 << 24)
        }
    }

    private struct PreScanFixture {
        let qkvRaw: MTLBuffer
        let convOut: MTLBuffer
        let convWeight: MTLBuffer
        let tail: MTLBuffer
        let y: MTLBuffer
        let z: MTLBuffer
        let gatedWeight: MTLBuffer
        let gatedOut: MTLBuffer
        let hidden: MTLBuffer
        let delta: MTLBuffer
        let normWeight: MTLBuffer
        let normed: MTLBuffer

        /// Private storage, like `PrefillChunkScratch`: a bandwidth-bound
        /// chain measured over shared buffers is measuring the wrong thing.
        init(context: MetalContext) {
            let device = context.device
            let T = PreScanShape.rows
            let C = PreScanShape.qkvDim
            let V = PreScanShape.valueDim
            let D = PreScanShape.hidden
            var rng = PreScanRNG(state: 0x51E1_7A93_C4B2_0D67)
            let cb = context.queue.makeCommandBuffer()!
            let blit = cb.makeBlitCommandEncoder()!
            func upload(_ values: [UInt16]) -> MTLBuffer {
                let bytes = values.count * 2
                let staging = device.makeBuffer(bytes: values, length: bytes,
                                                options: .storageModeShared)!
                let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate)!
                blit.copy(from: staging, sourceOffset: 0, to: buffer,
                          destinationOffset: 0, size: bytes)
                return buffer
            }
            func halves(_ count: Int, _ lo: Float, _ hi: Float) -> MTLBuffer {
                upload((0..<count).map { _ in Float16(rng.uniform(lo, hi)).bitPattern })
            }
            func bf16(_ count: Int, _ lo: Float, _ hi: Float) -> MTLBuffer {
                upload((0..<count).map { _ in Quantization.bf16Bits(rng.uniform(lo, hi)) })
            }
            func output(_ count: Int) -> MTLBuffer {
                device.makeBuffer(length: count * 2, options: .storageModePrivate)!
            }
            qkvRaw = halves(T * C, -1, 1)
            convOut = halves(T * C, -1, 1)
            convWeight = bf16(C * PreScanShape.taps, -0.5, 0.5)
            tail = halves((PreScanShape.taps - 1) * C, -1, 1)
            y = halves(T * V, -1, 1)
            z = halves(T * V, -1, 1)
            gatedWeight = bf16(PreScanShape.headDim, 0.5, 1.5)
            gatedOut = output(T * V)
            hidden = halves(T * D, -1, 1)
            delta = halves(T * D, -0.01, 0.01)
            normWeight = bf16(D, 0.5, 1.5)
            normed = output(T * D)
            blit.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
    }

    /// `launchesPerChunk` is how often the GDN role encodes this kernel per
    /// layer-chunk, which is what the chain total sums — not the launch count.
    private struct PreScanStep {
        let label: String
        let bytes: Double
        let launchesPerChunk: Int
        let encode: (MTLCommandBuffer) -> Void
    }

    private struct PreScanKernels {
        let fixture: PreScanFixture
        let conv: MTLComputePipelineState
        let convTail: MTLComputePipelineState
        let qkNorm: MTLComputePipelineState
        let gatedNorm: MTLComputePipelineState
        let rmsNorm: MTLComputePipelineState
        let residualAdd: MTLComputePipelineState

        init(fixture: PreScanFixture, context: MetalContext) throws {
            self.fixture = fixture
            let normConstant = [MetalFunctionConstant(
                index: 95, value: .uint32(UInt32(PreScanShape.normThreads)))]
            conv = try context.pipeline("gdn_conv_mix_prefill")
            convTail = try context.pipeline("gdn_conv_tail_update")
            qkNorm = try context.pipeline("gdn_qk_norm", constants: normConstant)
            gatedNorm = try context.pipeline("gdn_gated_norm", constants: normConstant)
            rmsNorm = try context.pipeline("prefill_rmsnorm_bf16w_block")
            residualAdd = try context.pipeline("residual_add_fp16")
        }

        private func encoder(_ cb: MTLCommandBuffer,
                             _ pipeline: MTLComputePipelineState) -> MTLComputeCommandEncoder {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            return enc
        }

        private func dispatch2D(_ enc: MTLComputeCommandEncoder,
                                _ pipeline: MTLComputePipelineState,
                                width: Int, height: Int) {
            let tgWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
            enc.endEncoding()
        }

        func encodeConv(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, conv)
            enc.setBuffer(fixture.tail, offset: 0, index: 0)
            enc.setBuffer(fixture.qkvRaw, offset: 0, index: 1)
            enc.setBuffer(fixture.convWeight, offset: 0, index: 2)
            enc.setBuffer(fixture.convOut, offset: 0, index: 3)
            var channels = UInt32(PreScanShape.qkvDim)
            var taps = UInt32(PreScanShape.taps)
            var rows = UInt32(PreScanShape.rows)
            enc.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 5)
            enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 6)
            dispatch2D(enc, conv, width: PreScanShape.qkvDim, height: PreScanShape.rows)
        }

        func encodeConvTail(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, convTail)
            enc.setBuffer(fixture.tail, offset: 0, index: 0)
            enc.setBuffer(fixture.qkvRaw, offset: 0, index: 1)
            var channels = UInt32(PreScanShape.qkvDim)
            var taps = UInt32(PreScanShape.taps)
            var rows = UInt32(PreScanShape.rows)
            enc.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&taps, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 4)
            dispatch2D(enc, convTail, width: PreScanShape.qkvDim,
                       height: PreScanShape.taps - 1)
        }

        func encodeQKNorm(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, qkNorm)
            enc.setBuffer(fixture.convOut, offset: 0, index: 0)
            var kHeads = UInt32(PreScanShape.kHeads)
            var keyDim = UInt32(PreScanShape.headDim)
            var rowStride = UInt32(PreScanShape.qkvDim)
            enc.setBytes(&kHeads, length: MemoryLayout<UInt32>.size, index: 1)
            enc.setBytes(&keyDim, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&rowStride, length: MemoryLayout<UInt32>.size, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: 2 * PreScanShape.kHeads, height: PreScanShape.rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: PreScanShape.normThreads, height: 1, depth: 1))
            enc.endEncoding()
        }

        func encodeGatedNorm(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, gatedNorm)
            enc.setBuffer(fixture.y, offset: 0, index: 0)
            enc.setBuffer(fixture.z, offset: 0, index: 1)
            enc.setBuffer(fixture.gatedWeight, offset: 0, index: 2)
            enc.setBuffer(fixture.gatedOut, offset: 0, index: 3)
            var vHeads = UInt32(PreScanShape.vHeads)
            var valueDim = UInt32(PreScanShape.headDim)
            enc.setBytes(&vHeads, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&valueDim, length: MemoryLayout<UInt32>.size, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: PreScanShape.vHeads, height: PreScanShape.rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: PreScanShape.normThreads, height: 1, depth: 1))
            enc.endEncoding()
        }

        func encodeRMSNorm(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, rmsNorm)
            enc.setBuffer(fixture.hidden, offset: 0, index: 0)
            enc.setBuffer(fixture.normWeight, offset: 0, index: 1)
            enc.setBuffer(fixture.normed, offset: 0, index: 2)
            var rows = UInt32(PreScanShape.rows)
            var dim = UInt32(PreScanShape.hidden)
            var eps = PreScanShape.rmsEps
            enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&eps, length: MemoryLayout<Float>.size, index: 5)
            let threads = min(rmsNorm.maxTotalThreadsPerThreadgroup, 256)
            enc.dispatchThreadgroups(
                MTLSize(width: PreScanShape.rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            enc.endEncoding()
        }

        func encodeResidualAdd(_ cb: MTLCommandBuffer) {
            let enc = encoder(cb, residualAdd)
            enc.setBuffer(fixture.hidden, offset: 0, index: 0)
            enc.setBuffer(fixture.delta, offset: 0, index: 1)
            var count = UInt32(PreScanShape.rows * PreScanShape.hidden)
            enc.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
            dispatch2D(enc, residualAdd,
                       width: PreScanShape.rows * PreScanShape.hidden, height: 1)
        }

        var steps: [PreScanStep] {
            let T = Double(PreScanShape.rows)
            let C = Double(PreScanShape.qkvDim)
            let V = Double(PreScanShape.valueDim)
            let D = Double(PreScanShape.hidden)
            let qk = 2.0 * Double(PreScanShape.kHeads * PreScanShape.headDim)
            return [
                PreScanStep(label: "gdn_conv_mix_prefill", bytes: 4 * T * C,
                            launchesPerChunk: 1, encode: encodeConv),
                PreScanStep(label: "gdn_conv_tail_update",
                            bytes: 4 * C * Double(PreScanShape.taps - 1),
                            launchesPerChunk: 1, encode: encodeConvTail),
                PreScanStep(label: "gdn_qk_norm", bytes: 4 * T * qk,
                            launchesPerChunk: 1, encode: encodeQKNorm),
                PreScanStep(label: "gdn_gated_norm", bytes: 6 * T * V,
                            launchesPerChunk: 1, encode: encodeGatedNorm),
                PreScanStep(label: "prefill_rmsnorm_bf16w_block", bytes: 4 * T * D,
                            launchesPerChunk: 2, encode: encodeRMSNorm),
                PreScanStep(label: "residual_add_fp16", bytes: 6 * T * D,
                            launchesPerChunk: 1, encode: encodeResidualAdd),
            ]
        }
    }

    /// `gdn_pre`: the GDN pre-scan chain at the ornith 4,096-row chunk, each
    /// kernel alone in its own command buffer — ms, bytes and GB/s per kernel,
    /// then the chain total against `residual_add_fp16`'s streaming floor.
    static func runGDNPreScan(iterations: Int, context: MetalContext) throws {
        let fixture = PreScanFixture(context: context)
        let kernels = try PreScanKernels(fixture: fixture, context: context)
        let runs = max(1, iterations)

        func time(_ encode: (MTLCommandBuffer) -> Void) -> Double {
            func once() -> Double {
                let cb = context.queue.makeCommandBuffer()!
                encode(cb)
                cb.commit()
                cb.waitUntilCompleted()
                precondition(cb.status == .completed,
                             "command buffer failed: \(cb.status.rawValue)")
                return cb.gpuEndTime - cb.gpuStartTime
            }
            for _ in 0..<2 { _ = once() }
            var total = 0.0
            for _ in 0..<runs { total += once() }
            return total / Double(runs)
        }

        print("gdn_pre: T=\(PreScanShape.rows) Hk=\(PreScanShape.kHeads) "
              + "Hv=\(PreScanShape.vHeads) Dk=Dv=\(PreScanShape.headDim) "
              + "C=\(PreScanShape.qkvDim) D=\(PreScanShape.hidden) "
              + "taps=\(PreScanShape.taps), \(runs) iterations per kernel")
        var chainMillis = 0.0
        var chainBytes = 0.0
        var floorGBs = 0.0
        var bestGBs = 0.0
        for step in kernels.steps {
            let seconds = time(step.encode)
            let gbs = step.bytes / seconds / 1.0e9
            if step.label == "residual_add_fp16" { floorGBs = gbs }
            if step.bytes > 1.0e6 { bestGBs = max(bestGBs, gbs) }
            chainMillis += seconds * 1e3 * Double(step.launchesPerChunk)
            chainBytes += step.bytes * Double(step.launchesPerChunk)
            print("kernel=\(step.label) per_launch_ms=\(String(format: "%.4f", seconds * 1e3)) "
                  + "bytes_mb=\(String(format: "%.2f", step.bytes / 1.0e6)) "
                  + "achieved_gb_s=\(String(format: "%.2f", gbs)) "
                  + "launches_per_chunk=\(step.launchesPerChunk) "
                  + "chain_ms=\(String(format: "%.4f", seconds * 1e3 * Double(step.launchesPerChunk)))")
        }
        print("gdn_pre chain: total_ms=\(String(format: "%.4f", chainMillis)) "
              + "total_mb=\(String(format: "%.2f", chainBytes / 1.0e6)) "
              + "achieved_gb_s=\(String(format: "%.2f", chainBytes / (chainMillis / 1e3) / 1.0e9)) "
              + "streaming_floor_gb_s=\(String(format: "%.2f", floorGBs)) "
              + "floor_ms=\(String(format: "%.4f", chainBytes / (floorGBs * 1.0e9) * 1e3)) "
              + "best_gb_s=\(String(format: "%.2f", bestGBs)) "
              + "best_floor_ms=\(String(format: "%.4f", chainBytes / (bestGBs * 1.0e9) * 1e3))")
    }
}
