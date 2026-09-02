import Foundation
import Metal
import Shrike

extension ShrikeBench {
    /// Mirror of `GDNChunkParams` in `gdn_chunked.metal`.
    private struct ScanParams {
        var kHeads: UInt32
        var vHeads: UInt32
        var keyDim: UInt32
        var valueDim: UInt32
        var rows: UInt32
        var rowStride: UInt32
        var chunkCount: UInt32
        var checkpointEnabled: UInt32
    }

    private struct XorShift {
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

    /// The ornith GDN shape at one 4,096-row chunk, with q/k rows carrying
    /// the folded norm scales like the real conv output.
    private struct ScanFixture {
        static let kHeads = 16
        static let vHeads = 32
        static let headDim = 128
        static let rows = 4096
        static var rowStride: Int { 2 * kHeads * headDim + vHeads * headDim }
        static var chunkCount: Int { rows / 64 }
        static var stateCount: Int { vHeads * headDim * headDim }

        let convOut: MTLBuffer
        let aProj: MTLBuffer
        let bProj: MTLBuffer
        let aLog: MTLBuffer
        let dtBias: MTLBuffer
        let factors: MTLBuffer
        let state0: [Float]

        init(device: MTLDevice) {
            let Hk = Self.kHeads, Hv = Self.vHeads, D = Self.headDim
            let rows = Self.rows, C = Self.rowStride
            func makeBuffer(_ bytes: Int) -> MTLBuffer {
                device.makeBuffer(length: bytes, options: .storageModeShared)!
            }
            var rng = XorShift(state: 0x6D5C_0FF1_CE12_3457)
            convOut = makeBuffer(rows * C * 2)
            let conv = convOut.contents().bindMemory(to: Float16.self, capacity: rows * C)
            var raw = [Float](repeating: 0, count: D)
            for r in 0..<rows {
                for head in 0..<(2 * Hk) {
                    var sumsq: Float = 0
                    for i in 0..<D {
                        raw[i] = rng.uniform(-1, 1)
                        sumsq += raw[i] * raw[i]
                    }
                    let scale = (head < Hk ? 1 / sqrtf(Float(D)) : 1) / sqrtf(sumsq)
                    for i in 0..<D { conv[r * C + head * D + i] = Float16(raw[i] * scale) }
                }
                for i in 0..<(Hv * D) {
                    conv[r * C + 2 * Hk * D + i] = Float16(rng.uniform(-1, 1))
                }
            }
            aProj = makeBuffer(rows * Hv * 2)
            bProj = makeBuffer(rows * Hv * 2)
            let aPtr = aProj.contents().bindMemory(to: Float16.self, capacity: rows * Hv)
            let bPtr = bProj.contents().bindMemory(to: Float16.self, capacity: rows * Hv)
            for i in 0..<(rows * Hv) {
                aPtr[i] = Float16(rng.uniform(-1, 1))
                bPtr[i] = Float16(rng.uniform(-1, 1))
            }
            aLog = makeBuffer(Hv * 2)
            dtBias = makeBuffer(Hv * 2)
            let aLogPtr = aLog.contents().bindMemory(to: UInt16.self, capacity: Hv)
            let dtBiasPtr = dtBias.contents().bindMemory(to: UInt16.self, capacity: Hv)
            for h in 0..<Hv {
                aLogPtr[h] = UInt16(truncatingIfNeeded: rng.uniform(-1, 1.5).bitPattern >> 16)
                dtBiasPtr[h] = UInt16(truncatingIfNeeded: rng.uniform(-0.5, 0.5).bitPattern >> 16)
            }
            state0 = (0..<Self.stateCount).map { _ in rng.uniform(-0.5, 0.5) }
            factors = makeBuffer(Hv * Self.chunkCount * 17_408)
        }

        func reset(_ state: MTLBuffer) {
            state0.withUnsafeBytes { bytes in
                _ = memcpy(state.contents(), bytes.baseAddress!, bytes.count)
            }
        }
    }

    private struct ScanKernels {
        let fixture: ScanFixture
        let serialPSO: MTLComputePipelineState
        let factorsPSO: MTLComputePipelineState
        let scanPSO: MTLComputePipelineState

        func encodeSerial(_ cb: MTLCommandBuffer, state: MTLBuffer, y: MTLBuffer) {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(serialPSO)
            enc.setBuffer(fixture.convOut, offset: 0, index: 0)
            enc.setBuffer(fixture.aProj, offset: 0, index: 1)
            enc.setBuffer(fixture.bProj, offset: 0, index: 2)
            enc.setBuffer(fixture.aLog, offset: 0, index: 3)
            enc.setBuffer(fixture.dtBias, offset: 0, index: 4)
            enc.setBuffer(state, offset: 0, index: 5)
            enc.setBuffer(y, offset: 0, index: 6)
            var dims: [UInt32] = [UInt32(ScanFixture.kHeads), UInt32(ScanFixture.vHeads),
                                  UInt32(ScanFixture.headDim), UInt32(ScanFixture.headDim),
                                  UInt32(ScanFixture.rows), UInt32(ScanFixture.rowStride)]
            for i in 0..<dims.count {
                enc.setBytes(&dims[i], length: MemoryLayout<UInt32>.size, index: 7 + i)
            }
            enc.setBuffer(state, offset: 0, index: 13)
            var checkpointEnabled = false
            enc.setBytes(&checkpointEnabled, length: MemoryLayout<Bool>.size, index: 14)
            enc.dispatchThreadgroups(
                MTLSize(width: ScanFixture.vHeads, height: ScanFixture.headDim / 4, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
            enc.endEncoding()
        }

        func encodeChunked(_ cb: MTLCommandBuffer, state: MTLBuffer, y: MTLBuffer) {
            var params = ScanParams(
                kHeads: UInt32(ScanFixture.kHeads), vHeads: UInt32(ScanFixture.vHeads),
                keyDim: UInt32(ScanFixture.headDim), valueDim: UInt32(ScanFixture.headDim),
                rows: UInt32(ScanFixture.rows), rowStride: UInt32(ScanFixture.rowStride),
                chunkCount: UInt32(ScanFixture.chunkCount), checkpointEnabled: 0)
            let enc = cb.makeComputeCommandEncoder()!
            let threads = MTLSize(width: 128, height: 1, depth: 1)
            enc.setComputePipelineState(factorsPSO)
            enc.setBuffer(fixture.convOut, offset: 0, index: 0)
            enc.setBuffer(fixture.aProj, offset: 0, index: 1)
            enc.setBuffer(fixture.bProj, offset: 0, index: 2)
            enc.setBuffer(fixture.aLog, offset: 0, index: 3)
            enc.setBuffer(fixture.dtBias, offset: 0, index: 4)
            enc.setBuffer(fixture.factors, offset: 0, index: 5)
            enc.setBytes(&params, length: MemoryLayout<ScanParams>.stride, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: ScanFixture.chunkCount, height: ScanFixture.vHeads, depth: 1),
                threadsPerThreadgroup: threads)
            enc.setComputePipelineState(scanPSO)
            enc.setBuffer(fixture.convOut, offset: 0, index: 0)
            enc.setBuffer(fixture.factors, offset: 0, index: 1)
            enc.setBuffer(state, offset: 0, index: 2)
            enc.setBuffer(y, offset: 0, index: 3)
            enc.setBuffer(state, offset: 0, index: 4)
            enc.setBytes(&params, length: MemoryLayout<ScanParams>.stride, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: ScanFixture.headDim / 32, height: ScanFixture.vHeads, depth: 1),
                threadsPerThreadgroup: threads)
            enc.endEncoding()
        }
    }

    private static func maxAbsAndScale(halves got: MTLBuffer, want: MTLBuffer,
                                       count: Int) -> (maxAbs: Float, scale: Float) {
        let g = got.contents().bindMemory(to: Float16.self, capacity: count)
        let w = want.contents().bindMemory(to: Float16.self, capacity: count)
        var maxAbs: Float = 0
        var scale: Float = 0
        for i in 0..<count {
            let wanted = Float(w[i])
            maxAbs = max(maxAbs, abs(Float(g[i]) - wanted))
            scale = max(scale, abs(wanted))
        }
        return (maxAbs, scale)
    }

    private static func maxAbsAndScale(floats got: MTLBuffer, want: MTLBuffer,
                                       count: Int) -> (maxAbs: Float, scale: Float) {
        let g = got.contents().bindMemory(to: Float.self, capacity: count)
        let w = want.contents().bindMemory(to: Float.self, capacity: count)
        var maxAbs: Float = 0
        var scale: Float = 0
        for i in 0..<count {
            maxAbs = max(maxAbs, abs(g[i] - w[i]))
            scale = max(scale, abs(w[i]))
        }
        return (maxAbs, scale)
    }

    /// `gdn_scan`: the prefill delta-rule scan at the ornith shape — the
    /// serial kernel against the chunked pair, one layer call each. Prints ms
    /// per call, µs per token, TFLOPS, and the chunked-vs-serial error from
    /// one call on the same inputs and state.
    static func runGDNScan(iterations: Int, context: MetalContext) throws {
        let serialPSO = try context.pipeline("gdn_delta_step_prefill")
        let factorsPSO: MTLComputePipelineState
        let scanPSO: MTLComputePipelineState
        do {
            factorsPSO = try context.pipeline("gdn_chunk_factors")
            scanPSO = try context.pipeline("gdn_chunk_scan")
        } catch {
            print("gdn_scan: chunked kernels unavailable: \(error)")
            return
        }
        let device = context.device
        let fixture = ScanFixture(device: device)
        let kernels = ScanKernels(fixture: fixture, serialPSO: serialPSO,
                                  factorsPSO: factorsPSO, scanPSO: scanPSO)
        let rows = ScanFixture.rows
        let yCount = rows * ScanFixture.vHeads * ScanFixture.headDim
        let stateSerial = device.makeBuffer(length: ScanFixture.stateCount * 4, options: .storageModeShared)!
        let stateChunked = device.makeBuffer(length: ScanFixture.stateCount * 4, options: .storageModeShared)!
        let ySerial = device.makeBuffer(length: yCount * 2, options: .storageModeShared)!
        let yChunked = device.makeBuffer(length: yCount * 2, options: .storageModeShared)!

        func run(_ encode: (MTLCommandBuffer) -> Void) -> Double {
            let cb = context.queue.makeCommandBuffer()!
            encode(cb)
            cb.commit()
            cb.waitUntilCompleted()
            precondition(cb.status == .completed, "command buffer failed: \(cb.status.rawValue)")
            return cb.gpuEndTime - cb.gpuStartTime
        }
        func time(_ encode: (MTLCommandBuffer) -> Void) -> Double {
            for _ in 0..<2 { _ = run(encode) }
            var total = 0.0
            for _ in 0..<max(1, iterations) { total += run(encode) }
            return total / Double(max(1, iterations))
        }

        fixture.reset(stateSerial)
        fixture.reset(stateChunked)
        _ = run { kernels.encodeSerial($0, state: stateSerial, y: ySerial) }
        _ = run { kernels.encodeChunked($0, state: stateChunked, y: yChunked) }
        let yError = maxAbsAndScale(halves: yChunked, want: ySerial, count: yCount)
        let stateError = maxAbsAndScale(floats: stateChunked, want: stateSerial,
                                        count: ScanFixture.stateCount)

        let serialSeconds = time { kernels.encodeSerial($0, state: stateSerial, y: ySerial) }
        let chunkedSeconds = time { kernels.encodeChunked($0, state: stateChunked, y: yChunked) }

        let D = Double(ScanFixture.headDim)
        let serialFlop = 2.0 * Double(rows * ScanFixture.vHeads) * 4.0 * D * D
        let perChunkMACs = 2.0 * 64 * 64 * D
            + Double(ScanFixture.headDim / 32) * (2.0 * 64 * 32 * D + 2.0 * 64 * 32 * 64 + 32.0 * D * 64)
        let chunkedFlop = 2.0 * Double(ScanFixture.vHeads * ScanFixture.chunkCount) * perChunkMACs

        print("gdn_scan: T=\(rows) Hv=\(ScanFixture.vHeads) Hk=\(ScanFixture.kHeads) "
              + "Dk=Dv=\(ScanFixture.headDim), \(iterations) iterations per variant")
        print(String(format: "serial : %8.3f ms/call  %6.2f µs/token  %5.2f GFLOP  %5.3f TFLOPS",
                     serialSeconds * 1e3, serialSeconds * 1e6 / Double(rows),
                     serialFlop / 1e9, serialFlop / serialSeconds / 1e12))
        print(String(format: "chunked: %8.3f ms/call  %6.2f µs/token  %5.2f GFLOP  %5.3f TFLOPS  speedup %.2fx",
                     chunkedSeconds * 1e3, chunkedSeconds * 1e6 / Double(rows),
                     chunkedFlop / 1e9, chunkedFlop / chunkedSeconds / 1e12,
                     serialSeconds / chunkedSeconds))
        print(String(format: "chunked vs serial: y maxAbs %.3e rel %.3e | state maxAbs %.3e rel %.3e",
                     yError.maxAbs, yError.maxAbs / max(yError.scale, 1e-6),
                     stateError.maxAbs, stateError.maxAbs / max(stateError.scale, 1e-6)))
    }
}
