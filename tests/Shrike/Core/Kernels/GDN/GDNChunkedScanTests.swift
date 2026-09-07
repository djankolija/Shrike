import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

/// The chunked gated delta rule (v12 P4) against the serial recurrence, at the
/// shape the kernels are compiled for (Dk = Dv = 128) with the smallest head
/// count that still exercises the Hv/Hk grouping.
@Suite struct GDNChunkedScanTests {
    static let cfg = LinearAttentionConfig(
        numKHeads: 1, numVHeads: 2, keyHeadDim: 128, valueHeadDim: 128,
        convKernelSize: 4)

    private static func bf16Value(_ x: Float) -> Float {
        Float(bitPattern: UInt32(UInt16(truncatingIfNeeded: x.bitPattern >> 16)) << 16)
    }

    private static func fp16(_ x: Float) -> Float { Float(Float16(x)) }

    private static func bf16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        return device.makeBuffer(bytes: bits, length: bits.count * 2,
                                 options: .storageModeShared)
    }

    private static func floatBuffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        values.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                              options: .storageModeShared)
        }
    }

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { ptr[$0] }
    }

    private static func readHalves(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    /// q and k carry the folded norm scales (q at `1/√Dk`, k at 1) so the
    /// magnitudes match what `GDNReference.normalize` produces.
    struct Fixture {
        let aLog: [Float]
        let dtBias: [Float]
        let normed: [[Float]]
        let a: [[Float]]
        let b: [[Float]]
        let state: [Float]

        init(rows: Int, seed: UInt64, cfg: LinearAttentionConfig) {
            var rng = SeedTree(seed).key("gdn-chunked-\(rows)")
            let Hk = cfg.numKHeads
            let Hv = cfg.numVHeads
            let Dk = cfg.keyHeadDim
            let Dv = cfg.valueHeadDim
            self.aLog = (0..<Hv).map { _ in GDNChunkedScanTests.bf16Value(rng.uniform(-1.0, 1.5)) }
            self.dtBias = (0..<Hv).map { _ in GDNChunkedScanTests.bf16Value(rng.uniform(-0.5, 0.5)) }

            func unit(_ scale: Float) -> [Float] {
                let raw = (0..<Dk).map { _ in rng.uniform(-1.0, 1.0) }
                let norm = sqrtf(raw.reduce(0) { $0 + $1 * $1 })
                return raw.map { GDNChunkedScanTests.fp16($0 / norm * scale) }
            }
            self.normed = (0..<rows).map { _ in
                var row: [Float] = []
                for _ in 0..<Hk { row += unit(1 / sqrtf(Float(Dk))) }
                for _ in 0..<Hk { row += unit(1) }
                for _ in 0..<(Hv * Dv) { row.append(GDNChunkedScanTests.fp16(rng.uniform(-1.0, 1.0))) }
                return row
            }
            self.a = (0..<rows).map { _ in
                (0..<Hv).map { _ in GDNChunkedScanTests.fp16(rng.uniform(-1.0, 1.0)) }
            }
            self.b = (0..<rows).map { _ in
                (0..<Hv).map { _ in GDNChunkedScanTests.fp16(rng.uniform(-1.0, 1.0)) }
            }
            self.state = (0..<(Hv * Dv * Dk)).map { _ in rng.uniform(-0.5, 0.5) }
        }
    }

    private static func serial(_ fixture: Fixture, cfg: LinearAttentionConfig)
        -> (y: [[Float]], checkpoint: [Float], state: [Float]) {
        var reference = GDNReference(
            cfg: cfg,
            convW: [Float](repeating: 0, count: cfg.qkvDim * cfg.convKernelSize),
            aLog: fixture.aLog, dtBias: fixture.dtBias,
            normW: [Float](repeating: 1, count: cfg.valueHeadDim))
        reference.state = fixture.state
        var y: [[Float]] = []
        var checkpoint: [Float] = []
        for t in 0..<fixture.normed.count {
            y.append(reference.deltaRule(normed: fixture.normed[t],
                                         a: fixture.a[t], b: fixture.b[t]))
            if t == 0 { checkpoint = reference.state }
        }
        return (y, checkpoint, reference.state)
    }

    @Test func chunkedReferenceMatchesSerialReference() {
        let cfg = Self.cfg
        for rows in [64, 200] {
            let fixture = Fixture(rows: rows, seed: 0x5EED, cfg: cfg)
            let want = Self.serial(fixture, cfg: cfg)
            let chunked = GDNChunkedReference(cfg: cfg, aLog: fixture.aLog,
                                              dtBias: fixture.dtBias)
            var state = fixture.state
            let got = chunked.run(normed: fixture.normed, a: fixture.a,
                                  b: fixture.b, state: &state)
            for t in 0..<rows {
                let scale = want.y[t].reduce(0) { max($0, abs($1)) }
                let diff = RelError.maxAbsDiff(got.y[t], want.y[t])
                #expect(diff <= 1e-3 + 1e-3 * scale,
                        "rows \(rows) row \(t): y maxAbs \(diff) (scale \(scale))")
            }
            let stateRel = RelError.compute(actual: state, reference: want.state)
            #expect(stateRel <= 1e-3, "rows \(rows): state rel \(stateRel)")
            let checkpointRel = RelError.compute(actual: got.checkpoint,
                                                 reference: want.checkpoint)
            #expect(checkpointRel <= 1e-4, "rows \(rows): checkpoint rel \(checkpointRel)")
        }
    }

    // MARK: - Kernels

    private struct KernelRun {
        let y: [Float]
        let state: [Float]
    }

    /// Runs either delta-step kernel on the fixture's normed rows; `convOut`
    /// is allocated at the 64-row multiple the chunked path requires.
    private static func runKernel(chunked: Bool, fixture: Fixture,
                                  ctx: MetalContext, gdn: GDN) throws -> KernelRun {
        let cfg = gdn.config
        let rows = fixture.normed.count
        let paddedRows = (rows + GDN.chunkTokens - 1) / GDN.chunkTokens * GDN.chunkTokens
        let C = cfg.qkvDim
        var convHalves = fixture.normed.flatMap { $0.map { Float16($0) } }
        convHalves += [Float16](repeating: Float16.nan, count: (paddedRows - rows) * C)
        let device = ctx.device
        guard let convOut = Fp16Buffer.make(device, halves: convHalves),
              let aProj = Fp16Buffer.make(device, halves: fixture.a.flatMap { $0.map { Float16($0) } }),
              let bProj = Fp16Buffer.make(device, halves: fixture.b.flatMap { $0.map { Float16($0) } }),
              let aLog = bf16Buffer(device, fixture.aLog),
              let dtBias = bf16Buffer(device, fixture.dtBias),
              let state = floatBuffer(device, fixture.state),
              let y = Fp16Buffer.make(device, count: rows * cfg.valueDim),
              let factors = device.makeBuffer(
                length: GDN.chunkFactorsBytes(config: cfg, prefillChunkTokens: rows),
                options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.noDevice
        }
        if chunked {
            try gdn.encodeDeltaStepPrefillChunked(
                commandBuffer: cb, convOut: convOut, aProj: aProj, bProj: bProj,
                aLog: aLog, aLogOffset: 0, dtBias: dtBias, dtBiasOffset: 0,
                state: state, y: y,
                rows: rows, factors: factors)
        } else {
            try gdn.encodeDeltaStepPrefill(
                commandBuffer: cb, convOut: convOut, aProj: aProj, bProj: bProj,
                aLog: aLog, aLogOffset: 0, dtBias: dtBias, dtBiasOffset: 0,
                state: state, y: y, rows: rows)
        }
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed, "command buffer status \(cb.status.rawValue)")
        return KernelRun(y: readHalves(y, count: rows * cfg.valueDim),
                         state: readFloats(state, count: fixture.state.count))
    }

    @Test func chunkedKernelMatchesSerialKernel() throws {
        let cfg = Self.cfg
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)
        try #require(gdn.chunkedScanAvailable,
                     "chunked scan unavailable: \(gdn.chunkedScanUnavailableReason ?? "")")
        for rows in [64, 200, 4096] {
            let fixture = Fixture(rows: rows, seed: 0xC4A7, cfg: cfg)
            let want = try Self.runKernel(chunked: false, fixture: fixture, ctx: ctx, gdn: gdn)
            let got = try Self.runKernel(chunked: true, fixture: fixture, ctx: ctx, gdn: gdn)
            #expect(got.y.allSatisfy { $0.isFinite }, "rows \(rows): non-finite y")
            #expect(got.state.allSatisfy { $0.isFinite }, "rows \(rows): non-finite state")
            let yAbs = RelError.maxAbsDiff(got.y, want.y)
            let yRel = RelError.compute(actual: got.y, reference: want.y)
            #expect(yAbs <= 2e-2 && yRel <= 2e-2,
                    "rows \(rows): y maxAbs \(yAbs) rel \(yRel)")
            let stateRel = RelError.compute(actual: got.state, reference: want.state)
            #expect(stateRel <= 2e-2, "rows \(rows): state rel \(stateRel)")
        }
    }

    /// Raw projection rows through the whole chain — conv, tail carry, q/k
    /// norm, the chunked scan, gated norm — against `GDNReference.step`.
    @Test func chunkedKernelMatchesReference() throws {
        let cfg = Self.cfg
        let rows = 200
        let ctx = try MetalContext()
        let gdn = try GDN(context: ctx, config: cfg)
        try #require(gdn.chunkedScanAvailable,
                     "chunked scan unavailable: \(gdn.chunkedScanUnavailableReason ?? "")")
        var rng = SeedTree(0xF00D).key("gdn-chunked-chain")
        let convW = (0..<(cfg.qkvDim * cfg.convKernelSize)).map { _ in
            Self.bf16Value(rng.uniform(-0.4, 0.4))
        }
        let aLog = (0..<cfg.numVHeads).map { _ in Self.bf16Value(rng.uniform(-1.0, 1.5)) }
        let dtBias = (0..<cfg.numVHeads).map { _ in Self.bf16Value(rng.uniform(-0.5, 0.5)) }
        let normW = (0..<cfg.valueHeadDim).map { _ in Self.bf16Value(rng.uniform(0.5, 1.5)) }
        let qkvRows = (0..<rows).map { _ in (0..<cfg.qkvDim).map { _ in Self.fp16(rng.uniform(-1.0, 1.0)) } }
        let aRows = (0..<rows).map { _ in (0..<cfg.numVHeads).map { _ in Self.fp16(rng.uniform(-1.0, 1.0)) } }
        let bRows = (0..<rows).map { _ in (0..<cfg.numVHeads).map { _ in Self.fp16(rng.uniform(-1.0, 1.0)) } }
        let zRows = (0..<rows).map { _ in (0..<cfg.valueDim).map { _ in Self.fp16(rng.uniform(-1.0, 1.0)) } }
        let state0 = (0..<(cfg.numVHeads * cfg.valueHeadDim * cfg.keyHeadDim)).map { _ in
            rng.uniform(-0.5, 0.5)
        }

        var reference = GDNReference(cfg: cfg, convW: convW, aLog: aLog,
                                     dtBias: dtBias, normW: normW)
        reference.state = state0
        let want = (0..<rows).map { t in
            reference.step(qkvRaw: qkvRows[t], a: aRows[t], b: bRows[t], z: zRows[t])
        }

        let paddedRows = (rows + GDN.chunkTokens - 1) / GDN.chunkTokens * GDN.chunkTokens
        let tailBytes = (cfg.convKernelSize - 1) * cfg.qkvDim * 2
        let device = ctx.device
        guard let tail = device.makeBuffer(length: tailBytes, options: .storageModeShared),
              let qkv = Fp16Buffer.make(device, halves: qkvRows.flatMap { $0.map { Float16($0) } }),
              let aProj = Fp16Buffer.make(device, halves: aRows.flatMap { $0.map { Float16($0) } }),
              let bProj = Fp16Buffer.make(device, halves: bRows.flatMap { $0.map { Float16($0) } }),
              let z = Fp16Buffer.make(device, halves: zRows.flatMap { $0.map { Float16($0) } }),
              let convWBuf = Self.bf16Buffer(device, convW),
              let aLogBuf = Self.bf16Buffer(device, aLog),
              let dtBiasBuf = Self.bf16Buffer(device, dtBias),
              let normWBuf = Self.bf16Buffer(device, normW),
              let stateBuf = Self.floatBuffer(device, state0),
              let convOut = Fp16Buffer.make(device, count: paddedRows * cfg.qkvDim),
              let y = Fp16Buffer.make(device, count: rows * cfg.valueDim),
              let out = Fp16Buffer.make(device, count: rows * cfg.valueDim),
              let factors = device.makeBuffer(
                length: GDN.chunkFactorsBytes(config: cfg, prefillChunkTokens: rows),
                options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer() else {
            throw MetalError.noDevice
        }
        memset(tail.contents(), 0, tailBytes)
        try gdn.encodeConvPrefill(commandBuffer: cb, tail: tail, qkvRows: qkv,
                                  convWeight: convWBuf, convWeightOffset: 0,
                                  out: convOut, rows: rows)
        try gdn.encodeConvTailUpdate(commandBuffer: cb, tail: tail, qkvRows: qkv, rows: rows)
        try gdn.encodeQKNorm(commandBuffer: cb, convOut: convOut, rows: rows)
        try gdn.encodeDeltaStepPrefillChunked(
            commandBuffer: cb, convOut: convOut, aProj: aProj, bProj: bProj,
            aLog: aLogBuf, aLogOffset: 0, dtBias: dtBiasBuf, dtBiasOffset: 0,
            state: stateBuf, y: y, rows: rows, factors: factors)
        try gdn.encodeGatedNorm(commandBuffer: cb, y: y, z: z,
                                weight: normWBuf, weightOffset: 0, out: out, rows: rows)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed, "command buffer status \(cb.status.rawValue)")

        let got = Self.readHalves(out, count: rows * cfg.valueDim)
        let gotState = Self.readFloats(stateBuf, count: state0.count)
        #expect(got.allSatisfy { $0.isFinite }, "non-finite gated output")
        #expect(gotState.allSatisfy { $0.isFinite }, "non-finite state")
        var worst: Float = 0
        for t in 0..<rows {
            for i in 0..<cfg.valueDim {
                let wanted = want[t][i]
                let diff = abs(got[t * cfg.valueDim + i] - wanted)
                let tolerance = max(2e-2, abs(wanted) * 4e-2)
                worst = max(worst, diff / tolerance)
                #expect(diff <= tolerance,
                        "row \(t) element \(i): chunked \(got[t * cfg.valueDim + i]), reference \(wanted)")
            }
        }
        #expect(worst <= 1, "worst diff/tolerance \(worst)")
        let stateRel = RelError.compute(actual: gotState, reference: reference.state)
        #expect(stateRel <= 2e-2, "state rel \(stateRel)")
    }

    @Test func chunkedScanShapeGate() throws {
        #expect(GDN.chunkedScanSupports(config: Self.cfg, perChannelDecay: false))
        #expect(!GDN.chunkedScanSupports(config: Self.cfg, perChannelDecay: true))
        let small = LinearAttentionConfig(numKHeads: 2, numVHeads: 4, keyHeadDim: 32,
                                          valueHeadDim: 32, convKernelSize: 4)
        #expect(!GDN.chunkedScanSupports(config: small, perChannelDecay: false))
        let mixed = LinearAttentionConfig(numKHeads: 1, numVHeads: 2, keyHeadDim: 128,
                                          valueHeadDim: 64, convKernelSize: 4)
        #expect(!GDN.chunkedScanSupports(config: mixed, perChannelDecay: false))
        #expect(GDN.chunkFactorsBytes(config: Self.cfg, prefillChunkTokens: 200) == 2 * 4 * 17_408)

        let ctx = try MetalContext()
        let gdnSmall = try GDN(context: ctx, config: small)
        #expect(!gdnSmall.chunkedScanAvailable)
        #expect(gdnSmall.chunkedScanUnavailableReason != nil)

        let gdn = try GDN(context: ctx, config: Self.cfg)
        try #require(gdn.chunkedScanAvailable,
                     "chunked scan unavailable: \(gdn.chunkedScanUnavailableReason ?? "")")
        let fixture = Fixture(rows: 63, seed: 0x63, cfg: Self.cfg)
        #expect(throws: GDNChunkedScanError.tooFewRows(63)) {
            _ = try Self.runKernel(chunked: true, fixture: fixture, ctx: ctx, gdn: gdn)
        }
    }
}
