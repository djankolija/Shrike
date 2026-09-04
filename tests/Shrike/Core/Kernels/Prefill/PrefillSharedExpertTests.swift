import Foundation
import Metal
import Testing
@testable import Shrike
import ShrikeValidationSupport

@Suite struct PrefillSharedExpertTests {
    private static let rows = 4
    private static let d = 128
    private static let f = 64
    private static let xStride = 144
    private static let yStride = 137
    private static let sentinel = Float16(-7.25)
    private static let chunkD = 2048
    private static let chunkF = 512

    @Test func blockSharedExpertMatchesRepeatedScalarRows() throws {
        try Self.runBlockSharedExpertMatchesRepeatedScalarRows()
    }

    @Test func sharedExpertPhaseSplitMatchesCurrentPrefillBlock() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-phase-split")
        let ctx = try MetalContext()
        let shared = try SharedExpertInt8(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)

        let x = Self.makeInputBlock(rng: &rng)
        let gate = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeWeights(rows: Self.d, cols: Self.f, rng: &rng)
        let yElements = Self.rows * Self.d

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let ySplit = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: Self.rows * Self.f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = Self.makeProjection(ctx: ctx, packed: gate, rows: Self.f, cols: Self.d)
        let upProj = Self.makeProjection(ctx: ctx, packed: up, rows: Self.f, cols: Self.d)
        let downProj = Self.makeProjection(ctx: ctx, packed: down, rows: Self.d, cols: Self.f)
        let halfBytes = MemoryLayout<Float16>.stride

        let refCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: refCB,
                                x: xBuf,
                                y: yRef,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: Self.rows,
                                d: Self.d,
                                intermediate: Self.f,
                                xStrideElements: Self.xStride,
                                yStrideElements: Self.d)
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let splitCB = ctx.queue.makeCommandBuffer()!
        let splitEncoder = splitCB.makeComputeCommandEncoder()!
        for row in 0..<Self.rows {
            try shared.encodePhase1(encoder: splitEncoder,
                                    x: xBuf,
                                    xOffset: row * Self.xStride * halfBytes,
                                    gate: gateProj,
                                    up: upProj,
                                    scratchAct: scratchAct,
                                    scratchActOffset: row * Self.f * halfBytes)
        }
        for row in 0..<Self.rows {
            try shared.encodeDown(encoder: splitEncoder,
                                  down: downProj,
                                  y: ySplit,
                                  yOffset: row * Self.d * halfBytes,
                                  scratchAct: scratchAct,
                                  scratchActOffset: row * Self.f * halfBytes)
        }
        splitEncoder.endEncoding()
        splitCB.commit()
        splitCB.waitUntilCompleted()
        #expect(splitCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: yElements)
        let got = Fp16Buffer.readHalf(ySplit, count: yElements)
        #expect(got == ref)
    }

    @Test func blockSharedExpertThenPostF1NormMatchesRepeatedScalarRows() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-postf1")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt8(context: ctx)
        let scalarRMS = try RMSNorm(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)
        let prefillRMS = try PrefillRMSNorm(context: ctx)

        let x = Self.makeInputBlock(rng: &rng)
        let gate = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeWeights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeWeights(rows: Self.d, cols: Self.f, rng: &rng)
        let postF1 = Self.makeBF16Weights(rng: &rng)
        let yElements = Self.rows * Self.d

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let yGot = Fp16Buffer.make(ctx.device, halves: Array(repeating: Self.sentinel, count: yElements)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: Self.f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: Self.f),
              let postF1Buf = ctx.device.makeBuffer(bytes: postF1,
                                                     length: postF1.count * MemoryLayout<UInt16>.stride,
                                                     options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = Self.makeProjection(ctx: ctx, packed: gate, rows: Self.f, cols: Self.d)
        let upProj = Self.makeProjection(ctx: ctx, packed: up, rows: Self.f, cols: Self.d)
        let downProj = Self.makeProjection(ctx: ctx, packed: down, rows: Self.d, cols: Self.f)

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<Self.rows {
            try scalar.encode(commandBuffer: refCB,
                              x: xBuf,
                              xOffset: row * Self.xStride * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: yRef,
                              yOffset: row * Self.d * halfBytes,
                              scratchAct: scratchAct)
            try scalarRMS.encodeBF16W(commandBuffer: refCB,
                                  x: yRef,
                                  xOffset: row * Self.d * halfBytes,
                                  weight: postF1Buf,
                                  out: yRef,
                                  outOffset: row * Self.d * halfBytes,
                                  d: UInt32(Self.d),
                                  eps: 1e-6)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: gotCB,
                                x: xBuf,
                                y: yGot,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: Self.rows,
                                d: Self.d,
                                intermediate: Self.f,
                                xStrideElements: Self.xStride,
                                yStrideElements: Self.d)
        try prefillRMS.encodeBF16W(commandBuffer: gotCB,
                               x: yGot,
                               weight: postF1Buf,
                               out: yGot,
                               t: UInt32(Self.rows),
                               d: UInt32(Self.d),
                               eps: 1e-6)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: yElements)
        let got = Fp16Buffer.readHalf(yGot, count: yElements)
        #expect(got == ref)
    }

    @Test(arguments: [64, 33], MPPPrefillInt4QMM.TileVariant.allCases)
    func chunkSharedExpertMatchesRowLoop(rows: Int, variant: MPPPrefillInt4QMM.TileVariant) throws {
        try Self.runChunkSharedExpertMatchesRowLoop(rows: rows, variant: variant)
    }

    @Test func chunkSharedExpertMatchesRowLoopWithVectorLoads() throws {
        try Self.runChunkSharedExpertMatchesRowLoop(rows: 33, variant: .n32b1, weightLoads: .vector)
    }

    @Test func chunkSharedExpertMatchesRowLoopAtALoweredMinimum() throws {
        try Self.runChunkSharedExpertMatchesRowLoop(rows: 21, variant: .n32b1, minimumRows: 16)
    }

    @Test func sharedExpertMatrixPathHonoursALoweredMinimum() throws {
        let ctx = try MetalContext()
        let int4 = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let mpp4 = MPPPrefillInt4QMM(context: ctx, weightBits: 4)

        #expect(int4.matrixPath(for: mpp4, queryCount: 21, d: Self.d, intermediate: Self.f) == nil)
        #expect((int4.matrixPath(for: mpp4, queryCount: 21, d: Self.d, intermediate: Self.f,
                                 minimumRows: 16) != nil) == mpp4.isAvailable)
        #expect(int4.matrixPath(for: mpp4, queryCount: 15, d: Self.d, intermediate: Self.f,
                                minimumRows: 16) == nil)
    }

    @Test func chunkSharedExpertRejectsShortChunks() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-short-chunk")
        let ctx = try MetalContext()
        let prefill = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4)

        let gate = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeInt4Weights(rows: Self.d, cols: Self.f, rng: &rng)
        guard let xBuf = Fp16Buffer.make(ctx.device, count: 8 * Self.d),
              let yBuf = Fp16Buffer.make(ctx.device, count: 8 * Self.d),
              let scratchGate = Fp16Buffer.make(ctx.device, count: 8 * Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: 8 * Self.f),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        #expect(throws: PrefillSharedExpertError.chunkTooShort(8)) {
            try prefill.encodeChunk(commandBuffer: cb,
                                    mpp: mpp,
                                    x: xBuf,
                                    y: yBuf,
                                    gate: Self.makeProjection(ctx: ctx, packed: gate,
                                                              rows: Self.f, cols: Self.d),
                                    up: Self.makeProjection(ctx: ctx, packed: up,
                                                            rows: Self.f, cols: Self.d),
                                    down: Self.makeProjection(ctx: ctx, packed: down,
                                                              rows: Self.d, cols: Self.f),
                                    scratchGate: scratchGate,
                                    scratchUp: scratchUp,
                                    queryCount: 8,
                                    d: Self.d,
                                    intermediate: Self.f)
        }
    }

    @Test func matrixPathSelectionRequiresAvailableMatchingMPP() throws {
        let ctx = try MetalContext()
        let int4 = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let int8 = try PrefillSharedExpert(context: ctx, weightBits: 8)
        let mpp4 = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
        let mpp8 = MPPPrefillInt4QMM(context: ctx, weightBits: 8)
        let rows = PrefillSharedExpert.matrixPathMinimumRows

        #expect(int4.matrixPath(for: nil, queryCount: rows, d: Self.d, intermediate: Self.f) == nil)
        #expect(int4.matrixPath(for: mpp4, queryCount: rows - 1, d: Self.d, intermediate: Self.f) == nil)
        #expect(int4.matrixPath(for: mpp8, queryCount: rows, d: Self.d, intermediate: Self.f) == nil)
        #expect(int8.matrixPath(for: mpp8, queryCount: rows, d: Self.d, intermediate: Self.f) == nil)
        // The row loop is the fallback exactly when the pipeline is missing:
        // true on a box with MPP TensorOps, vacuously true on one without.
        #expect((int4.matrixPath(for: mpp4, queryCount: rows, d: Self.d, intermediate: Self.f) != nil) == mpp4.isAvailable)
    }

    @Test func matrixPathRejectsNonMultipleOfTileK() throws {
        let ctx = try MetalContext()
        let int4 = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let mpp4 = MPPPrefillInt4QMM(context: ctx, weightBits: 4)
        let rows = PrefillSharedExpert.matrixPathMinimumRows

        #expect(int4.matrixPath(for: mpp4, queryCount: rows,
                                d: Self.d + 1, intermediate: Self.f) == nil)
        #expect(int4.matrixPath(for: mpp4, queryCount: rows,
                                d: Self.d, intermediate: Self.f + 1) == nil)
    }

    @Test func chunkSharedExpertRejectsMismatchedWeightBits() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-bits-mismatch")
        let ctx = try MetalContext()
        let prefill = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 8)
        let rows = PrefillSharedExpert.matrixPathMinimumRows

        let gate = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let up = Self.makeInt4Weights(rows: Self.f, cols: Self.d, rng: &rng)
        let down = Self.makeInt4Weights(rows: Self.d, cols: Self.f, rng: &rng)
        guard let xBuf = Fp16Buffer.make(ctx.device, count: rows * Self.d),
              let yBuf = Fp16Buffer.make(ctx.device, count: rows * Self.d),
              let scratchGate = Fp16Buffer.make(ctx.device, count: rows * Self.f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: rows * Self.f),
              let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }

        #expect(throws: PrefillSharedExpertError.weightBitsMismatch(expected: 4, got: 8)) {
            try prefill.encodeChunk(commandBuffer: cb,
                                    mpp: mpp,
                                    x: xBuf,
                                    y: yBuf,
                                    gate: Self.makeProjection(ctx: ctx, packed: gate,
                                                              rows: Self.f, cols: Self.d),
                                    up: Self.makeProjection(ctx: ctx, packed: up,
                                                            rows: Self.f, cols: Self.d),
                                    down: Self.makeProjection(ctx: ctx, packed: down,
                                                              rows: Self.d, cols: Self.f),
                                    scratchGate: scratchGate,
                                    scratchUp: scratchUp,
                                    queryCount: rows,
                                    d: Self.d,
                                    intermediate: Self.f)
        }
    }

    @Test func scalarGateRowsMatchesPerRowGEMV() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-scalar-gate-rows")
        let ctx = try MetalContext()
        let elementwise = try Elementwise(context: ctx)
        let gemv = try DequantInt8GEMV(context: ctx)
        let rows = 33
        let d = 2048

        let x = (0..<(rows * d)).map { _ in Float16(rng.uniform(-0.35, 0.35)) }
        let weights = Self.makeScalarGateView(ctx: ctx, d: d, rng: &rng)
        let hidden = (0..<(rows * d)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: hidden),
              let yRows = Fp16Buffer.make(ctx.device, halves: hidden),
              let gateRef = Fp16Buffer.make(ctx.device, count: rows),
              let gateRows = Fp16Buffer.make(ctx.device, count: rows) else {
            Issue.record("buffer allocation failed")
            return
        }

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            try gemv.encode(commandBuffer: refCB,
                            weights: weights.buffer, weightsOffset: Int(weights.offset),
                            scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                            x: xBuf, xOffset: row * d * halfBytes,
                            y: gateRef, yOffset: row * halfBytes,
                            m: 1, n: UInt32(d))
        }
        for row in 0..<rows {
            try elementwise.encodeSigmoidScalarMul(commandBuffer: refCB,
                                                   y: yRef, yOffset: row * d * halfBytes,
                                                   gate: gateRef, gateOffset: row * halfBytes,
                                                   count: d)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let rowsCB = ctx.queue.makeCommandBuffer()!
        try elementwise.encodeScalarGateRows(commandBuffer: rowsCB,
                                             weights: weights,
                                             x: xBuf,
                                             gate: gateRows,
                                             rows: rows, d: d)
        try elementwise.encodeSigmoidScalarMulRows(commandBuffer: rowsCB,
                                                   y: yRows,
                                                   gate: gateRows,
                                                   rows: rows, d: d)
        rowsCB.commit()
        rowsCB.waitUntilCompleted()
        #expect(rowsCB.error == nil)

        let gateExpected = Fp16Buffer.read(gateRef, count: rows)
        let gateActual = Fp16Buffer.read(gateRows, count: rows)
        let gateMaxAbs = RelError.maxAbsDiff(gateActual, gateExpected)
        #expect(gateMaxAbs <= 2e-2, "scalar gate maxAbs=\(gateMaxAbs)")

        let expected = Fp16Buffer.read(yRef, count: rows * d)
        let actual = Fp16Buffer.read(yRows, count: rows * d)
        let maxAbs = RelError.maxAbsDiff(actual, expected)
        let rel = RelError.compute(actual: actual, reference: expected)
        #expect(maxAbs <= 2e-2, "gated rows maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2)
    }

    private static func runChunkSharedExpertMatchesRowLoop(rows: Int,
                                                           variant: MPPPrefillInt4QMM.TileVariant,
                                                           weightLoads: MPPPrefillInt4QMM.WeightLoads = .byte,
                                                           minimumRows: Int = PrefillSharedExpert.matrixPathMinimumRows) throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert-chunk-\(rows)")
        let ctx = try MetalContext()
        let prefill = try PrefillSharedExpert(context: ctx, weightBits: 4, siluActivation: true)
        let mpp = MPPPrefillInt4QMM(context: ctx, weightBits: 4, variant: variant, weightLoads: weightLoads)
        #expect(mpp.isAvailable, "Requires runtime MPP TensorOps support (\(variant))")
        let d = chunkD
        let f = chunkF

        let x = makeInputBlock(rows: rows, d: d, rng: &rng)
        let gate = makeInt4Weights(rows: f, cols: d, rng: &rng)
        let up = makeInt4Weights(rows: f, cols: d, rng: &rng)
        let down = makeInt4Weights(rows: d, cols: f, rng: &rng)
        let yElements = rows * d

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yLoop = Fp16Buffer.make(ctx.device,
                                          halves: Array(repeating: sentinel, count: yElements)),
              let yChunk = Fp16Buffer.make(ctx.device,
                                           halves: Array(repeating: sentinel, count: yElements)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: rows * f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: rows * f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = makeProjection(ctx: ctx, packed: gate, rows: f, cols: d)
        let upProj = makeProjection(ctx: ctx, packed: up, rows: f, cols: d)
        let downProj = makeProjection(ctx: ctx, packed: down, rows: d, cols: f)

        let loopCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: loopCB,
                                x: xBuf,
                                y: yLoop,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: rows,
                                d: d,
                                intermediate: f,
                                xStrideElements: d,
                                yStrideElements: d)
        loopCB.commit()
        loopCB.waitUntilCompleted()
        #expect(loopCB.error == nil)

        let chunkCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeChunk(commandBuffer: chunkCB,
                                mpp: mpp,
                                x: xBuf,
                                y: yChunk,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                queryCount: rows,
                                d: d,
                                intermediate: f,
                                minimumRows: minimumRows)
        chunkCB.commit()
        chunkCB.waitUntilCompleted()
        #expect(chunkCB.error == nil)

        let reference = Fp16Buffer.read(yLoop, count: yElements)
        let actual = Fp16Buffer.read(yChunk, count: yElements)
        let maxAbs = RelError.maxAbsDiff(actual, reference)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(maxAbs <= 2e-2, "chunk shared expert maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel <= 2e-2)
    }

    private static func runBlockSharedExpertMatchesRepeatedScalarRows() throws {
        var rng = SeedTree(0xC0FFEE).key("prefill-shared-expert")
        let ctx = try MetalContext()
        let scalar = try SharedExpertInt8(context: ctx)
        let prefill = try PrefillSharedExpert(context: ctx)

        let x = makeInputBlock(rng: &rng)
        let gate = makeWeights(rows: f, cols: d, rng: &rng)
        let up = makeWeights(rows: f, cols: d, rng: &rng)
        let down = makeWeights(rows: d, cols: f, rng: &rng)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x),
              let yRef = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let yGot = Fp16Buffer.make(ctx.device, halves: Array(repeating: sentinel, count: rows * yStride)),
              let scratchGate = Fp16Buffer.make(ctx.device, count: f),
              let scratchUp = Fp16Buffer.make(ctx.device, count: f),
              let scratchAct = Fp16Buffer.make(ctx.device, count: f) else {
            Issue.record("buffer allocation failed")
            return
        }

        let gateProj = makeProjection(ctx: ctx, packed: gate, rows: f, cols: d)
        let upProj = makeProjection(ctx: ctx, packed: up, rows: f, cols: d)
        let downProj = makeProjection(ctx: ctx, packed: down, rows: d, cols: f)

        let halfBytes = MemoryLayout<Float16>.stride
        let refCB = ctx.queue.makeCommandBuffer()!
        for row in 0..<rows {
            try scalar.encode(commandBuffer: refCB,
                              x: xBuf,
                              xOffset: row * xStride * halfBytes,
                              gate: gateProj,
                              up: upProj,
                              down: downProj,
                              y: yRef,
                              yOffset: row * yStride * halfBytes,
                              scratchAct: scratchAct)
        }
        refCB.commit()
        refCB.waitUntilCompleted()
        #expect(refCB.error == nil)

        let gotCB = ctx.queue.makeCommandBuffer()!
        try prefill.encodeBlock(commandBuffer: gotCB,
                                x: xBuf,
                                y: yGot,
                                gate: gateProj,
                                up: upProj,
                                down: downProj,
                                scratchGate: scratchGate,
                                scratchUp: scratchUp,
                                scratchAct: scratchAct,
                                queryCount: rows,
                                d: d,
                                intermediate: f,
                                xStrideElements: xStride,
                                yStrideElements: yStride)
        gotCB.commit()
        gotCB.waitUntilCompleted()
        #expect(gotCB.error == nil)

        let ref = Fp16Buffer.readHalf(yRef, count: rows * yStride)
        let got = Fp16Buffer.readHalf(yGot, count: rows * yStride)
        #expect(got == ref)
        assertPaddingUnchanged(got)
    }

    private static func makeInputBlock(rng: inout SplitMix64) -> [Float16] {
        var block = Array(repeating: sentinel, count: rows * xStride)
        for row in 0..<rows {
            for col in 0..<d {
                block[row * xStride + col] = Float16(rng.uniform(-0.35, 0.35))
            }
        }
        return block
    }

    private static func makeWeights(rows: Int,
                                    cols: Int,
                                    rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let groupsPerRow = cols / Quantization.groupSize
        var packed = [UInt8](repeating: 0, count: rows * cols)
        var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)
        for row in 0..<rows {
            let values = (0..<cols).map { _ in rng.uniform(-0.4, 0.4) }
            let q = Quantization.quantizeInt8Affine(values)
            for col in 0..<cols {
                packed[row * cols + col] = q.packed[col]
            }
            for group in 0..<groupsPerRow {
                scales[row * groupsPerRow + group] = q.scales[group]
                biases[row * groupsPerRow + group] = q.biases[group]
            }
        }
        return (packed, scales, biases)
    }

    private static func makeInputBlock(rows: Int, d: Int, rng: inout SplitMix64) -> [Float16] {
        (0..<(rows * d)).map { _ in Float16(rng.uniform(-0.35, 0.35)) }
    }

    /// Scaled by 1/sqrt(fan-in) so a 2048-wide chunk lands on O(1) outputs —
    /// the chunk-vs-loop tolerance is absolute.
    private static func makeInt4Weights(rows: Int,
                                        cols: Int,
                                        rng: inout SplitMix64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16])
    {
        let groupsPerRow = cols / Quantization.groupSize
        let rowBytes = cols / 2
        var packed = [UInt8](repeating: 0, count: rows * rowBytes)
        var scales = [UInt16](repeating: 0, count: rows * groupsPerRow)
        var biases = [UInt16](repeating: 0, count: rows * groupsPerRow)
        let limit = 5.5 / Float(cols).squareRoot()
        for row in 0..<rows {
            let values = (0..<cols).map { _ in rng.uniform(-limit, limit) }
            let quantized = Quantization.quantizeInt4Affine(values)
            for byte in 0..<rowBytes {
                packed[row * rowBytes + byte] = quantized.packed[byte]
            }
            for group in 0..<groupsPerRow {
                scales[row * groupsPerRow + group] = quantized.scales[group]
                biases[row * groupsPerRow + group] = quantized.biases[group]
            }
        }
        return (packed, scales, biases)
    }

    /// Weights, scales and biases in one buffer, as a manifest stores them.
    private static func makeScalarGateView(ctx: MetalContext,
                                           d: Int,
                                           rng: inout SplitMix64) -> TensorView {
        let groups = d / Quantization.groupSize
        let limit = 5.5 / Float(d).squareRoot()
        let quantized = Quantization.quantizeInt8Affine(
            (0..<d).map { _ in rng.uniform(-limit, limit) })
        let scaleOffset = d
        let biasOffset = scaleOffset + groups * MemoryLayout<UInt16>.stride
        let length = biasOffset + groups * MemoryLayout<UInt16>.stride
        let buffer = ctx.device.makeBuffer(length: length, options: .storageModeShared)!
        buffer.contents().copyMemory(from: quantized.packed, byteCount: d)
        quantized.scales.withUnsafeBytes {
            buffer.contents().advanced(by: scaleOffset)
                .copyMemory(from: $0.baseAddress!, byteCount: groups * MemoryLayout<UInt16>.stride)
        }
        quantized.biases.withUnsafeBytes {
            buffer.contents().advanced(by: biasOffset)
                .copyMemory(from: $0.baseAddress!, byteCount: groups * MemoryLayout<UInt16>.stride)
        }
        return TensorView(buffer: buffer,
                          offset: 0, length: UInt64(d),
                          scaleOffset: UInt64(scaleOffset),
                          scaleLength: UInt64(groups * MemoryLayout<UInt16>.stride),
                          biasOffset: UInt64(biasOffset),
                          biasLength: UInt64(groups * MemoryLayout<UInt16>.stride),
                          shape: (1, UInt32(d), 0, 0),
                          dtype: 0)
    }

    private static func makeBF16Weights(rng: inout SplitMix64) -> [UInt16] {
        (0..<d).map { _ in Quantization.bf16Bits(rng.uniform(0.75, 1.25)) }
    }

    private static func makeProjection(
        ctx: MetalContext,
        packed: (packed: [UInt8], scales: [UInt16], biases: [UInt16]),
        rows: Int,
        cols: Int
    ) -> SharedExpertInt8Proj {
        let w = ctx.device.makeBuffer(bytes: packed.packed,
                                      length: packed.packed.count,
                                      options: .storageModeShared)!
        let s = ctx.device.makeBuffer(bytes: packed.scales,
                                      length: packed.scales.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        let b = ctx.device.makeBuffer(bytes: packed.biases,
                                      length: packed.biases.count * MemoryLayout<UInt16>.stride,
                                      options: .storageModeShared)!
        return SharedExpertInt8Proj(weights: w,
                                    scales: s,
                                    biases: b,
                                    rows: UInt32(rows),
                                    cols: UInt32(cols))
    }

    private static func assertPaddingUnchanged(_ values: [Float16]) {
        for row in 0..<rows {
            for col in d..<yStride {
                #expect(values[row * yStride + col] == sentinel)
            }
        }
    }
}
