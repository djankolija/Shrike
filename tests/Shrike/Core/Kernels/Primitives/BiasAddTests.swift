import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct BiasAddTests {

    private static func bf16(_ x: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: x.bitPattern >> 16)
    }

    private static func bf16Value(_ x: Float) -> Float {
        Float(bitPattern: UInt32(bf16(x)) << 16)
    }

    @Test(arguments: [(rows: 1, rowElems: 4096), (rows: 3, rowElems: 512)])
    func biasAddBroadcastsBF16RowOverTokenRows(_ shape: (rows: Int, rowElems: Int)) throws {
        var rng = SeedTree(0xB1A5).key("bias-add-r\(shape.rows)-e\(shape.rowElems)")
        let count = shape.rows * shape.rowElems
        let x = (0..<count).map { _ in rng.uniform(-1.0, 1.0) }
        let bias = (0..<shape.rowElems).map { _ in rng.uniform(-0.5, 0.5) }

        let ctx = try MetalContext()
        let elementwise = try Elementwise(context: ctx)
        let biasBits = bias.map { Self.bf16($0) }
        guard let xBuf = Fp16Buffer.make(ctx.device, values: x),
              let biasBuf = ctx.device.makeBuffer(bytes: biasBits,
                                                  length: biasBits.count * 2,
                                                  options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        try elementwise.encodeBiasAdd(commandBuffer: cb,
                                      x: xBuf,
                                      bias: biasBuf,
                                      rowElems: shape.rowElems,
                                      rows: shape.rows)
        cb.commit()
        cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(xBuf, count: count)
        let reference = (0..<count).map { i in
            Float(Float16(Float(Float16(x[i])) + Self.bf16Value(bias[i % shape.rowElems])))
        }
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.fp16ChainedReduction, "bias add rel=\(rel)")
    }
}
