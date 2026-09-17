import Foundation
import Metal
import Shrike
import ShrikeValidationSupport

final class SyntheticRows {
    let config: ArchConfig
    let maxSeq: Int
    let headDim: Int
    let numQHeads: Int
    let numKVHeads: Int
    let scale: Float
    let qBuf: MTLBuffer
    let keyView: KVView
    let valueView: KVView
    let outBuf: MTLBuffer
    private let rows: Int8KVRows

    init(context: MetalContext, maxSeq: Int, seed: UInt64) throws {
        let config = ArchConfig.qwen36_35B_A3B
        let rows = try Int8KVRows.make(context: context, config: config, seqLen: maxSeq, seed: seed)
        guard let outBuf = Fp16Buffer.make(context.device, count: config.numHeads * config.fullHeadDim) else {
            throw BenchError.allocation("output")
        }
        self.config = config
        self.maxSeq = maxSeq
        self.headDim = config.fullHeadDim
        self.numQHeads = config.numHeads
        self.numKVHeads = config.numFullKVHeads
        self.scale = 1 / Float(config.fullHeadDim).squareRoot()
        self.qBuf = rows.qBuf
        self.keyView = rows.keyView
        self.valueView = rows.valueView
        self.outBuf = outBuf
        self.rows = rows
    }

    var bytesPerPosition: Int { 2 * keyView.stride }
}
