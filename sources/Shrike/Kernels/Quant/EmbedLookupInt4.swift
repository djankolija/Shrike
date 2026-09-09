import Foundation
import Metal

/// 4-bit affine embedding lookup with fused output scale.
///
/// The embedding table (`embed_tokens.weight`) is `U32`/packed 4-bit with
/// `weightBits=4` in the manifest. The same table also drives the lm_head
/// GEMV via `DequantInt4GEMV` over the transposed access pattern. The
/// `outScale` parameter fuses the optional `sqrt(hidden_size)` post-embedding
/// scale so the per-token dequant + scale is one pass.
final class EmbedLookupInt4 {
    private enum TokenSource {
        case constant(UInt32)
        case buffer(MTLBuffer, Int)
    }

    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("embed_lookup_int4")
    }

    /// Encodes the lookup. `table`, `scales`, `biases` typically live inside
    /// one resident blob — pass that buffer with the per-region offsets.
    /// Pass `outScale = 1.0` to write the raw dequantized row. `vocab` is the
    /// row count; out-of-range token ids write zeros (K8).
    func encode(commandBuffer: MTLCommandBuffer,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float,
                       vocab: UInt32) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encode(encoder: encoder,
               table: table, tableOffset: tableOffset,
               scales: scales, scalesOffset: scalesOffset,
               biases: biases, biasesOffset: biasesOffset,
               out: out, outOffset: outOffset,
               tokenId: tokenId,
               d: d, outScale: outScale, vocab: vocab)
        encoder.endEncoding()
    }

    func encode(encoder: MTLComputeCommandEncoder,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenId: UInt32,
                       d: UInt32,
                       outScale: Float,
                       vocab: UInt32) {
        encodeTokenLookup(encoder: encoder,
                          table: table, tableOffset: tableOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          out: out, outOffset: outOffset,
                          tokenSource: .constant(tokenId),
                          d: d, outScale: outScale, vocab: vocab)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenBuffer: MTLBuffer, tokenOffset: Int = 0,
                       d: UInt32,
                       outScale: Float,
                       vocab: UInt32) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encode(encoder: encoder,
               table: table, tableOffset: tableOffset,
               scales: scales, scalesOffset: scalesOffset,
               biases: biases, biasesOffset: biasesOffset,
               out: out, outOffset: outOffset,
               tokenBuffer: tokenBuffer, tokenOffset: tokenOffset,
               d: d, outScale: outScale, vocab: vocab)
        encoder.endEncoding()
    }

    func encode(encoder: MTLComputeCommandEncoder,
                       table:  MTLBuffer, tableOffset:  Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       out:    MTLBuffer, outOffset: Int = 0,
                       tokenBuffer: MTLBuffer, tokenOffset: Int = 0,
                       d: UInt32,
                       outScale: Float,
                       vocab: UInt32) {
        encodeTokenLookup(encoder: encoder,
                          table: table, tableOffset: tableOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          out: out, outOffset: outOffset,
                          tokenSource: .buffer(tokenBuffer, tokenOffset),
                          d: d, outScale: outScale, vocab: vocab)
    }

    private func encodeTokenLookup(encoder: MTLComputeCommandEncoder,
                                    table:  MTLBuffer, tableOffset:  Int,
                                    scales: MTLBuffer, scalesOffset: Int,
                                    biases: MTLBuffer, biasesOffset: Int,
                                    out:    MTLBuffer, outOffset: Int,
                                    tokenSource: TokenSource,
                                    d: UInt32,
                                    outScale: Float,
                                    vocab: UInt32) {
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(table,  offset: tableOffset,  index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out,    offset: outOffset,    index: 3)
        switch tokenSource {
        case .constant(let tokenId):
            var tokenVar = tokenId
            encoder.setBytes(&tokenVar, length: MemoryLayout<UInt32>.size, index: 4)
        case .buffer(let tokenBuffer, let tokenOffset):
            encoder.setBuffer(tokenBuffer, offset: tokenOffset, index: 4)
        }
        var dVar     = d
        var sVar     = outScale
        var vocabVar = vocab
        encoder.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&sVar,     length: MemoryLayout<Float>.size,  index: 6)
        encoder.setBytes(&vocabVar, length: MemoryLayout<UInt32>.size, index: 7)

        let threadsPerGroup = min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
        let gridSize = MTLSize(width: Int(d), height: 1, depth: 1)
        let tgSize   = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    }
}
