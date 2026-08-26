import Foundation
import Metal

/// Kimi-Linear MLA absorbed-form projections (`mla.metal`): the batched
/// per-head embed (`mlaQ[h] = [W_UKᵀ[h]·q_nope[h] | q_pe[h]]`, int8 weights)
/// and unembed (`out[h] = W_UV[h]·attn[h]`, source-precision int4 weights).
/// One dispatch covers every (token, head, output row), so decode (T = 1)
/// and prefill share the same entry points.
final class MLA {
    private let embedPSO: MTLComputePipelineState
    private let unembedPSO: MTLComputePipelineState

    let config: MLAConfig
    let numHeads: Int

    private static let rowsPerTG = 8

    init(context: MetalContext, config: MLAConfig, numHeads: Int) throws {
        precondition(config.qkNopeDim % 64 == 0 && config.latentDim % 64 == 0,
                     "MLA projection widths must be multiples of the quant group")
        self.config = config
        self.numHeads = numHeads
        self.embedPSO = try context.pipeline("mla_embed_q")
        self.unembedPSO = try context.pipeline("mla_unembed")
    }

    /// `qRaw` rows are `[numHeads * (nope + rope)]` halves; `y` rows are
    /// `[numHeads * (latent + rope)]`. `embedQ` is the repacked
    /// `self_attn.embed_q` view ([H * latent, nope], 8-bit affine g64).
    func encodeEmbedQ(commandBuffer: MTLCommandBuffer,
                      embedQ: TensorView,
                      qRaw: MTLBuffer, qRawOffset: Int = 0,
                      y: MTLBuffer, yOffset: Int = 0,
                      tokens: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(embedPSO)
        enc.setBuffer(embedQ.buffer, offset: Int(embedQ.offset), index: 0)
        enc.setBuffer(embedQ.buffer, offset: Int(embedQ.scaleOffset), index: 1)
        enc.setBuffer(embedQ.buffer, offset: Int(embedQ.biasOffset), index: 2)
        enc.setBuffer(qRaw, offset: qRawOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var heads = UInt32(numHeads)
        var nope = UInt32(config.qkNopeDim)
        var rope = UInt32(config.qkRopeDim)
        var latent = UInt32(config.latentDim)
        var tokenCount = UInt32(tokens)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nope, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&rope, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&latent, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&tokenCount, length: MemoryLayout<UInt32>.size, index: 9)
        let rows = tokens * numHeads * (config.latentDim + config.qkRopeDim)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerTG - 1) / Self.rowsPerTG,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerTG,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `attn` rows are `[numHeads * latent]` halves; `y` rows are
    /// `[numHeads * vHeadDim]`. `unembedOut` is the repacked
    /// `self_attn.unembed_out` view ([H * vDim, latent], int4 affine g64).
    func encodeUnembed(commandBuffer: MTLCommandBuffer,
                       unembedOut: TensorView,
                       attn: MTLBuffer, attnOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       tokens: Int) throws {
        precondition(Int(unembedOut.offset) % 2 == 0,
                     "mla_unembed needs a 2-aligned weights offset")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(unembedPSO)
        enc.setBuffer(unembedOut.buffer, offset: Int(unembedOut.offset), index: 0)
        enc.setBuffer(unembedOut.buffer, offset: Int(unembedOut.scaleOffset), index: 1)
        enc.setBuffer(unembedOut.buffer, offset: Int(unembedOut.biasOffset), index: 2)
        enc.setBuffer(attn, offset: attnOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var heads = UInt32(numHeads)
        var latent = UInt32(config.latentDim)
        var vDim = UInt32(config.valueHeadDim)
        var tokenCount = UInt32(tokens)
        enc.setBytes(&heads, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&latent, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&tokenCount, length: MemoryLayout<UInt32>.size, index: 8)
        let rows = tokens * numHeads * config.valueHeadDim
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerTG - 1) / Self.rowsPerTG,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerTG,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}
