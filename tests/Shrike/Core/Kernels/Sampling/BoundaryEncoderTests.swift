import Testing
import Foundation
import Metal
@testable import Shrike
import ShrikeValidationSupport

@Suite struct BoundaryEncoderTests {

    private struct Sizes {
        static let V = 256
        static let D = 128
        static let groupsPerRow = D / Quantization.groupSize
        static let argmaxIndex = 123
    }

    private static func buildEmbedTable(seed: UInt64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var rng = SeedTree(seed).key("boundary-encoder-embed-table")
        var rows: [[Float]] = []
        rows.reserveCapacity(Sizes.V)
        for _ in 0..<Sizes.V {
            rows.append((0..<Sizes.D).map { _ in rng.uniform(-1.0, 1.0) })
        }
        var packed = [UInt8](repeating: 0, count: Sizes.V * (Sizes.D / 2))
        var scales = [UInt16](repeating: 0, count: Sizes.V * Sizes.groupsPerRow)
        var biases = [UInt16](repeating: 0, count: Sizes.V * Sizes.groupsPerRow)
        for v in 0..<Sizes.V {
            let q = Quantization.quantizeInt4Affine(rows[v])
            for i in 0..<(Sizes.D / 2) { packed[v * (Sizes.D / 2) + i] = q.packed[i] }
            for g in 0..<Sizes.groupsPerRow {
                scales[v * Sizes.groupsPerRow + g] = q.scales[g]
                biases[v * Sizes.groupsPerRow + g] = q.biases[g]
            }
        }
        return (packed, scales, biases)
    }

    private final class Rig {
        let context: MetalContext
        let sampler: Sampler
        let embed: EmbedLookupInt4
        let tableBuf: MTLBuffer
        let scalesBuf: MTLBuffer
        let biasesBuf: MTLBuffer

        init() throws {
            self.context = try MetalContext()
            self.sampler = try Sampler(context: context, vocab: Sizes.V)
            self.embed = try EmbedLookupInt4(context: context)
            let (packed, scales, biases) = BoundaryEncoderTests.buildEmbedTable(seed: 0x8A11)
            guard let tableBuf = context.device.makeBuffer(
                    bytes: packed, length: packed.count,
                    options: .storageModeShared),
                  let scalesBuf = context.device.makeBuffer(
                    bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared),
                  let biasesBuf = context.device.makeBuffer(
                    bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                    options: .storageModeShared)
            else {
                throw MetalError.noDevice
            }
            self.tableBuf = tableBuf
            self.scalesBuf = scalesBuf
            self.biasesBuf = biasesBuf
        }
    }

    private static func makeLogits(_ device: MTLDevice) -> MTLBuffer? {
        var values = [Float](repeating: -1.0, count: Sizes.V)
        values[Sizes.argmaxIndex] = 8.0
        return Fp16Buffer.make(device, values: values)
    }

    private static func makeSentinelTokenBuffer(_ device: MTLDevice) -> MTLBuffer? {
        guard let buf = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            return nil
        }
        buf.contents().storeBytes(of: UInt32(0xFFFF_FFFF), as: UInt32.self)
        return buf
    }

    private static func runSeparateEncoders(rig: Rig, config: GenerationConfig) throws
        -> (token: UInt32, embed: [Float16]) {
        guard let logits = Self.makeLogits(rig.context.device),
              let probs = Fp16Buffer.make(rig.context.device, count: Sizes.V),
              let outToken = Self.makeSentinelTokenBuffer(rig.context.device),
              let embedOut = Fp16Buffer.make(rig.context.device, count: Sizes.D),
              let cb = rig.context.queue.makeCommandBuffer() else {
            Issue.record("Failed to allocate Metal resources")
            return (0, [])
        }

        try rig.sampler.sample(commandBuffer: cb,
                               logits: logits, probs: probs,
                               history: [], config: config, position: 0,
                               outToken: outToken)
        try rig.embed.encode(commandBuffer: cb,
                             table: rig.tableBuf, scales: rig.scalesBuf, biases: rig.biasesBuf,
                             out: embedOut,
                             tokenBuffer: outToken, d: UInt32(Sizes.D),
                             outScale: 1.0, vocab: UInt32(Sizes.V))
        cb.commit()
        cb.waitUntilCompleted()

        let token = outToken.contents().load(as: UInt32.self)
        return (token, Fp16Buffer.readHalf(embedOut, count: Sizes.D))
    }

    private static func runOneEncoder(rig: Rig, config: GenerationConfig) throws
        -> (token: UInt32, embed: [Float16]) {
        guard let logits = Self.makeLogits(rig.context.device),
              let probs = Fp16Buffer.make(rig.context.device, count: Sizes.V),
              let outToken = Self.makeSentinelTokenBuffer(rig.context.device),
              let embedOut = Fp16Buffer.make(rig.context.device, count: Sizes.D),
              let cb = rig.context.queue.makeCommandBuffer(),
              let encoder = cb.makeComputeCommandEncoder() else {
            Issue.record("Failed to allocate Metal resources")
            return (0, [])
        }

        try rig.sampler.sample(encoder: encoder,
                               logits: logits, probs: probs,
                               history: [], config: config, position: 0,
                               outToken: outToken)
        rig.embed.encode(encoder: encoder,
                         table: rig.tableBuf, scales: rig.scalesBuf, biases: rig.biasesBuf,
                         out: embedOut,
                         tokenBuffer: outToken, tokenOffset: 0, d: UInt32(Sizes.D),
                         outScale: 1.0, vocab: UInt32(Sizes.V))
        encoder.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let token = outToken.contents().load(as: UInt32.self)
        return (token, Fp16Buffer.readHalf(embedOut, count: Sizes.D))
    }

    @Test func samplerAndEmbedOnOneEncoderMatchSeparateEncoders() throws {
        let rig = try Rig()
        let configs: [GenerationConfig] = [
            GenerationConfig(temperature: 0, topK: nil, topP: nil, seed: 42),
            GenerationConfig(temperature: 1.0, topK: 8, topP: 1.0, seed: 42)
        ]

        for config in configs {
            let separate = try Self.runSeparateEncoders(rig: rig, config: config)
            let merged = try Self.runOneEncoder(rig: rig, config: config)

            #expect(separate.token == merged.token,
                    "temperature \(config.temperature): separate token \(separate.token) vs merged \(merged.token)")
            #expect(separate.embed.map(\.bitPattern) == merged.embed.map(\.bitPattern),
                    "temperature \(config.temperature): embed output bytes differ between separate and merged encoders")
        }
    }
}
