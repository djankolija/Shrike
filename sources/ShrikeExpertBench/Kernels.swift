import Foundation
import Metal
import Shrike

final class ExpertKernels {
    static let hidden: UInt32 = 2048
    static let intermediate: UInt32 = 512
    static let topK: UInt32 = 8
    static let rowsPerThreadgroup = 16

    let context: MetalContext
    let production: MTLComputePipelineState
    let ioReady: MTLBuffer

    init(context: MetalContext) throws {
        self.context = context
        self.production = try context.pipeline("moe_phase1_gate_up_act_u16load", constants: [
            MetalFunctionConstant(index: 0, value: .uint32(Self.hidden)),
            MetalFunctionConstant(index: 1, value: .uint32(Self.intermediate)),
            MetalFunctionConstant(index: 2, value: .uint32(Self.topK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(true)),
        ])
        guard let ready = context.device.makeBuffer(length: 16, options: .storageModeShared) else {
            throw BenchError.allocation("io status")
        }
        ready.contents().storeBytes(of: UInt32(1), as: UInt32.self)
        self.ioReady = ready
    }

    static func argumentBuffer(device: MTLDevice, blobs: [MTLBuffer]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: 8 * 8, options: .storageModeShared) else {
            throw BenchError.allocation("argument buffer")
        }
        let p = buffer.contents().assumingMemoryBound(to: UInt64.self)
        for i in 0..<8 { p[i] = blobs[min(i, blobs.count - 1)].gpuAddress }
        return buffer
    }

    func encodeProduction(_ cb: MTLCommandBuffer, arguments: MTLBuffer, blobs: [MTLBuffer],
                          offsets: PlainOffsets, x: MTLBuffer, acts: MTLBuffer) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else { throw BenchError.encoder }
        defer { encoder.endEncoding() }
        encoder.setComputePipelineState(production)
        var d = Self.hidden, f = Self.intermediate, k = Self.topK
        encoder.setBuffer(arguments, offset: 0, index: 0)
        for blob in blobs { encoder.useResource(blob, usage: .read) }
        var o = offsets
        encoder.setBytes(&o, length: MemoryLayout<PlainOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.setBytes(&f, length: 4, index: 5)
        encoder.setBytes(&k, length: 4, index: 6)
        encoder.setBuffer(ioReady, offset: 0, index: 7)
        let rows = Int(Self.topK * Self.intermediate)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }
}

/// Mirrors `ExpertOffsets` in moe.metal field for field.
struct PlainOffsets {
    var gateW: UInt32 = 0
    var gateS: UInt32 = 0
    var gateB: UInt32 = 0
    var upW: UInt32 = 0
    var upS: UInt32 = 0
    var upB: UInt32 = 0
    var downW: UInt32 = 0
    var downS: UInt32 = 0
    var downB: UInt32 = 0
    var gateAB: UInt32 = 0
    var upAB: UInt32 = 0
    var downAB: UInt32 = 0
}
