import Foundation
import Metal
import Shrike

/// The coded kernel is compiled behind moe.metal's own source so its helpers
/// are the production functions, not copies.
final class ExpertKernels {
    static let hidden: UInt32 = 2048
    static let intermediate: UInt32 = 512
    static let topK: UInt32 = 8
    static let rowsPerThreadgroup = 16

    let context: MetalContext
    let production: MTLComputePipelineState
    let coded: MTLComputePipelineState
    let codedAux: MTLComputePipelineState
    let ioReady: MTLBuffer

    init(context: MetalContext) throws {
        self.context = context
        let shared: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(Self.hidden)),
            MetalFunctionConstant(index: 1, value: .uint32(Self.intermediate)),
            MetalFunctionConstant(index: 2, value: .uint32(Self.topK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
            MetalFunctionConstant(index: 4, value: .bool(true)),
            MetalFunctionConstant(index: 6, value: .bool(true)),
        ]
        self.production = try context.pipeline("moe_phase1_gate_up_act_u16load", constants: shared)

        let executableDir = Bundle.main.executableURL!.deletingLastPathComponent()
        let moeURL = executableDir.appendingPathComponent("Shrike_Shrike.bundle/Metal/MoE/moe.metal")
        guard let moeSource = try? String(contentsOf: moeURL, encoding: .utf8) else {
            throw BenchError.missingResource(moeURL.path)
        }
        guard let expertURL = Bundle.module.url(forResource: "expert", withExtension: "metal",
                                                subdirectory: "Metal") else {
            throw BenchError.missingResource("Metal/expert.metal")
        }
        let expertSource = try String(contentsOf: expertURL, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library = try context.device.makeLibrary(source: moeSource + "\n" + expertSource, options: options)
        func codedPipeline(aux: Bool) throws -> MTLComputePipelineState {
            let values = MTLFunctionConstantValues()
            var d = Self.hidden, f = Self.intermediate, k = Self.topK
            var yes = true, auxValue = aux
            values.setConstantValue(&d, type: .uint, index: 0)
            values.setConstantValue(&f, type: .uint, index: 1)
            values.setConstantValue(&k, type: .uint, index: 2)
            values.setConstantValue(&yes, type: .bool, index: 3)
            values.setConstantValue(&yes, type: .bool, index: 4)
            values.setConstantValue(&yes, type: .bool, index: 6)
            values.setConstantValue(&auxValue, type: .bool, index: 20)
            let function = try library.makeFunction(name: "moe_phase1_coded_gate_up_act", constantValues: values)
            return try context.device.makeComputePipelineState(function: function)
        }
        self.coded = try codedPipeline(aux: false)
        self.codedAux = try codedPipeline(aux: true)
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

    private func bind(_ encoder: MTLComputeCommandEncoder, arguments: MTLBuffer, blobs: [MTLBuffer],
                      x: MTLBuffer, acts: MTLBuffer) {
        var d = Self.hidden, f = Self.intermediate, k = Self.topK
        encoder.setBuffer(arguments, offset: 0, index: 0)
        for blob in blobs { encoder.useResource(blob, usage: .read) }
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&d, length: 4, index: 4)
        encoder.setBytes(&f, length: 4, index: 5)
        encoder.setBytes(&k, length: 4, index: 6)
        encoder.setBuffer(ioReady, offset: 0, index: 7)
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder) {
        let rows = Int(Self.topK * Self.intermediate)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    func encodeProduction(_ cb: MTLCommandBuffer, arguments: MTLBuffer, blobs: [MTLBuffer],
                          offsets: PlainOffsets, x: MTLBuffer, acts: MTLBuffer) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else { throw BenchError.encoder }
        defer { encoder.endEncoding() }
        encoder.setComputePipelineState(production)
        bind(encoder, arguments: arguments, blobs: blobs, x: x, acts: acts)
        var o = offsets
        encoder.setBytes(&o, length: MemoryLayout<PlainOffsets>.stride, index: 1)
        dispatch(encoder)
    }

    func encodeCoded(_ cb: MTLCommandBuffer, aux: Bool, arguments: MTLBuffer, blobs: [MTLBuffer],
                     offsets: CodedOffsets, table: MTLBuffer, auxTables: MTLBuffer,
                     x: MTLBuffer, acts: MTLBuffer) throws {
        guard let encoder = cb.makeComputeCommandEncoder() else { throw BenchError.encoder }
        defer { encoder.endEncoding() }
        encoder.setComputePipelineState(aux ? codedAux : coded)
        bind(encoder, arguments: arguments, blobs: blobs, x: x, acts: acts)
        var o = offsets
        encoder.setBytes(&o, length: MemoryLayout<CodedOffsets>.stride, index: 1)
        encoder.setBuffer(table, offset: 0, index: 8)
        encoder.setBuffer(auxTables, offset: 0, index: 9)
        dispatch(encoder)
    }
}
