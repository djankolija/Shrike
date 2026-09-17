import Foundation
import Metal
import Shrike

/// Dispatched on the production geometry so its times compare with the wrapper's.
final class LadderKernel {
    static let numChunks = 64
    static let threadsPerGroup = 256

    private let device: MTLDevice
    private let source: String
    private let library: MTLLibrary
    private var safeLibrary: MTLLibrary?
    private var pipelines: [LadderSwitches: MTLComputePipelineState] = [:]
    let mBuf: MTLBuffer
    let dBuf: MTLBuffer
    let oBuf: MTLBuffer
    let partialCount: Int
    let oFloats: Int

    init(device: MTLDevice, numQHeads: Int, headDim: Int) throws {
        guard let url = Bundle.module.url(forResource: "ladder", withExtension: "metal",
                                          subdirectory: "Metal") else {
            throw BenchError.missingResource("Metal/ladder.metal")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        self.device = device
        self.source = source
        self.library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        self.partialCount = numQHeads * Self.numChunks
        self.oFloats = partialCount * headDim
        let loadOnlyFloats = 2 * Self.numChunks * Self.threadsPerGroup
        guard let mBuf = device.makeBuffer(length: partialCount * 4, options: .storageModeShared),
              let dBuf = device.makeBuffer(length: partialCount * 4, options: .storageModeShared),
              let oBuf = device.makeBuffer(length: max(oFloats, loadOnlyFloats) * 4,
                                           options: .storageModeShared) else {
            throw BenchError.allocation("partial scratch")
        }
        self.mBuf = mBuf
        self.dBuf = dBuf
        self.oBuf = oBuf
    }

    func pipeline(_ sw: LadderSwitches) throws -> MTLComputePipelineState {
        if let pso = pipelines[sw] { return pso }
        guard sw.threadgroupBytes <= device.maxThreadgroupMemoryLength else {
            throw BenchError.usage(
                "arm needs \(sw.threadgroupBytes) bytes of threadgroup memory, the device allows \(device.maxThreadgroupMemoryLength)")
        }
        let values = MTLFunctionConstantValues()
        var qRegs = sw.qRegs, posBlock = sw.posBlock, doubleBuffer = sw.doubleBuffer
        var noSoftmax = sw.noSoftmax, noV = sw.noV, loadOnly = sw.loadOnly
        var fullRow = sw.fullRow, loadBytes = sw.loadBytes, staticLoops = sw.staticLoops
        var oForm = sw.oForm, dForm = sw.dForm
        values.setConstantValue(&qRegs, type: .bool, index: 0)
        values.setConstantValue(&posBlock, type: .uint, index: 1)
        values.setConstantValue(&doubleBuffer, type: .bool, index: 2)
        values.setConstantValue(&noSoftmax, type: .bool, index: 3)
        values.setConstantValue(&noV, type: .bool, index: 4)
        values.setConstantValue(&loadOnly, type: .bool, index: 5)
        values.setConstantValue(&fullRow, type: .bool, index: 6)
        values.setConstantValue(&loadBytes, type: .uint, index: 7)
        values.setConstantValue(&staticLoops, type: .bool, index: 8)
        values.setConstantValue(&oForm, type: .uint, index: 9)
        values.setConstantValue(&dForm, type: .uint, index: 10)
        let function = try (sw.safeMath ? safeMathLibrary() : library)
            .makeFunction(name: "ladder_partial", constantValues: values)
        let pso = try device.makeComputePipelineState(function: function)
        pipelines[sw] = pso
        return pso
    }

    private func safeMathLibrary() throws -> MTLLibrary {
        if let lib = safeLibrary { return lib }
        let options = MTLCompileOptions()
        options.mathMode = .safe
        let lib = try device.makeLibrary(source: source, options: options)
        safeLibrary = lib
        return lib
    }

    func encode(commandBuffer: MTLCommandBuffer, rows: SyntheticRows,
                seqLen: Int, switches sw: LadderSwitches) throws {
        let pso = try pipeline(sw)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchError.encoder
        }
        defer { encoder.endEncoding() }
        let chunks = Self.numChunks
        let chunkLen = (seqLen + chunks - 1) / chunks
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(rows.qBuf, offset: 0, index: 0)
        encoder.setBuffer(rows.keyView.buffer, offset: rows.keyView.offset, index: 1)
        encoder.setBuffer(rows.valueView.buffer, offset: rows.valueView.offset, index: 2)
        encoder.setBuffer(mBuf, offset: 0, index: 3)
        encoder.setBuffer(dBuf, offset: 0, index: 4)
        encoder.setBuffer(oBuf, offset: 0, index: 5)
        var sl = UInt32(seqLen), cl = UInt32(chunkLen), nc = UInt32(chunks), sc = rows.scale
        encoder.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&cl, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&sc, length: MemoryLayout<Float>.size, index: 9)
        encoder.setThreadgroupMemoryLength(sw.threadgroupBytes, index: 0)
        let groups = sw.fullRow ? chunks : rows.numKVHeads * chunks
        encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup,
                                                                    height: 1, depth: 1))
    }

    /// Mirrors attention_decode_combine so a ladder arm's output can meet the production's.
    func combineOnCPU(numQHeads: Int, headDim: Int) -> [Float] {
        let chunks = Self.numChunks
        let m = mBuf.contents().assumingMemoryBound(to: Float.self)
        let d = dBuf.contents().assumingMemoryBound(to: Float.self)
        let o = oBuf.contents().assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: numQHeads * headDim)
        for head in 0..<numQHeads {
            let base = head * chunks
            var mGlob = -Float.infinity
            for c in 0..<chunks { mGlob = max(mGlob, m[base + c]) }
            var denom: Float = 0
            for c in 0..<chunks where m[base + c] > -Float.infinity {
                denom += d[base + c] * expf(m[base + c] - mGlob)
            }
            for i in 0..<headDim {
                var acc: Float = 0
                for c in 0..<chunks where m[base + c] > -Float.infinity {
                    acc += o[(base + c) * headDim + i] * expf(m[base + c] - mGlob)
                }
                out[head * headDim + i] = acc / denom
            }
        }
        return out
    }
}
