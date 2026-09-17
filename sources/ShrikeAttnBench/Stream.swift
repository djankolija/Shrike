import Foundation
import Metal
import Shrike

struct StreamSwitches: Hashable {
    var headsPerSimdgroup: UInt32 = 4
    var noLoad = false
    var unroll2 = false
    var lazy = false

    func validate() throws {
        if ![2, 4, 8].contains(headsPerSimdgroup) {
            throw BenchError.usage("heads per simdgroup are 2, 4 or 8")
        }
    }
}

/// Shares the ladder's partial scratch so the same combine and hash apply.
final class StreamKernel {
    private let device: MTLDevice
    private let library: MTLLibrary
    private let scratch: LadderKernel
    private var pipelines: [StreamSwitches: MTLComputePipelineState] = [:]

    init(device: MTLDevice, scratch: LadderKernel) throws {
        guard let url = Bundle.module.url(forResource: "stream", withExtension: "metal",
                                          subdirectory: "Metal") else {
            throw BenchError.missingResource("Metal/stream.metal")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        self.device = device
        self.library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        self.scratch = scratch
    }

    func pipeline(_ sw: StreamSwitches) throws -> MTLComputePipelineState {
        if let pso = pipelines[sw] { return pso }
        let values = MTLFunctionConstantValues()
        var hpt = sw.headsPerSimdgroup, noLoad = sw.noLoad
        var unroll2 = sw.unroll2, lazy = sw.lazy
        values.setConstantValue(&hpt, type: .uint, index: 0)
        values.setConstantValue(&noLoad, type: .bool, index: 1)
        values.setConstantValue(&unroll2, type: .bool, index: 2)
        values.setConstantValue(&lazy, type: .bool, index: 3)
        let function = try library.makeFunction(name: "stream_partial", constantValues: values)
        let pso = try device.makeComputePipelineState(function: function)
        pipelines[sw] = pso
        return pso
    }

    func encode(commandBuffer: MTLCommandBuffer, rows: SyntheticRows,
                seqLen: Int, switches sw: StreamSwitches) throws {
        let pso = try pipeline(sw)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw BenchError.encoder
        }
        defer { encoder.endEncoding() }
        let chunks = LadderKernel.numChunks / Int(sw.headsPerSimdgroup)
        let chunkLen = (seqLen + chunks - 1) / chunks
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(rows.qBuf, offset: 0, index: 0)
        encoder.setBuffer(rows.keyView.buffer, offset: rows.keyView.offset, index: 1)
        encoder.setBuffer(rows.valueView.buffer, offset: rows.valueView.offset, index: 2)
        encoder.setBuffer(scratch.mBuf, offset: 0, index: 3)
        encoder.setBuffer(scratch.dBuf, offset: 0, index: 4)
        encoder.setBuffer(scratch.oBuf, offset: 0, index: 5)
        var sl = UInt32(seqLen), cl = UInt32(chunkLen), nc = UInt32(chunks), sc = rows.scale
        encoder.setBytes(&sl, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&cl, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&nc, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&sc, length: MemoryLayout<Float>.size, index: 9)
        encoder.dispatchThreadgroups(MTLSize(width: rows.numKVHeads * chunks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: LadderKernel.threadsPerGroup,
                                                                    height: 1, depth: 1))
    }
}
