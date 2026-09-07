import Foundation
import Metal

enum MPPPrefillInt4QMMError: Error, CustomStringConvertible {
    case pipelineUnavailable(reason: String)
    case invalidArguments(String)

    public var description: String {
        switch self {
        case .pipelineUnavailable(let reason):
            return "MPP prefill QMM path unavailable: \(reason)"
        case .invalidArguments(let detail):
            return "MPP prefill QMM invalid arguments: \(detail)"
        }
    }
}

final class MPPPrefillInt4QMM {
    enum Path: String, Sendable {
        case affineThreadgroupF16 = "affine-threadgroup-f16"
        case affineGroupedF16 = "affine-grouped-f16"
        case unavailable
    }

    static let tileM = 64
    static let tileN = 32
    /// The grouped kernel's row tile: 32 exists for the kernel's own K tile
    /// only and is chosen per dispatch.
    enum GroupedRowTile: Int, Sendable {
        case m64 = 64
        case m32 = 32
    }
    /// The matrix path's admission unit (`PrefillSharedExpert` and
    /// `PrefillGroupedRoutedMoE` refuse a `d`/`intermediate` that is not a
    /// multiple of it), not the kernel's K tile: the kernel carries the
    /// narrower rungs and picks per dispatch.
    static let tileK = Quantization.groupSize
    /// `n32k256b1`: one 32 × 256 weight tile per threadgroup, dequantized
    /// into one buffer.
    static let kernelTileK = 256
    static let kernelName = "mpp_prefill_affine_threadgroup_f16_n32k256b1"
    static let groupedKernelName = "mpp_prefill_affine_grouped_f16_n32k256b1"
    /// Widest first; gpt-oss's K 2880 and Kimi's 128-wide low-rank legs need them.
    private static let narrowerRungs: [(tileK: Int, kernelName: String, groupedKernelName: String)] = [
        (128, "mpp_prefill_affine_threadgroup_f16_n32k128b1", "mpp_prefill_affine_grouped_f16_n32k128b1"),
        (64, "mpp_prefill_affine_threadgroup_f16", "mpp_prefill_affine_grouped_f16"),
    ]
    /// Bounds a grouped dispatch's grid height and its block count: 64 tiles
    /// is 2,048 rows at the 32-row tile, above the 1,024-row block the caller's
    /// staging loops over.
    static let groupedMaxRowTiles = 64

    private struct Rung {
        let tileK: Int
        let pipeline: MTLComputePipelineState?
        let grouped: MTLComputePipelineState?
    }

    private var pipeline: MTLComputePipelineState?
    private var groupedPipeline: MTLComputePipelineState?
    private let groupedPipelineM32: MTLComputePipelineState?
    private let groupedM32UnavailableReason: String
    private let narrowRungs: [Rung]
    /// K6: the compile failure reason, recorded once at init so an explicit
    /// MPP request can throw the real cause instead of silently degrading.
    private let unavailableReason: String
    private let groupedUnavailableReason: String
    /// The grouped kernel reads an argument buffer encoded by the prefill
    /// module's encoder, so the caller checks the two lengths agree.
    let groupedArgumentEncodedLength: Int

    /// The bit width baked into the pipeline (function constant 78); a caller
    /// reusing this instance for another tensor must match it.
    let weightBits: Int

    init(context: MetalContext, weightBits: Int = 4) {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        let constants = MTLFunctionConstantValues()
        var bits = UInt32(weightBits)
        constants.setConstantValue(&bits, type: .uint, index: 78)
        var library: MTLLibrary?
        do {
            library = try Self.compileTensorOpsLibrary(device: context.device)
            self.pipeline = try Self.makePipeline(
                library: library, name: Self.kernelName, constants: constants).pipeline
            self.unavailableReason = ""
        } catch {
            // Capability probe: this path is optional on non-Apple10 hardware,
            // so init stays non-throwing. Record the reason so a later
            // explicit request can throw it (K6).
            self.pipeline = nil
            self.unavailableReason = "\(error)"
        }
        do {
            let grouped = try Self.makePipeline(
                library: library, name: Self.groupedKernelName, constants: constants)
            self.groupedPipeline = grouped.pipeline
            self.groupedArgumentEncodedLength =
                grouped.function.makeArgumentEncoder(bufferIndex: 0).encodedLength
            self.groupedUnavailableReason = ""
        } catch {
            self.groupedPipeline = nil
            self.groupedArgumentEncodedLength = 0
            self.groupedUnavailableReason = "\(error)"
        }
        do {
            self.groupedPipelineM32 = try Self.makePipeline(
                library: library, name: Self.groupedKernelName + "_m32", constants: constants).pipeline
            self.groupedM32UnavailableReason = ""
        } catch {
            self.groupedPipelineM32 = nil
            self.groupedM32UnavailableReason = "\(error)"
        }
        self.narrowRungs = Self.narrowerRungs.map { rung in
            Rung(tileK: rung.tileK,
                 pipeline: try? Self.makePipeline(
                     library: library, name: rung.kernelName, constants: constants).pipeline,
                 grouped: try? Self.makePipeline(
                     library: library, name: rung.groupedKernelName, constants: constants).pipeline)
        }
    }

    /// `tilesPerRow` is `K / TILE_K`: a tile wider than `k` divides would drop the K tail.
    private func pipeline(forK k: Int) -> MTLComputePipelineState? {
        if k.isMultiple(of: Self.kernelTileK) { return pipeline }
        return narrowRungs.first { k.isMultiple(of: $0.tileK) && $0.pipeline != nil }?.pipeline
    }

    private func groupedPipeline(forK k: Int, rowTile: GroupedRowTile) -> MTLComputePipelineState? {
        switch rowTile {
        case .m64:
            if k.isMultiple(of: Self.kernelTileK) { return groupedPipeline }
            return narrowRungs.first { k.isMultiple(of: $0.tileK) && $0.grouped != nil }?.grouped
        case .m32:
            return k.isMultiple(of: Self.kernelTileK) ? groupedPipelineM32 : nil
        }
    }

    /// The 32-row grouped instantiation exists for the kernel's own K tile
    /// only; a ragged K has no narrow rung at 32 rows.
    func groupedRowTile32Available(forK k: Int) -> Bool {
        groupedPipelineM32 != nil && k.isMultiple(of: Self.kernelTileK)
    }

    private static func makePipeline(library: MTLLibrary?,
                                     name: String,
                                     constants: MTLFunctionConstantValues) throws
        -> (pipeline: MTLComputePipelineState, function: MTLFunction) {
        guard let library else {
            throw MPPPrefillInt4QMMError.pipelineUnavailable(reason: "tensorops library failed to compile")
        }
        let function = try library.makeFunction(name: name, constantValues: constants)
        return (try library.device.makeComputePipelineState(function: function), function)
    }

    var isAvailable: Bool {
        pipeline != nil
    }

    var groupedAvailable: Bool {
        groupedPipeline != nil
    }

    /// `required: true` makes an unavailable path a thrown error instead of a
    /// silent `.unavailable` fallback — use it when the caller explicitly
    /// requests the MPP path. Auto-selected callers keep `required: false`
    /// and check the returned `Path`.
    @discardableResult
    func encode(commandBuffer: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       biases: MTLBuffer, biasesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       m: Int,
                       n: Int,
                       k: Int,
                       required: Bool = false) throws -> Path {
        guard m > 0,
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride) else {
            if required {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "m=\(m) n=\(n) k=\(k) offsets \(weightsOffset)/\(scalesOffset)/\(biasesOffset)/\(xOffset)/\(yOffset)")
            }
            return .unavailable
        }
        guard let pipeline = pipeline(forK: k) else {
            if required {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(
                    reason: unavailableReason.isEmpty
                        ? "MPP pipeline failed to compile"
                        : unavailableReason)
            }
            return .unavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            if required { throw MetalError.commandEncoderFailed }
            return .unavailable
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        var loads = vectorLoadsFlag(weightsOffset: weightsOffset)
        encoder.setBytes(&loads, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: (m + Self.tileM - 1) / Self.tileM,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return .affineThreadgroupF16
    }

    /// The block tables go inline with the encoder so a later wave cannot
    /// overwrite them under a command buffer that has not run yet.
    @discardableResult
    func encodeGrouped(commandBuffer: MTLCommandBuffer,
                       experts: MTLBuffer,
                       expertViews: [TensorView],
                       blocks: [PrefillRoutedExpertBlock],
                       rowTileBlock: [UInt32],
                       weightsOffset: Int,
                       scalesOffset: Int,
                       biasesOffset: Int,
                       x: MTLBuffer, xOffset: Int = 0,
                       y: MTLBuffer, yOffset: Int = 0,
                       paddedRows: Int,
                       n: Int,
                       k: Int,
                       rowTile: GroupedRowTile = .m64,
                       regionOrigin: Int = 0,
                       regionRows: Int? = nil,
                       required: Bool = true,
                       encoder existing: MTLComputeCommandEncoder? = nil) throws -> Path {
        let regionRows = regionRows ?? paddedRows
        let rowTiles = regionRows / rowTile.rawValue
        let tileM = UInt32(rowTile.rawValue)
        guard paddedRows > 0,
              regionRows > 0,
              regionOrigin >= 0,
              regionOrigin + regionRows <= paddedRows,
              regionRows.isMultiple(of: rowTile.rawValue),
              rowTiles <= Self.groupedMaxRowTiles,
              rowTileBlock.count == rowTiles,
              !blocks.isEmpty,
              blocks.count <= Self.groupedMaxRowTiles,
              rowTileBlock.allSatisfy({ Int($0) < blocks.count }),
              blocks.allSatisfy({ block in
                  Int(block.slot) < expertViews.count
                      && block.rows > 0
                      && Int(block.stagingRow) >= regionOrigin
                      && (Int(block.stagingRow) - regionOrigin).isMultiple(of: Int(tileM))
                      && Int(block.rowTileStart) == (Int(block.stagingRow) - regionOrigin) / Int(tileM)
                      && Int(block.stagingRow) + Int(block.rows) <= regionOrigin + regionRows
              }),
              n > 0,
              k > 0,
              k.isMultiple(of: Self.tileK),
              weightsOffset >= 0,
              scalesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              biasesOffset.isMultiple(of: MemoryLayout<UInt16>.stride),
              xOffset.isMultiple(of: MemoryLayout<Float16>.stride),
              yOffset.isMultiple(of: MemoryLayout<Float16>.stride) else {
            if required {
                throw MPPPrefillInt4QMMError.invalidArguments(
                    "grouped paddedRows=\(paddedRows) region=\(regionOrigin)+\(regionRows)"
                        + " tile=\(rowTile.rawValue) blocks=\(blocks.count) rowTiles=\(rowTileBlock.count)"
                        + " n=\(n) k=\(k) offsets \(weightsOffset)/\(scalesOffset)/\(biasesOffset)/\(xOffset)/\(yOffset)")
            }
            return .unavailable
        }
        guard let groupedPipeline = groupedPipeline(forK: k, rowTile: rowTile) else {
            if required {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(
                    reason: rowTile == .m32
                        ? "no 32-row grouped instantiation at K \(k)"
                            + (groupedM32UnavailableReason.isEmpty ? "" : ": \(groupedM32UnavailableReason)")
                        : groupedUnavailableReason.isEmpty
                        ? "MPP grouped pipeline failed to compile"
                        : groupedUnavailableReason)
            }
            return .unavailable
        }
        guard let encoder = existing ?? commandBuffer.makeComputeCommandEncoder() else {
            if required { throw MetalError.commandEncoderFailed }
            return .unavailable
        }

        encoder.setComputePipelineState(groupedPipeline)
        encoder.setBuffer(experts, offset: 0, index: 0)
        blocks.withUnsafeBufferPointer { table in
            encoder.setBytes(table.baseAddress!,
                             length: table.count * MemoryLayout<PrefillRoutedExpertBlock>.stride,
                             index: 1)
        }
        rowTileBlock.withUnsafeBufferPointer { table in
            encoder.setBytes(table.baseAddress!,
                             length: table.count * MemoryLayout<UInt32>.stride,
                             index: 2)
        }
        try bindGroupedArguments(encoder: encoder, expertViews: expertViews,
                                 x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                                 n: n, k: k, weightsOffset: weightsOffset,
                                 scalesOffset: scalesOffset, biasesOffset: biasesOffset,
                                 paddedRows: paddedRows)
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + Self.tileN - 1) / Self.tileN,
                    height: rowTiles,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: groupedPipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        if existing == nil { encoder.endEncoding() }
        return .affineGroupedF16
    }

    private func bindGroupedArguments(encoder: MTLComputeCommandEncoder,
                                      expertViews: [TensorView],
                                      x: MTLBuffer, xOffset: Int,
                                      y: MTLBuffer, yOffset: Int,
                                      n: Int, k: Int,
                                      weightsOffset: Int, scalesOffset: Int, biasesOffset: Int,
                                      paddedRows: Int) throws {
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        var wOff = UInt32(weightsOffset)
        var sOff = UInt32(scalesOffset)
        var bOff = UInt32(biasesOffset)
        var mValue = UInt32(paddedRows)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&wOff, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&sOff, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.setBytes(&bOff, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 10)
        var loads = expertViews.allSatisfy({ $0.offset.isMultiple(of: 16) })
            ? vectorLoadsFlag(weightsOffset: weightsOffset) : 0
        encoder.setBytes(&loads, length: MemoryLayout<UInt32>.size, index: 11)
        for view in expertViews {
            encoder.useResource(view.buffer, usage: .read)
        }
    }

    /// The row stride is already a multiple of 16 (from the `k % 64` guard), so
    /// the base offset alone decides whether the vector body may run.
    private func vectorLoadsFlag(weightsOffset: Int) -> UInt32 {
        weightsOffset.isMultiple(of: 16) ? 1 : 0
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }
}
