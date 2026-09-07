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
    /// The grouped kernel's row tile: 64 is every variant's; 32 exists for the
    /// default variant only and is chosen per dispatch.
    enum GroupedRowTile: Int, Sendable {
        case m64 = 64
        case m32 = 32
    }
    /// The matrix path's admission unit (`PrefillSharedExpert` and
    /// `PrefillGroupedRoutedMoE` refuse a `d`/`intermediate` that is not a
    /// multiple of it), not the kernel's K tile: a wide-K variant carries the
    /// narrower rungs and picks per dispatch.
    static let tileK = Quantization.groupSize
    /// `SHRIKE_MPP_TILE_N` (32|64), `SHRIKE_MPP_TILE_K` (64|128|256) and
    /// `SHRIKE_MPP_DEQUANT_BUFFERS` (1|2) name the variant; anything
    /// unrecognised — including a wide K with N 64 or two buffers, which
    /// is what `SHRIKE_MPP_TILE_N=64` alone asks for against this default —
    /// keeps the measured choice.
    static let tileVariant: TileVariant = {
        let environment = ProcessInfo.processInfo.environment
        return TileVariant(tileN: environment["SHRIKE_MPP_TILE_N"],
                           tileK: environment["SHRIKE_MPP_TILE_K"],
                           buffers: environment["SHRIKE_MPP_DEQUANT_BUFFERS"],
                           fallback: .n32k256b1) ?? .n32k256b1
    }()
    /// `SHRIKE_MPP_WEIGHT_LOADS` (byte|vector) is a per-dispatch override;
    /// the vector body runs only when the weight base is 16-byte aligned.
    static let weightLoads: WeightLoads =
        ProcessInfo.processInfo.environment["SHRIKE_MPP_WEIGHT_LOADS"]
            .flatMap(WeightLoads.init(rawValue:)) ?? .vector
    let variant: TileVariant
    let weightLoads: WeightLoads
    var tileN: Int { variant.tileN }
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

    init(context: MetalContext, weightBits: Int = 4,
         variant: TileVariant = MPPPrefillInt4QMM.tileVariant,
         weightLoads: WeightLoads = MPPPrefillInt4QMM.weightLoads) {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.variant = variant
        self.weightLoads = weightLoads
        let constants = MTLFunctionConstantValues()
        var bits = UInt32(weightBits)
        constants.setConstantValue(&bits, type: .uint, index: 78)
        var library: MTLLibrary?
        do {
            library = try Self.compileTensorOpsLibrary(device: context.device)
            self.pipeline = try Self.makePipeline(
                library: library, name: variant.kernelName, constants: constants).pipeline
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
                library: library, name: variant.groupedKernelName, constants: constants)
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
                library: library, name: variant.groupedKernelName + "_m32", constants: constants).pipeline
            self.groupedM32UnavailableReason = ""
        } catch {
            self.groupedPipelineM32 = nil
            self.groupedM32UnavailableReason = "\(error)"
        }
        self.narrowRungs = variant.narrowerRungs.map { rung in
            Rung(tileK: rung.tileK,
                 pipeline: try? Self.makePipeline(
                     library: library, name: rung.kernelName, constants: constants).pipeline,
                 grouped: try? Self.makePipeline(
                     library: library, name: rung.groupedKernelName, constants: constants).pipeline)
        }
    }

    /// `tilesPerRow` is `K / TILE_K`: a tile wider than `k` divides would drop the K tail.
    private func pipeline(forK k: Int) -> MTLComputePipelineState? {
        if k.isMultiple(of: variant.tileK) { return pipeline }
        return narrowRungs.first { k.isMultiple(of: $0.tileK) && $0.pipeline != nil }?.pipeline
    }

    private func groupedPipeline(forK k: Int, rowTile: GroupedRowTile) -> MTLComputePipelineState? {
        switch rowTile {
        case .m64:
            if k.isMultiple(of: variant.tileK) { return groupedPipeline }
            return narrowRungs.first { k.isMultiple(of: $0.tileK) && $0.grouped != nil }?.grouped
        case .m32:
            return k.isMultiple(of: variant.tileK) ? groupedPipelineM32 : nil
        }
    }

    /// The 32-row grouped instantiation exists for the variant's own K tile
    /// only; a ragged K has no narrow rung at 32 rows.
    func groupedRowTile32Available(forK k: Int) -> Bool {
        groupedPipelineM32 != nil && k.isMultiple(of: variant.tileK)
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
            MTLSize(width: (n + tileN - 1) / tileN,
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
                        ? "no 32-row grouped instantiation for \(variant) at K \(k)"
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
            MTLSize(width: (n + tileN - 1) / tileN,
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
        weightLoads == .vector && weightsOffset.isMultiple(of: 16) ? 1 : 0
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }
}

extension MPPPrefillInt4QMM {
    /// `n<tileN>[k<tileK>]b<buffers>`: the N and K widths of one weight tile
    /// and how many weight tiles the threadgroup alternates between. `n32b1`
    /// is the kernel P6 shipped and keeps its bare names; the numbers restate
    /// the Metal instantiations' template arguments.
    enum TileVariant: String, CaseIterable, Sendable {
        case n32b1, n32b2, n64b1, n64b2, n32k128b1, n32k256b1

        var tileN: Int {
            switch self {
            case .n32b1, .n32b2, .n32k128b1, .n32k256b1: 32
            case .n64b1, .n64b2: 64
            }
        }
        var tileK: Int {
            switch self {
            case .n32b1, .n32b2, .n64b1, .n64b2: 64
            case .n32k128b1: 128
            case .n32k256b1: 256
            }
        }
        var dequantBuffers: Int {
            switch self {
            case .n32b1, .n64b1, .n32k128b1, .n32k256b1: 1
            case .n32b2, .n64b2: 2
            }
        }
        /// Widest first; gpt-oss's K 2880 and Kimi's 128-wide low-rank legs need them.
        var narrowerRungs: [TileVariant] {
            switch self {
            case .n32k256b1: [.n32k128b1, .n32b1]
            case .n32k128b1: [.n32b1]
            case .n32b1, .n32b2, .n64b1, .n64b2: []
            }
        }
        var kernelName: String {
            self == .n32b1
                ? "mpp_prefill_affine_threadgroup_f16"
                : "mpp_prefill_affine_threadgroup_f16_\(rawValue)"
        }
        var groupedKernelName: String {
            self == .n32b1
                ? "mpp_prefill_affine_grouped_f16"
                : "mpp_prefill_affine_grouped_f16_\(rawValue)"
        }

        /// A missing value takes the fallback's; an unrecognised one — or a
        /// 128- or 256-wide K tile with anything but N 32 / one buffer — rejects
        /// the whole selection so a typo cannot pick a variant by accident.
        init?(tileN: String?, tileK: String?, buffers: String?, fallback: TileVariant) {
            let width: Int
            switch tileN {
            case nil: width = fallback.tileN
            case "32"?: width = 32
            case "64"?: width = 64
            default: return nil
            }
            let depth: Int
            switch tileK {
            case nil: depth = fallback.tileK
            case "64"?: depth = 64
            case "128"?: depth = 128
            case "256"?: depth = 256
            default: return nil
            }
            let count: Int
            switch buffers {
            case nil: count = fallback.dequantBuffers
            case "1"?: count = 1
            case "2"?: count = 2
            default: return nil
            }
            if depth != 64 {
                guard width == 32, count == 1 else { return nil }
                self = depth == 128 ? .n32k128b1 : .n32k256b1
                return
            }
            self.init(rawValue: "n\(width)b\(count)")
        }
    }

    enum WeightLoads: String, CaseIterable, Sendable {
        case byte, vector
    }
}
