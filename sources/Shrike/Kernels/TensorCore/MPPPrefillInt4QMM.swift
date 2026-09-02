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
    static let tileK = Quantization.groupSize
    /// `SHRIKE_MPP_TILE_N` (32|64) and `SHRIKE_MPP_DEQUANT_BUFFERS` (1|2) name
    /// the variant; anything unrecognised keeps the measured choice.
    static let tileVariant: TileVariant = {
        let environment = ProcessInfo.processInfo.environment
        return TileVariant(tileN: environment["SHRIKE_MPP_TILE_N"],
                           buffers: environment["SHRIKE_MPP_DEQUANT_BUFFERS"],
                           fallback: .n32b1) ?? .n32b1
    }()
    let variant: TileVariant
    var tileN: Int { variant.tileN }
    /// A wave is at most 2,048 staging rows, twice the runtime's staging block.
    static let groupedMaxRowTiles = 32

    private var pipeline: MTLComputePipelineState?
    private var groupedPipeline: MTLComputePipelineState?
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
         variant: TileVariant = MPPPrefillInt4QMM.tileVariant) {
        precondition([4, 8].contains(weightBits))
        self.weightBits = weightBits
        self.variant = variant
        let constants = MTLFunctionConstantValues()
        var bits = UInt32(weightBits)
        constants.setConstantValue(&bits, type: .uint, index: 78)
        var library: MTLLibrary?
        do {
            library = try Self.compileTensorOpsLibrary(device: context.device)
            let function = try library!.makeFunction(
                name: variant.kernelName,
                constantValues: constants)
            self.pipeline = try context.device.makeComputePipelineState(function: function)
            self.unavailableReason = ""
        } catch {
            // Capability probe: this path is optional on non-Apple10 hardware,
            // so init stays non-throwing. Record the reason so a later
            // explicit request can throw it (K6).
            self.pipeline = nil
            self.unavailableReason = "\(error)"
        }
        do {
            guard let library else {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(reason: unavailableReason)
            }
            let function = try library.makeFunction(
                name: variant.groupedKernelName,
                constantValues: constants)
            self.groupedPipeline = try context.device.makeComputePipelineState(function: function)
            self.groupedArgumentEncodedLength = function.makeArgumentEncoder(bufferIndex: 0).encodedLength
            self.groupedUnavailableReason = ""
        } catch {
            self.groupedPipeline = nil
            self.groupedArgumentEncodedLength = 0
            self.groupedUnavailableReason = "\(error)"
        }
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
        guard let pipeline else {
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
                       required: Bool = true) throws -> Path {
        let rowTiles = paddedRows / Self.tileM
        let tileM = UInt32(Self.tileM)
        guard paddedRows > 0,
              paddedRows.isMultiple(of: Self.tileM),
              rowTiles <= Self.groupedMaxRowTiles,
              rowTileBlock.count == rowTiles,
              !blocks.isEmpty,
              blocks.count <= Self.groupedMaxRowTiles,
              rowTileBlock.allSatisfy({ Int($0) < blocks.count }),
              blocks.allSatisfy({ block in
                  Int(block.slot) < expertViews.count
                      && block.rows > 0
                      && block.stagingRow.isMultiple(of: tileM)
                      && block.rowTileStart == block.stagingRow / tileM
                      && Int(block.stagingRow) + Int(block.rows) <= paddedRows
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
                    "grouped paddedRows=\(paddedRows) blocks=\(blocks.count) rowTiles=\(rowTileBlock.count)"
                        + " n=\(n) k=\(k) offsets \(weightsOffset)/\(scalesOffset)/\(biasesOffset)/\(xOffset)/\(yOffset)")
            }
            return .unavailable
        }
        guard let groupedPipeline else {
            if required {
                throw MPPPrefillInt4QMMError.pipelineUnavailable(
                    reason: groupedUnavailableReason.isEmpty
                        ? "MPP grouped pipeline failed to compile"
                        : groupedUnavailableReason)
            }
            return .unavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
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
        for view in expertViews {
            encoder.useResource(view.buffer, usage: .read)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: (n + tileN - 1) / tileN,
                    height: rowTiles,
                    depth: 1),
            threadsPerThreadgroup: MTLSize(width: groupedPipeline.threadExecutionWidth * 4,
                                           height: 1,
                                           depth: 1))
        encoder.endEncoding()
        return .affineGroupedF16
    }

    private static func compileTensorOpsLibrary(device: MTLDevice) throws -> MTLLibrary {
        try MetalContext.moduleLibrary(device: device, module: "tensorops")
    }
}

extension MPPPrefillInt4QMM {
    /// `n<tileN>b<buffers>`: the N width of one weight tile and how many
    /// weight tiles the threadgroup alternates between. `n32b1` is the
    /// kernel P6 shipped and keeps its bare names; the numbers restate the
    /// Metal instantiations' template arguments.
    enum TileVariant: String, CaseIterable, Sendable {
        case n32b1, n32b2, n64b1, n64b2

        var tileN: Int {
            switch self {
            case .n32b1, .n32b2: 32
            case .n64b1, .n64b2: 64
            }
        }
        var dequantBuffers: Int {
            switch self {
            case .n32b1, .n64b1: 1
            case .n32b2, .n64b2: 2
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

        /// A missing value takes the fallback's; an unrecognised one rejects
        /// the whole selection so a typo cannot pick a variant by accident.
        init?(tileN: String?, buffers: String?, fallback: TileVariant) {
            let width: Int
            switch tileN {
            case nil: width = fallback.tileN
            case "32"?: width = 32
            case "64"?: width = 64
            default: return nil
            }
            let count: Int
            switch buffers {
            case nil: count = fallback.dequantBuffers
            case "1"?: count = 1
            case "2"?: count = 2
            default: return nil
            }
            self.init(rawValue: "n\(width)b\(count)")
        }
    }
}
