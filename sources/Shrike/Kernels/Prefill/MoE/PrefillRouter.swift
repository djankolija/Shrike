import Foundation
import Metal

@frozen
public struct PrefillTokenExpertPair: Equatable, Sendable {
    public var token: UInt32
    public var expert: UInt32
    public var rank: UInt32
    public var weightBitsAndReserved: UInt32

    public init(token: UInt32, expert: UInt32, rank: UInt32, weight: Float16) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = UInt32(weight.bitPattern)
    }

    public init(token: UInt32, expert: UInt32, rank: UInt32, weightBitsAndReserved: UInt32) {
        self.token = token
        self.expert = expert
        self.rank = rank
        self.weightBitsAndReserved = weightBitsAndReserved
    }

    public var weight: Float16 {
        Float16(bitPattern: UInt16(truncatingIfNeeded: weightBitsAndReserved))
    }
}

enum PrefillRouterError: Error, CustomStringConvertible {
    case threadgroupTooNarrow(maxThreads: Int, experts: Int)

    var description: String {
        switch self {
        case .threadgroupTooNarrow(let maxThreads, let experts):
            return "prefill router: the tiled pipeline allows \(maxThreads) threads per threadgroup, fewer than the \(experts) experts"
        }
    }
}

final class PrefillRouter {
    /// `block` is one threadgroup per token (the kernel P0 shipped), kept as the
    /// reference and the A/B arm; `tiled` gives a threadgroup `tokenBlock`
    /// tokens against every expert (v12 P14) and is the default.
    enum Kind: String, Sendable {
        case block, tiled
    }

    static let maxTokenBlock = 24
    static let defaultTokenBlock = 12

    /// `SHRIKE_PREFILL_ROUTER=block` keeps the per-token kernel for the
    /// same-binary A/B; anything else takes `tiled`.
    static func environmentKind() -> Kind {
        ProcessInfo.processInfo.environment["SHRIKE_PREFILL_ROUTER"] == "block" ? .block : .tiled
    }

    /// `SHRIKE_PREFILL_ROUTER_TOKENS` (4...24, a multiple of 4) sets the tiled
    /// kernel's token block.
    static func environmentTokenBlock() -> Int {
        guard let raw = ProcessInfo.processInfo.environment["SHRIKE_PREFILL_ROUTER_TOKENS"],
              let block = Int(raw.trimmingCharacters(in: .whitespaces)) else {
            return defaultTokenBlock
        }
        return max(4, min(maxTokenBlock, block / 4 * 4))
    }

    private let pso: MTLComputePipelineState
    private let sigmoidRouterScores: Bool
    private let routedScalingFactor: Float
    let kind: Kind
    let tokenBlock: Int
    let weightBits: Int

    init(context: MetalContext, weightBits: Int = 8,
         sigmoidRouterScores: Bool = false,
         routedScalingFactor: Float = 1.0,
         kind: Kind = PrefillRouter.environmentKind(),
         tokenBlock: Int = PrefillRouter.environmentTokenBlock()) throws {
        precondition([4, 8].contains(weightBits))
        precondition((4...PrefillRouter.maxTokenBlock).contains(tokenBlock) && tokenBlock % 4 == 0)
        self.sigmoidRouterScores = sigmoidRouterScores
        self.routedScalingFactor = routedScalingFactor
        self.kind = kind
        self.tokenBlock = tokenBlock
        self.weightBits = weightBits
        let name: String
        switch (kind, sigmoidRouterScores) {
        case (.block, false): name = "prefill_router_block"
        case (.block, true): name = "prefill_router_block_sigmoid"
        case (.tiled, false): name = "prefill_router_block_tiled"
        case (.tiled, true): name = "prefill_router_block_tiled_sigmoid"
        }
        var constants = [MetalFunctionConstant(index: 79, value: .uint32(UInt32(weightBits)))]
        if kind == .tiled {
            constants.append(MetalFunctionConstant(index: 123, value: .uint32(UInt32(tokenBlock))))
        }
        self.pso = try context.pipeline(name, constants: constants)
    }

    var description: String {
        kind == .tiled ? "tiled tokens=\(tokenBlock) bits=\(weightBits)" : "block bits=\(weightBits)"
    }

    func threadgroupWidth(numExperts: Int) -> Int {
        min(max(numExperts, 32), pso.maxTotalThreadsPerThreadgroup)
    }

    func threadgroups(queryCount: Int) -> Int {
        kind == .tiled ? (queryCount + tokenBlock - 1) / tokenBlock : queryCount
    }

    func encodeBlock(commandBuffer: MTLCommandBuffer,
                                  weights: MTLBuffer,
                                  weightsOffset: Int = 0,
                                  scales: MTLBuffer,
                                  scalesOffset: Int = 0,
                                  biases: MTLBuffer,
                                  biasesOffset: Int = 0,
                                  hidden: MTLBuffer,
                                  hiddenOffset: Int = 0,
                                  effectiveScale: MTLBuffer,
                                  effectiveScaleOffset: Int = 0,
                                  perExpertScale: MTLBuffer,
                                  perExpertScaleOffset: Int = 0,
                                  logitBias: MTLBuffer,
                                  logitBiasOffset: Int = 0,
                                  outIndices: MTLBuffer,
                                  outIndicesOffset: Int = 0,
                                  outWeights: MTLBuffer,
                                  outWeightsOffset: Int = 0,
                                  queryCount: UInt32,
                                  numExperts: UInt32,
                                  d: UInt32,
                                  topK: UInt32,
                                  hiddenStrideElements: UInt32) throws {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(numExperts <= 256, "numExperts > 256 is not supported")
        precondition(topK > 0 && topK <= 64, "topK must be in 1...64")
        precondition(d % UInt32(Quantization.groupSize) == 0,
                     "D must be a multiple of \(Quantization.groupSize)")
        precondition(hiddenStrideElements >= d, "hidden stride is too small")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 3)
        enc.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        enc.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 5)
        enc.setBuffer(outIndices, offset: outIndicesOffset, index: 6)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 7)
        var tVar = queryCount
        var neVar = numExperts
        var dVar = d
        var topKVar = topK
        var strideVar = hiddenStrideElements
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&neVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&strideVar, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBuffer(logitBias, offset: logitBiasOffset, index: 13)
        if sigmoidRouterScores {
            var scaling = routedScalingFactor
            enc.setBytes(&scaling, length: MemoryLayout<Float>.size, index: 14)
        }
        let tgWidth = threadgroupWidth(numExperts: Int(numExperts))
        if kind == .tiled {
            guard tgWidth >= Int(numExperts) else {
                enc.endEncoding()
                throw PrefillRouterError.threadgroupTooNarrow(maxThreads: tgWidth,
                                                              experts: Int(numExperts))
            }
            let floats = MemoryLayout<Float>.stride
            var vectorLoads: UInt32 = weightsOffset.isMultiple(of: 16) ? 1 : 0
            enc.setBytes(&vectorLoads, length: MemoryLayout<UInt32>.size, index: 15)
            enc.setThreadgroupMemoryLength(
                Self.roundedThreadgroupBytes(tokenBlock * Self.maxExperts * floats),
                index: 0)
            enc.setThreadgroupMemoryLength(
                Self.roundedThreadgroupBytes((tokenBlock * Quantization.groupSize + tokenBlock) * floats),
                index: 1)
        }
        enc.dispatchThreadgroups(MTLSize(width: threadgroups(queryCount: Int(queryCount)),
                                         height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
    }

    private static let maxExperts = 256

    private static func roundedThreadgroupBytes(_ bytes: Int) -> Int {
        (bytes + 15) / 16 * 16
    }

    static func makeTokenExpertPairs(indices: [UInt32],
                                            weights: [Float16],
                                            queryCount: Int,
                                            topK: Int) -> [PrefillTokenExpertPair] {
        precondition(queryCount >= 0, "queryCount must be non-negative")
        precondition(topK >= 0, "topK must be non-negative")
        precondition(indices.count == queryCount * topK, "indices count mismatch")
        precondition(weights.count == queryCount * topK, "weights count mismatch")
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(indices.count)
        for token in 0..<queryCount {
            for rank in 0..<topK {
                let i = token * topK + rank
                pairs.append(PrefillTokenExpertPair(token: UInt32(token),
                                                    expert: indices[i],
                                                    rank: UInt32(rank),
                                                    weight: weights[i]))
            }
        }
        return pairs
    }
}
