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
    /// One threadgroup owns `tokenBlock` tokens against every expert (v12 P14).
    static let tokenBlock = 12

    private let pso: MTLComputePipelineState
    private let sigmoidRouterScores: Bool
    private let routedScalingFactor: Float
    let weightBits: Int

    init(context: MetalContext, weightBits: Int = 8,
         sigmoidRouterScores: Bool = false,
         routedScalingFactor: Float = 1.0) throws {
        precondition([4, 8].contains(weightBits))
        self.sigmoidRouterScores = sigmoidRouterScores
        self.routedScalingFactor = routedScalingFactor
        self.weightBits = weightBits
        let name = sigmoidRouterScores
            ? "prefill_router_block_tiled_sigmoid"
            : "prefill_router_block_tiled"
        let constants = [
            MetalFunctionConstant(index: 79, value: .uint32(UInt32(weightBits))),
            MetalFunctionConstant(index: 123, value: .uint32(UInt32(Self.tokenBlock))),
        ]
        self.pso = try context.pipeline(name, constants: constants)
    }

    func threadgroupWidth(numExperts: Int) -> Int {
        min(max(numExperts, 32), pso.maxTotalThreadsPerThreadgroup)
    }

    func threadgroups(queryCount: Int) -> Int {
        (queryCount + Self.tokenBlock - 1) / Self.tokenBlock
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
        guard tgWidth >= Int(numExperts) else {
            enc.endEncoding()
            throw PrefillRouterError.threadgroupTooNarrow(maxThreads: tgWidth,
                                                          experts: Int(numExperts))
        }
        let floats = MemoryLayout<Float>.stride
        let tokenBlock = Self.tokenBlock
        var vectorLoads: UInt32 = weightsOffset.isMultiple(of: 16) ? 1 : 0
        enc.setBytes(&vectorLoads, length: MemoryLayout<UInt32>.size, index: 15)
        enc.setThreadgroupMemoryLength(
            Self.roundedThreadgroupBytes(tokenBlock * Self.maxExperts * floats),
            index: 0)
        enc.setThreadgroupMemoryLength(
            Self.roundedThreadgroupBytes((tokenBlock * Quantization.groupSize + tokenBlock) * floats),
            index: 1)
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
