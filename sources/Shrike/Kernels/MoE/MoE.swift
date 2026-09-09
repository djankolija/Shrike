import Foundation
import Metal

@frozen
public struct MoEExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var gateBOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var upBOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32
    public var downBOff: UInt32
    /// Additive per-expert bias offsets (gpt-oss); zero when the family has
    /// none. Must mirror `ExpertOffsets` in moe.metal field-for-field.
    public var gateABOff: UInt32
    public var upABOff: UInt32
    public var downABOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32, gateBOff: UInt32,
                upWOff: UInt32, upSOff: UInt32, upBOff: UInt32,
                downWOff: UInt32, downSOff: UInt32, downBOff: UInt32,
                gateABOff: UInt32 = 0, upABOff: UInt32 = 0,
                downABOff: UInt32 = 0) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.gateBOff = gateBOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.upBOff = upBOff
        self.downWOff = downWOff
        self.downSOff = downSOff
        self.downBOff = downBOff
        self.gateABOff = gateABOff
        self.upABOff = upABOff
        self.downABOff = downABOff
    }
}

final class MoE {
    static let maxStreamedExperts = 8
    /// Ceiling for the phase-1 threadgroup-staged hidden vector; must match
    /// `kMoEXMaxD` in moe.metal.
    static let maxStagedHiddenD: UInt32 = 2880

    private let realDecodeD: UInt32
    /// Kimi router: selection on sigmoid(logit) + correction bias, weights
    /// renormalized original sigmoid scores × `routedScalingFactor`.
    private let sigmoidRouterScores: Bool
    private let routedScalingFactor: Float
    private let realDecodeF: UInt32
    private let realDecodeTopK: UInt32
    private let realDecodeNumExperts: UInt32

    private let routerGemvPSO: MTLComputePipelineState
    private let routerGemvSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PSO: MTLComputePipelineState
    private let routerSelectK8SpecializedPSO: MTLComputePipelineState
    private let routerGemvPairPSO: MTLComputePipelineState
    private let routerGemvPairSpecializedPSO: MTLComputePipelineState
    private let routerSelectK8PairPSO: MTLComputePipelineState
    private let routerSelectK8PairSpecializedPSO: MTLComputePipelineState
    private let routerLogitsPair: MTLBuffer
    private let residencyClassifySpecPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1U16PSO: MTLComputePipelineState
    private let specPhase1PSO: MTLComputePipelineState
    private let specPhase1SpecializedPSO: MTLComputePipelineState
    private let specPhase2PSO: MTLComputePipelineState
    private let specPhase2SpecializedPSO: MTLComputePipelineState
    private let phase1U16SpecializedPSO: MTLComputePipelineState
    private let phase1SubsetU16PSO: MTLComputePipelineState
    private let phase1SubsetU16SpecializedPSO: MTLComputePipelineState
    private let phase2ReduceK8PSO: MTLComputePipelineState
    private let phase2ReduceK8SpecializedPSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer
    private let alwaysReadyIOStatus: MTLBuffer

    /// `specializedD`/`specializedF`/`specializedNumExperts` describe the
    /// production shape this instance specializes for (the specialized
    /// defaults 2816/704/128 predate Qwen-only support; Qwen 3.6 passes
    /// 2048/512/256). `siluActivation` selects the expert FFN activation
    /// (false = gelu_pytorch_tanh, true = silu).
    init(context: MetalContext,
         siluActivation: Bool = false,
         routedWeightBits: Int = 4,
         routerWeightBits: Int = 8,
         eventGatedIO: Bool = false,
         specializedD: UInt32 = 2816,
         specializedF: UInt32 = 704,
         specializedNumExperts: UInt32 = 128,
         specializedTopK: UInt32 = 8,
         expertAdditiveBiases: Bool = false,
         clampedSwiGLU: Bool = false,
         sigmoidRouterScores: Bool = false,
         routedScalingFactor: Float = 1.0) throws {
        self.realDecodeD = specializedD
        self.realDecodeF = specializedF
        self.realDecodeNumExperts = specializedNumExperts
        self.realDecodeTopK = specializedTopK
        self.sigmoidRouterScores = sigmoidRouterScores
        self.routedScalingFactor = routedScalingFactor
        precondition([4, 8].contains(routedWeightBits))
        precondition([4, 8].contains(routerWeightBits))
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(specializedTopK))
        precondition(specializedD <= Self.maxStagedHiddenD)
        let biasConstants: [MetalFunctionConstant] = expertAdditiveBiases
            ? [MetalFunctionConstant(index: 7, value: .bool(true))]
            : []
        var activationConstants: [MetalFunctionConstant] = siluActivation
            ? [MetalFunctionConstant(index: 4, value: .bool(true))]
            : []
        if clampedSwiGLU {
            activationConstants.append(
                MetalFunctionConstant(index: 8, value: .bool(true)))
        }
        let weightConstants = (routedWeightBits == 4 ? [] : [
            MetalFunctionConstant(index: 5, value: .uint32(UInt32(routedWeightBits)))
        ]) + biasConstants
        let ioConstants = [MetalFunctionConstant(index: 6, value: .bool(eventGatedIO))]
        let moeConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 0, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 1, value: .uint32(specializedF)),
            MetalFunctionConstant(index: 2, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 3, value: .bool(true)),
        ] + activationConstants + weightConstants + ioConstants
        let routerConstants: [MetalFunctionConstant] = [
            MetalFunctionConstant(index: 40, value: .uint32(specializedNumExperts)),
            MetalFunctionConstant(index: 41, value: .uint32(specializedD)),
            MetalFunctionConstant(index: 42, value: .uint32(specializedTopK)),
            MetalFunctionConstant(index: 43, value: .bool(true)),
            MetalFunctionConstant(index: 44, value: .uint32(UInt32(routerWeightBits))),
        ]
        let routerPipelines = try Self.makeRouterPipelines(
            context: context, constants: routerConstants,
            routerWeightBits: routerWeightBits, sigmoidRouterScores: sigmoidRouterScores)
        self.routerGemvPSO = routerPipelines.gemv
        self.routerGemvSpecializedPSO = routerPipelines.gemvSpecialized
        self.routerSelectK8PSO = routerPipelines.select
        self.routerSelectK8SpecializedPSO = routerPipelines.selectSpecialized
        self.routerGemvPairPSO = routerPipelines.gemvPair
        self.routerGemvPairSpecializedPSO = routerPipelines.gemvPairSpecialized
        self.routerSelectK8PairPSO = routerPipelines.selectPair
        self.routerSelectK8PairSpecializedPSO = routerPipelines.selectPairSpecialized
        self.residencyClassifySpecPSO = try context.pipeline(
            "moe_classify_expert_residency_spec")
        let phase1Name = routedWeightBits == 4
            ? "moe_phase1_gate_up_act_u16load" : "moe_affine_phase1_gate_up_act"
        let phase1SubsetName = routedWeightBits == 4
            ? "moe_phase1_gate_up_act_subset_u16load" : "moe_affine_phase1_gate_up_act_subset"
        let phase2Name = routedWeightBits == 4
            ? "moe_phase2_down_reduce_k8" : "moe_affine_phase2_down_reduce_k8"
        self.phase1U16PSO = try context.pipeline(
            phase1Name, constants: activationConstants + weightConstants + ioConstants)
        self.phase1U16SpecializedPSO = try context.pipeline(
            phase1Name,
            constants: moeConstants)
        self.phase1SubsetU16PSO = try context.pipeline(
            phase1SubsetName, constants: activationConstants + weightConstants + ioConstants)
        self.phase1SubsetU16SpecializedPSO = try context.pipeline(
            phase1SubsetName,
            constants: moeConstants)
        self.phase2ReduceK8PSO = try context.pipeline(
            phase2Name, constants: weightConstants + ioConstants)
        self.phase2ReduceK8SpecializedPSO = try context.pipeline(
            phase2Name,
            constants: moeConstants)
        self.specPhase1PSO = try context.pipeline(
            "moe_phase1_gate_up_act_spec_u16load",
            constants: activationConstants + weightConstants)
        self.specPhase1SpecializedPSO = try context.pipeline(
            "moe_phase1_gate_up_act_spec_u16load",
            constants: moeConstants)
        self.specPhase2PSO = try context.pipeline(
            "moe_phase2_down_reduce_spec_k8",
            constants: weightConstants)
        self.specPhase2SpecializedPSO = try context.pipeline(
            "moe_phase2_down_reduce_spec_k8",
            constants: moeConstants)

        guard let logits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared),
              let readyStatus = context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared),
              let phase1Function = context.library.makeFunction(name: phase1Name) else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        guard let pairLogits = context.device.makeBuffer(
            length: 256 * MemoryLayout<Float>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.routerLogitsPair = pairLogits
        readyStatus.contents().storeBytes(of: UInt32(1), as: UInt32.self)
        self.alwaysReadyIOStatus = readyStatus
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    func encodeRouter(commandBuffer: MTLCommandBuffer,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   logitBias: MTLBuffer, logitBiasOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeRouter(encoder: encoder,
                     weights: weights, weightsOffset: weightsOffset,
                     scales: scales, scalesOffset: scalesOffset,
                     biases: biases, biasesOffset: biasesOffset,
                     hidden: hidden,
                     effectiveScale: effectiveScale,
                     effectiveScaleOffset: effectiveScaleOffset,
                     perExpertScale: perExpertScale,
                     perExpertScaleOffset: perExpertScaleOffset,
                     logitBias: logitBias, logitBiasOffset: logitBiasOffset,
                     outIndices: outIndices, outWeights: outWeights,
                     numExperts: numExperts, d: d, topK: topK)
        encoder.endEncoding()
    }

    func encodeRouter(encoder: MTLComputeCommandEncoder,
                                   weights: MTLBuffer, weightsOffset: Int = 0,
                                   scales: MTLBuffer, scalesOffset: Int = 0,
                                   biases: MTLBuffer, biasesOffset: Int = 0,
                                   hidden: MTLBuffer,
                                   effectiveScale: MTLBuffer, effectiveScaleOffset: Int = 0,
                                   perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                                   logitBias: MTLBuffer, logitBiasOffset: Int = 0,
                                   outIndices: MTLBuffer,
                                   outWeights: MTLBuffer,
                                   numExperts: UInt32,
                                   d: UInt32,
                                   topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts <= 256)
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(topK))
        precondition(topK <= numExperts)
        // K16: `router_gemv_r4` multiplies every hidden element by
        // `effective_scale[idx]` and `router_topk_select_k8` multiplies every
        // weight by `per_expert_scale[expert]`. Qwen 3.6 has no router scale
        // tensors, so the runner synthesizes 1.0-filled buffers for both —
        // they must always be supplied with at least the addressed element
        // count, never nil/undersized, or the kernels read out of bounds.
        precondition(effectiveScale.length >= Int(d) * MemoryLayout<UInt16>.stride,
                     "encodeRouter: effectiveScale must cover [d] BF16 (runner synthesizes a 1.0 buffer for Qwen; the kernel always reads it)")
        precondition(perExpertScale.length >= Int(numExperts) * MemoryLayout<UInt16>.stride,
                     "encodeRouter: perExpertScale must cover [numExperts] BF16 (runner synthesizes a 1.0 buffer for Qwen; router_topk_select_k8 always dereferences it)")
        precondition(logitBias.length >= Int(numExperts) * MemoryLayout<UInt16>.stride,
                     "encodeRouter: logitBias must cover [numExperts] BF16 (runner synthesizes a 0.0 buffer for families without a router bias; the selector always dereferences it)")

        var expertCount = numExperts
        var dimension = d
        var topKValue = topK
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
            && topK == realDecodeTopK
        encoder.setComputePipelineState(
            useSpecialized ? routerGemvSpecializedPSO : routerGemvPSO)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(effectiveScale, offset: effectiveScaleOffset, index: 4)
        encoder.setBuffer(routerLogits, offset: 0, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        let selector = encoder
        selector.setComputePipelineState(
            useSpecialized ? routerSelectK8SpecializedPSO : routerSelectK8PSO)
        selector.setBuffer(routerLogits, offset: 0, index: 0)
        selector.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        selector.setBuffer(outIndices, offset: 0, index: 2)
        selector.setBuffer(outWeights, offset: 0, index: 3)
        selector.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        selector.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 5)
        selector.setBuffer(logitBias, offset: logitBiasOffset, index: 6)
        if sigmoidRouterScores {
            var scaling = routedScalingFactor
            selector.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 7)
        }
        selector.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private struct RouterPipelines {
        let gemv: MTLComputePipelineState
        let gemvSpecialized: MTLComputePipelineState
        let select: MTLComputePipelineState
        let selectSpecialized: MTLComputePipelineState
        let gemvPair: MTLComputePipelineState
        let gemvPairSpecialized: MTLComputePipelineState
        let selectPair: MTLComputePipelineState
        let selectPairSpecialized: MTLComputePipelineState
    }

    private static func makeRouterPipelines(context: MetalContext,
                                            constants: [MetalFunctionConstant],
                                            routerWeightBits: Int,
                                            sigmoidRouterScores: Bool) throws -> RouterPipelines {
        let bits = [MetalFunctionConstant(index: 44, value: .uint32(UInt32(routerWeightBits)))]
        let selectName = sigmoidRouterScores
            ? "router_topk_select_sigmoid_k8" : "router_topk_select_k8"
        return RouterPipelines(
            gemv: try context.pipeline("router_gemv_r4", constants: bits,
                                       maxTotalThreadsPerThreadgroup: 512),
            gemvSpecialized: try context.pipeline("router_gemv_r4", constants: constants,
                                                  maxTotalThreadsPerThreadgroup: 512),
            select: try context.pipeline(selectName),
            selectSpecialized: try context.pipeline(selectName, constants: constants),
            gemvPair: try context.pipeline("router_gemv_r4_pair", constants: bits,
                                           maxTotalThreadsPerThreadgroup: 512),
            gemvPairSpecialized: try context.pipeline("router_gemv_r4_pair", constants: constants,
                                                      maxTotalThreadsPerThreadgroup: 512),
            selectPair: try context.pipeline(selectName + "_pair"),
            selectPairSpecialized: try context.pipeline(selectName + "_pair", constants: constants))
    }

    struct RouterOperands {
        let weights: MTLBuffer
        var weightsOffset = 0
        let scales: MTLBuffer
        var scalesOffset = 0
        let biases: MTLBuffer
        var biasesOffset = 0
        let effectiveScale: MTLBuffer
        var effectiveScaleOffset = 0
        let logitBias: MTLBuffer
        var logitBiasOffset = 0
        let outIndices: MTLBuffer
        let outWeights: MTLBuffer
    }

    func encodeRouterPair(commandBuffer: MTLCommandBuffer,
                          first: RouterOperands, second: RouterOperands,
                          hidden: MTLBuffer,
                          perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                          numExperts: UInt32, d: UInt32, topK: UInt32) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeRouterPair(encoder: encoder, first: first, second: second, hidden: hidden,
                         perExpertScale: perExpertScale, perExpertScaleOffset: perExpertScaleOffset,
                         numExperts: numExperts, d: d, topK: topK)
        encoder.endEncoding()
    }

    /// The grid's second row is the second router; the first router's results are bit for bit `encodeRouter`'s.
    func encodeRouterPair(encoder: MTLComputeCommandEncoder,
                          first: RouterOperands, second: RouterOperands,
                          hidden: MTLBuffer,
                          perExpertScale: MTLBuffer, perExpertScaleOffset: Int = 0,
                          numExperts: UInt32, d: UInt32, topK: UInt32) {
        precondition(d.isMultiple(of: UInt32(Quantization.groupSize)))
        precondition(numExperts <= 256)
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(topK))
        precondition(topK <= numExperts)
        for operands in [first, second] {
            precondition(operands.effectiveScale.length >= Int(d) * MemoryLayout<UInt16>.stride)
            precondition(operands.logitBias.length >= Int(numExperts) * MemoryLayout<UInt16>.stride)
        }
        precondition(perExpertScale.length >= Int(numExperts) * MemoryLayout<UInt16>.stride)
        var expertCount = numExperts
        var dimension = d
        var topKValue = topK
        let useSpecialized = numExperts == realDecodeNumExperts
            && d == realDecodeD
            && topK == realDecodeTopK
        encoder.setComputePipelineState(
            useSpecialized ? routerGemvPairSpecializedPSO : routerGemvPairPSO)
        encoder.setBuffer(first.weights, offset: first.weightsOffset, index: 0)
        encoder.setBuffer(first.scales, offset: first.scalesOffset, index: 1)
        encoder.setBuffer(first.biases, offset: first.biasesOffset, index: 2)
        encoder.setBuffer(hidden, offset: 0, index: 3)
        encoder.setBuffer(first.effectiveScale, offset: first.effectiveScaleOffset, index: 4)
        encoder.setBuffer(routerLogits, offset: 0, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(second.weights, offset: second.weightsOffset, index: 8)
        encoder.setBuffer(second.scales, offset: second.scalesOffset, index: 9)
        encoder.setBuffer(second.biases, offset: second.biasesOffset, index: 10)
        encoder.setBuffer(second.effectiveScale, offset: second.effectiveScaleOffset, index: 11)
        encoder.setBuffer(routerLogitsPair, offset: 0, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(numExperts) + 3) / 4, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))

        encoder.setComputePipelineState(
            useSpecialized ? routerSelectK8PairSpecializedPSO : routerSelectK8PairPSO)
        encoder.setBuffer(routerLogits, offset: 0, index: 0)
        encoder.setBuffer(perExpertScale, offset: perExpertScaleOffset, index: 1)
        encoder.setBuffer(first.outIndices, offset: 0, index: 2)
        encoder.setBuffer(first.outWeights, offset: 0, index: 3)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBuffer(first.logitBias, offset: first.logitBiasOffset, index: 6)
        if sigmoidRouterScores {
            var scaling = routedScalingFactor
            encoder.setBytes(&scaling, length: MemoryLayout<Float>.stride, index: 7)
        }
        encoder.setBuffer(routerLogitsPair, offset: 0, index: 8)
        encoder.setBuffer(second.outIndices, offset: 0, index: 9)
        encoder.setBuffer(second.outWeights, offset: 0, index: 10)
        encoder.setBuffer(second.logitBias, offset: second.logitBiasOffset, index: 11)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 2, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                         topK: UInt32,
                                         routedBufferOffsets: [Int]? = nil) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
            length: routedArgEncoder.encodedLength,
            options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs,
                                   routedBufferOffsets: routedBufferOffsets)
        return buffer
    }

    /// `arguments` receives three MTLDispatchThreadgroupsIndirectArguments
    /// (phase-1 at offset 0, phase-2 at `specPhase2ArgsOffset`, residual tail
    /// at `specTailArgsOffset`); the grids are what the classifier publishes
    /// when every routed expert is resident.
    struct SpeculativeDispatchArguments {
        let arguments: MTLBuffer
        let phase1Threadgroups: MTLSize
        let phase2Threadgroups: MTLSize
        let tailThreadgroups: MTLSize
    }

    static let specDispatchArgsLength = MemoryLayout<UInt32>.stride * 9
    static let specPhase2ArgsOffset = MemoryLayout<UInt32>.stride * 3
    static let specTailArgsOffset = MemoryLayout<UInt32>.stride * 6

    /// The classifier's tagged copy of the host's readback; `RouterHostReadback` gives the layout.
    struct RouterHostReadbackArguments {
        let buffer: MTLBuffer
        let tag: UInt32
        let topKWeights: MTLBuffer
        let predictedIndices: MTLBuffer
    }

    func encodeResidencyClassification(
        commandBuffer: MTLCommandBuffer,
        topKIndices: MTLBuffer,
        residencyTable: MTLBuffer,
        hitCount: MTLBuffer,
        hitPositions: MTLBuffer,
        missCount: MTLBuffer,
        missPositions: MTLBuffer,
        missExperts: MTLBuffer,
        resolvedSlots: MTLBuffer,
        topK: UInt32,
        numExperts: UInt32,
        speculative: SpeculativeDispatchArguments,
        hostReadback: RouterHostReadbackArguments? = nil
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeResidencyClassification(
            encoder: encoder, topKIndices: topKIndices,
            residencyTable: residencyTable,
            hitCount: hitCount, hitPositions: hitPositions,
            missCount: missCount, missPositions: missPositions,
            missExperts: missExperts, resolvedSlots: resolvedSlots,
            topK: topK, numExperts: numExperts,
            speculative: speculative,
            hostReadback: hostReadback)
        encoder.endEncoding()
    }

    func encodeResidencyClassification(
        encoder: MTLComputeCommandEncoder,
        topKIndices: MTLBuffer,
        residencyTable: MTLBuffer,
        hitCount: MTLBuffer,
        hitPositions: MTLBuffer,
        missCount: MTLBuffer,
        missPositions: MTLBuffer,
        missExperts: MTLBuffer,
        resolvedSlots: MTLBuffer,
        topK: UInt32,
        numExperts: UInt32,
        speculative: SpeculativeDispatchArguments,
        hostReadback: RouterHostReadbackArguments? = nil
    ) {
        precondition(topK <= UInt32(Self.maxStreamedExperts))
        precondition(speculative.arguments.length >= Self.specDispatchArgsLength)
        if let hostReadback {
            precondition(hostReadback.tag != 0)
            precondition(hostReadback.buffer.length
                >= RouterHostReadback.wordCount(topK: Int(topK)) * MemoryLayout<UInt32>.stride)
            precondition(hostReadback.topKWeights.length >= Int(topK) * MemoryLayout<UInt16>.stride)
            precondition(hostReadback.predictedIndices.length >= Int(topK) * MemoryLayout<UInt32>.stride)
        }
        var topKValue = topK
        var expertCount = numExperts
        encoder.setComputePipelineState(residencyClassifySpecPSO)
        encoder.setBuffer(topKIndices, offset: 0, index: 0)
        encoder.setBuffer(residencyTable, offset: 0, index: 1)
        encoder.setBuffer(hitCount, offset: 0, index: 2)
        encoder.setBuffer(hitPositions, offset: 0, index: 3)
        encoder.setBuffer(missCount, offset: 0, index: 4)
        encoder.setBuffer(missPositions, offset: 0, index: 5)
        encoder.setBuffer(missExperts, offset: 0, index: 6)
        encoder.setBuffer(resolvedSlots, offset: 0, index: 7)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 10)
        var grids: [UInt32] = [
            UInt32(speculative.phase1Threadgroups.width),
            UInt32(speculative.phase1Threadgroups.height),
            UInt32(speculative.phase1Threadgroups.depth),
            UInt32(speculative.phase2Threadgroups.width),
            UInt32(speculative.phase2Threadgroups.height),
            UInt32(speculative.phase2Threadgroups.depth),
            UInt32(speculative.tailThreadgroups.width),
            UInt32(speculative.tailThreadgroups.height),
            UInt32(speculative.tailThreadgroups.depth),
        ]
        encoder.setBytes(&grids, length: Self.specDispatchArgsLength, index: 11)
        encoder.setBuffer(speculative.arguments, offset: 0, index: 12)
        // A zero tag tells the kernel there is no copy to write; the bindings
        // stay valid either way.
        var readbackTag = hostReadback?.tag ?? 0
        encoder.setBuffer(hostReadback?.buffer ?? topKIndices, offset: 0, index: 14)
        encoder.setBytes(&readbackTag, length: MemoryLayout<UInt32>.stride, index: 15)
        encoder.setBuffer(hostReadback?.topKWeights ?? topKIndices, offset: 0, index: 16)
        encoder.setBuffer(hostReadback?.predictedIndices ?? topKIndices, offset: 0, index: 17)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                               topK: UInt32,
                                               routedBufferOffsets: [Int]? = nil) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs,
                                   routedBufferOffsets: routedBufferOffsets)
        return reusableRoutedArgBuffer
    }

    func encodeRoutedPersistentPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        xOffset: Int = 0,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeRoutedPersistentPhase1U16Load(
            encoder: encoder,
            routedArgBuffer: routedArgBuffer,
            routedBlobs: routedBlobs,
            routedOffsets: routedOffsets,
            x: x, xOffset: xOffset,
            acts: acts, actsOffset: actsOffset,
            d: d, f: f, topK: topK,
            ioStatus: ioStatus, ioStatusOffset: ioStatusOffset)
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1U16Load(
        encoder: MTLComputeCommandEncoder,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        xOffset: Int = 0,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(d <= Self.maxStagedHiddenD)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1U16SpecializedPSO
                : phase1U16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(acts, offset: actsOffset, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 7)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x), so the
        // dispatch is (topK*f)/16 groups of 512 threads.
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        guard activeCount > 0 else { return }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeRoutedPersistentPhase1SubsetU16Load(
            encoder: encoder,
            routedArgBuffer: routedArgBuffer,
            routedBlobs: routedBlobs,
            routedOffsets: routedOffsets,
            x: x, acts: acts,
            activeSlots: activeSlots,
            activeSlotIndices: activeSlotIndices,
            activeCount: activeCount,
            d: d, f: f, topK: topK,
            ioStatus: ioStatus, ioStatusOffset: ioStatusOffset)
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase1SubsetU16Load(
        encoder: MTLComputeCommandEncoder,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        activeSlots: MTLBuffer,
        activeSlotIndices: [UInt32],
        activeCount: UInt32,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase1SubsetU16SpecializedPSO
                : phase1SubsetU16PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 9)
        // Phase-1 uses 16 rows per threadgroup (threadgroup-staged x).
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 15) / 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    func encodeRoutedPersistentPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        routingWeights: MTLBuffer,
        routingWeightsOffset: Int = 0,
        residual: MTLBuffer,
        residualOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeRoutedPersistentPhase2Reduce(
            encoder: encoder,
            routedArgBuffer: routedArgBuffer,
            routedBlobs: routedBlobs,
            routedOffsets: routedOffsets,
            acts: acts, actsOffset: actsOffset,
            routingWeights: routingWeights, routingWeightsOffset: routingWeightsOffset,
            residual: residual, residualOffset: residualOffset,
            y: y, yOffset: yOffset,
            d: d, f: f, topK: topK,
            ioStatus: ioStatus, ioStatusOffset: ioStatusOffset)
        encoder.endEncoding()
    }

    func encodeRoutedPersistentPhase2Reduce(
        encoder: MTLComputeCommandEncoder,
        routedArgBuffer: MTLBuffer,
        routedBlobs: [MTLBuffer],
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        actsOffset: Int = 0,
        routingWeights: MTLBuffer,
        routingWeightsOffset: Int = 0,
        residual: MTLBuffer,
        residualOffset: Int = 0,
        y: MTLBuffer,
        yOffset: Int = 0,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        ioStatus: MTLBuffer? = nil,
        ioStatusOffset: Int = 0
    ) {
        validate(routedBlobs: routedBlobs, topK: topK)
        var dimension = d
        var intermediate = f
        var topKValue = topK
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? phase2ReduceK8SpecializedPSO
                : phase2ReduceK8PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: actsOffset, index: 2)
        encoder.setBuffer(routingWeights, offset: routingWeightsOffset, index: 3)
        encoder.setBuffer(residual, offset: residualOffset, index: 4)
        encoder.setBuffer(y, offset: yOffset, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(ioStatus ?? alwaysReadyIOStatus,
                          offset: ioStatus == nil ? 0 : ioStatusOffset,
                          index: 8)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 9)
        // One simdgroup per selected expert; the kernel reduces partial[0..<topK].
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Int(topK), height: 1, depth: 1))
    }

    static func specPhase1FullGrid(f: UInt32, topK: UInt32) -> MTLSize {
        MTLSize(width: (Int(topK * f) + 15) / 16, height: 1, depth: 1)
    }

    static func specPhase2FullGrid(d: UInt32) -> MTLSize {
        MTLSize(width: Int(d), height: 1, depth: 1)
    }

    static func specTailFullGrid(d: UInt32, threadgroupWidth: Int) -> MTLSize {
        MTLSize(width: (Int(d) + threadgroupWidth - 1) / threadgroupWidth,
                height: 1, depth: 1)
    }

    func encodeSpecPhase1U16Load(
        commandBuffer: MTLCommandBuffer,
        expertPool: MTLBuffer,
        poolSlotStride: UInt64,
        resolvedSlots: MTLBuffer,
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        indirectArguments: MTLBuffer,
        indirectOffset: Int = 0
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeSpecPhase1U16Load(
            encoder: encoder,
            expertPool: expertPool,
            poolSlotStride: poolSlotStride,
            resolvedSlots: resolvedSlots,
            routedOffsets: routedOffsets,
            x: x, acts: acts,
            d: d, f: f, topK: topK,
            indirectArguments: indirectArguments,
            indirectOffset: indirectOffset)
        encoder.endEncoding()
    }

    func encodeSpecPhase1U16Load(
        encoder: MTLComputeCommandEncoder,
        expertPool: MTLBuffer,
        poolSlotStride: UInt64,
        resolvedSlots: MTLBuffer,
        routedOffsets: MoEExpertOffsets,
        x: MTLBuffer,
        acts: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        indirectArguments: MTLBuffer,
        indirectOffset: Int = 0
    ) {
        precondition(d <= Self.maxStagedHiddenD)
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(topK))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var stride = poolSlotStride
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? specPhase1SpecializedPSO
                : specPhase1PSO)
        encoder.setBuffer(expertPool, offset: 0, index: 0)
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(resolvedSlots, offset: 0, index: 7)
        encoder.setBytes(&stride, length: MemoryLayout<UInt64>.stride, index: 8)
        encoder.dispatchThreadgroups(
            indirectBuffer: indirectArguments,
            indirectBufferOffset: indirectOffset,
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
    }

    func encodeSpecPhase2Reduce(
        commandBuffer: MTLCommandBuffer,
        expertPool: MTLBuffer,
        poolSlotStride: UInt64,
        resolvedSlots: MTLBuffer,
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        indirectArguments: MTLBuffer,
        indirectOffset: Int = MoE.specPhase2ArgsOffset
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        encodeSpecPhase2Reduce(
            encoder: encoder,
            expertPool: expertPool,
            poolSlotStride: poolSlotStride,
            resolvedSlots: resolvedSlots,
            routedOffsets: routedOffsets,
            acts: acts,
            routingWeights: routingWeights,
            residual: residual,
            y: y,
            d: d, f: f, topK: topK,
            indirectArguments: indirectArguments,
            indirectOffset: indirectOffset)
        encoder.endEncoding()
    }

    func encodeSpecPhase2Reduce(
        encoder: MTLComputeCommandEncoder,
        expertPool: MTLBuffer,
        poolSlotStride: UInt64,
        resolvedSlots: MTLBuffer,
        routedOffsets: MoEExpertOffsets,
        acts: MTLBuffer,
        routingWeights: MTLBuffer,
        residual: MTLBuffer,
        y: MTLBuffer,
        d: UInt32,
        f: UInt32,
        topK: UInt32,
        indirectArguments: MTLBuffer,
        indirectOffset: Int = MoE.specPhase2ArgsOffset
    ) {
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(topK))
        var dimension = d
        var intermediate = f
        var topKValue = topK
        var stride = poolSlotStride
        encoder.setComputePipelineState(
            useRealDecodeConstants(d: d, f: f, topK: topK)
                ? specPhase2SpecializedPSO
                : specPhase2PSO)
        encoder.setBuffer(expertPool, offset: 0, index: 0)
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<MoEExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBuffer(resolvedSlots, offset: 0, index: 8)
        encoder.setBytes(&topKValue, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.setBytes(&stride, length: MemoryLayout<UInt64>.stride, index: 10)
        encoder.dispatchThreadgroups(
            indirectBuffer: indirectArguments,
            indirectBufferOffset: indirectOffset,
            threadsPerThreadgroup: MTLSize(width: 32 * Int(topK), height: 1, depth: 1))
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition((1...UInt32(Self.maxStreamedExperts)).contains(topK))
        precondition(routedBlobs.count == Int(topK))
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer],
                                            routedBufferOffsets: [Int]?) {
        precondition(routedBufferOffsets == nil
                     || routedBufferOffsets?.count == routedBlobs.count)
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(
                blob,
                offset: routedBufferOffsets?[index] ?? 0,
                index: index)
        }
    }

    private func useRealDecodeConstants(d: UInt32, f: UInt32, topK: UInt32) -> Bool {
        d == realDecodeD && f == realDecodeF && topK == realDecodeTopK
    }
}
