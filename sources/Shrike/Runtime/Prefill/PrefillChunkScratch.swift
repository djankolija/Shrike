import Metal

struct PrefillChunkScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let maxQElementsPerToken: Int
    let maxKVElementsPerToken: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    let routedPairMicrobatchRows: Int
    /// Rows of the widest per-token projection written into `q`: the packed
    /// [query ; gate] q_proj when `attnOutputGate`, and the gated-DeltaNet
    /// `in_proj_qkv` rows on architectures with linear-attention layers.
    let qProjElementsPerToken: Int
    /// Non-zero when the architecture gates attention output (Qwen): sizes
    /// the split q / gate buffers.
    let attnGateElementsPerToken: Int
    /// Gated-DeltaNet dims (all zero when the mask has no linear layers).
    let gdnQKVDim: Int
    let gdnValueDim: Int
    let gdnVHeads: Int
    /// Per-token width of the decay input `a`: [Hv] scalar-decay, [Hv * Dk]
    /// per-channel (Kimi KDA).
    let gdnADim: Int
    /// Kimi KDA low-rank stage (f_a / g_a output, keyHeadDim); zero otherwise.
    let gdnLowRankDim: Int
    /// Kimi MLA per-token widths (zero when the mask has no MLA layers):
    /// the absorbed Q rows [H * (latent + rope)] and the unembed output
    /// [H * vHeadDim]. q_proj output rides the `q` buffer, attention output
    /// rides `attentionOutput`.
    let mlaQDim: Int
    let mlaUnembedDim: Int
    /// Non-zero when the shared expert output is scalar-gated (Qwen).
    let sharedScalarGateElements: Int
    /// v12 P4: the chunked delta-rule scan's factors scratch, sized only for
    /// the compiled shape and chunks of 64+ rows — the 32-token MTP draft
    /// chunk stays on the serial kernel and allocates nothing here.
    let gdnChunkFactorBytes: Int

    init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32) {
        let tokens = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        self.chunkTokens = tokens
        self.hiddenSize = config.hiddenSize
        self.maxQElementsPerToken = config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        self.sharedIntermediate = max(config.intermediateSize,
                                      config.denseIntermediateSize)
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = max(1, min(routedPairMicrobatchRows, 128))
        let qDim = config.numHeads * max(config.headDim, config.fullHeadDim)
        let hasLinear = config.hasLinearAttentionLayers
        self.qProjElementsPerToken = max(config.attnOutputGate ? 2 * qDim : qDim,
                                         hasLinear ? config.linearAttention.qkvDim : 0)
        self.attnGateElementsPerToken = config.attnOutputGate ? qDim : 0
        self.gdnQKVDim = hasLinear ? config.linearAttention.qkvDim : 0
        self.gdnValueDim = hasLinear ? config.linearAttention.valueDim : 0
        self.gdnVHeads = hasLinear ? config.linearAttention.numVHeads : 0
        let perChannel = hasLinear && config.linearAttentionPerChannelDecay
        self.gdnADim = perChannel
            ? config.linearAttention.numVHeads * config.linearAttention.keyHeadDim
            : (hasLinear ? config.linearAttention.numVHeads : 0)
        self.gdnLowRankDim = perChannel ? config.linearAttention.keyHeadDim : 0
        let chunkedScan = hasLinear && !perChannel
            && GDN.chunkedScanSupports(config: config.linearAttention, perChannelDecay: false)
            && tokens >= GDN.chunkTokens
        self.gdnChunkFactorBytes = chunkedScan
            ? GDN.chunkFactorsBytes(config: config.linearAttention, prefillChunkTokens: tokens)
            : 0
        if let mla = config.mla, config.hasMLALayers {
            self.mlaQDim = config.numHeads * (mla.latentDim + mla.qkRopeDim)
            self.mlaUnembedDim = config.numHeads * mla.valueHeadDim
        } else {
            self.mlaQDim = 0
            self.mlaUnembedDim = 0
        }
        self.sharedScalarGateElements = config.sharedExpertGated ? 1 : 0
    }

    init(config: ArchConfig, runtime: PrefillRuntimeConfig) {
        self.init(config: config,
                  chunkTokens: runtime.chunkTokens)
    }

    var hiddenElements: Int { chunkTokens * hiddenSize }
    var normedElements: Int { hiddenElements }
    var qElements: Int { chunkTokens * qProjElementsPerToken }
    var attnElements: Int { chunkTokens * maxQElementsPerToken }
    var attnQElements: Int { chunkTokens * attnGateElementsPerToken }
    var attnGateElements: Int { attnQElements }
    var gdnConvOutElements: Int { chunkTokens * gdnQKVDim }
    var gdnZElements: Int { chunkTokens * gdnValueDim }
    var gdnAElements: Int { chunkTokens * gdnADim }
    var gdnBElements: Int { chunkTokens * gdnVHeads }
    var gdnYElements: Int { gdnZElements }
    var gdnLowRankElements: Int { chunkTokens * gdnLowRankDim }
    var mlaQElements: Int { chunkTokens * mlaQDim }
    var mlaUnembedElements: Int { chunkTokens * mlaUnembedDim }
    var sharedScalarGateBufferElements: Int { chunkTokens * sharedScalarGateElements }
    var kStageElements: Int { chunkTokens * maxKVElementsPerToken }
    var vStageElements: Int { kStageElements }
    var attentionOutputElements: Int { attnElements }
    var denseXElements: Int { hiddenElements }
    var routedXElements: Int { hiddenElements }
    var routerXElements: Int { hiddenElements }
    var h1Elements: Int { hiddenElements }
    var h2Elements: Int { hiddenElements }
    var routePartialElements: Int { chunkTokens * topK * hiddenSize }
    var routeIDElements: Int { chunkTokens * topK }
    var routeWeightElements: Int { routeIDElements }
    /// The matrix path folds its activation back into the gate buffer, so only
    /// gate and up hold the whole chunk.
    var sharedExpertScratchElements: Int { chunkTokens * sharedIntermediate }
    var sharedExpertActScratchElements: Int { sharedIntermediate }
    var routedGateUpActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    var routedDownOutputElements: Int { routedPairMicrobatchRows * hiddenSize }
    /// The strict `>` keeps the 32-token MTP draft chunk off the matrix path
    /// so its hard memory budget stays unchanged; a dense architecture has no
    /// routed experts at all. This same predicate gates the allocation below
    /// and the branch in `RealForwardRunner.encodeRoutedTileExperts`.
    var usesRoutedExpertMatrixPath: Bool {
        routedIntermediate > 0
            && topK > 0
            && chunkTokens > PrefillGroupedRoutedMoE.matrixPathMinimumRows
    }
    /// The matrix path loops over row blocks of this size, so the staging never
    /// needs more rows than this however long the chunk is.
    static let routedExpertGEMMRowBlock = 512
    var routedExpertStagingRows: Int {
        usesRoutedExpertMatrixPath ? min(Self.routedExpertGEMMRowBlock, chunkTokens) : 0
    }
    var routedExpertHiddenStagingElements: Int { routedExpertStagingRows * hiddenSize }
    var routedExpertActStagingElements: Int { routedExpertStagingRows * routedIntermediate }

    var devicePrivateBytes: Int {
        let fp16Elements = hiddenElements
            + normedElements
            + qElements
            + kStageElements
            + vStageElements
            + attentionOutputElements
            + denseXElements
            + routedXElements
            + routerXElements
            + h1Elements
            + h2Elements
            + routePartialElements
            + 2 * sharedExpertScratchElements
            + sharedExpertActScratchElements
            + routedGateUpActElements
            + routedDownOutputElements
            + 2 * routedExpertHiddenStagingElements
            + 2 * routedExpertActStagingElements
            + attnQElements
            + attnGateElements
            + gdnConvOutElements
            + gdnZElements
            + gdnAElements
            + gdnBElements
            + gdnYElements
            + gdnLowRankElements
            + mlaQElements
            + mlaUnembedElements
            + sharedScalarGateBufferElements
        return fp16Elements * MemoryLayout<Float16>.stride + gdnChunkFactorBytes
    }

    var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
    }

    var totalPersistentBytes: Int {
        devicePrivateBytes + sharedMetadataBytes
    }
}

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let kStage: MTLBuffer
    let vStage: MTLBuffer
    let attentionOutput: MTLBuffer
    let denseX: MTLBuffer
    let routedX: MTLBuffer
    let routerX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routePartials: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedGateUpActScratch: MTLBuffer
    let routedDownScratch: MTLBuffer
    let routedExpertStaging: PrefillExpertStaging
    // Qwen 3.6 additions. Placeholder-sized (1 element) when the arch does
    // not use them, so the struct stays non-optional.
    let attnQ: MTLBuffer
    let attnGate: MTLBuffer
    let gdnConvOut: MTLBuffer
    let gdnZ: MTLBuffer
    let gdnA: MTLBuffer
    let gdnB: MTLBuffer
    let gdnY: MTLBuffer
    let gdnChunkFactors: MTLBuffer?
    let gdnLowRank: MTLBuffer
    let mlaQ: MTLBuffer
    let mlaUnembed: MTLBuffer
    let sharedScalarGate: MTLBuffer

    static func allocate(device: MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func privateBytes(_ bytes: Int, label: String) throws -> MTLBuffer? {
            guard bytes > 0 else { return nil }
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(layout.hiddenElements, label: "prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kStageElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.vStageElements, label: "prefill.vStage"),
            attentionOutput: try privateBuffer(layout.attentionOutputElements, label: "prefill.attnOut"),
            denseX: try privateBuffer(layout.denseXElements, label: "prefill.denseX"),
            routedX: try privateBuffer(layout.routedXElements, label: "prefill.routedX"),
            routerX: try privateBuffer(layout.routerXElements, label: "prefill.routerX"),
            h1: try privateBuffer(layout.h1Elements, label: "prefill.h1"),
            h2: try privateBuffer(layout.h2Elements, label: "prefill.h2"),
            routePartials: try privateBuffer(layout.routePartialElements, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeWeightElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                 label: "prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                               label: "prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertActScratchElements,
                                                label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(layout.routedGateUpActElements,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(layout.routedDownOutputElements,
                                                 label: "prefill.routedDownScratch"),
            routedExpertStaging: try PrefillExpertStaging.allocate(
                device: device,
                rowBlock: layout.routedExpertStagingRows,
                hiddenSize: layout.hiddenSize,
                intermediate: layout.routedIntermediate),
            attnQ: try privateBuffer(layout.attnQElements, label: "prefill.attnQ"),
            attnGate: try privateBuffer(layout.attnGateElements, label: "prefill.attnGate"),
            gdnConvOut: try privateBuffer(layout.gdnConvOutElements, label: "prefill.gdnConvOut"),
            gdnZ: try privateBuffer(layout.gdnZElements, label: "prefill.gdnZ"),
            gdnA: try privateBuffer(layout.gdnAElements, label: "prefill.gdnA"),
            gdnB: try privateBuffer(layout.gdnBElements, label: "prefill.gdnB"),
            gdnY: try privateBuffer(layout.gdnYElements, label: "prefill.gdnY"),
            gdnChunkFactors: try privateBytes(layout.gdnChunkFactorBytes,
                                              label: "prefill.gdnChunkFactors"),
            gdnLowRank: try privateBuffer(layout.gdnLowRankElements,
                                          label: "prefill.gdnLowRank"),
            mlaQ: try privateBuffer(layout.mlaQElements, label: "prefill.mlaQ"),
            mlaUnembed: try privateBuffer(layout.mlaUnembedElements,
                                          label: "prefill.mlaUnembed"),
            sharedScalarGate: try privateBuffer(layout.sharedScalarGateBufferElements,
                                                label: "prefill.sharedScalarGate"))
    }
}
