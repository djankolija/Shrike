import Foundation

/// Model family discriminator, mirrored into `manifest.json -> arch.family`.
/// Raw values match the runtime's `ModelFamily`.
enum RepackModelFamily: String, Sendable, Equatable {
    case qwen36 = "qwen36"
    case qwen36MTP = "qwen36_mtp"
    case gptOss20b = "gpt_oss_20b"
    case kimiLinear48b = "kimi_linear_48b"
}

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention,
/// 3 = multi-head latent attention (MLA).
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `sliding_attention`, 2 if `linear_attention`,
    /// 3 if MLA.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String

    // Family-dependent extensions. Defaults describe the compatible
    // Qwen3.5-MoE text architecture used by Qwen 3.6 and Ornith 1.5.
    let family: RepackModelFamily
    let attnOutputGate: Bool
    let attentionScale: Double
    let embeddingScaledBySqrtHidden: Bool
    let routerScaled: Bool
    let ffnSandwichNorms: Bool
    let sharedExpertGated: Bool
    let ropeNeoxSubdim: Bool
    let linearNumKHeads: Int
    let linearNumVHeads: Int
    let linearKeyHeadDim: Int
    let linearValueHeadDim: Int
    let linearConvKernelSize: Int

    // Non-Qwen extensions. Zero for families they do not apply to.
    let numLeadingDenseLayers: Int
    let denseIntermediateSize: Int
    let mlaKVLoraRank: Int
    let mlaQKNopeDim: Int
    let mlaQKRopeDim: Int
    let mlaVHeadDim: Int

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        switch root["model_type"] as? String {
        case "qwen3_5_mtp", "qwen3_5_moe":
            guard let tc = root["text_config"] as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
            }
            if (root["model_type"] as? String) == "qwen3_5_mtp" {
                return try loadQwen36MTP(configPath: configPath, tc: tc)
            }
            return try loadQwen35MoE(configPath: configPath, tc: tc)
        case "gpt_oss":
            return try loadGptOss(configPath: configPath, tc: root)
        case "kimi_linear":
            return try loadKimiLinear(configPath: configPath, tc: root)
        default:
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "unsupported model_type (expected qwen3_5_moe, qwen3_5_mtp, "
                    + "gpt_oss, or kimi_linear)")
        }
    }

    // MARK: - Shared field readers (flat configs)

    private static func intField(_ tc: [String: Any], _ k: String,
                                 configPath: String) throws -> Int {
        guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
        }
        return n
    }

    private static func doubleField(_ tc: [String: Any], _ k: String,
                                    configPath: String) throws -> Double {
        guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
        }
        return n
    }

    // MARK: - Qwen3.5-MoE text (`model_type == "qwen3_5_moe"`)

    private static func loadQwen35MoE(configPath: String,
                                     tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "linear_attention": mask.append(2)
            case "full_attention":   mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        guard let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.rope_theta")
        }
        guard let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing rope_parameters.partial_rotary_factor")
        }
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let gate = (tc["attn_output_gate"] as? Bool) ?? false
        let act = (tc["hidden_act"] as? String) ?? "silu"
        let headDim = try i("head_dim")

        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_key_value_heads"),
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: tie,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            family: .qwen36,
            attnOutputGate: gate,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: try i("linear_num_key_heads"),
            linearNumVHeads: try i("linear_num_value_heads"),
            linearKeyHeadDim: try i("linear_key_head_dim"),
            linearValueHeadDim: try i("linear_value_head_dim"),
            linearConvKernelSize: try i("linear_conv_kernel_dim"),
            numLeadingDenseLayers: 0,
            denseIntermediateSize: 0,
            mlaKVLoraRank: 0,
            mlaQKNopeDim: 0,
            mlaQKRopeDim: 0,
            mlaVHeadDim: 0)
        try crossCheckProductionQwen35MoE(arch, configPath: configPath)
        return arch
    }

    /// Qwen3.6 MTP is a single full-attention decoder layer. It intentionally
    /// carries neither an embedding table nor an LM head: both are shared from
    /// the verified target model at runtime. Treating it as a distinct family
    /// keeps a draft sidecar from ever being accepted as a standalone target.
    private static func loadQwen36MTP(configPath: String,
                                      tc: [String: Any]) throws -> ArchInfo {
        var base = try loadQwen35MoE(configPath: configPath, tc: tc)
        guard let count = (tc["mtp_num_hidden_layers"] as? Int)
            ?? (tc["mtp_num_hidden_layers"] as? NSNumber)?.intValue,
              count == 1 else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "Qwen3.6 MTP requires mtp_num_hidden_layers == 1")
        }
        guard (tc["mtp_use_dedicated_embeddings"] as? Bool) == false else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "Qwen3.6 MTP must reuse the target embedding and head")
        }
        // MTP contract (mirrors the runtime's `qwen36MTP` arch config):
        // `numExperts` and `ropeNeoxSubdim` are deliberately kept from the
        // target baseline. The draft layer shares the target's router shape
        // (numExperts 256 drives the sidecar's per-expert layout), and the
        // MTP layer applies the same rotary embedding variant as the target,
        // so ropeNeoxSubdim stays true. The linear-attention parameters are
        // zeroed because the MTP layer is pure full-attention and carries no
        // DeltaNet bundle. numLayers collapses to 1 and the MTP arch reports
        // no embedding/head of its own (tieWordEmbeddings false).
        base = ArchInfo(
            hiddenSize: base.hiddenSize,
            intermediateSize: base.intermediateSize,
            moeIntermediateSize: base.moeIntermediateSize,
            numHeads: base.numHeads,
            numKVHeads: base.numKVHeads,
            numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim,
            fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize,
            slidingWindow: 65_536,
            finalLogitSoftcap: base.finalLogitSoftcap,
            ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: 1,
            numExperts: base.numExperts,
            topKExperts: base.topKExperts,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [1],
            hiddenActivation: base.hiddenActivation,
            family: .qwen36MTP,
            attnOutputGate: base.attnOutputGate,
            attentionScale: base.attentionScale,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            numLeadingDenseLayers: 0,
            denseIntermediateSize: 0,
            mlaKVLoraRank: 0,
            mlaQKNopeDim: 0,
            mlaQKRopeDim: 0,
            mlaVHeadDim: 0)
        return base
    }

    /// Production Qwen3.5-MoE 35B-A3B contract (mirrors the runtime's
    /// `ArchConfig.qwen36_35B_A3B`; the repack target has no dependency on the
    /// runtime module). A config that matches the production shape
    /// (hidden 2048, 40 layers) must agree on every field; toy/synthetic
    /// configs are exempt.
    private static func crossCheckProductionQwen35MoE(_ a: ArchInfo,
                                                      configPath: String) throws {
        guard a.hiddenSize == 2048, a.numLayers == 40 else { return }
        var expectedMask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { expectedMask[i] = 1 }
        let expected = ArchInfo(
            hiddenSize: 2048,
            intermediateSize: 512,
            moeIntermediateSize: 512,
            numHeads: 16,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 256,
            fullHeadDim: 256,
            vocabSize: 248_320,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 40,
            numExperts: 256,
            topKExperts: 8,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: expectedMask,
            hiddenActivation: "silu",
            family: .qwen36,
            attnOutputGate: true,
            attentionScale: 0.0625,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearNumKHeads: 16,
            linearNumVHeads: 32,
            linearKeyHeadDim: 128,
            linearValueHeadDim: 128,
            linearConvKernelSize: 4,
            numLeadingDenseLayers: 0,
            denseIntermediateSize: 0,
            mlaKVLoraRank: 0,
            mlaQKNopeDim: 0,
            mlaQKRopeDim: 0,
            mlaVHeadDim: 0)
        guard a == expected else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "qwen3_5_moe config does not match the supported "
                    + "35B-A3B architecture contract")
        }
    }

    // MARK: - gpt-oss (`model_type == "gpt_oss"`, flat config, no text_config)

    private static func loadGptOss(configPath: String,
                                   tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int { try intField(tc, k, configPath: configPath) }
        func d(_ k: String) throws -> Double { try doubleField(tc, k, configPath: configPath) }
        guard let layerTypes = tc["layer_types"] as? [String] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing layer_types")
        }
        var mask: [UInt8] = []
        mask.reserveCapacity(layerTypes.count)
        for t in layerTypes {
            switch t {
            case "sliding_attention": mask.append(0)
            case "full_attention":    mask.append(1)
            default:
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "unknown layer_types entry \"\(t)\"")
            }
        }
        let headDim = try i("head_dim")
        let numKVHeads = try i("num_key_value_heads")
        let theta = try d("rope_theta")
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: 0,
            moeIntermediateSize: try i("intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: numKVHeads,
            numFullKVHeads: numKVHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: 1.0,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_local_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .gptOss20b,
            attnOutputGate: false,
            attentionScale: 1.0 / Double(headDim).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            linearNumKHeads: 0,
            linearNumVHeads: 0,
            linearKeyHeadDim: 0,
            linearValueHeadDim: 0,
            linearConvKernelSize: 0,
            numLeadingDenseLayers: 0,
            denseIntermediateSize: 0,
            mlaKVLoraRank: 0,
            mlaQKNopeDim: 0,
            mlaQKRopeDim: 0,
            mlaVHeadDim: 0)
        try crossCheckProductionGptOss(arch, tc: tc, configPath: configPath)
        return arch
    }

    /// Production gpt-oss-20b contract. The runtime's `ArchConfig.gptOss20b`
    /// bakes in YaRN factor 32 over original 4096 and the clamped SwiGLU with
    /// limit 7.0, so a production-shaped config that disagrees on the trained
    /// facts must be rejected here rather than repacked into a silently wrong
    /// model. Toy/synthetic configs (different hidden/layer count) are exempt.
    private static func crossCheckProductionGptOss(_ a: ArchInfo,
                                                   tc: [String: Any],
                                                   configPath: String) throws {
        guard a.hiddenSize == 2880, a.numLayers == 24 else { return }
        func fail(_ detail: String) -> RepackError {
            RepackError.configJsonInvalid(
                path: configPath,
                detail: "gpt_oss config does not match the supported gpt-oss-20b "
                    + "architecture contract: \(detail)")
        }
        let expectedMask = (0..<24).map { UInt8($0 % 2 == 0 ? 0 : 1) }
        guard a.numHeads == 64, a.numKVHeads == 8, a.headDim == 64,
              a.vocabSize == 201_088, a.slidingWindow == 128,
              a.numExperts == 32, a.topKExperts == 4,
              a.moeIntermediateSize == 2880, a.ropeTheta == 150_000,
              a.fullAttentionLayerMask == expectedMask else {
            throw fail("core dimensions")
        }
        guard let limit = (tc["swiglu_limit"] as? Double)
            ?? (tc["swiglu_limit"] as? NSNumber)?.doubleValue, limit == 7.0 else {
            throw fail("swiglu_limit must be 7.0")
        }
        guard let scaling = tc["rope_scaling"] as? [String: Any],
              (scaling["rope_type"] as? String) == "yarn",
              ((scaling["factor"] as? Double)
                ?? (scaling["factor"] as? NSNumber)?.doubleValue) == 32.0,
              ((scaling["original_max_position_embeddings"] as? Int)
                ?? (scaling["original_max_position_embeddings"] as? NSNumber)?.intValue)
                == 4096 else {
            throw fail("rope_scaling must be yarn, factor 32, original 4096")
        }
    }

    // MARK: - Kimi-Linear (`model_type == "kimi_linear"`, flat config)

    private static func kimiLayerMask(_ tc: [String: Any],
                                      numLayers: Int,
                                      configPath: String) throws -> [UInt8] {
        guard let lac = tc["linear_attn_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing linear_attn_config")
        }
        func layerList(_ k: String) throws -> [Int] {
            guard let raw = lac[k] as? [Any] else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "missing linear_attn_config.\(k)")
            }
            return try raw.map {
                guard let n = ($0 as? Int) ?? ($0 as? NSNumber)?.intValue else {
                    throw RepackError.configJsonInvalid(
                        path: configPath, detail: "non-integer entry in \(k)")
                }
                return n
            }
        }
        // Both lists are 1-indexed in the checkpoint config and must partition
        // 1...numLayers exactly.
        var mask = [UInt8](repeating: 255, count: numLayers)
        for l in try layerList("kda_layers") {
            guard l >= 1, l <= numLayers, mask[l - 1] == 255 else {
                throw RepackError.configJsonInvalid(
                    path: configPath, detail: "kda_layers entry \(l) out of range or duplicate")
            }
            mask[l - 1] = 2
        }
        for l in try layerList("full_attn_layers") {
            guard l >= 1, l <= numLayers, mask[l - 1] == 255 else {
                throw RepackError.configJsonInvalid(
                    path: configPath,
                    detail: "full_attn_layers entry \(l) out of range or duplicate")
            }
            mask[l - 1] = 3
        }
        guard !mask.contains(255) else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "kda_layers and full_attn_layers must cover every layer")
        }
        return mask
    }

    private static func loadKimiLinear(configPath: String,
                                       tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int { try intField(tc, k, configPath: configPath) }
        func d(_ k: String) throws -> Double { try doubleField(tc, k, configPath: configPath) }
        let numLayers = try i("num_hidden_layers")
        let mask = try kimiLayerMask(tc, numLayers: numLayers, configPath: configPath)
        guard let lac = tc["linear_attn_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(
                path: configPath, detail: "missing linear_attn_config")
        }
        func li(_ k: String) throws -> Int { try intField(lac, k, configPath: configPath) }
        let kvLoraRank = try i("kv_lora_rank")
        let qkNope = try i("qk_nope_head_dim")
        let qkRope = try i("qk_rope_head_dim")
        // MLA runs as MQA over one [latent | rope] row per token; the score
        // scale stays the original per-head q dimension (nope + rope).
        let mqaDim = kvLoraRank + qkRope
        let moeIntermediate = try i("moe_intermediate_size")
        let linearHeads = try li("num_heads")
        let linearHeadDim = try li("head_dim")
        let theta = try d("rope_theta")
        let arch = ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("num_shared_experts") * moeIntermediate,
            moeIntermediateSize: moeIntermediate,
            numHeads: try i("num_attention_heads"),
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: mqaDim,
            fullHeadDim: mqaDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: 0.0,
            numLayers: numLayers,
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_token"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: (tc["hidden_act"] as? String) ?? "silu",
            family: .kimiLinear48b,
            attnOutputGate: false,
            attentionScale: 1.0 / Double(qkNope + qkRope).squareRoot(),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: false,
            linearNumKHeads: linearHeads,
            linearNumVHeads: linearHeads,
            linearKeyHeadDim: linearHeadDim,
            linearValueHeadDim: linearHeadDim,
            linearConvKernelSize: try li("short_conv_kernel_size"),
            numLeadingDenseLayers: try i("first_k_dense_replace"),
            denseIntermediateSize: try i("intermediate_size"),
            mlaKVLoraRank: kvLoraRank,
            mlaQKNopeDim: qkNope,
            mlaQKRopeDim: qkRope,
            mlaVHeadDim: try i("v_head_dim"))
        try crossCheckProductionKimiLinear(arch, tc: tc, configPath: configPath)
        return arch
    }

    /// Production Kimi-Linear-48B-A3B contract. The runtime's ArchConfig bakes
    /// in the sigmoid router with correction bias, renormalization, and the
    /// 2.446 routed scaling factor, plus NoPE MLA — reject a production-shaped
    /// config that disagrees. Toy/synthetic configs are exempt.
    private static func crossCheckProductionKimiLinear(_ a: ArchInfo,
                                                       tc: [String: Any],
                                                       configPath: String) throws {
        guard a.hiddenSize == 2304, a.numLayers == 27 else { return }
        func fail(_ detail: String) -> RepackError {
            RepackError.configJsonInvalid(
                path: configPath,
                detail: "kimi_linear config does not match the supported "
                    + "Kimi-Linear-48B-A3B architecture contract: \(detail)")
        }
        guard a.numHeads == 32, a.vocabSize == 163_840,
              a.numExperts == 256, a.topKExperts == 8,
              a.moeIntermediateSize == 1024, a.intermediateSize == 1024,
              a.denseIntermediateSize == 9216, a.numLeadingDenseLayers == 1,
              a.mlaKVLoraRank == 512, a.mlaQKNopeDim == 128,
              a.mlaQKRopeDim == 64, a.mlaVHeadDim == 128,
              a.linearNumKHeads == 32, a.linearKeyHeadDim == 128,
              a.linearConvKernelSize == 4,
              a.fullAttentionLayerMask.filter({ $0 == 3 }).count == 7,
              a.fullAttentionLayerMask.filter({ $0 == 2 }).count == 20 else {
            throw fail("core dimensions")
        }
        guard (tc["moe_router_activation_func"] as? String) == "sigmoid",
              (tc["moe_renormalize"] as? Bool) == true,
              (tc["mla_use_nope"] as? Bool) == true,
              ((tc["routed_scaling_factor"] as? Double)
                ?? (tc["routed_scaling_factor"] as? NSNumber)?.doubleValue) == 2.446,
              ((tc["num_expert_group"] as? Int)
                ?? (tc["num_expert_group"] as? NSNumber)?.intValue) == 1,
              ((tc["topk_group"] as? Int)
                ?? (tc["topk_group"] as? NSNumber)?.intValue) == 1 else {
            throw fail("router/MLA behavior flags")
        }
    }
}
