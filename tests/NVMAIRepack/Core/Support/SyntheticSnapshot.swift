import Foundation

/// Synthesises a tiny MLX-affine-quantized safetensors snapshot inside a
/// temporary directory. The remote repack tests need only deterministic bytes
/// plus matching `config.json` and `model.safetensors.index.json` metadata.
enum SyntheticSnapshot {

    struct Snapshot {
        let shardPath: String
    }

    /// Build the snapshot. `seed` controls the pseudo-random payload bytes so
    /// tests can pre-compute byte-fidelity expectations. NVMAI is Qwen-only,
    /// so this produces the same qwen3_5_moe-shaped snapshot as `buildQwen`.
    static func build(at dir: String, seed: UInt64 = 0xA17B_EEF1_5FAC_E202) throws -> Snapshot {
        try buildQwen(at: dir, seed: seed)
    }

    // MARK: - Qwen 3.6 variant

    /// Tiny qwen3_5_moe-shaped architecture: a hybrid of three gated-DeltaNet
    /// linear-attention layers and one full-attention layer, two routed
    /// experts, and an untied lm_head.
    struct QwenArch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let sharedIntermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let numExperts: Int = 2
        let topK: Int = 2
        let groupSize: Int = 64
        let linearNumKHeads: Int = 2
        let linearNumVHeads: Int = 4
        let linearKeyHeadDim: Int = 32
        let linearValueHeadDim: Int = 32
        let linearConvKernelSize: Int = 4
        // layers 0-2 = linear attention, layer 3 = full attention
        let layerTypes: [String] = ["linear_attention", "linear_attention",
                                    "linear_attention", "full_attention"]
        /// Fused qkv rows: 2 * K-dim + V-dim. Also the conv1d channel count.
        var qkvDim: Int { 2 * linearNumKHeads * linearKeyHeadDim + linearNumVHeads * linearValueHeadDim }
        var valueDim: Int { linearNumVHeads * linearValueHeadDim }
    }

    static func buildQwen(at dir: String,
                          weightBits: Int = 4,
                          seed: UInt64 = 0xC0FF_EE00_9A11_AB1E) throws -> Snapshot {
        precondition([4, 8].contains(weightBits))
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = QwenArch()
        var rng = SplitMix64(seed: seed)

        var tensors: [(String, String, [Int], [UInt8])] = []
        tensors.reserveCapacity(96)

        // -- Embedding + untied lm_head (4-bit, group=64)
        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "language_model.lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            if arch.layerTypes[li] == "full_attention" {
                // q_proj emits per-head [query ; gate] halves (attn_output_gate).
                appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                      outerShape: [2 * arch.numHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [arch.numKVHeads * arch.headDim],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.numHeads * arch.headDim, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                      shape: [arch.headDim], into: &tensors, rng: &rng)
            } else {
                // Gated-DeltaNet linear attention bundle.
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_qkv",
                                      outerShape: [arch.qkvDim],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_z",
                                      outerShape: [arch.valueDim],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_a",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_b",
                                      outerShape: [arch.linearNumVHeads],
                                      innerLogical: arch.hidden, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.conv1d.weight",
                                      shape: [arch.qkvDim, arch.linearConvKernelSize, 1],
                                      into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.A_log",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.dt_bias",
                                      shape: [arch.linearNumVHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.norm.weight",
                                      shape: [arch.linearValueHeadDim], into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.out_proj",
                                      outerShape: [arch.hidden],
                                      innerLogical: arch.valueDim, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }

            // Router + sigmoid-gated shared expert — 8-bit gates, 4-bit MLP.
            appendQuantizedWeight(name: prefix + ".mlp.gate",
                                  outerShape: [arch.numExperts], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert_gate",
                                  outerShape: [1], innerLogical: arch.hidden,
                                  bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.gate_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: weightBits, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.up_proj",
                                  outerShape: [arch.sharedIntermediate], innerLogical: arch.hidden,
                                  bits: weightBits, groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.down_proj",
                                  outerShape: [arch.hidden], innerLogical: arch.sharedIntermediate,
                                  bits: weightBits, groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — stacked expert-major, 4-bit.
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.gate_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.up_proj",
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.down_proj",
                                  outerShape: [arch.numExperts, arch.hidden],
                                  innerLogical: arch.moeIntermediate, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)

            // Plain pre-norm layer norms (no sandwich norms).
            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        // Final norm
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        // Vision-tower tensors included to prove the text-only repacker drops them.
        appendUnquantizedBF16(name: "vision_tower.blocks.0.norm1.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)
        appendUnquantizedBF16(name: "vision_tower.patch_embed.proj.weight",
                              shape: [arch.hidden, arch.hidden], into: &tensors, rng: &rng)

        // -- Encode safetensors.
        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        // -- Write config.json with 8-bit overrides for the router + gate.
        var quant: [String: Any] = [
            "bits": weightBits, "group_size": arch.groupSize, "mode": "affine"
        ]
        for li in 0..<arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            for k in ["mlp.gate", "mlp.shared_expert_gate"] {
                quant[prefix + "." + k] = ["bits": 8, "group_size": arch.groupSize]
            }
        }

        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "moe_intermediate_size": arch.moeIntermediate,
            "shared_expert_intermediate_size": arch.sharedIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "num_experts_per_tok": arch.topK,
            "layer_types": arch.layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25
            ],
            "linear_num_key_heads": arch.linearNumKHeads,
            "linear_num_value_heads": arch.linearNumVHeads,
            "linear_key_head_dim": arch.linearKeyHeadDim,
            "linear_value_head_dim": arch.linearValueHeadDim,
            "linear_conv_kernel_dim": arch.linearConvKernelSize,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu"
        ]
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": quant,
            "text_config": textConfig
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        // -- Write model.safetensors.index.json.
        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let indexObj: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))
        try writeTokenizerStubs(at: dir)
        return Snapshot(shardPath: shardPath)
    }


    /// Minimal tokenizer sidecar files so local-import tests exercise the
    /// same required-sidecar policy as a real snapshot.
    static func writeTokenizerStubs(at dir: String) throws {
        let tokenizer: [String: Any] = [
            "version": "1.0",
            "added_tokens": [],
            "decoder": ["type": "ByteLevel"],
            "model": ["type": "BPE", "vocab": [:], "merges": []],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("tokenizer.json")))
        let config: [String: Any] = [
            "tokenizer_class": "TokenizersBackend",
            "clean_up_tokenization_spaces": false,
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (dir as NSString)
                .appendingPathComponent("tokenizer_config.json")))
    }

    /// Tiny one-layer native-MTP sidecar with the same flattened tensor names
    /// used by both Qwen and locally prepared Ornith snapshots.
    static func buildQwenMTP(at dir: String,
                             weightBits: Int = 4,
                             seed: UInt64 = 0x0A11_CE55_1DE0_0001) throws -> Snapshot {
        precondition([4, 8].contains(weightBits))
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)
        let arch = QwenArch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []
        let prefix = "layers.0"
        appendQuantizedWeight(name: "fc", outerShape: [arch.hidden],
                              innerLogical: 2 * arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for name in ["pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight"] {
            appendUnquantizedBF16(name: name, shape: [arch.hidden],
                                  into: &tensors, rng: &rng)
        }
        appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                              outerShape: [2 * arch.numHeads * arch.headDim],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for projection in ["k_proj", "v_proj"] {
            appendQuantizedWeight(name: prefix + ".self_attn." + projection,
                                  outerShape: [arch.numKVHeads * arch.headDim],
                                  innerLogical: arch.hidden, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
        }
        appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                              outerShape: [arch.hidden],
                              innerLogical: arch.numHeads * arch.headDim,
                              bits: weightBits, groupSize: arch.groupSize,
                              into: &tensors, rng: &rng)
        for name in ["q_norm.weight", "k_norm.weight"] {
            appendUnquantizedBF16(name: prefix + ".self_attn." + name,
                                  shape: [arch.headDim], into: &tensors, rng: &rng)
        }
        appendQuantizedWeight(name: prefix + ".mlp.gate",
                              outerShape: [arch.numExperts],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: prefix + ".mlp.shared_expert_gate",
                              outerShape: [1], innerLogical: arch.hidden,
                              bits: weightBits, groupSize: arch.groupSize,
                              into: &tensors, rng: &rng)
        for role in ["gate_proj", "up_proj"] {
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert." + role,
                                  outerShape: [arch.sharedIntermediate],
                                  innerLogical: arch.hidden, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
        }
        appendQuantizedWeight(name: prefix + ".mlp.shared_expert.down_proj",
                              outerShape: [arch.hidden],
                              innerLogical: arch.sharedIntermediate, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for role in ["gate_proj", "up_proj"] {
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp." + role,
                                  outerShape: [arch.numExperts, arch.moeIntermediate],
                                  innerLogical: arch.hidden, bits: weightBits,
                                  groupSize: arch.groupSize, into: &tensors, rng: &rng)
        }
        appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.down_proj",
                              outerShape: [arch.numExperts, arch.hidden],
                              innerLogical: arch.moeIntermediate, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            appendUnquantizedBF16(name: prefix + "." + name,
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        appendUnquantizedBF16(name: "norm.weight", shape: [arch.hidden],
                              into: &tensors, rng: &rng)

        let shardName = "model.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)
        let textConfig: [String: Any] = [
            "hidden_size": arch.hidden,
            "moe_intermediate_size": arch.moeIntermediate,
            "shared_expert_intermediate_size": arch.sharedIntermediate,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "num_experts_per_tok": arch.topK,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "layer_types": arch.layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
            ],
            "linear_num_key_heads": arch.linearNumKHeads,
            "linear_num_value_heads": arch.linearNumVHeads,
            "linear_key_head_dim": arch.linearKeyHeadDim,
            "linear_value_head_dim": arch.linearValueHeadDim,
            "linear_conv_kernel_dim": arch.linearConvKernelSize,
            "attn_output_gate": true,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
        ]
        let config: [String: Any] = [
            "model_type": "qwen3_5_mtp",
            "quantization": [
                "bits": weightBits,
                "group_size": arch.groupSize,
                "mode": "affine",
            ],
            "text_config": textConfig,
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("config.json")))
        let weightMap = Dictionary(uniqueKeysWithValues: tensors.map {
            ($0.0, shardName)
        })
        let index: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap,
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("model.safetensors.index.json")))
        try writeTokenizerStubs(at: dir)
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - gpt-oss variant

    /// Tiny gpt_oss-shaped architecture: four alternating sliding/full
    /// attention layers, additive biases on attention, router, and experts,
    /// attention sinks, and flat (`model.`-prefixed) tensor names.
    struct GptOssArch {
        let hidden: Int = 128
        let intermediate: Int = 64
        let numHeads: Int = 2
        let numKVHeads: Int = 2
        let headDim: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 4
        let numExperts: Int = 2
        let topK: Int = 2
        let groupSize: Int = 64
        let slidingWindow: Int = 8
        let layerTypes: [String] = ["sliding_attention", "full_attention",
                                    "sliding_attention", "full_attention"]
    }

    static func buildGptOss(at dir: String,
                            weightBits: Int = 4,
                            seed: UInt64 = 0x6F55_20B0_0000_0001) throws -> Snapshot {
        precondition([4, 8].contains(weightBits))
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = GptOssArch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        appendQuantizedWeight(name: "model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            let qDim = arch.numHeads * arch.headDim
            let kvDim = arch.numKVHeads * arch.headDim
            for (proj, rows, inner) in [("q_proj", qDim, arch.hidden),
                                        ("k_proj", kvDim, arch.hidden),
                                        ("v_proj", kvDim, arch.hidden),
                                        ("o_proj", arch.hidden, qDim)] {
                appendQuantizedWeight(name: prefix + ".self_attn." + proj,
                                      outerShape: [rows], innerLogical: inner,
                                      bits: weightBits, groupSize: arch.groupSize,
                                      into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.\(proj).bias",
                                      shape: [rows], into: &tensors, rng: &rng)
            }
            appendUnquantizedBF16(name: prefix + ".self_attn.sinks",
                                  shape: [arch.numHeads], into: &tensors, rng: &rng)

            appendQuantizedWeight(name: prefix + ".mlp.router",
                                  outerShape: [arch.numExperts], innerLogical: arch.hidden,
                                  bits: weightBits, groupSize: arch.groupSize,
                                  into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".mlp.router.bias",
                                  shape: [arch.numExperts], into: &tensors, rng: &rng)

            for (role, rows, inner) in [("gate_proj", arch.intermediate, arch.hidden),
                                        ("up_proj", arch.intermediate, arch.hidden),
                                        ("down_proj", arch.hidden, arch.intermediate)] {
                appendQuantizedWeight(name: prefix + ".mlp.experts." + role,
                                      outerShape: [arch.numExperts, rows],
                                      innerLogical: inner, bits: weightBits,
                                      groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".mlp.experts.\(role).bias",
                                      shape: [arch.numExperts, rows],
                                      into: &tensors, rng: &rng)
            }

            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        appendUnquantizedBF16(name: "model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        let config: [String: Any] = [
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
            "hidden_size": arch.hidden,
            "intermediate_size": arch.intermediate,
            "head_dim": arch.headDim,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "vocab_size": arch.vocab,
            "num_hidden_layers": arch.numLayers,
            "num_local_experts": arch.numExperts,
            "num_experts_per_tok": arch.topK,
            "sliding_window": arch.slidingWindow,
            "layer_types": arch.layerTypes,
            "rope_theta": 150_000.0,
            "rope_scaling": [
                "rope_type": "yarn", "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0, "beta_slow": 1.0,
            ],
            "swiglu_limit": 7.0,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "quantization": [
                "bits": weightBits, "group_size": arch.groupSize, "mode": "affine",
            ],
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("config.json")))

        let weightMap = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, shardName) })
        let index: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap,
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("model.safetensors.index.json")))
        try writeTokenizerStubs(at: dir)
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Tensor builders

    private static func appendQuantizedWeight(name: String,
                                              outerShape: [Int],
                                              innerLogical: Int,
                                              bits: Int,
                                              groupSize: Int,
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        precondition(innerLogical % groupSize == 0)
        precondition((innerLogical * bits).isMultiple(of: 32))
        let innerSource = innerLogical * bits / 32
        let shape = outerShape + [innerSource]
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".weight", "U32", shape, bytes))

        // Companions are finite BF16 values, not raw random bytes: random bit
        // patterns include NaN/Inf, which would poison numeric checks that
        // dequantize synthetic tensors.
        let groups = innerLogical / groupSize
        let companionShape = outerShape + [groups]
        let companionElems = companionShape.reduce(1, *)
        var sb = [UInt8]()
        sb.reserveCapacity(companionElems * 2)
        for _ in 0..<companionElems {
            let half = bf16Bits(Float(1 + (rng.next() % 255)) / 128.0)
            sb.append(UInt8(half & 0xFF))
            sb.append(UInt8(half >> 8))
        }
        tensors.append((name + ".scales", "BF16", companionShape, sb))
        var bb = [UInt8]()
        bb.reserveCapacity(companionElems * 2)
        for _ in 0..<companionElems {
            let half = bf16Bits(Float(Int64(rng.next() % 255) - 127) / 256.0)
            bb.append(UInt8(half & 0xFF))
            bb.append(UInt8(half >> 8))
        }
        tensors.append((name + ".biases", "BF16", companionShape, bb))
    }

    private static func bf16Bits(_ value: Float) -> UInt16 {
        UInt16(truncatingIfNeeded: value.bitPattern >> 16)
    }

    private static func appendUnquantizedBF16(name: String, shape: [Int],
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 2)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "BF16", shape, bytes))
    }

    private static func appendUnquantizedF32(name: String, shape: [Int],
                                             into tensors: inout [(String, String, [Int], [UInt8])],
                                             rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for element in 0..<elements {
            // Small finite values keep F32 payloads NaN-free for references.
            var value = Float(Int64(rng.next() & 0xFF) - 128) / 64.0
            withUnsafeBytes(of: &value) { raw in
                for (i, b) in raw.enumerated() { bytes[element * 4 + i] = b }
            }
        }
        tensors.append((name, "F32", shape, bytes))
    }

    // MARK: - Kimi-Linear variant

    /// Tiny kimi_linear-shaped architecture: five layers — four KDA and one
    /// MLA (1-indexed layer 4) — with a dense layer 0, four routed experts
    /// behind an 8-bit sigmoid router, and one ungated shared expert.
    struct KimiArch {
        let hidden: Int = 64
        let denseIntermediate: Int = 128
        let moeIntermediate: Int = 64
        let numHeads: Int = 2
        let vocab: Int = 256
        let numLayers: Int = 5
        let numExperts: Int = 4
        let topK: Int = 2
        let groupSize: Int = 64
        let linearHeads: Int = 2
        let linearHeadDim: Int = 32
        let convKernel: Int = 4
        let lowRank: Int = 64
        let kvLoraRank: Int = 64
        let qkNope: Int = 64
        let qkRope: Int = 64
        let vHeadDim: Int = 64
        let kdaLayers1Indexed: [Int] = [1, 2, 3, 5]
        let mlaLayers1Indexed: [Int] = [4]
        var kdaDim: Int { linearHeads * linearHeadDim }
    }

    static func buildKimiLinear(at dir: String,
                                weightBits: Int = 4,
                                seed: UInt64 = 0x4B12_11FE_A123_0007) throws -> Snapshot {
        precondition([4, 8].contains(weightBits))
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = KimiArch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        appendQuantizedWeight(name: "model.embed_tokens",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "lm_head",
                              outerShape: [arch.vocab],
                              innerLogical: arch.hidden, bits: weightBits,
                              groupSize: arch.groupSize, into: &tensors, rng: &rng)

        for li in 0..<arch.numLayers {
            let prefix = "model.layers.\(li)"
            if arch.mlaLayers1Indexed.contains(li + 1) {
                appendQuantizedWeight(
                    name: prefix + ".self_attn.q_proj",
                    outerShape: [arch.numHeads * (arch.qkNope + arch.qkRope)],
                    innerLogical: arch.hidden, bits: weightBits,
                    groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(
                    name: prefix + ".self_attn.kv_a_proj_with_mqa",
                    outerShape: [arch.kvLoraRank + arch.qkRope],
                    innerLogical: arch.hidden, bits: weightBits,
                    groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.kv_a_layernorm.weight",
                                      shape: [arch.kvLoraRank], into: &tensors, rng: &rng)
                appendQuantizedWeight(
                    name: prefix + ".self_attn.kv_b_proj",
                    outerShape: [arch.numHeads * (arch.qkNope + arch.vHeadDim)],
                    innerLogical: arch.kvLoraRank, bits: weightBits,
                    groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(
                    name: prefix + ".self_attn.o_proj",
                    outerShape: [arch.hidden],
                    innerLogical: arch.numHeads * arch.vHeadDim, bits: weightBits,
                    groupSize: arch.groupSize, into: &tensors, rng: &rng)
            } else {
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    appendQuantizedWeight(
                        name: prefix + ".self_attn." + proj,
                        outerShape: [arch.kdaDim],
                        innerLogical: arch.hidden, bits: weightBits,
                        groupSize: arch.groupSize, into: &tensors, rng: &rng)
                }
                for conv in ["q_conv", "k_conv", "v_conv"] {
                    appendUnquantizedBF16(name: prefix + ".self_attn.\(conv).conv.weight",
                                          shape: [arch.kdaDim, arch.convKernel, 1],
                                          into: &tensors, rng: &rng)
                }
                for (proj, rows, inner) in [("f_a_proj", arch.lowRank, arch.hidden),
                                            ("f_b_proj", arch.kdaDim, arch.lowRank),
                                            ("g_a_proj", arch.lowRank, arch.hidden),
                                            ("g_b_proj", arch.kdaDim, arch.lowRank),
                                            ("b_proj", arch.linearHeads, arch.hidden)] {
                    appendQuantizedWeight(
                        name: prefix + ".self_attn." + proj,
                        outerShape: [rows], innerLogical: inner, bits: weightBits,
                        groupSize: arch.groupSize, into: &tensors, rng: &rng)
                }
                appendUnquantizedF32(name: prefix + ".self_attn.A_log",
                                     shape: [1, 1, arch.linearHeads, 1],
                                     into: &tensors, rng: &rng)
                appendUnquantizedF32(name: prefix + ".self_attn.dt_bias",
                                     shape: [arch.kdaDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.o_norm.weight",
                                      shape: [arch.linearHeadDim], into: &tensors, rng: &rng)
                appendQuantizedWeight(
                    name: prefix + ".self_attn.o_proj",
                    outerShape: [arch.hidden],
                    innerLogical: arch.kdaDim, bits: weightBits,
                    groupSize: arch.groupSize, into: &tensors, rng: &rng)
            }

            if li == 0 {
                for (role, rows, inner) in [("gate_proj", arch.denseIntermediate, arch.hidden),
                                            ("up_proj", arch.denseIntermediate, arch.hidden),
                                            ("down_proj", arch.hidden, arch.denseIntermediate)] {
                    appendQuantizedWeight(
                        name: prefix + ".mlp." + role,
                        outerShape: [rows], innerLogical: inner, bits: weightBits,
                        groupSize: arch.groupSize, into: &tensors, rng: &rng)
                }
            } else {
                appendQuantizedWeight(
                    name: prefix + ".mlp.gate",
                    outerShape: [arch.numExperts], innerLogical: arch.hidden,
                    bits: 8, groupSize: arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".mlp.e_score_correction_bias",
                                      shape: [arch.numExperts], into: &tensors, rng: &rng)
                for (role, rows, inner) in [("gate_proj", arch.moeIntermediate, arch.hidden),
                                            ("up_proj", arch.moeIntermediate, arch.hidden),
                                            ("down_proj", arch.hidden, arch.moeIntermediate)] {
                    appendQuantizedWeight(
                        name: prefix + ".mlp.switch_mlp." + role,
                        outerShape: [arch.numExperts, rows], innerLogical: inner,
                        bits: weightBits, groupSize: arch.groupSize,
                        into: &tensors, rng: &rng)
                    appendQuantizedWeight(
                        name: prefix + ".mlp.shared_experts." + role,
                        outerShape: [rows], innerLogical: inner, bits: weightBits,
                        groupSize: arch.groupSize, into: &tensors, rng: &rng)
                }
            }

            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [arch.hidden], into: &tensors, rng: &rng)
        }
        appendUnquantizedBF16(name: "model.norm.weight",
                              shape: [arch.hidden], into: &tensors, rng: &rng)

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        var quantization: [String: Any] = [
            "bits": weightBits, "group_size": arch.groupSize, "mode": "affine",
        ]
        for li in 1..<arch.numLayers {
            quantization["model.layers.\(li).mlp.gate"] = [
                "bits": 8, "group_size": arch.groupSize,
            ]
        }
        let config: [String: Any] = [
            "architectures": ["KimiLinearForCausalLM"],
            "model_type": "kimi_linear",
            "hidden_size": arch.hidden,
            "intermediate_size": arch.denseIntermediate,
            "moe_intermediate_size": arch.moeIntermediate,
            "num_shared_experts": 1,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numHeads,
            "num_hidden_layers": arch.numLayers,
            "num_experts": arch.numExperts,
            "num_experts_per_token": arch.topK,
            "first_k_dense_replace": 1,
            "kv_lora_rank": arch.kvLoraRank,
            "qk_nope_head_dim": arch.qkNope,
            "qk_rope_head_dim": arch.qkRope,
            "v_head_dim": arch.vHeadDim,
            "vocab_size": arch.vocab,
            "rope_theta": 10_000.0,
            "rms_norm_eps": 1e-5,
            "mla_use_nope": true,
            "moe_renormalize": true,
            "moe_router_activation_func": "sigmoid",
            "routed_scaling_factor": 2.446,
            "num_expert_group": 1,
            "topk_group": 1,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "linear_attn_config": [
                "full_attn_layers": arch.mlaLayers1Indexed,
                "kda_layers": arch.kdaLayers1Indexed,
                "head_dim": arch.linearHeadDim,
                "num_heads": arch.linearHeads,
                "short_conv_kernel_size": arch.convKernel,
            ],
            "quantization": quantization,
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("config.json")))

        let weightMap = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, shardName) })
        let index: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap,
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath:
                (dir as NSString).appendingPathComponent("model.safetensors.index.json")))
        try writeTokenizerStubs(at: dir)
        return Snapshot(shardPath: shardPath)
    }

    // MARK: - Safetensors writer

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerEntries: [(String, [String: Any])] = []
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerEntries.append((name, [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end]
            ]))
            off = end
        }
        var headerDict: [String: Any] = [:]
        for (n, e) in headerEntries { headerDict[n] = e }
        headerDict["__metadata__"] = ["format": "mlx"]
        // Ensure deterministic key ordering for the header — JSONSerialization
        // sortedKeys handles that for us.
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        // Pad header so payload starts on an 8-byte boundary (matches MLX
        // convention and trips fewer downstream surprises).
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) } // space pad

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }
}

/// Tiny deterministic PRNG. We do not need crypto quality — just stable
/// byte streams across test runs.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
