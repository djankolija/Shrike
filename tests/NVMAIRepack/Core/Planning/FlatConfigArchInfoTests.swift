import Darwin
import Foundation
import Testing
@testable import NVMAIRepackCore

/// `ArchInfo.load` for the two flat-config families (no `text_config` block,
/// flat `model.` tensor names): gpt-oss-20b and Kimi-Linear-48B-A3B. The
/// fixture JSON is the production checkpoint config, verbatim except that
/// Kimi's per-tensor quantization overrides are trimmed to one representative
/// entry (ArchInfo never reads the quantization block).
@Suite
struct FlatConfigArchInfoTests {

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("nvmai-flat-arch-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func writeConfig(_ json: String, slug: String) throws -> (root: String, path: String) {
        let root = temporaryRoot(slug)
        let path = (root as NSString).appendingPathComponent("config.json")
        try json.data(using: .utf8)!.write(to: URL(fileURLWithPath: path))
        return (root, path)
    }

    private func mutatedConfig(_ json: String,
                               mutate: (inout [String: Any]) -> Void) throws -> String {
        var obj = try JSONSerialization.jsonObject(with: json.data(using: .utf8)!)
            as! [String: Any]  // lint:allow-force fixture JSON is a literal in this file
        mutate(&obj)
        let data = try JSONSerialization.data(withJSONObject: obj)
        return String(data: data, encoding: .utf8)!
    }

    // MARK: - gpt-oss-20b

    @Test func gptOssArchInfoLoadsFromRealConfig() throws {
        let (root, path) = try writeConfig(Self.gptOssConfig, slug: "gptoss-arch")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let arch = try ArchInfo.load(configPath: path)

        #expect(arch.family == .gptOss20b)
        #expect(arch.hiddenSize == 2880)
        #expect(arch.numLayers == 24)
        #expect(arch.numHeads == 64)
        #expect(arch.numKVHeads == 8)
        #expect(arch.numFullKVHeads == 8)
        #expect(arch.headDim == 64)
        #expect(arch.fullHeadDim == 64)
        #expect(arch.vocabSize == 201_088)
        #expect(arch.slidingWindow == 128)
        #expect(arch.numExperts == 32)
        #expect(arch.topKExperts == 4)
        #expect(arch.moeIntermediateSize == 2880)
        #expect(arch.intermediateSize == 0)
        #expect(arch.ropeTheta == 150_000)
        #expect(arch.partialRotaryFactor == 1.0)
        #expect(arch.attentionScale == 0.125)  // 64^-0.5
        #expect(arch.attnOutputGate == false)
        #expect(arch.sharedExpertGated == false)
        #expect(arch.ropeNeoxSubdim == true)
        #expect(arch.tieWordEmbeddings == false)
        #expect(arch.fullAttentionLayerMask.count == 24)
        for (i, v) in arch.fullAttentionLayerMask.enumerated() {
            #expect(v == (i % 2 == 0 ? 0 : 1))
        }
        #expect(arch.linearNumKHeads == 0)
        #expect(arch.numLeadingDenseLayers == 0)
        #expect(arch.mlaKVLoraRank == 0)
    }

    @Test func gptOssRejectsAlteredRopeScaling() throws {
        let mutated = try mutatedConfig(Self.gptOssConfig) { obj in
            var scaling = obj["rope_scaling"] as! [String: Any]  // lint:allow-force fixture literal
            scaling["factor"] = 8.0
            obj["rope_scaling"] = scaling
        }
        let (root, path) = try writeConfig(mutated, slug: "gptoss-arch-bad-rope")
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func gptOssRejectsAlteredSwigluLimit() throws {
        let mutated = try mutatedConfig(Self.gptOssConfig) { obj in
            obj["swiglu_limit"] = 30.0
        }
        let (root, path) = try writeConfig(mutated, slug: "gptoss-arch-bad-swiglu")
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    // MARK: - Kimi-Linear-48B-A3B

    @Test func kimiArchInfoLoadsFromRealConfig() throws {
        let (root, path) = try writeConfig(Self.kimiConfig, slug: "kimi-arch")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let arch = try ArchInfo.load(configPath: path)

        #expect(arch.family == .kimiLinear48b)
        #expect(arch.hiddenSize == 2304)
        #expect(arch.numLayers == 27)
        #expect(arch.numHeads == 32)
        #expect(arch.numKVHeads == 1)
        #expect(arch.numFullKVHeads == 1)
        #expect(arch.headDim == 576)   // kv_lora_rank + qk_rope_head_dim
        #expect(arch.fullHeadDim == 576)
        #expect(arch.vocabSize == 163_840)
        #expect(arch.slidingWindow == 0)
        #expect(arch.numExperts == 256)
        #expect(arch.topKExperts == 8)
        #expect(arch.moeIntermediateSize == 1024)
        #expect(arch.intermediateSize == 1024)  // one ungated shared expert
        #expect(arch.denseIntermediateSize == 9216)
        #expect(arch.numLeadingDenseLayers == 1)
        #expect(arch.attentionScale == 1.0 / Double(192).squareRoot())
        #expect(arch.attnOutputGate == false)
        #expect(arch.sharedExpertGated == false)
        #expect(arch.ropeNeoxSubdim == false)
        #expect(arch.partialRotaryFactor == 0.0)
        #expect(arch.mlaKVLoraRank == 512)
        #expect(arch.mlaQKNopeDim == 128)
        #expect(arch.mlaQKRopeDim == 64)
        #expect(arch.mlaVHeadDim == 128)
        #expect(arch.linearNumKHeads == 32)
        #expect(arch.linearNumVHeads == 32)
        #expect(arch.linearKeyHeadDim == 128)
        #expect(arch.linearValueHeadDim == 128)
        #expect(arch.linearConvKernelSize == 4)
        let fullLayers = arch.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 3 }.map { $0.offset }
        #expect(fullLayers == [3, 7, 11, 15, 19, 23, 26])
        #expect(arch.fullAttentionLayerMask.filter { $0 == 2 }.count == 20)
    }

    @Test func kimiRejectsWrongRouterActivation() throws {
        let mutated = try mutatedConfig(Self.kimiConfig) { obj in
            obj["moe_router_activation_func"] = "softmax"
        }
        let (root, path) = try writeConfig(mutated, slug: "kimi-arch-bad-router")
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func kimiRejectsIncompleteLayerPartition() throws {
        let mutated = try mutatedConfig(Self.kimiConfig) { obj in
            var lac = obj["linear_attn_config"] as! [String: Any]  // lint:allow-force fixture literal
            var kda = lac["kda_layers"] as! [Int]  // lint:allow-force fixture literal
            kda.removeLast()
            lac["kda_layers"] = kda
            obj["linear_attn_config"] = lac
        }
        let (root, path) = try writeConfig(mutated, slug: "kimi-arch-bad-partition")
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    @Test func unknownModelTypeIsRejected() throws {
        let (root, path) = try writeConfig(
            #"{"model_type": "llama", "hidden_size": 64}"#, slug: "unknown-arch")
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(throws: RepackError.self) {
            _ = try ArchInfo.load(configPath: path)
        }
    }

    // MARK: - Fixtures

    static let gptOssConfig = """
    {
        "architectures": ["GptOssForCausalLM"],
        "attention_bias": true,
        "attention_dropout": 0.0,
        "eos_token_id": 200002,
        "experts_per_token": 4,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 2880,
        "initial_context_length": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 2880,
        "layer_types": [
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            "sliding_attention", "full_attention", "sliding_attention", "full_attention"
        ],
        "max_position_embeddings": 131072,
        "model_type": "gpt_oss",
        "num_attention_heads": 64,
        "num_experts_per_tok": 4,
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "output_router_logits": false,
        "pad_token_id": 199999,
        "quantization": {"group_size": 64, "bits": 4},
        "quantization_config": {"group_size": 64, "bits": 4},
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "rope_type": "yarn",
            "truncate": false
        },
        "rope_theta": 150000,
        "router_aux_loss_coef": 0.9,
        "sliding_window": 128,
        "swiglu_limit": 7.0,
        "tie_word_embeddings": false,
        "transformers_version": "4.55.0.dev0",
        "use_cache": true,
        "vocab_size": 201088
    }
    """

    static let kimiConfig = """
    {
        "architectures": ["KimiLinearForCausalLM"],
        "bos_token_id": 163584,
        "dtype": "bfloat16",
        "eos_token_id": 163586,
        "first_k_dense_replace": 1,
        "head_dim": 72,
        "hidden_act": "silu",
        "hidden_size": 2304,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "kv_lora_rank": 512,
        "linear_attn_config": {
            "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
            "head_dim": 128,
            "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19,
                           21, 22, 23, 25, 26],
            "num_heads": 32,
            "short_conv_kernel_size": 4
        },
        "mla_use_nope": true,
        "model_max_length": 1048576,
        "model_type": "kimi_linear",
        "moe_intermediate_size": 1024,
        "moe_layer_freq": 1,
        "moe_renormalize": true,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 32,
        "num_expert_group": 1,
        "num_experts": 256,
        "num_experts_per_token": 8,
        "num_hidden_layers": 27,
        "num_key_value_heads": 32,
        "num_nextn_predict_layers": 0,
        "num_shared_experts": 1,
        "pad_token_id": 163839,
        "q_lora_rank": null,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            "model.layers.1.mlp.gate": {"group_size": 64, "bits": 8}
        },
        "rms_norm_eps": 1e-05,
        "rope_scaling": null,
        "rope_theta": 10000.0,
        "routed_scaling_factor": 2.446,
        "tie_word_embeddings": false,
        "topk_group": 1,
        "transformers_version": "4.57.1",
        "use_cache": true,
        "use_grouped_topk": true,
        "v_head_dim": 128,
        "vocab_size": 163840
    }
    """
}
