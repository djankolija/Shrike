import Foundation
import Testing
@testable import ShrikeRepackCore

/// `RepackPlanner.classify` and name normalization for the flat-name families.
@Suite
struct FlatNameClassificationTests {

    @Test func gptOssNamesClassify() {
        let f = RepackModelFamily.gptOss20b
        func c(_ n: String) -> RepackPlanner.Bucket {
            RepackPlanner.classify(n, numLayers: 24, family: f)
        }
        #expect(c("model.embed_tokens.weight") == .lmResident)
        #expect(c("model.norm.weight") == .lmResident)
        #expect(c("lm_head.weight") == .lmResident)
        #expect(c("model.layers.0.self_attn.q_proj.weight") == .lmResident)
        #expect(c("model.layers.0.self_attn.q_proj.bias") == .lmResident)
        #expect(c("model.layers.7.self_attn.sinks") == .lmResident)
        #expect(c("model.layers.3.mlp.router.weight") == .lmResident)
        #expect(c("model.layers.3.mlp.router.bias") == .lmResident)
        #expect(c("model.layers.23.input_layernorm.weight") == .lmResident)
        #expect(c("model.layers.5.mlp.experts.gate_proj.weight")
            == .routedExpert(role: "gate", layer: 5))
        #expect(c("model.layers.5.mlp.experts.up_proj.weight")
            == .routedExpert(role: "up", layer: 5))
        #expect(c("model.layers.5.mlp.experts.down_proj.weight")
            == .routedExpert(role: "down", layer: 5))
        #expect(RepackPlanner.isRoutedExpertAdditiveBias(
            "model.layers.5.mlp.experts.gate_proj.bias", family: f))
        #expect(!RepackPlanner.isRoutedExpertAdditiveBias(
            "model.layers.5.mlp.router.bias", family: f))
        #expect(!RepackPlanner.isRoutedExpertAdditiveBias(
            "model.layers.5.self_attn.q_proj.bias", family: f))
        #expect(c("language_model.model.layers.0.self_attn.q_proj.weight") == .unknown)
        #expect(c("vision_tower.blocks.0.attn.qkv.weight") == .unknown)
    }

    @Test func kimiNamesClassify() {
        let f = RepackModelFamily.kimiLinear48b
        func c(_ n: String) -> RepackPlanner.Bucket {
            RepackPlanner.classify(n, numLayers: 27, family: f)
        }
        #expect(c("model.embed_tokens.weight") == .lmResident)
        #expect(c("model.layers.0.mlp.gate_proj.weight") == .lmResident)
        #expect(c("model.layers.1.mlp.gate.weight") == .lmResident)
        #expect(c("model.layers.1.mlp.e_score_correction_bias") == .lmResident)
        #expect(c("model.layers.1.mlp.shared_experts.gate_proj.weight") == .lmResident)
        #expect(c("model.layers.2.self_attn.q_conv.conv.weight") == .lmResident)
        #expect(c("model.layers.2.self_attn.A_log") == .lmResident)
        #expect(c("model.layers.3.self_attn.kv_b_proj.weight") == .lmResident)
        #expect(c("model.layers.1.mlp.switch_mlp.gate_proj.weight")
            == .routedExpert(role: "gate", layer: 1))
        #expect(c("model.layers.26.mlp.switch_mlp.down_proj.weight")
            == .routedExpert(role: "down", layer: 26))
        #expect(!RepackPlanner.isRoutedExpertAdditiveBias(
            "model.layers.1.mlp.switch_mlp.gate_proj.weight", family: f))
    }

    @Test func flatNamesNormalizeToContractPrefix() {
        for f in [RepackModelFamily.gptOss20b, .kimiLinear48b] {
            #expect(RepackPlanner.residentDestinationName(
                "model.layers.3.self_attn.q_proj.weight", family: f)
                == "language_model.model.layers.3.self_attn.q_proj.weight")
            #expect(RepackPlanner.residentDestinationName(
                "model.embed_tokens.weight", family: f)
                == "language_model.model.embed_tokens.weight")
            #expect(RepackPlanner.residentDestinationName(
                "model.norm.weight", family: f)
                == "language_model.model.norm.weight")
            #expect(RepackPlanner.residentDestinationName(
                "lm_head.weight", family: f)
                == "language_model.lm_head.weight")
        }
        #expect(RepackPlanner.residentDestinationName(
            "language_model.model.norm.weight", family: .qwen36)
            == "language_model.model.norm.weight")
        #expect(RepackPlanner.residentDestinationName(
            "layers.0.self_attn.q_proj.weight", family: .qwen36MTP)
            == "language_model.model.layers.0.self_attn.q_proj.weight")
    }

    @Test func qwenClassificationUnchanged() {
        func c(_ n: String) -> RepackPlanner.Bucket {
            RepackPlanner.classify(n, numLayers: 40, family: .qwen36)
        }
        #expect(c("language_model.model.layers.0.linear_attn.in_proj_qkv.weight") == .lmResident)
        #expect(c("language_model.model.layers.9.mlp.switch_mlp.up_proj.weight")
            == .routedExpert(role: "up", layer: 9))
        #expect(c("model.layers.0.self_attn.q_proj.weight") == .unknown)
        #expect(c("vision_tower.blocks.0.attn.qkv.weight") == .excludedMultimodal)
    }
}
