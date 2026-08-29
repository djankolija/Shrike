import Foundation
import Testing
@testable import Shrike

@Suite("Kimi ArchConfig descriptors")
struct KimiArchConfigTests {

    @Test func mlaDimsMatchThePinnedCheckpoint() throws {
        let arch = ArchConfig.kimiLinear48bA3b
        let mla = try #require(arch.mla)
        #expect(mla.latentDim == 512)
        #expect(mla.qkNopeDim == 128)
        #expect(mla.qkRopeDim == 64)
        #expect(mla.valueHeadDim == 128)
        #expect(arch.headDim == mla.latentDim + mla.qkRopeDim)
        #expect(ArchConfig.qwen36_35B_A3B.mla == nil)
        #expect(ArchConfig.gptOss20b.mla == nil)
    }

    @Test func kdaDecayIsPerChannelOnlyForKimi() {
        #expect(ArchConfig.kimiLinear48bA3b.linearAttentionPerChannelDecay)
        #expect(!ArchConfig.qwen36_35B_A3B.linearAttentionPerChannelDecay)
    }

    @Test func sigmoidRouterDescriptor() {
        let arch = ArchConfig.kimiLinear48bA3b
        #expect(arch.routerUsesSigmoidScores)
        #expect(arch.routerHasCorrectionBias)
        #expect(arch.routedScalingFactor == 2.446)
        #expect(!ArchConfig.qwen36_35B_A3B.routerUsesSigmoidScores)
        #expect(ArchConfig.gptOss20b.routedScalingFactor == 1.0)
    }

    @Test func layerMaskCarriesSevenMLALayers() {
        let mask = ArchConfig.kimiLinear48bA3b.fullAttentionLayerMask
        #expect(mask.count == 27)
        #expect(mask.filter { $0 == 3 }.count == 7)
        #expect(mask.filter { $0 == 2 }.count == 20)
        for oneIndexed in [4, 8, 12, 16, 20, 24, 27] {
            #expect(mask[oneIndexed - 1] == 3)
        }
        #expect(ArchConfig.kimiLinear48bA3b.hasMLALayers)
        #expect(ArchConfig.kimiLinear48bA3b.hasLinearAttentionLayers)
        #expect(!ArchConfig.kimiLinear48bA3b.hasSlidingWindowLayers)
    }
}
