import Foundation
import Testing
import ShrikeFormat
@testable import ShrikeRepackCore

/// A layer whose MLP is dense carries an empty expert list and no layer file
/// (Kimi-Linear layer 0). The structural validator and the manifest/layout
/// cross-check must accept that shape without loosening anything else.
@Suite
struct DenseLayerLayoutTests {

    private func subTensor() -> GTurboSubTensorV1 {
        GTurboSubTensorV1(offset: 0, size: 64, dtype: "U32",
                          shape: [8, 4], bits: 4)
    }

    private func layout(withEmptyLayer emptyLayer: Int?) -> GTurboPackedExpertsLayoutV1 {
        let stride: UInt64 = 16_384
        var layers: [GTurboLayerV1] = []
        for L in 0..<3 {
            if L == emptyLayer {
                layers.append(GTurboLayerV1(layer: L, file: "layer_0\(L).bin", experts: []))
                continue
            }
            let experts = (0..<2).map { e in
                GTurboExpertV1(expert: e, physicalRank: e,
                               offset: UInt64(e) * stride, size: stride,
                               tensors: ["gate": subTensor()])
            }
            layers.append(GTurboLayerV1(layer: L, file: "layer_0\(L).bin", experts: experts))
        }
        return GTurboPackedExpertsLayoutV1(expertStride: stride, numLayers: 3,
                                           expertsPerLayer: 2, layers: layers)
    }

    @Test func emptyLayerValidatesStructurally() throws {
        try GTurboV1StructuralValidator.validate(layout(withEmptyLayer: 0))
    }

    @Test func partialExpertCountStillRejected() {
        let stride: UInt64 = 16_384
        let one = [GTurboExpertV1(expert: 0, physicalRank: 0, offset: 0, size: stride,
                                  tensors: ["gate": subTensor()])]
        let bad = GTurboPackedExpertsLayoutV1(
            expertStride: stride, numLayers: 1, expertsPerLayer: 2,
            layers: [GTurboLayerV1(layer: 0, file: "layer_00.bin", experts: one)])
        #expect(throws: (any Error).self) {
            try GTurboV1StructuralValidator.validate(bad)
        }
    }

    @Test func crossValidateSkipsEmptyLayerFile() throws {
        let l = layout(withEmptyLayer: 0)
        let sizes: [String: UInt64] = [
            "packed_experts/layer_01.bin": 2 * 16_384,
            "packed_experts/layer_02.bin": 2 * 16_384,
        ]
        try GTurboV1StructuralValidator.crossValidate(
            manifestNumLayers: 3, manifestExpertsPerLayer: 2,
            manifestExpertStride: 16_384, manifestFileSizes: sizes, layout: l)
    }

    @Test func crossValidateStillRequiresPopulatedLayerFiles() {
        let l = layout(withEmptyLayer: 0)
        let sizes: [String: UInt64] = [
            "packed_experts/layer_01.bin": 2 * 16_384,
        ]
        #expect(throws: (any Error).self) {
            try GTurboV1StructuralValidator.crossValidate(
                manifestNumLayers: 3, manifestExpertsPerLayer: 2,
                manifestExpertStride: 16_384, manifestFileSizes: sizes, layout: l)
        }
    }
}
