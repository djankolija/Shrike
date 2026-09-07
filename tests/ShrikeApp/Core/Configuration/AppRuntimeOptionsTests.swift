import Foundation
import Testing
import Shrike
@testable import ShrikeAppCore

@Suite struct AppRuntimeOptionsTests {
    @Test func defaultsMatchProduction() throws {
        let options = AppRuntimeOptions()
        #expect(options.expertCacheSlots == 64)
        #expect(options.prefillEnabled)
        #expect(options.prefillChunkTokens == 4096)
        #expect(options.modelVerification == .fullSha256)
        #expect(options.kvCachePrecision == .int8)
        #expect(options.ropeScalingMode == .none)
        #expect(options.thinkingMode == .off)

        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime.expertCacheSlots == RuntimeConfiguration.production.expertCacheSlots)
        #expect(runtime.prefillConfig.chunkTokens == 4096)
        #expect(runtime.headPath == RuntimeConfiguration.production.headPath)
        #expect(options.resultSummary ==
            "Cache 64, prefill 4096, 8-bit KV, native RoPE, thinking off, full SHA-256")
    }

    @Test func everyPublicChoiceMapsToRuntime() throws {
        for slots in AppRuntimeOptions.allowedSlotCounts {
            let runtime = try AppRuntimeOptions(expertCacheSlots: slots)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.expertCacheSlots == slots)
        }
        for chunk in AppRuntimeOptions.allowedPrefillChunkTokens {
            let runtime = try AppRuntimeOptions(prefillChunkTokens: chunk)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.prefillConfig.chunkTokens == chunk)
        }
        for precision in KVCachePrecision.allCases {
            let runtime = try AppRuntimeOptions(kvCachePrecision: precision)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.kvCachePrecision == precision)
        }
    }

    @Test func runtimeAndTrustChoicesAreExplicit() throws {
        let options = AppRuntimeOptions(
            expertCacheSlots: 32,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            modelVerification: .trustedInstall)
        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.headPath == .logits)
        #expect(options.modelVerification.runtimeValue == .sizeCheckTrustedReceipt)
    }

    @Test func validationRejectsValuesOutsideClosedSets() {
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(expertCacheSlots: 12).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(prefillChunkTokens: 96).validate()
        }
    }

    @Test func loadedRuntimeKeyTracksOnlyLoadTimeChoices() {
        let directory = URL(fileURLWithPath: "/tmp/model.gturbo")
        let base = AppRuntimeOptions()
        let baseline = AppLoadedRuntimeKey(
            modelDirectory: directory, maxContextTokens: 4096, options: base)

        var variants: [AppRuntimeOptions] = []
        var value = base
        value.expertCacheSlots = 24; variants.append(value)
        value = base; value.modelVerification = .trustedInstall; variants.append(value)
        value = base; value.kvCachePrecision = .int4; variants.append(value)
        value = base; value.ropeScalingMode = .yarn; variants.append(value)
        value = base; value.thinkingMode = .on; variants.append(value)

        for variant in variants {
            #expect(AppLoadedRuntimeKey(
                modelDirectory: directory,
                maxContextTokens: 4096,
                options: variant) != baseline)
        }
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: base,
            forceLogitsHead: true) != baseline)

        value = base; value.prefillEnabled = false
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: value) == baseline)
        value = base; value.prefillChunkTokens = 64
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: value) == baseline)
    }
}
