import Testing
@testable import Shrike

@Suite struct RuntimeConfigurationTests {
    @Test func publicContextChoicesReachQwenMaximum() {
        #expect(RuntimeConfiguration.supportedContextTokens
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
        #expect(RuntimeConfiguration.supportedContextTokens.last
            == RuntimeConfiguration.nativeMaximumContextTokens)
        #expect(RuntimeConfiguration.supportedYaRNContextTokens == [524_288, 1_048_576])
    }

    @Test func productionDefaultsAreStable() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 16,
            prefillEnabled: true,
            prefillChunkTokens: 128,
            forceLogitsHead: false)
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.headPath == .fusedRows)
        #expect(runtime.kvCachePrecision == .int8)
        #expect(runtime.ropeScalingMode == .none)
    }

    @Test func contextScalingValidationIsFailClosed() throws {
        let native = try RuntimeConfiguration()
        try native.validate(maxContext: 262_144)
        #expect(throws: RuntimeConfigurationError.self) {
            try native.validate(maxContext: 524_288)
        }
        let yarn = try RuntimeConfiguration(ropeScalingMode: .yarn,
                                            yarnContextTokens: 524_288)
        try yarn.validate(maxContext: 524_288)
        #expect(throws: RuntimeConfigurationError.self) {
            try yarn.validate(maxContext: 1_048_576)
        }
    }

    @Test func retainedControlsReachTypedRuntime() throws {
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: 32,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.headPath == .logits)
    }

    @Test func prefetchTracePathReadsTheEnvironment() {
        #expect(RuntimeConfiguration.environmentPrefetchTracePath([:]) == nil)
        #expect(RuntimeConfiguration.environmentPrefetchTracePath(["SHRIKE_PREFETCH_TRACE": ""]) == nil)
        #expect(RuntimeConfiguration.environmentPrefetchTracePath(
            ["SHRIKE_PREFETCH_TRACE": "/tmp/prefetch.jsonl"]) == "/tmp/prefetch.jsonl")
        #expect(RuntimeConfiguration.production.prefetchTracePath == nil)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) throws {
        let runtime = try RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
