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
            expertCachePolicy: .lfu,
            rdadvisePolicy: .off,
            prefillEnabled: true,
            prefillChunkTokens: 128,
            prefillAttentionPath: .fullTensorOps2DPreferred,
            forceLogitsHead: false)
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(runtime.decodeExpertExecution == .speculative)
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
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            decodeExpertExecution: .barrier)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.decodeExpertExecution == .barrier)
    }

    @Test func decodeExpertExecutionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeDecodeExpertExecution.environmentValue([:]) == .speculative)
        #expect(try RuntimeDecodeExpertExecution.environmentValue([
            "SHRIKE_DECODE_EXPERT_EXECUTION": "barrier",
        ]) == .barrier)
        #expect(try RuntimeDecodeExpertExecution.environmentValue([
            "SHRIKE_DECODE_EXPERT_EXECUTION": "gpu-residency",
        ]) == .gpuResidency)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeDecodeExpertExecution.environmentValue([
                "SHRIKE_DECODE_EXPERT_EXECUTION": "typo",
            ])
        }
    }

    @Test func specPhase1CoverageEnvironmentIsFailClosed() throws {
        #expect(try RuntimeSpecPhase1Coverage.environmentValue([:]) == .allHit)
        #expect(try RuntimeSpecPhase1Coverage.environmentValue([
            "SHRIKE_SPEC_PHASE1": "hits",
        ]) == .hits)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeSpecPhase1Coverage.environmentValue([
                "SHRIKE_SPEC_PHASE1": "typo",
            ])
        }
    }

    @Test func routerWakeEnvironmentIsFailClosed() throws {
        #expect(try RuntimeRouterWake.environmentValue([:]) == .word)
        #expect(try RuntimeRouterWake.environmentValue([
            "SHRIKE_ROUTER_WAKE": "status",
        ]) == .status)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeRouterWake.environmentValue([
                "SHRIKE_ROUTER_WAKE": "typo",
            ])
        }
    }

    @Test func prefetchEnvironmentIsFailClosed() throws {
        #expect(try RuntimePrefetch.environmentValue([:]) == .production)
        #expect(RuntimePrefetch.production
            == RuntimePrefetch(enabled: true, topM: nil, inFlight: 1, placement: .after,
                               distance: 1, tracePath: nil, adoption: .blit, joinMicros: 400))
        #expect(try RuntimePrefetch.environmentValue(["SHRIKE_PREDICTIVE_PREFETCH": "0"]) == .off)
        #expect(try RuntimePrefetch.environmentValue(["SHRIKE_PREDICTIVE_PREFETCH": "1"]) == .production)
        #expect(try RuntimePrefetch.environmentValue([
            "SHRIKE_PREDICTIVE_PREFETCH": "1",
            "SHRIKE_PREFETCH_TOP_M": "8",
            "SHRIKE_PREFETCH_INFLIGHT": "2",
            "SHRIKE_PREFETCH_PLACEMENT": "beside",
            "SHRIKE_PREFETCH_PROBE_DISTANCE": "2",
            "SHRIKE_PREFETCH_TRACE": "/tmp/prefetch.jsonl",
        ]) == RuntimePrefetch(enabled: true, topM: 8, inFlight: 2, placement: .beside,
                              distance: 2, tracePath: "/tmp/prefetch.jsonl"))
        #expect(try RuntimePrefetch.environmentValue(["SHRIKE_PREFETCH_TRACE": ""]).tracePath == nil)
        #expect(RuntimePrefetch.production.adoption == .blit)
        #expect(RuntimePrefetch.production.joinMicros == 400)
        let copyUnjoined = try RuntimePrefetch.environmentValue([
            "SHRIKE_PREFETCH_ADOPT": "copy", "SHRIKE_PREFETCH_JOIN_US": "0",
        ])
        #expect(copyUnjoined.adoption == .copy)
        #expect(copyUnjoined.joinMicros == 0)
        #expect(try RuntimePrefetch.environmentValue(["SHRIKE_PREFETCH_JOIN_US": "250"]).joinMicros == 250)
        let bad: [[String: String]] = [
            ["SHRIKE_PREDICTIVE_PREFETCH": "yes"],
            ["SHRIKE_PREFETCH_TOP_M": "0"],
            ["SHRIKE_PREFETCH_TOP_M": "many"],
            ["SHRIKE_PREFETCH_INFLIGHT": "0"],
            ["SHRIKE_PREFETCH_INFLIGHT": "9"],
            ["SHRIKE_PREFETCH_PLACEMENT": "typo"],
            ["SHRIKE_PREFETCH_PROBE_DISTANCE": "0"],
            ["SHRIKE_PREFETCH_PROBE_DISTANCE": "far"],
            ["SHRIKE_PREFETCH_ADOPT": "memcpy"],
            ["SHRIKE_PREFETCH_JOIN_US": "-1"],
            ["SHRIKE_PREFETCH_JOIN_US": "2001"],
            ["SHRIKE_PREFETCH_JOIN_US": "soon"],
        ]
        for environment in bad {
            #expect(throws: RuntimeConfigurationError.self) {
                try RuntimePrefetch.environmentValue(environment)
            }
        }
    }

    @Test func configurationDefaultsMatchTheEnvironmentDefaults() throws {
        #expect(RuntimeConfiguration.production.prefetch == .production)
        #expect(RuntimeConfiguration.production.prefetch.enabled)
        #expect(RuntimeConfiguration.production.prefetch
            == (try RuntimePrefetch.environmentValue([:])))
        #expect(RuntimeConfiguration.production.routerWake == .word)
        #expect(RuntimeConfiguration.production.specPhase1Coverage == .allHit)
        #expect(RuntimeConfiguration.production.routerWake
            == (try RuntimeRouterWake.environmentValue([:])))
        #expect(RuntimeConfiguration.production.specPhase1Coverage
            == (try RuntimeSpecPhase1Coverage.environmentValue([:])))
    }

    @Test func expertIOSynchronizationEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSynchronization.environmentValue([:]) == .event)
        #expect(try RuntimeExpertIOSynchronization.environmentValue([
            "SHRIKE_EXPERT_IO_SYNC": "event",
        ]) == .event)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSynchronization.environmentValue([
                "SHRIKE_EXPERT_IO_SYNC": "typo",
            ])
        }
    }

    @Test func expertIOSubmissionEnvironmentIsFailClosed() throws {
        #expect(try RuntimeExpertIOSubmission.environmentValue([:]) == .immediate)
        #expect(try RuntimeExpertIOSubmission.environmentValue([
            "SHRIKE_EXPERT_IO_SUBMISSION": "immediate",
        ]) == .immediate)
        #expect(throws: RuntimeConfigurationError.self) {
            try RuntimeExpertIOSubmission.environmentValue([
                "SHRIKE_EXPERT_IO_SUBMISSION": "typo",
            ])
        }
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) throws {
        let runtime = try RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
