import Foundation
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

    @Test func refusesEveryDeletedKnobByName() {
        let deleted = [
            "SHRIKE_DECODE_EXPERT_EXECUTION", "SHRIKE_SPEC_PHASE1", "SHRIKE_ROUTER_WAKE",
            "SHRIKE_HOST_WAIT", "SHRIKE_EXPERT_IO_SYNC", "SHRIKE_EXPERT_IO_SUBMISSION",
            "SHRIKE_RDADVISE_POLICY",
            "SHRIKE_EXPERT_CACHE_LAYOUT", "SHRIKE_EXPERT_IO_BACKEND", "SHRIKE_BOUNDED_IO",
            "SHRIKE_PARALLEL_IO", "SHRIKE_EXPERT_IO_THREADS", "SHRIKE_EXPERT_IO_BATCH_DEPTH",
            "SHRIKE_EXPERT_CACHE_POLICY", "SHRIKE_EXPERT_CACHE_PROTECT", "SHRIKE_NO_PIN",
            "SHRIKE_PREDICTIVE_PREFETCH", "SHRIKE_PREFETCH_TOP_M", "SHRIKE_PREFETCH_INFLIGHT",
            "SHRIKE_PREFETCH_PROBE_DISTANCE", "SHRIKE_PREFETCH_JOIN_US",
            "SHRIKE_PREFETCH_PLACEMENT", "SHRIKE_PREFETCH_PROBE", "SHRIKE_PREFETCH_ADOPT",
            "SHRIKE_ATTN_MATRIX_TILE", "SHRIKE_MPP_TILE_N", "SHRIKE_MPP_TILE_K",
            "SHRIKE_MPP_DEQUANT_BUFFERS", "SHRIKE_MPP_WEIGHT_LOADS", "SHRIKE_PREFILL_ATTENTION",
            "SHRIKE_PREFILL_ROUTER", "SHRIKE_PREFILL_ROUTER_TOKENS", "SHRIKE_PREFILL_ROUTED_GEMM",
            "SHRIKE_PREFILL_ROUTE_OVERLAP", "SHRIKE_PREFILL_POOL_RESIDENCY",
            "SHRIKE_PREFILL_TAIL_TILE", "SHRIKE_PREFILL_TILE_BATCH", "SHRIKE_PREFILL_TILE_DEPTH",
            "SHRIKE_PREFILL_FETCH_DEPTH", "SHRIKE_PREFILL_MATRIX_MIN_ROWS", "SHRIKE_PREFILL_SWEEP",
            "SHRIKE_PREFILL_SWEEP_TAIL", "SHRIKE_GDN_PREFILL_SCAN", "SHRIKE_ATTN_FULL_CHUNKS",
            "SHRIKE_SAMPLER_PATH",
            "SHRIKE_MTP_VERIFY", "SHRIKE_MTP_EXPERT_SLOTS",
            "SHRIKE_EXPERT_CACHE_SLOTS",
            "SHRIKE_LAYER_TRACE", "SHRIKE_GPU_CAPTURE_DIR", "SHRIKE_CACHE_DIAG",
            "SHRIKE_GEN_DIAG", "SHRIKE_PHASES",
        ]
        #expect(deleted.count == 53)
        var environment = Dictionary(uniqueKeysWithValues: deleted.map { ($0, "1") })
        environment["PATH"] = "/usr/bin"
        #expect(throws: RuntimeConfigurationError.unknownEnvironment(deleted.sorted())) {
            try RuntimeConfiguration.refuseUnknownEnvironment(environment)
        }
    }

    @Test func theSurvivingFifteenPass() throws {
        let names = RuntimeConfiguration.knownEnvironmentNames
        #expect(names.count == 15)
        try RuntimeConfiguration.refuseUnknownEnvironment(
            Dictionary(uniqueKeysWithValues: names.map { ($0, "1") }))
    }

    @Test func expertPolicyParsesAndRefuses() throws {
        #expect(try RuntimeConfiguration.environmentExpertPolicy([:]) == .agingLFU)
        #expect(try RuntimeConfiguration.environmentExpertPolicy(["SHRIKE_EXPERT_POLICY": "aging-lfu"]) == .agingLFU)
        #expect(try RuntimeConfiguration.environmentExpertPolicy(["SHRIKE_EXPERT_POLICY": "slru"])
            == .slru(protectedShare: 0.5))
        #expect(try RuntimeConfiguration.environmentExpertPolicy(["SHRIKE_EXPERT_POLICY": "SLRU:0.6"])
            == .slru(protectedShare: 0.6))
        for bad in ["lru", "slru:1", "slru:0", "slru:x", "slru:"] {
            #expect(throws: RuntimeConfigurationError.self) {
                _ = try RuntimeConfiguration.environmentExpertPolicy(["SHRIKE_EXPERT_POLICY": bad])
            }
        }
        #expect(ExpertEvictionPolicy.slru(protectedShare: 0.34).protectedCapacity(slots: 3) == 1)
        #expect(ExpertEvictionPolicy.slru(protectedShare: 0.5).protectedCapacity(slots: 128) == 64)
        #expect(ExpertEvictionPolicy.agingLFU.protectedCapacity(slots: 128) == 0)
    }

    @Test func expertSlotTableParsesAndRefuses() throws {
        let fortyUniform = Array(repeating: 128, count: 40)
        let list = fortyUniform.map(String.init).joined(separator: ",")
        let parsed = try RuntimeConfiguration.environmentExpertSlotTable(
            ["SHRIKE_EXPERT_SLOT_TABLE": list], layers: 40, uniformSlots: 128, leadingDenseLayers: 0)
        #expect(parsed == fortyUniform)
        #expect(try RuntimeConfiguration.environmentExpertSlotTable(
            [:], layers: 40, uniformSlots: 128, leadingDenseLayers: 0) == nil)
        var split = fortyUniform
        split[0] += 40
        split[20] -= 40
        #expect(try RuntimeConfiguration.environmentExpertSlotTable(
            ["SHRIKE_EXPERT_SLOT_TABLE": split.map(String.init).joined(separator: ",")],
            layers: 40, uniformSlots: 128, leadingDenseLayers: 0) == split)
        let short = Array(repeating: 128, count: 39).map(String.init).joined(separator: ",")
        #expect(throws: RuntimeConfigurationError.self) {
            _ = try RuntimeConfiguration.environmentExpertSlotTable(
                ["SHRIKE_EXPERT_SLOT_TABLE": short], layers: 40, uniformSlots: 128, leadingDenseLayers: 0)
        }
        var overBudget = fortyUniform
        overBudget[0] += 1
        #expect(throws: RuntimeConfigurationError.self) {
            _ = try RuntimeConfiguration.environmentExpertSlotTable(
                ["SHRIKE_EXPERT_SLOT_TABLE": overBudget.map(String.init).joined(separator: ",")],
                layers: 40, uniformSlots: 128, leadingDenseLayers: 0)
        }
        var tooFew = fortyUniform
        tooFew[3] = 4
        tooFew[4] += 124
        #expect(throws: RuntimeConfigurationError.self) {
            _ = try RuntimeConfiguration.environmentExpertSlotTable(
                ["SHRIKE_EXPERT_SLOT_TABLE": tooFew.map(String.init).joined(separator: ",")],
                layers: 40, uniformSlots: 128, leadingDenseLayers: 0)
        }
        #expect(throws: RuntimeConfigurationError.self) {
            _ = try RuntimeConfiguration.environmentExpertSlotTable(
                ["SHRIKE_EXPERT_SLOT_TABLE": "128,x"], layers: 2, uniformSlots: 128, leadingDenseLayers: 0)
        }
        let dense = try RuntimeConfiguration.environmentExpertSlotTable(
            ["SHRIKE_EXPERT_SLOT_TABLE": "0,200,56"], layers: 3, uniformSlots: 128, leadingDenseLayers: 1)
        #expect(dense == [0, 200, 56])
        #expect(throws: RuntimeConfigurationError.self) {
            _ = try RuntimeConfiguration.environmentExpertSlotTable(
                ["SHRIKE_EXPERT_SLOT_TABLE": "8,192,56"], layers: 3, uniformSlots: 128, leadingDenseLayers: 1)
        }
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("shrike-slot-table-\(UUID().uuidString).json")
        try Data("{\"0\": 200, \"1\": 56}".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        #expect(try RuntimeConfiguration.environmentExpertSlotTable(
            ["SHRIKE_EXPERT_SLOT_TABLE": file.path], layers: 2, uniformSlots: 128,
            leadingDenseLayers: 0) == [200, 56])
    }

    @Test func ignoresVariablesOutsideThePrefix() throws {
        try RuntimeConfiguration.refuseUnknownEnvironment([:])
        try RuntimeConfiguration.refuseUnknownEnvironment(
            ["PATH": "/usr/bin", "SHRIKEX": "1", "MY_SHRIKE_MODEL": "x", "shrike_phases": "1"])
    }

    @Test func theRefusalListsTheNamesSortedAndNamesTheChapter() {
        #expect(throws: RuntimeConfigurationError.unknownEnvironment(["SHRIKE_AA", "SHRIKE_ZZ"])) {
            try RuntimeConfiguration.refuseUnknownEnvironment(["SHRIKE_ZZ": "1", "SHRIKE_AA": ""])
        }
        #expect(RuntimeConfigurationError.unknownEnvironment(["SHRIKE_AA", "SHRIKE_ZZ"]).description
            == "SHRIKE_AA, SHRIKE_ZZ: not read by this build "
                + "(removed in v17; docs/v17-consolidation.md is the record)")
    }
}
