import ArgumentParser
import Foundation
import Shrike
import ShrikeArgumentSupport
import ShrikeCatalog

extension ServerPromptCacheMode: ExpressibleByArgument {}

public struct ShrikeServerCommand: AsyncParsableCommand, Sendable {
    public static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve one or many .gturbo models over an OpenAI-compatible HTTP API.")

    /// Serve exactly this model, ignoring any config or roster. nil selects
    /// config mode: scan a models directory and serve everything found.
    @Option(help: ArgumentHelp("""
        Serve exactly this model, ignoring any config or roster. Without it the \
        server scans a models directory and serves every bundle found, selected \
        per request by the OpenAI "model" field.
        """,
        valueName: "dir"))
    public var model: String?

    @Option(name: .customLong("config"),
            help: ArgumentHelp("""
                Multi-model config file (default ~/.shrike/config.json when it \
                exists). Names, defaults and the default model. Cannot be \
                combined with --model.
                """,
                valueName: "path"))
    public var configPath: String?

    @Option(help: ArgumentHelp("""
        Directory scanned for *.gturbo bundles (default: the config file's \
        models_dir, else ~/shrike-runtime/models). Cannot be combined with --model.
        """,
        valueName: "dir"))
    public var modelsDir: String?

    @Flag(help: "Load the default model at startup instead of on the first request.")
    public var preload = false

    @Option(help: ArgumentHelp("Loopback port.", valueName: "1...65535"))
    public var port = 8080

    /// Explicit --model-id value; nil derives the API ID from the installed
    /// manifest (for example qwen3.6-35b-a3b or ornith-1.5-35b-a3b).
    @Option(name: .customLong("model-id"), parsing: .unconditional,
            help: ArgumentHelp("API model identifier (default derived from the "
                + "installed model manifest).", valueName: "id"))
    public var modelIDOverride: String?

    @Option(name: .customLong("max-context"),
            help: ArgumentHelp("""
                Native context: \(Self.nativeContextList) (default 262144). With \
                YaRN: \(Self.yaRNContextList) (default 1048576).
                """,
                valueName: "tokens"))
    var maxContextOption: Int?

    @Option(name: .customLong("rope-scaling"),
            help: ArgumentHelp("Context scaling: none or yarn.", valueName: "mode"))
    public var ropeScalingMode: RuntimeRoPEScalingMode = .none

    @Option(help: ArgumentHelp("Maximum queued requests, 1...64.", valueName: "count"))
    public var queueLimit = 4

    @Option(name: .customLong("prompt-cache-mode"),
            help: ArgumentHelp("Prompt KV reuse mode.", valueName: "mode"))
    public var promptCacheMode: ServerPromptCacheMode = .multiPrefix

    @Option(name: .customLong("prompt-cache-entries"),
            help: ArgumentHelp("Maximum retained prefixes, 1...64.", valueName: "count"))
    public var promptCacheMaximumEntries = 4

    @Option(name: .customLong("prompt-cache-memory-mib"),
            help: ArgumentHelp("RAM snapshot budget, 0...4096.", valueName: "MiB"))
    public var promptCacheMemoryMiB = 256

    @Option(name: .customLong("prompt-cache-disk"),
            help: ArgumentHelp("Optional persistent SSD cache directory.", valueName: "dir"))
    public var promptCacheDiskDirectory: String?

    @Option(name: .customLong("prompt-cache-disk-mib"),
            help: ArgumentHelp("SSD snapshot budget, 0...65536.", valueName: "MiB"))
    public var promptCacheDiskMiB = 8_192

    @Option(name: .customLong("prefill-chunk"),
            help: ArgumentHelp("""
                Prefill chunk size: 32, 64, 128, 256, 512, 1024, 2048 or 4096 \
                (default 4096 for supported 35B-A3B text models).
                """,
                valueName: "tokens"))
    public var prefillChunkTokens: Int?

    @Option(name: .customLong("kv-bits"),
            help: ArgumentHelp("KV-cache storage precision: 4, 8 or 16.", valueName: "bits"))
    public var kvCachePrecision: KVCachePrecision = .int8

    @Option(name: .customLong("thinking"),
            help: ArgumentHelp("""
                Ornith/Qwen reasoning mode: off, on or adaptive (default off, or \
                SHRIKE_THINKING_MODE). Adaptive injects nothing and lets the model \
                decide. The model does not expose low/medium/high effort levels.
                """,
                valueName: "mode"))
    var thinkingModeOption: ModelThinkingMode?

    @Option(name: .customLong("reasoning-effort"),
            help: ArgumentHelp("""
                Harmony deliberation level: low, medium or high (default medium, \
                or SHRIKE_REASONING_EFFORT; --thinking off on a Harmony model \
                implies low).
                """,
                valueName: "level"))
    public var reasoningEffort: ReasoningEffort?

    @Option(name: .customLong("reasoning-retention"),
            help: ArgumentHelp("""
                History-turn render form: as-generated or stripped (default \
                as-generated, or SHRIKE_REASONING_RETENTION). as-generated renders \
                turns as the model produced them (no settle rewrites); stripped \
                keeps the v6 canonical re-render. Harmony always strips.
                """,
                valueName: "form"))
    public var reasoningRetention: ReasoningRetention?

    @Option(name: .customLong("expert-cache-slots"),
            help: ArgumentHelp("""
                Routed-expert cache slots per layer: 8, 16, 24, 32, 64, 96, 128, \
                160, 192, 224 or 256 (default: derived from --ram-budget).
                """,
                valueName: "count"))
    public var expertCacheSlots: Int?

    /// Bytes the routed-expert cache may use. Slots are derived from it and the
    /// model's own expert stride, so this is the knob and the slot count is the
    /// outcome. `--expert-cache-slots` still wins if both are given.
    @Option(name: .customLong("ram-budget"),
            help: ArgumentHelp("""
                Bytes the routed-expert cache may use, e.g. 8G, 2G, 512M. Slots are \
                derived from this and the model's expert stride, so this is the knob \
                and the slot count is the result. Default 8G, which holds the \
                measured routing working set; smaller budgets are markedly slower \
                because expert reads bypass the page cache and have no fallback. \
                --expert-cache-slots overrides this.
                """,
                valueName: "size"),
            transform: Self.budgetBytes)
    public var expertCacheBudgetBytes: Int?

    /// Defer the model load to the first inference request. This is the
    /// default behaviour; the flag remains accepted for compatibility.
    @Flag(help: """
        Defer the model load to the first inference request. This is the default; \
        the flag remains accepted for compatibility.
        """)
    public var lazyLoad = false

    @Option(name: .customLong("idle-unload-seconds"),
            help: ArgumentHelp("""
                Release the model weights after n seconds with no requests, \
                0...86400 (default 0, disabled). The next request reloads \
                transparently. Implies --lazy-load. Pair with --prompt-cache-disk, \
                since unloading discards the in-memory prefix cache.
                """,
                valueName: "n"))
    var idleUnloadSecondsOption: Int?

    public init() {}

    /// Three defaults ArgumentParser cannot state on the property: two depend on
    /// another flag, and the third on the environment, which parsing no longer
    /// reads.
    public var maxContext: Int {
        if let maxContextOption { return maxContextOption }
        return ropeScalingMode == .yarn
            ? RuntimeConfiguration.defaultYaRNContextTokens
            : RuntimeConfiguration.nativeMaximumContextTokens
    }

    /// Release the weights after this many idle seconds; 0 disables unloading.
    public var idleUnloadSeconds: Int { idleUnloadSecondsOption ?? 0 }

    public var thinkingMode: ModelThinkingMode { thinkingModeOption ?? .off }

    /// Idle unloading discards the in-memory prefix cache with the session.
    /// With a disk cache configured the entries rehydrate on reload; without
    /// one, every unload costs a full cold prefill on the next request.
    public var unloadDiscardsWarmCache: Bool {
        idleUnloadSeconds > 0
            && promptCacheMode != .off
            && promptCacheDiskDirectory == nil
    }

    static let nativeContextList = RuntimeConfiguration.supportedContextTokens
        .map(String.init).joined(separator: ", ")
    static let yaRNContextList = RuntimeConfiguration.supportedYaRNContextTokens
        .map(String.init).joined(separator: " or ")

    static func budgetBytes(_ value: String) throws -> Int {
        guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
            throw ValidationError(
                "--ram-budget must be a positive size such as 2G, 512M or a byte count")
        }
        return parsed
    }

    public func validate() throws {
        if model != nil, configPath != nil || modelsDir != nil {
            throw ValidationError(
                "--model serves exactly one model; it cannot be combined with --config or --models-dir")
        }
        if model == nil, modelIDOverride != nil {
            throw ValidationError(
                "--model-id requires --model; config mode names models in the config file")
        }
        if preload, lazyLoad {
            throw ValidationError("--preload and --lazy-load contradict each other")
        }
        guard (1...65_535).contains(port) else {
            throw ValidationError("--port must be between 1 and 65535")
        }
        guard (1...64).contains(queueLimit) else {
            throw ValidationError("--queue-limit must be between 1 and 64")
        }
        guard (1...64).contains(promptCacheMaximumEntries) else {
            throw ValidationError("--prompt-cache-entries must be between 1 and 64")
        }
        guard (0...4_096).contains(promptCacheMemoryMiB) else {
            throw ValidationError("--prompt-cache-memory-mib must be between 0 and 4096")
        }
        guard (0...65_536).contains(promptCacheDiskMiB) else {
            throw ValidationError("--prompt-cache-disk-mib must be between 0 and 65536")
        }
        if let idleUnloadSecondsOption, !(0...86_400).contains(idleUnloadSecondsOption) {
            throw ValidationError("--idle-unload-seconds must be between 0 and 86400")
        }
        try validateOptionalMemberships()
        try Self.validateMaxContext(maxContext, ropeScalingMode: ropeScalingMode)
    }

    private func validateOptionalMemberships() throws {
        if let prefillChunkTokens,
           !RuntimeConfiguration.allowedPrefillChunkTokens.contains(prefillChunkTokens) {
            throw ValidationError("--prefill-chunk must be one of "
                + RuntimeConfiguration.allowedPrefillChunkTokens.map(String.init)
                    .joined(separator: ", "))
        }
        if let expertCacheSlots,
           !RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) {
            throw ValidationError("--expert-cache-slots must be one of "
                + RuntimeConfiguration.allowedExpertCacheSlots.map(String.init)
                    .joined(separator: ", "))
        }
        for (value, flag) in [(configPath, "--config"), (modelsDir, "--models-dir"),
                              (promptCacheDiskDirectory, "--prompt-cache-disk"),
                              (modelIDOverride, "--model-id")] {
            if let value, value.isEmpty {
                throw ValidationError("\(flag) must not be empty")
            }
        }
    }

    static func validateMaxContext(_ value: Int,
                                   ropeScalingMode: RuntimeRoPEScalingMode) throws {
        if ropeScalingMode == .yarn {
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(value) else {
                throw ValidationError("--max-context with YaRN must be "
                    + Self.yaRNContextList)
            }
        } else {
            guard RuntimeConfiguration.supportedContextTokens.contains(value) else {
                throw ValidationError("--max-context must be one of "
                    + Self.nativeContextList + "; for more, enable --rope-scaling yarn")
            }
        }
    }

    /// Resolved in memory: a flag beats a config default (`max_context`,
    /// `ram_budget`, `idle_unload_seconds`) and beats an environment setting (the
    /// three `SHRIKE_*` below); no setting has both layers. A flag never writes
    /// back into the config file, and parsing itself reads neither.
    public func merging(
        configDefaults defaults: ShrikeConfig.Defaults,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ShrikeServerCommand {
        var effective = self
        if maxContextOption == nil, let value = defaults.maxContext {
            try Self.validateMaxContext(value, ropeScalingMode: ropeScalingMode)
            effective.maxContextOption = value
        }
        if idleUnloadSecondsOption == nil, let value = defaults.idleUnloadSeconds {
            effective.idleUnloadSecondsOption = value
        }
        if expertCacheBudgetBytes == nil, let text = defaults.ramBudget {
            effective.expertCacheBudgetBytes = RuntimeConfiguration.parseBudgetBytes(text)
        }
        try effective.resolveFromEnvironment(environment)
        return effective
    }

    /// Separate from resolution so a launch can reject a malformed value before
    /// it reads a config file or scans a models directory, as it did when
    /// parsing still read the environment itself.
    static func validateEnvironment(_ environment: [String: String]) throws {
        if let raw = environment["SHRIKE_REASONING_EFFORT"],
           ReasoningEffort(rawValue: raw.lowercased()) == nil {
            throw ValidationError("SHRIKE_REASONING_EFFORT must be low, medium or high")
        }
        if let raw = environment["SHRIKE_REASONING_RETENTION"],
           ReasoningRetention(rawValue: raw.lowercased()) == nil {
            throw ValidationError("SHRIKE_REASONING_RETENTION must be as-generated or stripped")
        }
    }

    private mutating func resolveFromEnvironment(_ environment: [String: String]) throws {
        try Self.validateEnvironment(environment)
        if thinkingModeOption == nil {
            thinkingModeOption = ModelThinkingMode.resolved(environment: environment)
        }
        if reasoningEffort == nil {
            reasoningEffort = ReasoningEffort.resolved(environment: environment)
        }
        if reasoningRetention == nil {
            reasoningRetention = ReasoningRetention.resolved(environment: environment)
        }
    }
}
