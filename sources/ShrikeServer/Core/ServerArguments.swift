import Foundation
import Shrike

public struct ServerArguments: Equatable, Sendable {
    /// Serve exactly this model, ignoring any config or roster. nil selects
    /// config mode: scan a models directory and serve everything found.
    public let model: String?
    public let port: Int
    /// Explicit --model-id value; nil derives the API ID from the installed
    /// manifest (for example qwen3.6-35b-a3b or ornith-1.5-35b-a3b).
    public let modelIDOverride: String?
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryMiB: Int
    public let promptCacheDiskDirectory: String?
    public let promptCacheDiskMiB: Int
    public let prefillChunkTokens: Int?
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let thinkingMode: ModelThinkingMode
    public let reasoningEffort: ReasoningEffort?
    public let reasoningRetention: ReasoningRetention?
    public let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use. Slots are derived from it and the
    /// model's own expert stride, so this is the knob and the slot count is the
    /// outcome. `--expert-cache-slots` still wins if both are given.
    public let expertCacheBudgetBytes: Int?
    /// Defer the model load to the first inference request. This is the
    /// default behaviour; the flag remains accepted for compatibility.
    public let lazyLoad: Bool
    /// Release the weights after this many idle seconds; 0 disables unloading.
    public let idleUnloadSeconds: Int
    /// Multi-model config file path; nil tries ~/.shrike/server.json.
    public let configPath: String?
    /// Directory scanned for *.gturbo bundles; nil defers to the config
    /// file's models_dir, else ~/shrike-runtime/models.
    public let modelsDir: String?
    /// Load the default model at startup instead of on the first request.
    public let preload: Bool
    /// Whether the flag was typed, so config defaults know not to override it.
    let maxContextWasSet: Bool
    let idleUnloadWasSet: Bool

    /// Idle unloading discards the in-memory prefix cache with the session.
    /// With a disk cache configured the entries rehydrate on reload; without
    /// one, every unload costs a full cold prefill on the next request.
    public var unloadDiscardsWarmCache: Bool {
        idleUnloadSeconds > 0
            && promptCacheMode != .off
            && promptCacheDiskDirectory == nil
    }

    public static let usage = """
    usage: ShrikeServer [--model <completed .gturbo directory>] [options]

      --model <dir>          Serve exactly this model, ignoring any config or
                             roster. Without it the server scans a models
                             directory and serves every bundle found, selected
                             per request by the OpenAI "model" field.
      --config <path>        Multi-model config file (default ~/.shrike/server.json
                             when it exists). Names, defaults and the default
                             model. Cannot be combined with --model.
      --models-dir <dir>     Directory scanned for *.gturbo bundles (default:
                             the config file's models_dir, else
                             ~/shrike-runtime/models). Cannot be combined with
                             --model.
      --preload              Load the default model at startup instead of on
                             the first request.
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default derived from the
                             installed model manifest).
      --max-context <tokens> Native: 4096...262144 (default 262144).
                             With YaRN: 524288 or 1048576 (default 1048576).
      --rope-scaling <mode>  Context scaling: none or yarn (default none).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix|multi-prefix>
                             Prompt KV reuse mode (default multi-prefix).
      --prompt-cache-entries <count>
                             Maximum retained prefixes, 1...64 (default 4).
      --prompt-cache-memory-mib <MiB>
                             RAM snapshot budget, 0...4096 (default 256).
      --prompt-cache-disk <dir>
                             Optional persistent SSD cache directory.
      --prompt-cache-disk-mib <MiB>
                             SSD snapshot budget, 0...65536 (default 8192).
      --prefill-chunk <tokens>
                             Prefill chunk size: 32, 64, 128, 256, 512,
                             1024, 2048, or 4096 (default 4096 for supported
                             35B-A3B text models).
      --kv-bits <4|8|16>     KV-cache storage precision (default 8).
      --thinking <off|on|adaptive>
                             Ornith/Qwen reasoning mode (default off, or
                             SHRIKE_THINKING_MODE). Adaptive injects nothing
                             and lets the model decide. The model does not
                             expose low/medium/high effort levels.
      --reasoning-effort <low|medium|high>
                             Harmony deliberation level: low, medium or high
                             (default medium, or SHRIKE_REASONING_EFFORT;
                             --thinking off on a Harmony model implies low).
      --reasoning-retention <as-generated|stripped>
                             History-turn render form (default as-generated,
                             or SHRIKE_REASONING_RETENTION). as-generated
                             renders turns as the model produced them (no
                             settle rewrites); stripped keeps the v6
                             canonical re-render. Harmony always strips.
      --expert-cache-slots <count>
                             Routed-expert cache slots per layer: 8, 16, 24,
                             32, 64, 96, or 128 (default: derived from
                             --ram-budget).
      --ram-budget <size>    Bytes the routed-expert cache may use, e.g. 8G,
                             2G, 512M. Slots are derived from this and the
                             model's expert stride, so this is the knob and
                             the slot count is the result. Default 8G, which
                             holds the measured routing working set; smaller
                             budgets are markedly slower because expert reads
                             bypass the page cache and have no fallback.
                             --expert-cache-slots overrides this.
      --lazy-load            Defer the model load to the first inference
                             request. This is the default; the flag remains
                             accepted for compatibility.
      --idle-unload-seconds <n>
                             Release the model weights after n seconds with no
                             requests, 0...86400 (default 0, disabled). The
                             next request reloads transparently. Implies
                             --lazy-load. Pair with --prompt-cache-disk, since
                             unloading discards the in-memory prefix cache.
      --help                 Show this help.
    """

    /// A flag table: one `case` per option plus its validation.
    public static func parse(
        _ input: [String],
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ServerArguments {
        var context = try ParseContext(environment: environment)
        try context.applyFlags(input)
        try context.validate()
        return context.makeArguments()
    }

    private struct ParseContext {
        var model: String?
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 262_144
        var maxContextWasSet = false
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .multiPrefix
        var promptCacheMaximumEntries = 4
        var promptCacheMemoryMiB = 256
        var promptCacheDiskDirectory: String?
        var promptCacheDiskMiB = 8_192
        var prefillChunkTokens: Int?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var thinkingMode: ModelThinkingMode
        var reasoningEffort: ReasoningEffort?
        var reasoningRetention: ReasoningRetention?
        var expertCacheSlots: Int?
        var expertCacheBudgetBytes: Int?
        var lazyLoad = false
        var idleUnloadSeconds = 0
        var idleUnloadWasSet = false
        var configPath: String?
        var modelsDir: String?
        var preload = false

        init(environment: [String: String]) throws {
            thinkingMode = ModelThinkingMode.resolved(environment: environment)
            if let raw = environment["SHRIKE_REASONING_EFFORT"],
               ReasoningEffort(rawValue: raw.lowercased()) == nil {
                throw ServerArgumentError.invalid(
                    "SHRIKE_REASONING_EFFORT must be low, medium or high")
            }
            reasoningEffort = ReasoningEffort.resolved(environment: environment)
            if let raw = environment["SHRIKE_REASONING_RETENTION"],
               ReasoningRetention(rawValue: raw.lowercased()) == nil {
                throw ServerArgumentError.invalid(
                    "SHRIKE_REASONING_RETENTION must be as-generated or stripped")
            }
            reasoningRetention = ReasoningRetention.resolved(environment: environment)
        }

        mutating func applyFlags(_ input: [String]) throws {
            var index = 0
            while index < input.count {
                let flag = input[index]
                if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
                // Valueless flags are consumed before the "requires a value" guard
                // below; otherwise `--lazy-load --port 9999` would swallow --port
                // as this flag's value and then reject it as unknown.
                if flag == "--lazy-load" {
                    lazyLoad = true
                    index += 1
                    continue
                }
                if flag == "--preload" {
                    preload = true
                    index += 1
                    continue
                }
                guard index + 1 < input.count else {
                    throw ServerArgumentError.invalid("\(flag) requires a value")
                }
                let value = input[index + 1]
                index += 2
                try apply(flag: flag, value: value)
            }
        }

        mutating func apply(flag: String, value: String) throws {
            switch flag {
            case "--model":
                model = value
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                modelIDOverride = try requireNonEmpty(value, flag: flag)
            case "--max-context":
                guard let parsed = Int(value),
                      (1...RuntimeConfiguration.maximumContextTokens).contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--rope-scaling must be none or yarn")
                }
                ropeScalingMode = parsed
            case "--queue-limit":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid("--queue-limit must be between 1 and 64")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off, single-prefix, or multi-prefix")
                }
                promptCacheMode = parsed
            case "--prompt-cache-entries":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-entries must be between 1 and 64")
                }
                promptCacheMaximumEntries = parsed
            case "--prompt-cache-memory-mib":
                guard let parsed = Int(value), (0...4_096).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-memory-mib must be between 0 and 4096")
                }
                promptCacheMemoryMiB = parsed
            case "--prompt-cache-disk":
                promptCacheDiskDirectory = try requireNonEmpty(value, flag: flag)
            case "--prompt-cache-disk-mib":
                guard let parsed = Int(value), (0...65_536).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk-mib must be between 0 and 65536")
                }
                promptCacheDiskMiB = parsed
            case "--prefill-chunk":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid("--prefill-chunk is not supported")
                }
                prefillChunkTokens = parsed
            case "--kv-bits":
                guard let bits = Int(value),
                      let parsed = KVCachePrecision(rawValue: bits) else {
                    throw ServerArgumentError.invalid("--kv-bits must be 4, 8, or 16")
                }
                kvCachePrecision = parsed
            case "--thinking":
                guard let parsed = ModelThinkingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--thinking must be off, on or adaptive")
                }
                thinkingMode = parsed
            case "--reasoning-effort":
                guard let parsed = ReasoningEffort(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning-effort must be low, medium or high")
                }
                reasoningEffort = parsed
            case "--reasoning-retention":
                guard let parsed = ReasoningRetention(rawValue: value.lowercased()) else {
                    throw ServerArgumentError.invalid(
                        "--reasoning-retention must be as-generated or stripped")
                }
                reasoningRetention = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be one of \(RuntimeConfiguration.allowedExpertCacheSlots)")
                }
                expertCacheSlots = parsed
            case "--ram-budget":
                guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be a positive size such as 2G, 512M or a byte count")
                }
                expertCacheBudgetBytes = parsed
            case "--idle-unload-seconds":
                guard let parsed = Int(value), (0...86_400).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--idle-unload-seconds must be between 0 and 86400")
                }
                idleUnloadSeconds = parsed
                idleUnloadWasSet = true
            case "--config":
                configPath = try requireNonEmpty(value, flag: flag)
            case "--models-dir":
                modelsDir = try requireNonEmpty(value, flag: flag)
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }

        mutating func validate() throws {
            if model != nil, configPath != nil || modelsDir != nil {
                throw ServerArgumentError.invalid(
                    "--model serves exactly one model; it cannot be combined with --config or --models-dir")
            }
            if model == nil, modelIDOverride != nil {
                throw ServerArgumentError.invalid(
                    "--model-id requires --model; config mode names models in the config file")
            }
            if preload, lazyLoad {
                throw ServerArgumentError.invalid("--preload and --lazy-load contradict each other")
            }
            if ropeScalingMode == .yarn, !maxContextWasSet {
                maxContext = RuntimeConfiguration.defaultYaRNContextTokens
            }
            try validateMaxContext(maxContext, ropeScalingMode: ropeScalingMode)
        }

        func makeArguments() -> ServerArguments {
            return ServerArguments(model: model,
                                   port: port,
                                   modelIDOverride: modelIDOverride,
                                   maxContext: maxContext,
                                   queueLimit: queueLimit,
                                   promptCacheMode: promptCacheMode,
                                   promptCacheMaximumEntries: promptCacheMaximumEntries,
                                   promptCacheMemoryMiB: promptCacheMemoryMiB,
                                   promptCacheDiskDirectory: promptCacheDiskDirectory,
                                   promptCacheDiskMiB: promptCacheDiskMiB,
                                   prefillChunkTokens: prefillChunkTokens,
                                   kvCachePrecision: kvCachePrecision,
                                   ropeScalingMode: ropeScalingMode,
                                   thinkingMode: thinkingMode,
                                   reasoningEffort: reasoningEffort,
                                   reasoningRetention: reasoningRetention,
                                   expertCacheSlots: expertCacheSlots,
                                   expertCacheBudgetBytes: expertCacheBudgetBytes,
                                   lazyLoad: lazyLoad,
                                   idleUnloadSeconds: idleUnloadSeconds,
                                   configPath: configPath,
                                   modelsDir: modelsDir,
                                   preload: preload,
                                   maxContextWasSet: maxContextWasSet,
                                   idleUnloadWasSet: idleUnloadWasSet)
        }
    }

    private static func requireNonEmpty(_ value: String, flag: String) throws -> String {
        guard !value.isEmpty else {
            throw ServerArgumentError.invalid("\(flag) must not be empty")
        }
        return value
    }

    private static func validateMaxContext(_ value: Int,
                                           ropeScalingMode: RuntimeRoPEScalingMode) throws {
        if ropeScalingMode == .yarn {
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(value) else {
                throw ServerArgumentError.invalid(
                    "YaRN --max-context must be 524288 or 1048576")
            }
        } else {
            guard RuntimeConfiguration.supportedContextTokens.contains(value) else {
                throw ServerArgumentError.invalid("--max-context is not supported")
            }
        }
    }

    /// Config defaults merge under flag > config > built-in, resolved in
    /// memory. A flag never writes back into the config file.
    public func merging(configDefaults defaults: ServerConfig.Defaults) throws -> ServerArguments {
        var maxContext = self.maxContext
        if !maxContextWasSet, let value = defaults.maxContext {
            try Self.validateMaxContext(value, ropeScalingMode: ropeScalingMode)
            maxContext = value
        }
        var idleUnloadSeconds = self.idleUnloadSeconds
        if !idleUnloadWasSet, let value = defaults.idleUnloadSeconds {
            idleUnloadSeconds = value
        }
        var expertCacheBudgetBytes = self.expertCacheBudgetBytes
        if expertCacheBudgetBytes == nil, let text = defaults.ramBudget {
            expertCacheBudgetBytes = RuntimeConfiguration.parseBudgetBytes(text)
        }
        return ServerArguments(model: model,
                               port: port,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               promptCacheMaximumEntries: promptCacheMaximumEntries,
                               promptCacheMemoryMiB: promptCacheMemoryMiB,
                               promptCacheDiskDirectory: promptCacheDiskDirectory,
                               promptCacheDiskMiB: promptCacheDiskMiB,
                               prefillChunkTokens: prefillChunkTokens,
                               kvCachePrecision: kvCachePrecision,
                               ropeScalingMode: ropeScalingMode,
                               thinkingMode: thinkingMode,
                               reasoningEffort: reasoningEffort,
                               reasoningRetention: reasoningRetention,
                               expertCacheSlots: expertCacheSlots,
                               expertCacheBudgetBytes: expertCacheBudgetBytes,
                               lazyLoad: lazyLoad,
                               idleUnloadSeconds: idleUnloadSeconds,
                               configPath: configPath,
                               modelsDir: modelsDir,
                               preload: preload,
                               maxContextWasSet: maxContextWasSet,
                               idleUnloadWasSet: idleUnloadWasSet)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
