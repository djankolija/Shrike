import ArgumentParser
import Foundation
import Shrike
import ShrikeArgumentSupport
import ShrikeCatalog

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

    @Option(help: ArgumentHelp("Loopback port.", valueName: "1...65535"))
    public var port = 8080

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

    @Option(name: .customLong("ram-budget"),
            help: ExpertCacheBudgetArgument.help,
            transform: ExpertCacheBudgetArgument.bytes)
    public var expertCacheBudgetBytes: Int?

    public init() {}

    /// Two defaults ArgumentParser cannot state on the property: one depends on
    /// another flag, and the other on the environment, which parsing no longer
    /// reads.
    public var maxContext: Int {
        if let maxContextOption { return maxContextOption }
        return ropeScalingMode == .yarn
            ? RuntimeConfiguration.defaultYaRNContextTokens
            : RuntimeConfiguration.nativeMaximumContextTokens
    }

    public var thinkingMode: ModelThinkingMode { thinkingModeOption ?? .off }

    static let nativeContextList = RuntimeConfiguration.supportedContextTokens
        .map(String.init).joined(separator: ", ")
    static let yaRNContextList = RuntimeConfiguration.supportedYaRNContextTokens
        .map(String.init).joined(separator: " or ")

    public func validate() throws {
        if model != nil, configPath != nil {
            throw ValidationError(
                "--model serves exactly one model; it cannot be combined with --config")
        }
        guard (1...65_535).contains(port) else {
            throw ValidationError("--port must be between 1 and 65535")
        }
        if let configPath, configPath.isEmpty {
            throw ValidationError("--config must not be empty")
        }
        try Self.validateMaxContext(maxContext, ropeScalingMode: ropeScalingMode)
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
    /// `ram_budget`) and beats an environment setting (the two `SHRIKE_*`
    /// below); no setting has both layers. A flag never writes back into the
    /// config file, and parsing itself reads neither.
    public func merging(
        configDefaults defaults: ShrikeConfig.Defaults,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ShrikeServerCommand {
        var effective = self
        if maxContextOption == nil, let value = defaults.maxContext {
            try Self.validateMaxContext(value, ropeScalingMode: ropeScalingMode)
            effective.maxContextOption = value
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
    }

    private mutating func resolveFromEnvironment(_ environment: [String: String]) throws {
        try Self.validateEnvironment(environment)
        if thinkingModeOption == nil {
            thinkingModeOption = ModelThinkingMode.resolved(environment: environment)
        }
        if reasoningEffort == nil {
            reasoningEffort = ReasoningEffort.resolved(environment: environment)
        }
    }
}
