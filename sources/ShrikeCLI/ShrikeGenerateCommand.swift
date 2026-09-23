import ArgumentParser
import Shrike
import ShrikeCatalog
import ShrikeArgumentSupport

public enum TopKChoice: Equatable, Sendable {
    case off
    case limit(Int)

    static let kernelLimit = 256

    public var tokens: Int? {
        switch self {
        case .off: return nil
        case .limit(let count): return count
        }
    }
}

extension TopKChoice: ExpressibleByArgument {
    public init?(argument: String) {
        guard let parsed = Int(argument), (0...Self.kernelLimit).contains(parsed) else { return nil }
        self = parsed == 0 ? .off : .limit(parsed)
    }

    public var defaultValueDescription: String {
        switch self {
        case .off: return "0"
        case .limit(let count): return String(count)
        }
    }
}

public struct ShrikeGenerateCommand: ParsableCommand, Sendable {
    public static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Qwen3.5-MoE 35B-A3B text generation.")

    @Option(help: ArgumentHelp("""
        A .gturbo directory, or the id of one in the models directory. Without \
        it, the configured default model is used, or the only installed one.
        """,
        valueName: "dir|id"))
    public var model: String?

    @Option(parsing: .unconditional,
            help: ArgumentHelp("Raw-completion prompt.", valueName: "string"))
    public var prompt: String?

    @Option(help: ArgumentHelp("JSON chat messages with role and content fields.",
                               valueName: "path"))
    public var messagesFile: String?

    @Option(help: ArgumentHelp("Generated-token limit.", valueName: "int"))
    public var maxNew = 1_024

    @Option(name: .customLong("max-context"),
            help: ArgumentHelp("""
                Native context limit, 1...262144 (default 4096). With YaRN: \
                524288 or 1048576 (default 1048576).
                """,
                valueName: "int"))
    var maxContextOption: Int?

    @Option(name: .customLong("rope-scaling"),
            help: ArgumentHelp("Context scaling: none or yarn.", valueName: "mode"))
    public var ropeScalingMode: RuntimeRoPEScalingMode = .none

    @Option(help: ArgumentHelp("Sampling temperature; 0 is greedy.", valueName: "float"))
    public var temperature: Float = GenerationDefaults.temperature

    @Option(name: .customLong("top-k"),
            help: ArgumentHelp("Top-k truncation, 1...256; 0 turns it off.", valueName: "int"))
    public var topK: TopKChoice = .limit(GenerationDefaults.topK)

    @Option(name: .customLong("top-p"),
            help: ArgumentHelp("Nucleus truncation.", valueName: "float"))
    public var topP: Float = GenerationDefaults.topP

    @Option(help: ArgumentHelp("Repetition penalty.", valueName: "float"))
    public var repetitionPenalty: Float = 1.0

    @Option(help: ArgumentHelp("Deterministic sampling seed (default off).", valueName: "uint64"))
    public var seed: UInt64?

    @Option(name: .customLong("stop"), parsing: .unconditionalSingleValue,
            help: ArgumentHelp("Stop substring (repeatable).", valueName: "string"))
    public var stops: [String] = []

    @Option(help: ArgumentHelp("""
        Routed-expert cache slots per layer: 8, 16, 24, 32, 64, 96, 128, 160, \
        192, 224 or 256. More slots raise the hit rate but use more memory.
        """,
        valueName: "n"))
    public var expertCacheSlots = 64

    @Option(name: .customLong("kv-bits"),
            help: ArgumentHelp("KV-cache storage precision: 4, 8 or 16.", valueName: "bits"))
    public var kvCachePrecision: KVCachePrecision = .int8

    @Option(name: .customLong("thinking"),
            help: ArgumentHelp("""
                Ornith/Qwen reasoning mode: off, on or adaptive. Adaptive injects \
                nothing and lets the model decide. These models do not define \
                effort levels.
                """,
                valueName: "mode"))
    public var thinkingMode: ModelThinkingMode = .off

    @Flag(help: "Suppress the timing footer.")
    public var quiet = false

    @Option(name: .customLong("dump-logits"),
            help: ArgumentHelp("""
                Write every position's fp16 logits as raw rows to <path> and a \
                JSON sidecar to <path>.json.
                """,
                valueName: "path"))
    public var dumpLogitsPath: String?

    @Option(name: .customLong("dump-hidden"),
            help: ArgumentHelp("""
                Write every position's fp16 residual before the final norm, the \
                prompt's rows then the answer's, as raw rows to <path> and a JSON \
                sidecar to <path>.json.
                """,
                valueName: "path"))
    public var dumpHiddenPath: String?

    @Option(name: .customLong("tokenize"),
            help: ArgumentHelp("""
                Render the prompt exactly as a run would, write its ids and their \
                pieces as JSON to <path>, and exit without loading the model.
                """,
                valueName: "path"))
    public var tokenizePath: String?

    public init() {}

    public var maxContext: Int {
        if let maxContextOption { return maxContextOption }
        return ropeScalingMode == .yarn
            ? RuntimeConfiguration.defaultYaRNContextTokens
            : 4096
    }

    public func validate() throws {
        if prompt != nil, messagesFile != nil {
            throw ValidationError("--prompt and --messages-file are mutually exclusive")
        }
        if prompt == nil, messagesFile == nil {
            throw ValidationError("one of --prompt or --messages-file is required")
        }
        guard (1...RuntimeConfiguration.maximumContextTokens).contains(maxNew) else {
            throw ValidationError("--max-new must be between 1 and "
                + "\(RuntimeConfiguration.maximumContextTokens)")
        }
        guard (0...2).contains(temperature) else {
            throw ValidationError("--temperature must be between 0 and 2")
        }
        guard topP > 0, topP <= 1 else {
            throw ValidationError("--top-p must be above 0 and at most 1")
        }
        guard repetitionPenalty > 0 else {
            throw ValidationError("--repetition-penalty must be above 0")
        }
        guard RuntimeConfiguration.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw ValidationError("--expert-cache-slots must be one of "
                + RuntimeConfiguration.allowedExpertCacheSlots.map(String.init)
                    .joined(separator: ", "))
        }
        if temperature > 0, topK == .off, topP < 1 {
            throw ValidationError("--top-p \(topP) requires --top-k between 1 and 256")
        }
        try validateContext()
    }

    private func validateContext() throws {
        if ropeScalingMode == .yarn {
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                throw ValidationError("--max-context with YaRN must be one of "
                    + RuntimeConfiguration.supportedYaRNContextTokens.map(String.init)
                        .joined(separator: ", "))
            }
        } else {
            guard (1...RuntimeConfiguration.nativeMaximumContextTokens).contains(maxContext) else {
                throw ValidationError("--max-context must be between 1 and "
                    + "\(RuntimeConfiguration.nativeMaximumContextTokens)")
            }
        }
    }

    public func run() throws {
        let modelURL = try ModelResolver.resolve(requested: model)
        let code = drive(self, modelPath: modelURL.path)
        if code != 0 { throw ExitCode(code) }
    }
}
