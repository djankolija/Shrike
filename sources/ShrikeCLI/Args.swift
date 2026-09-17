import Shrike

public enum PrefillChunkChoice: Equatable, Sendable {
    case fixed(Int)
    case auto
}

public struct Args: Equatable, Sendable {
    public var model: String
    public var prompt: String?
    public var messagesFile: String?
    public var maxNew: Int
    public var maxContext: Int
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stops: [String]
    public var quiet: Bool
    public var concise: Bool
    public var thinkingMode: ModelThinkingMode
    public var expertCacheSlots: Int
    public var prefillChunk: PrefillChunkChoice?
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode
    public var forceTokensPath: String?
    public var dumpLogitsPath: String?
    public var logitsHead: Bool
    public var tokenizePath: String?

    public init(model: String,
                prompt: String? = nil,
                messagesFile: String? = nil,
                maxNew: Int = 1_024,
                maxContext: Int = 4096,
                temperature: Float = GenerationDefaults.temperature,
                topK: Int? = GenerationDefaults.topK,
                topP: Float? = GenerationDefaults.topP,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stops: [String] = [],
                quiet: Bool = false,
                concise: Bool = false,
                thinkingMode: ModelThinkingMode = .off,
                expertCacheSlots: Int = 64,
                prefillChunk: PrefillChunkChoice? = nil,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none,
                forceTokensPath: String? = nil,
                dumpLogitsPath: String? = nil,
                logitsHead: Bool = false,
                tokenizePath: String? = nil) {
        self.model = model
        self.forceTokensPath = forceTokensPath
        self.dumpLogitsPath = dumpLogitsPath
        self.logitsHead = logitsHead
        self.tokenizePath = tokenizePath
        self.prompt = prompt
        self.messagesFile = messagesFile
        self.maxNew = maxNew
        self.maxContext = maxContext
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.expertCacheSlots = expertCacheSlots
        self.prefillChunk = prefillChunk
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.seed = seed
        self.stops = stops
        self.quiet = quiet
        self.concise = concise
        self.thinkingMode = thinkingMode
    }
}

public enum ArgsError: Error, Equatable, CustomStringConvertible {
    case helpRequested
    case unknownFlag(String)
    case missingValue(flag: String)
    case invalidValue(flag: String, value: String)
    case requiredMissing(String)
    case mutuallyExclusive(String, String)
    case modeMissing

    public var description: String {
        switch self {
        case .helpRequested: return "help requested"
        case .unknownFlag(let flag): return "unknown flag: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .invalidValue(let flag, let value): return "invalid value for \(flag): \(value)"
        case .requiredMissing(let flag): return "required flag missing: \(flag)"
        case .mutuallyExclusive(let a, let b): return "\(a) and \(b) are mutually exclusive"
        case .modeMissing: return "one of --prompt or --messages-file is required"
        }
    }
}

extension Args {
    public static let usage = """
    ShrikeCLI — Qwen3.5-MoE 35B-A3B text generation

    usage: ShrikeCLI --model <dir> (--prompt <string> | --messages-file <path>) [options]

    required:
      --model <dir>             Path to a .gturbo model directory.
      --prompt <string>         Raw-completion prompt.
      --messages-file <path>    JSON chat messages with role and content fields.

    options:
      --max-new <int>           Generated-token limit (default 1024).
      --max-context <int>       Native context limit, 1...262144 (default 4096).
                                With YaRN: 524288 or 1048576 (default 1048576).
      --rope-scaling <mode>     Context scaling: none or yarn (default none).
      --temperature <float>     Sampling temperature (default 0.6; 0 = greedy).
      --top-k <int>             Top-k truncation, 1...256 (default 20; 0 = off).
      --top-p <float>           Nucleus truncation (default 0.95).
      --repetition-penalty <f>  Repetition penalty (default 1.0).
      --seed <uint64>           Deterministic sampling seed (default off).
      --stop <string>           Stop substring (repeatable).
      --expert-cache-slots <n>  Routed-expert cache slots per layer: 8, 16,
                                24, 32, 64, 96, or 128 (default 64). More
                                slots raise the hit rate but use more memory.
      --prefill-chunk <n|auto>  Prefill chunk tokens. Larger chunks reduce
                                routed-expert file sweeps but use more GPU
                                scratch. Allowed: 32, 64, 128, 256, 512,
                                1024, 2048, 4096; auto covers the prompt with
                                the smallest allowed chunk.
      --kv-bits <4|8|16>        KV-cache storage precision (default 8).
      --concise                 Inject the per-quantization concise-mode
                                system prompt (answers without preamble,
                                filler, or closing codas).
      --thinking <off|on|adaptive>
                                Ornith/Qwen reasoning mode (default off).
                                Adaptive injects nothing and lets the model
                                decide. These models do not define effort
                                levels.
      --quiet                   Suppress the timing footer.
      --logits-head             Run the server's logits head even at
                                temperature 0 (the fused greedy head is the
                                default there).
      --force-tokens <path>     Feed these ids (one per line) in place of the
                                sampler's and stop when they run out; the
                                class-2 gate's instrument.
      --dump-logits <path>      Write every position's fp16 logits as raw rows
                                to <path> and a JSON sidecar to <path>.json.
      --tokenize <path>         Render the prompt exactly as a run would, write
                                its ids and their pieces as JSON to <path>, and
                                exit without loading the model.
      --help                    Show this message.
    """

    /// Same shape as ServerArguments.parse: a flag table.
    public static func parse(_ argv: [String]) throws -> Args {
        var context = ParseContext()
        try context.applyFlags(argv)
        guard let model = context.model else { throw ArgsError.requiredMissing("--model") }
        try context.validate()
        return context.makeArgs(model: model)
    }

    private struct ParseContext {
        var model: String?
        var prompt: String?
        var messagesFile: String?
        var maxNew = 1_024
        var maxContext = 4096
        var maxContextWasSet = false
        var temperature: Float = GenerationDefaults.temperature
        var topK: Int? = GenerationDefaults.topK
        var topP: Float? = GenerationDefaults.topP
        var repetitionPenalty: Float = 1.0
        var seed: UInt64?
        var stops: [String] = []
        var quiet = false
        var concise = false
        var thinkingMode: ModelThinkingMode = .off
        var expertCacheSlots = 64
        var prefillChunk: PrefillChunkChoice?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var forceTokensPath: String?
        var dumpLogitsPath: String?
        var logitsHead = false
        var tokenizePath: String?

        mutating func applyFlags(_ argv: [String]) throws {
            var index = 0
            while index < argv.count {
                let flag = argv[index]
                switch flag {
                case "--help":
                    throw ArgsError.helpRequested
                case "--quiet":
                    quiet = true
                    index += 1
                case "--concise":
                    concise = true
                    index += 1
                case "--thinking":
                    thinkingMode = try takeRawValue(argv, &index, flag: flag)
                case "--model":
                    model = try takeValue(argv, &index, flag: flag)
                case "--prompt":
                    prompt = try takeValue(argv, &index, flag: flag)
                case "--messages-file":
                    messagesFile = try takeValue(argv, &index, flag: flag)
                case "--max-new":
                    maxNew = try takeInt(argv, &index, flag: flag,
                                         in: 1...RuntimeConfiguration.maximumContextTokens)
                case "--max-context":
                    maxContext = try takeInt(argv, &index, flag: flag,
                                             in: 1...RuntimeConfiguration.maximumContextTokens)
                    maxContextWasSet = true
                case "--rope-scaling":
                    ropeScalingMode = try takeRawValue(argv, &index, flag: flag)
                case "--temperature":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let parsed = Float(value), parsed >= 0, parsed <= 2 else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    temperature = parsed
                case "--top-k":
                    let parsed = try takeInt(argv, &index, flag: flag, in: 0...256)
                    topK = parsed == 0 ? nil : parsed
                case "--top-p":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let parsed = Float(value), parsed > 0, parsed <= 1 else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    topP = parsed
                case "--repetition-penalty":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let parsed = Float(value), parsed > 0 else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    repetitionPenalty = parsed
                case "--seed":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let parsed = UInt64(value) else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    seed = parsed
                case "--expert-cache-slots":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let parsed = Int(value),
                          RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    expertCacheSlots = parsed
                case "--prefill-chunk":
                    let value = try takeValue(argv, &index, flag: flag)
                    if value == "auto" {
                        prefillChunk = .auto
                    } else if let parsed = Int(value),
                              RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) {
                        prefillChunk = .fixed(parsed)
                    } else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                case "--kv-bits":
                    let value = try takeValue(argv, &index, flag: flag)
                    guard let bits = Int(value),
                          let parsed = KVCachePrecision(rawValue: bits) else {
                        throw ArgsError.invalidValue(flag: flag, value: value)
                    }
                    kvCachePrecision = parsed
                case "--stop":
                    stops.append(try takeValue(argv, &index, flag: flag))
                case "--force-tokens":
                    forceTokensPath = try takeValue(argv, &index, flag: flag)
                case "--dump-logits":
                    dumpLogitsPath = try takeValue(argv, &index, flag: flag)
                case "--logits-head":
                    logitsHead = true
                    index += 1
                case "--tokenize":
                    tokenizePath = try takeValue(argv, &index, flag: flag)
                default:
                    throw ArgsError.unknownFlag(flag)
                }
            }
        }

        mutating func validate() throws {
            if prompt != nil && messagesFile != nil {
                throw ArgsError.mutuallyExclusive("--prompt", "--messages-file")
            }
            if prompt == nil && messagesFile == nil { throw ArgsError.modeMissing }
            if temperature > 0, topK == nil, let topP, topP < 1 {
                throw ArgsError.invalidValue(
                    flag: "--top-p",
                    value: "\(topP) requires --top-k between 1 and 256")
            }
            if ropeScalingMode == .yarn {
                if !maxContextWasSet {
                    maxContext = RuntimeConfiguration.defaultYaRNContextTokens
                }
                guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                    throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
                }
            } else if maxContext > RuntimeConfiguration.nativeMaximumContextTokens {
                throw ArgsError.invalidValue(flag: "--max-context", value: String(maxContext))
            }
        }

        func makeArgs(model: String) -> Args {
            return Args(model: model,
                        prompt: prompt,
                        messagesFile: messagesFile,
                        maxNew: maxNew,
                        maxContext: maxContext,
                        temperature: temperature,
                        topK: topK,
                        topP: topP,
                        repetitionPenalty: repetitionPenalty,
                        seed: seed,
                        stops: stops,
                        quiet: quiet,
                        concise: concise,
                        thinkingMode: thinkingMode,
                        expertCacheSlots: expertCacheSlots,
                        prefillChunk: prefillChunk,
                        kvCachePrecision: kvCachePrecision,
                        ropeScalingMode: ropeScalingMode,
                        forceTokensPath: forceTokensPath,
                        dumpLogitsPath: dumpLogitsPath,
                        logitsHead: logitsHead,
                        tokenizePath: tokenizePath)
        }
    }

    private static func takeValue(_ argv: [String],
                                  _ index: inout Int,
                                  flag: String) throws -> String {
        guard index + 1 < argv.count else { throw ArgsError.missingValue(flag: flag) }
        let value = argv[index + 1]
        index += 2
        return value
    }

    private static func takeInt(_ argv: [String],
                                _ index: inout Int,
                                flag: String,
                                in range: ClosedRange<Int>) throws -> Int {
        let value = try takeValue(argv, &index, flag: flag)
        guard let parsed = Int(value), range.contains(parsed) else {
            throw ArgsError.invalidValue(flag: flag, value: value)
        }
        return parsed
    }

    private static func takeRawValue<T: RawRepresentable>(_ argv: [String],
                                                          _ index: inout Int,
                                                          flag: String) throws -> T
    where T.RawValue == String {
        let value = try takeValue(argv, &index, flag: flag)
        guard let parsed = T(rawValue: value) else {
            throw ArgsError.invalidValue(flag: flag, value: value)
        }
        return parsed
    }
}
