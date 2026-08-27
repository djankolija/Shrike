import Foundation
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
/// The case set is the extension point for a non-ChatML architecture; a
/// tokenizer matching no case is rejected at load rather than rendered as
/// ChatML by default.
public enum ChatDialect: String, Sendable {
    case chatml
    case harmony
    case kimi
}

/// The reasoning switch for compatible Qwen/Ornith chat templates. `off`
/// injects a closed empty think block, `on` injects an open `<think>` (the
/// bundled template's `enable_thinking` branches), and `adaptive` injects
/// nothing so the model decides per prompt. The models do not define
/// low/medium/high effort levels or a thinking-token budget.
public enum ModelThinkingMode: String, Codable, CaseIterable, Sendable {
    case off
    case on
    case adaptive

    public var isEnabled: Bool { self == .on }

    /// Backwards-compatible resolution for processes that still configure the
    /// runtime through `NVMAI_THINKING_MODE`. Unknown values retain the
    /// historical safe default of off.
    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelThinkingMode {
        switch environment["NVMAI_THINKING_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return .on
        case "adaptive": return .adaptive
        default: return .off
        }
    }
}

/// Tokenizer wrapper for the compatible Qwen3.5-MoE ChatML model family.
///
/// Loads tokenizer sidecars in a completed `.gturbo/tokenizer/` directory.
/// Exposes typed accessors for the IDs the generator actually needs (BOS / EOS /
/// pad / end-of-turn) and adapts encode/decode to Int32 to match the buffer
/// types kernels consume.
///
/// NVMAI owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
/// unchecked-invariant: immutable after `load`. The stored token ids and the
/// underlying swift-transformers tokenizer are never mutated afterwards, so
/// concurrent encode/decode calls only read.
public struct GFTokenizer: @unchecked Sendable {
    /// Nominal BOS. This is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    /// ChatML `<tool_call>` / `</tool_call>` / `<tool_response>` markers.
    /// Harmony frames tool calls with headers instead, so these are nil there.
    public let toolCallStartID: Int32?
    public let toolCallEndID: Int32?
    public let toolResponseID: Int32?
    public let toolResponseEndID: Int32?
    /// Channel framing: ChatML `<think>` / `</think>`, Harmony `<|channel|>` /
    /// `<|end|>`.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    /// Harmony framing tokens; nil for other dialects.
    public let harmonyStartID: Int32?
    public let harmonyMessageID: Int32?
    public let harmonyConstrainID: Int32?
    public let harmonyCallID: Int32?
    public let harmonyReturnID: Int32?
    /// Kimi tool-section markers; nil for other dialects. Literal ByteLevel
    /// barriers (`special: false` upstream) like ChatML's `<tool_call>`, so
    /// the streaming decoder keys on their IDs and sees their text in the
    /// delta.
    public let kimiToolSectionBeginID: Int32?
    public let kimiToolSectionEndID: Int32?
    public let kimiToolCallBeginID: Int32?
    public let kimiToolArgumentBeginID: Int32?
    public let kimiToolCallEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    public let dialect: ChatDialect
    public let thinkingMode: ModelThinkingMode

    /// Generation-prompt suffix appended after the last message: derived from
    /// the tokenizer's bundled `chat_template.jinja`
    /// (`add_generation_prompt` per the thinking mode) when available,
    /// falling back to the pinned constant otherwise (R6). Internal so the
    /// structured decoder can prime itself with the injected prefix.
    let generationSuffix: String

    @usableFromInline
    let tokenizer: any Tokenizer
    let byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration

    public static func load(
        from folder: URL,
        thinkingMode: ModelThinkingMode = .off
    ) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(
            .local(folder.standardizedFileURL.path, thinkingMode))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            thinkingMode: ModelThinkingMode = .off,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        guard let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) else {
            throw GFTokenizerError.missingToolTemplate
        }
        return try await load(from: folder, thinkingMode: thinkingMode)
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(
        from folder: URL,
        thinkingMode: ModelThinkingMode
    ) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        let decoder = try GFByteLevelDecoderConfiguration.load(
            from: folder.appendingPathComponent("tokenizer.json"),
            tokenizer: underlying)
        return try GFTokenizer(tokenizer: underlying,
                               byteLevelDecoderConfiguration: decoder,
                               thinkingMode: thinkingMode)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(
        tokenizer: any Tokenizer,
        thinkingMode: ModelThinkingMode = .off
    ) throws {
        try self.init(
            tokenizer: tokenizer,
            byteLevelDecoderConfiguration: .knownFramingTokens(tokenizer: tokenizer),
            thinkingMode: thinkingMode)
    }

    init(tokenizer: any Tokenizer,
         byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration,
         thinkingMode: ModelThinkingMode = .off) throws {
        self.tokenizer = tokenizer
        self.byteLevelDecoderConfiguration = byteLevelDecoderConfiguration

        self.dialect = try Self.resolveDialect(tokenizer)
        let resolved: ResolvedSpecialTokens
        switch dialect {
        case .chatml:
            resolved = try Self.resolveChatMLTokens(tokenizer)
            try Self.validateStreamingDecoder(byteLevelDecoderConfiguration,
                                              tokenizer: tokenizer,
                                              resolved: resolved)
        case .harmony:
            resolved = try Self.resolveHarmonyTokens(tokenizer)
            try Self.validateHarmonyStreamingDecoder(byteLevelDecoderConfiguration,
                                                     tokenizer: tokenizer,
                                                     resolved: resolved)
        case .kimi:
            resolved = try Self.resolveKimiTokens(tokenizer)
            try Self.validateKimiStreamingDecoder(byteLevelDecoderConfiguration,
                                                  tokenizer: tokenizer,
                                                  resolved: resolved)
        }
        self.bosID = resolved.bosID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.harmonyStartID = resolved.harmonyStartID
        self.harmonyMessageID = resolved.harmonyMessageID
        self.harmonyConstrainID = resolved.harmonyConstrainID
        self.harmonyCallID = resolved.harmonyCallID
        self.harmonyReturnID = resolved.harmonyReturnID
        self.kimiToolSectionBeginID = resolved.kimiToolSectionBeginID
        self.kimiToolSectionEndID = resolved.kimiToolSectionEndID
        self.kimiToolCallBeginID = resolved.kimiToolCallBeginID
        self.kimiToolArgumentBeginID = resolved.kimiToolArgumentBeginID
        self.kimiToolCallEndID = resolved.kimiToolCallEndID
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
        self.thinkingMode = thinkingMode
        self.generationSuffix = switch dialect {
        case .harmony: Self.harmonyGenerationSuffix
        case .kimi: Self.kimiGenerationSuffix
        case .chatml: thinkingMode == .adaptive
            ? Self.adaptiveChatMLGenerationSuffix
            : Self.deriveGenerationSuffix(
                tokenizer, thinkingEnabled: thinkingMode.isEnabled)
        }
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32?
        let toolCallEndID: Int32?
        let toolResponseID: Int32?
        let toolResponseEndID: Int32?
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        var harmonyStartID: Int32?
        var harmonyMessageID: Int32?
        var harmonyConstrainID: Int32?
        var harmonyCallID: Int32?
        var harmonyReturnID: Int32?
        var kimiToolSectionBeginID: Int32?
        var kimiToolSectionEndID: Int32?
        var kimiToolCallBeginID: Int32?
        var kimiToolArgumentBeginID: Int32?
        var kimiToolCallEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    private static func validateStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let literalMarkers: [(Int32?, String)] = [
            (resolved.toolCallStartID, "<tool_call>"),
            (resolved.toolCallEndID, "</tool_call>"),
            (resolved.toolResponseID, "<tool_response>"),
            (resolved.toolResponseEndID, "</tool_response>"),
            (resolved.channelStartID, "<think>"),
            (resolved.channelEndID, "</think>"),
        ]
        for (id, content) in literalMarkers {
            guard let id,
                  let added = decoder.addedTokens[id],
                  added.content == content, !added.special else {
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML control token \(content) must be a literal ByteLevel barrier")
            }
        }

        let filteredMarkers = [resolved.eosID, resolved.endOfTurnID]
        for id in filteredMarkers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML stop token \(token) must be marked special")
            }
        }
    }

    /// Identifies the chat dialect from the tokenizer's framing tokens,
    /// most-specific-first: a Kimi vocabulary could also carry ChatML-like
    /// tokens, so its `<|im_middle|>` is tested before `<|im_start|>`.
    private static func resolveDialect(_ tokenizer: any Tokenizer) throws -> ChatDialect {
        if specialTokenID(tokenizer, Self.harmonyChannelMark) != nil { return .harmony }
        if specialTokenID(tokenizer, Self.kimiMiddleMark) != nil { return .kimi }
        if specialTokenID(tokenizer, Self.imStartMark) != nil { return .chatml }
        throw GFTokenizerError.unsupportedForDialect(
            "no recognized chat framing tokens (\(Self.harmonyChannelMark), "
                + "\(Self.kimiMiddleMark), or \(Self.imStartMark))")
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    /// The model's padded embedding/lm_head row count. The tokenizer's own
    /// vocab (248 077 for Qwen) is smaller; logits buffers and the
    /// embedding/lm_head are sized to the padded rows, and `vocabSize`
    /// reports at least this many.
    private static let paddedLogitsVocabSize = 248_320

    /// Derive the tokenizer's actual vocab by probing `convertIdToToken` for
    /// the first invalid id. Standard vocab files keep ids dense from 0, so
    /// the first nil is the vocab count. Bounded so a pathological tokenizer
    /// cannot make init scan forever; nil means "no reliable derivation".
    private static func derivedVocabSize(_ tokenizer: any Tokenizer) -> Int? {
        // 2,097,152 — far above any shipping vocab.
        let upperBound = 1 << 21
        for id in 0..<upperBound where tokenizer.convertIdToToken(id) == nil {
            return id
        }
        return nil
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            // At least the model's padded embedding/lm_head rows; larger when
            // the tokenizer's own vocab (derived from `convertIdToToken`)
            // exceeds them.
            vocabSize: max(Self.derivedVocabSize(tokenizer) ?? 0,
                           Self.paddedLogitsVocabSize))
    }

    /// gpt-oss embedding/lm_head row count; the o200k_harmony vocab is dense
    /// up to it, so no separate padding applies.
    private static let harmonyLogitsVocabSize = 201_088

    private static func resolveHarmonyTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let startOfText = try id("<|startoftext|>")
        let endOfText = try id("<|endoftext|>")
        let returnMark = try id(Self.harmonyReturnMark)
        let constrain = try id(Self.harmonyConstrainMark)
        let channel = try id(Self.harmonyChannelMark)
        let start = try id(Self.harmonyStartMark)
        let end = try id(Self.harmonyEndMark)
        let message = try id(Self.harmonyMessageMark)
        let call = try id(Self.harmonyCallMark)
        return ResolvedSpecialTokens(
            bosID: startOfText,
            eosID: returnMark,
            padID: endOfText,
            endOfTurnID: end,
            toolCallStartID: nil,
            toolCallEndID: nil,
            toolResponseID: nil,
            toolResponseEndID: nil,
            channelStartID: channel,
            channelEndID: end,
            thinkStartID: nil,
            thinkEndID: nil,
            harmonyStartID: start,
            harmonyMessageID: message,
            harmonyConstrainID: constrain,
            harmonyCallID: call,
            harmonyReturnID: returnMark,
            stopTokenIDs: [returnMark, call],
            vocabSize: max(Self.derivedVocabSize(tokenizer) ?? 0,
                           Self.harmonyLogitsVocabSize))
    }

    /// Every Harmony framing token must be a special added token so the
    /// streaming detokenizer filters it and the structured decoder sees an
    /// empty delta for it.
    private static func validateHarmonyStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let markers = [resolved.bosID, resolved.padID, resolved.eosID,
                       resolved.endOfTurnID, resolved.channelStartID]
            + [resolved.harmonyStartID, resolved.harmonyMessageID,
               resolved.harmonyConstrainID, resolved.harmonyCallID].compactMap { $0 }
        for id in markers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "Harmony control token \(token) must be marked special")
            }
        }
    }

    /// Kimi-Linear embedding/lm_head row count: the 163 584-token base vocab
    /// plus the 256 reserved special slots.
    private static let kimiLogitsVocabSize = 163_840

    private static func resolveKimiTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        let bos = try id("[BOS]")
        let eos = try id("[EOS]")
        let pad = try id("[PAD]")
        let imEnd = try id(Self.imEndMark)
        let middle = try id(Self.kimiMiddleMark)
        // The role marks have no stored properties; rendering relies on the
        // tokenizer recognizing their text.
        _ = try id(Self.kimiUserMark)
        _ = try id(Self.kimiAssistantMark)
        _ = try id(Self.kimiSystemMark)
        return ResolvedSpecialTokens(
            bosID: bos,
            eosID: eos,
            padID: pad,
            endOfTurnID: imEnd,
            toolCallStartID: nil,
            toolCallEndID: nil,
            toolResponseID: nil,
            toolResponseEndID: nil,
            channelStartID: middle,
            channelEndID: imEnd,
            thinkStartID: nil,
            thinkEndID: nil,
            kimiToolSectionBeginID: try id(Self.kimiToolSectionBeginMark),
            kimiToolSectionEndID: try id(Self.kimiToolSectionEndMark),
            kimiToolCallBeginID: try id(Self.kimiToolCallBeginMark),
            kimiToolArgumentBeginID: try id(Self.kimiToolArgumentBeginMark),
            kimiToolCallEndID: try id(Self.kimiToolCallEndMark),
            stopTokenIDs: [imEnd, eos],
            vocabSize: max(Self.derivedVocabSize(tokenizer) ?? 0,
                           Self.kimiLogitsVocabSize))
    }

    /// Kimi framing tokens must be special (filtered to empty deltas) while
    /// the five tool-section markers must be literal ByteLevel barriers so
    /// the structured decoder sees their text in the delta.
    private static func validateKimiStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let specialMarkers = [resolved.bosID, resolved.eosID, resolved.padID,
                              resolved.endOfTurnID, resolved.channelStartID]
            + [Self.kimiUserMark, Self.kimiAssistantMark, Self.kimiSystemMark]
                .compactMap { specialTokenID(tokenizer, $0).map(Int32.init) }
        for id in specialMarkers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "Kimi control token \(token) must be marked special")
            }
        }
        let literalMarkers: [(Int32?, String)] = [
            (resolved.kimiToolSectionBeginID, Self.kimiToolSectionBeginMark),
            (resolved.kimiToolSectionEndID, Self.kimiToolSectionEndMark),
            (resolved.kimiToolCallBeginID, Self.kimiToolCallBeginMark),
            (resolved.kimiToolArgumentBeginID, Self.kimiToolArgumentBeginMark),
            (resolved.kimiToolCallEndID, Self.kimiToolCallEndMark),
        ]
        for (id, content) in literalMarkers {
            guard let id,
                  let added = decoder.addedTokens[id],
                  added.content == content, !added.special else {
                throw GFTokenizerError.unsupportedForDialect(
                    "Kimi tool marker \(content) must be a literal ByteLevel barrier")
            }
        }
    }

    /// Encode UTF-8 text to token IDs.
    ///
    /// ChatML has no BOS, so `addBOS` is a no-op; BOS is never prepended.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Codable, Sendable {
        case system, developer, user, assistant, tool
    }
    public struct HistoricalToolCall: Codable, Sendable, Equatable {
        public let id: String
        public let name: String
        /// The JSON text the model emitted, verbatim. Not a parsed object:
        /// the KV was built from these bytes, so anything that re-serialises
        /// them can reorder the parameters and break the prefix match.
        public let arguments: String

        public init(id: String, name: String, arguments: String) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Codable, Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Codable, Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?
        /// Harmony analysis text replayed inside a tool loop
        /// (`reasoning_content` on the wire); other dialects ignore it.
        public let thinking: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
            self.thinking = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil,
                    thinking: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
            self.thinking = thinking
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    private static let harmonyChannelMark = "<|channel|>"
    private static let harmonyStartMark = "<|start|>"
    private static let harmonyEndMark = "<|end|>"
    private static let harmonyMessageMark = "<|message|>"
    private static let harmonyConstrainMark = "<|constrain|>"
    private static let harmonyCallMark = "<|call|>"
    private static let harmonyReturnMark = "<|return|>"
    private static let kimiMiddleMark = "<|im_middle|>"
    private static let kimiUserMark = "<|im_user|>"
    private static let kimiAssistantMark = "<|im_assistant|>"
    private static let kimiSystemMark = "<|im_system|>"
    static let kimiToolSectionBeginMark = "<|tool_calls_section_begin|>"
    static let kimiToolSectionEndMark = "<|tool_calls_section_end|>"
    static let kimiToolCallBeginMark = "<|tool_call_begin|>"
    static let kimiToolArgumentBeginMark = "<|tool_call_argument_begin|>"
    static let kimiToolCallEndMark = "<|tool_call_end|>"
    static let kimiGenerationSuffix = "<|im_assistant|>assistant<|im_middle|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch. Used only
    /// when the tokenizer has no chat template or template rendering fails
    /// (R6); `generationSuffix` carries the template-derived value otherwise.
    private static let fallbackChatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    /// Same as `fallbackChatMLGenerationSuffix`, but for thinking mode ON:
    /// the template's `enable_thinking=true` branch leaves the `<think>`
    /// block open so the model must reason before answering.
    private static let fallbackChatMLGenerationSuffixThinking =
        "<|im_start|>assistant\n<think>\n"
    /// Adaptive thinking injects nothing after the role header — the model
    /// decides whether to open a `<think>` block. Pinned rather than derived:
    /// the template's `enable_thinking` boolean can only choose between the
    /// two injected forms.
    static let adaptiveChatMLGenerationSuffix = "<|im_start|>assistant\n"

    /// Derive the generation-prompt suffix from the tokenizer's bundled
    /// `chat_template.jinja` (`add_generation_prompt` with thinking per
    /// `thinkingEnabled`), falling back to the pinned constant when no
    /// template is available or rendering fails. The probe renders one empty
    /// user turn both with and without the generation prompt; the generation
    /// prompt is appended after the message loop, so the suffix is the
    /// token-level difference between the two renders.
    private static func deriveGenerationSuffix(_ tokenizer: any Tokenizer,
                                               thinkingEnabled: Bool) -> String {
        let fallback = thinkingEnabled
            ? Self.fallbackChatMLGenerationSuffixThinking
            : Self.fallbackChatMLGenerationSuffix
        guard tokenizer.hasChatTemplate else {
            return fallback
        }
        let probe: [Tokenizers.Message] = [["role": "user", "content": ""]]
        do {
            let withPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: ["enable_thinking": thinkingEnabled])
            let withoutPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: false,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: ["enable_thinking": thinkingEnabled])
            guard withPrompt.count > withoutPrompt.count else {
                return fallback
            }
            // The generation prompt is appended after the message loop, so the
            // with-prompt render is the without-prompt render plus the suffix.
            let suffixIDs = Array(withPrompt[withoutPrompt.count...])
            return tokenizer.decode(tokens: suffixIDs, skipSpecialTokens: false)
        } catch {
            return fallback
        }
    }

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch dialect {
        case .chatml: return try chatMLChatTemplate(messages)
        case .harmony: return try harmonyChatTemplate(messages, tools: [])
        case .kimi: return try kimiChatTemplate(messages, tools: [])
        }
    }

    private func chatMLChatTemplate(_ messages: [Message],
                                    addGenerationPrompt: Bool = true) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            // The bundled Jinja template trims every message's rendered
            // content (`render_content(...)|trim`); the manual renderer
            // mirrors that exactly so both paths agree byte-for-byte.
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        if addGenerationPrompt { s += generationSuffix }
        return s
    }

    // MARK: - Harmony rendering

    /// Hand port of the gpt-oss `chat_template.jinja` (Harmony), byte-exact
    /// against jinja2 renders of the real template — including its whitespace
    /// artifacts in nested TypeScript types — except that object properties
    /// render in sorted key order (JSON order does not survive decoding) and
    /// `tojson` output is compact with sorted keys.
    static let harmonyModelIdentity =
        "You are ChatGPT, a large language model trained by OpenAI."
    static let harmonyGenerationSuffix = "<|start|>assistant"
    /// Template-source indentation retained by non-trimming Jinja tags inside
    /// nested type renders.
    private static let harmonyNestedTypeBreak = "\n                "

    func harmonyChatTemplate(_ messages: [Message],
                             tools: [FunctionDefinition],
                             currentDate: String? = nil,
                             addGenerationPrompt: Bool = true) throws -> String {
        var s = Self.harmonySystemBlock(
            hasTools: !tools.isEmpty,
            currentDate: currentDate ?? Self.harmonyCurrentDate())
        var loop = messages[...]
        var developerInstructions: String?
        if let first = loop.first, first.role == .system || first.role == .developer {
            guard let content = first.content else {
                throw GFTokenizerError.invalidChatTemplate(
                    "system messages require content")
            }
            developerInstructions = content
            loop = loop.dropFirst()
        }
        s += Self.harmonyDeveloperBlock(instructions: developerInstructions,
                                        tools: tools)
        var lastToolCallName: String?
        for message in loop {
            s += try Self.harmonyMessageBlock(
                message, lastToolCallName: &lastToolCallName)
        }
        if addGenerationPrompt { s += Self.harmonyGenerationSuffix }
        return s
    }

    private static func harmonyCurrentDate() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: Date())
    }

    private static func harmonySystemBlock(hasTools: Bool,
                                           currentDate: String) -> String {
        var s = "<|start|>system<|message|>"
        s += Self.harmonyModelIdentity + "\n"
        s += "Knowledge cutoff: 2024-06\n"
        s += "Current date: " + currentDate + "\n\n"
        s += "Reasoning: medium\n\n"
        s += "# Valid channels: analysis, commentary, final. "
        s += "Channel must be included for every message."
        if hasTools {
            s += "\nCalls to these tools must go to the commentary channel: 'functions'."
        }
        s += "<|end|>"
        return s
    }

    private static func harmonyDeveloperBlock(
        instructions: String?,
        tools: [FunctionDefinition]
    ) -> String {
        let hasInstructions = !(instructions ?? "").isEmpty
        guard hasInstructions || !tools.isEmpty else { return "" }
        var s = "<|start|>developer<|message|>"
        if hasInstructions, let instructions {
            s += "# Instructions\n\n" + instructions
        }
        if !tools.isEmpty {
            s += "\n\n# Tools\n\n" + Self.harmonyToolNamespace(tools)
        }
        s += "<|end|>"
        return s
    }

    private static func harmonyMessageBlock(
        _ message: Message,
        lastToolCallName: inout String?
    ) throws -> String {
        switch message.role {
        case .system, .developer:
            throw GFTokenizerError.invalidChatTemplate("system message must be first")
        case .user:
            guard let content = message.content else {
                throw GFTokenizerError.invalidChatTemplate("user messages require content")
            }
            return "<|start|>user<|message|>" + content + "<|end|>"
        case .assistant:
            return try harmonyAssistantBlock(
                message, lastToolCallName: &lastToolCallName)
        case .tool:
            guard let name = lastToolCallName else {
                throw GFTokenizerError.invalidChatTemplate(
                    "tool message without a preceding assistant tool call")
            }
            guard let content = message.content else {
                throw GFTokenizerError.invalidChatTemplate("tool messages require content")
            }
            return "<|start|>functions.\(name) to=assistant<|channel|>commentary<|message|>"
                + (try JSONValue.string(content).encoded()) + "<|end|>"
        }
    }

    private static func harmonyAssistantBlock(
        _ message: Message,
        lastToolCallName: inout String?
    ) throws -> String {
        for field in [message.content, message.thinking] {
            guard let field else { continue }
            if field.contains("<|channel|>analysis<|message|>")
                || field.contains("<|channel|>final<|message|>") {
                throw GFTokenizerError.invalidChatTemplate(
                    "assistant analysis belongs in thinking and final text in "
                        + "content, not inline <|channel|> markup")
            }
        }
        guard !message.toolCalls.isEmpty else {
            guard let content = message.content else {
                throw GFTokenizerError.invalidChatTemplate(
                    "assistant messages require content")
            }
            lastToolCallName = nil
            return "<|start|>assistant<|channel|>final<|message|>" + content + "<|end|>"
        }
        guard message.toolCalls.count == 1 else {
            throw GFTokenizerError.invalidChatTemplate(
                "at most one tool call per assistant message")
        }
        let content = message.content ?? ""
        let thinking = message.thinking ?? ""
        if !content.isEmpty, !thinking.isEmpty {
            throw GFTokenizerError.invalidChatTemplate(
                "assistant tool-call messages take analysis in content or "
                    + "thinking, not both")
        }
        var s = ""
        let analysis = content.isEmpty ? thinking : content
        if !analysis.isEmpty {
            s += "<|start|>assistant<|channel|>analysis<|message|>" + analysis + "<|end|>"
        }
        let call = message.toolCalls[0]
        s += "<|start|>assistant to=functions.\(call.name)"
        s += "<|channel|>commentary json<|message|>"
        s += call.arguments
        s += "<|call|>"
        lastToolCallName = call.name
        return s
    }

    private static func harmonyToolNamespace(_ tools: [FunctionDefinition]) -> String {
        var s = "## functions\n\nnamespace functions {\n\n"
        for tool in tools {
            s += "// " + tool.description + "\n"
            s += "type " + tool.name + " = "
            let parameters = tool.parameters.objectValue ?? [:]
            let properties = parameters["properties"]?.objectValue ?? [:]
            if !parameters.isEmpty, !properties.isEmpty {
                s += "(_: {\n"
                s += harmonyParameterLines(properties, requiredValue: parameters["required"])
                s += "}) => any;\n\n"
            } else {
                s += "() => any;\n\n"
            }
        }
        s += "} // namespace functions"
        return s
    }

    private static func harmonyParameterLines(
        _ properties: [String: JSONValue],
        requiredValue: JSONValue?
    ) -> String {
        let required = harmonyRequiredNames(requiredValue)
        var s = ""
        let sorted = properties.sorted { $0.key < $1.key }
        for (index, (name, spec)) in sorted.enumerated() {
            let object = spec.objectValue ?? [:]
            if case .string(let description) = object["description"], !description.isEmpty {
                s += "// " + description + "\n"
            }
            s += name
            if !required.contains(name) { s += "?" }
            s += ": "
            s += harmonyTypeScriptType(spec)
            if let defaultValue = object["default"] {
                if harmonyTruthy(object["enum"]) {
                    s += ", // default: " + harmonyRawText(defaultValue)
                } else if harmonyTruthy(object["oneOf"]) {
                    s += "// default: " + harmonyRawText(defaultValue)
                } else {
                    s += ", // default: " + ((try? defaultValue.encoded()) ?? "null")
                }
            }
            s += index == sorted.count - 1 ? "\n" : ",\n"
        }
        return s
    }

    private static func harmonyTypeScriptType(_ spec: JSONValue) -> String {
        let object = spec.objectValue ?? [:]
        let type = object["type"]
        if case .string("array") = type {
            return harmonyArrayType(object)
        }
        if case .array(let types) = type, !types.isEmpty {
            return types.map(harmonyRawText).joined(separator: " | ")
        }
        if case .array(let variants) = object["oneOf"], !variants.isEmpty {
            return harmonyOneOfType(variants)
        }
        switch type {
        case .string("string"):
            if case .array(let values) = object["enum"], !values.isEmpty {
                return "\"" + values.map(harmonyRawText).joined(separator: "\" | \"") + "\""
            }
            return harmonyTruthy(object["nullable"]) ? "string | null" : "string"
        case .string("number"), .string("integer"):
            return "number"
        case .string("boolean"):
            return "boolean"
        case .string("object"):
            guard let properties = object["properties"]?.objectValue,
                  !properties.isEmpty else { return "object" }
            let sorted = properties.sorted { $0.key < $1.key }
            let required = harmonyRequiredNames(object["required"])
            var s = "{\n"
            for (index, (name, propertySpec)) in sorted.enumerated() {
                s += name
                if !required.contains(name) { s += "?" }
                s += ": " + Self.harmonyNestedTypeBreak
                s += harmonyTypeScriptType(propertySpec)
                if index != sorted.count - 1 { s += ", " }
            }
            return s + "}"
        default:
            return "any"
        }
    }

    private static func harmonyArrayType(_ object: [String: JSONValue]) -> String {
        let nullable = harmonyTruthy(object["nullable"]) ? " | null" : ""
        guard harmonyTruthy(object["items"]), let items = object["items"] else {
            return "any[]" + nullable
        }
        switch items.objectValue?["type"] {
        case .string("string"): return "string[]" + nullable
        case .string("number"), .string("integer"): return "number[]" + nullable
        case .string("boolean"): return "boolean[]" + nullable
        default:
            let inner = harmonyTypeScriptType(items)
            let collapsed = inner == "object | object" || inner.count > 50
            return (collapsed ? "any[]" : inner + "[]") + nullable
        }
    }

    private static func harmonyOneOfType(_ variants: [JSONValue]) -> String {
        let hasObjectVariants = variants.contains {
            $0.objectValue?["type"] == .string("object")
        }
        if hasObjectVariants, variants.count > 1 { return "any" }
        var s = ""
        for (index, variant) in variants.enumerated() {
            s += harmonyTypeScriptType(variant)
            let object = variant.objectValue ?? [:]
            if case .string(let description) = object["description"], !description.isEmpty {
                s += "// " + description
            }
            if let defaultValue = object["default"] {
                s += "\n                    // default: "
                    + ((try? defaultValue.encoded()) ?? "null")
            }
            if index != variants.count - 1 {
                s += " | " + Self.harmonyNestedTypeBreak
            }
        }
        return s
    }

    private static func harmonyRequiredNames(_ value: JSONValue?) -> Set<String> {
        guard case .array(let names) = value else { return [] }
        return Set(names.compactMap {
            if case .string(let name) = $0 { return name }
            return nil
        })
    }

    /// Jinja string concatenation of a template value (`+ param_spec.default`):
    /// strings pass through bare, everything else via its JSON text.
    private static func harmonyRawText(_ value: JSONValue) -> String {
        if case .string(let text) = value { return text }
        return (try? value.encoded()) ?? "null"
    }

    /// Jinja truthiness for the template's `if` checks.
    private static func harmonyTruthy(_ value: JSONValue?) -> Bool {
        switch value {
        case .none, .null: return false
        case .bool(let value): return value
        case .string(let value): return !value.isEmpty
        case .array(let value): return !value.isEmpty
        case .object(let value): return !value.isEmpty
        case .integer(let value): return value != 0
        case .unsignedInteger(let value): return value != 0
        case .number(let value): return value != 0
        case .decimal(let value): return value != 0
        }
    }

    // MARK: - Kimi rendering

    /// Hand port of the Kimi-Linear `chat_template.jinja`, byte-exact against
    /// jinja2 renders of the shipped template — except that `tojson` output
    /// is compact with sorted keys (JSON order does not survive decoding),
    /// the same normalization the Harmony renderer pins.
    func kimiChatTemplate(_ messages: [Message],
                          tools: [FunctionDefinition],
                          addGenerationPrompt: Bool = true) throws -> String {
        var s = ""
        if !tools.isEmpty {
            s += Self.kimiSystemMark + "tool_declare" + Self.kimiMiddleMark
                + (try Self.kimiToolsJSON(tools)) + Self.imEndMark
        }
        for message in messages {
            s += try Self.kimiMessageBlock(message)
        }
        if addGenerationPrompt { s += Self.kimiGenerationSuffix }
        return s
    }

    private static func kimiMessageBlock(_ message: Message) throws -> String {
        // The template labels each turn `name or role`, so an empty name
        // falls back to the role.
        let roleName = message.name.flatMap { $0.isEmpty ? nil : $0 }
            ?? message.role.rawValue
        let mark = switch message.role {
        case .user: kimiUserMark
        case .assistant: kimiAssistantMark
        case .system, .developer, .tool: kimiSystemMark
        }
        return mark + roleName + kimiMiddleMark
            + (try kimiMessageBody(message)) + imEndMark
    }

    private static func kimiMessageBody(_ message: Message) throws -> String {
        if message.role == .assistant, !message.toolCalls.isEmpty {
            var s = message.content ?? ""
            s += kimiToolSectionBeginMark
            for call in message.toolCalls {
                s += kimiToolCallBeginMark + call.id
                    + kimiToolArgumentBeginMark
                    + call.arguments
                    + kimiToolCallEndMark
            }
            return s + kimiToolSectionEndMark
        }
        guard let content = message.content else {
            throw GFTokenizerError.invalidChatTemplate(
                "\(message.role.rawValue) messages require content")
        }
        if message.role == .tool {
            guard let id = message.toolCallID else {
                throw GFTokenizerError.invalidChatTemplate(
                    "tool messages require tool_call_id")
            }
            return "## Return of \(id)\n" + content
        }
        return content
    }

    /// The template's tool declaration: the OpenAI wire shape rendered by
    /// `tojson(separators=(',', ':'))`.
    private static func kimiToolsJSON(_ tools: [FunctionDefinition]) throws -> String {
        try JSONValue.array(tools.map { tool in
            .object(["type": .string("function"),
                     "function": .object([
                        "name": .string(tool.name),
                        "description": .string(tool.description),
                        "parameters": tool.parameters,
                     ])])
        }).encoded()
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        if dialect == .harmony {
            return encode(try harmonyChatTemplate(messages, tools: tools),
                          addBOS: false)
        }
        if dialect == .kimi {
            return encode(try kimiChatTemplate(messages, tools: tools),
                          addBOS: false)
        }
        let rendered = try upstreamJinjaRender(
            messages,
            tools: tools,
            // Adaptive appends the bare role header itself: the template's
            // `enable_thinking` boolean can only pick an injected form, and
            // the generation prompt is appended after the message loop, so
            // the split render is a token-exact substitute (same property the
            // suffix derivation probe relies on).
            addGenerationPrompt: thinkingMode != .adaptive)
        guard thinkingMode == .adaptive else { return rendered }
        return rendered + encode(generationSuffix, addBOS: false)
    }

    /// The ChatML tools path: the tokenizer's bundled `chat_template.jinja`,
    /// which the hand-written `chatMLChatTemplate` does not reproduce.
    private func upstreamJinjaRender(
        _ messages: [Message],
        tools: [FunctionDefinition],
        addGenerationPrompt: Bool
    ) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try JSONValue.orderedJinjaObject(call.arguments),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            if let thinking = message.thinking { value["reasoning_content"] = thinking }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: addGenerationPrompt,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: ["enable_thinking": thinkingMode.isEnabled]
        ).map(Int32.init)
    }

    // MARK: - Settled boundary

    /// Token count of the render truncated at the last user query — the
    /// boundary between the settled region, whose bytes no later turn can
    /// move, and the live region the in-flight request rewrites. The result is
    /// a token prefix of the full render of the same `messages` and `tools`.
    public func settledBoundaryTokenCount(messages: [Message],
                                          tools: [FunctionDefinition]) throws -> Int {
        guard let queryIndex = lastQueryIndex(messages) else {
            throw GFTokenizerError.invalidChatTemplate("no user query found in messages")
        }
        let settled = Array(messages[...queryIndex])
        switch dialect {
        case .chatml:
            guard tools.isEmpty else {
                return try upstreamJinjaRender(settled, tools: tools,
                                               addGenerationPrompt: false).count
            }
            return encode(try chatMLChatTemplate(settled, addGenerationPrompt: false),
                          addBOS: false).count
        case .harmony:
            return encode(try harmonyChatTemplate(settled, tools: tools,
                                                  addGenerationPrompt: false),
                          addBOS: false).count
        case .kimi:
            return encode(try kimiChatTemplate(settled, tools: tools,
                                               addGenerationPrompt: false),
                          addBOS: false).count
        }
    }

    /// The ChatML branch is the shipped template's `last_query_index` scan;
    /// Harmony and Kimi have no equivalent and take the last user message.
    private func lastQueryIndex(_ messages: [Message]) -> Int? {
        switch dialect {
        case .chatml:
            return messages.lastIndex {
                $0.role == .user && !Self.isToolResponseWrapper($0)
            }
        case .harmony, .kimi:
            return messages.lastIndex { $0.role == .user }
        }
    }

    private static func isToolResponseWrapper(_ message: Message) -> Bool {
        let content = (message.content ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return content.hasPrefix("<tool_response>") && content.hasSuffix("</tool_response>")
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        // The template trims user content (`render_content(...)|trim`), so the
        // continuation bridge mirrors it; see `chatMLChatTemplate`.
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return [endOfTurnID] + encode(
            "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                + generationSuffix,
            addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        // The ChatML template's `<think>` stripping depends on each assistant
        // turn's position relative to the last user query, so a re-rendered
        // prefix is not guaranteed to be a token prefix of the full render.
        // Callers (ServerPromptCache) fall back to prefix matching; the
        // tool-result KV continuation is unsupported for ChatML.
        throw GFTokenizerError.unsupportedForDialect("tool-result KV continuation")
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case local(String, ModelThinkingMode)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .local(let path, let thinkingMode):
                return try await GFTokenizer.loadUncached(
                    from: URL(fileURLWithPath: path),
                    thinkingMode: thinkingMode)
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
