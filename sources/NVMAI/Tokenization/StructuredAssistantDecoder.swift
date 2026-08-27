import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case thinking(String)
    case toolCall(ParsedToolCall)
}

/// unchecked-invariant: one decoder per generation, driven only from that
/// generation's task. Its channel/tool-token state is a running parse of a
/// single token stream and would be meaningless shared, so exclusive
/// ownership -- not locking -- is what makes it safe.
public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
    }

    /// Harmony generation is a sequence of
    /// `<|start|>{header}<|channel|>{header}<|message|>{content}` blocks; the
    /// initial state is `header` because the prompt already ends with
    /// `<|start|>assistant`. All markers are special tokens (empty deltas), so
    /// transitions key on token IDs alone.
    private enum HarmonyState {
        case expectStart
        case header(recipient: String)
        case channelHeader(recipient: String, channel: String)
        case thought
        case visible
        case toolArguments(name: String, body: String)
        case ended
    }

    /// Kimi generation is visible content optionally followed by one
    /// tool-call section. The five section markers are literal ByteLevel
    /// barriers (validated non-special), so transitions key on their token
    /// IDs while the marker text arrives at the end of the delta, exactly
    /// like ChatML's `<tool_call>`.
    private enum KimiState {
        case content
        case section
        case callID(String)
        case arguments(id: String, body: String)
        case sectionEnded
    }

    private static let maximumHarmonyHeaderBytes = 4096
    private static let maximumKimiCallIDBytes = 4096

    private let tokenizer: GFTokenizer
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var harmonyState: HarmonyState = .header(recipient: "")
    private var kimiState: KimiState = .content
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
        if tokenizer.dialect == .chatml {
            primeChatMLInjectedPrefix()
        }
    }

    /// The ChatML generation prompt may inject think markup (`<think>` open
    /// for thinking-on, a closed empty block for off, nothing for adaptive).
    /// Feeding those prompt tokens through the same consume path makes the
    /// decoder indifferent to whether a marker was injected or generated —
    /// the stream simply starts prefilled. Prompt-echo events are discarded;
    /// without this, injected-open thinking streamed as visible content.
    private func primeChatMLInjectedPrefix() {
        var detokenizer = GFDetokenizer(tokenizer: tokenizer)
        for id in tokenizer.encode(tokenizer.generationSuffix, addBOS: false) {
            guard let delta = try? detokenizer.push(id),
                  (try? consumeChatML(tokenID: id, delta: delta)) != nil else {
                assertionFailure("generation prompt failed to parse")
                failed = true
                return
            }
        }
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        switch tokenizer.dialect {
        case .chatml:
            return try consumeChatML(tokenID: tokenID, delta: delta)
        case .harmony:
            return try consumeHarmony(tokenID: tokenID, delta: delta)
        case .kimi:
            return try consumeKimi(tokenID: tokenID, delta: delta)
        }
    }

    /// Harmony transitions key on the framing tokens' IDs; their deltas are
    /// always empty (validated special tokens), so a non-empty delta on a
    /// marker means a mis-built tokenizer and fails closed.
    private func consumeHarmony(tokenID: Int32,
                                delta: String) throws -> [StructuredAssistantEvent] {
        if [tokenizer.harmonyStartID, tokenizer.channelStartID,
            tokenizer.harmonyMessageID, tokenizer.harmonyConstrainID,
            tokenizer.endOfTurnID, tokenizer.harmonyCallID,
            tokenizer.harmonyReturnID].contains(tokenID) {
            guard delta.isEmpty else {
                failed = true
                throw ToolCallParserError.malformed
            }
            return try consumeHarmonyMarker(tokenID: tokenID)
        }
        switch harmonyState {
        case .expectStart, .ended:
            guard delta.isEmpty else {
                failed = true
                throw ToolCallParserError.malformed
            }
            return []
        case .header(let recipient):
            harmonyState = .header(recipient: try appendingHeader(recipient, delta))
            return []
        case .channelHeader(let recipient, let channel):
            harmonyState = .channelHeader(recipient: recipient,
                                          channel: try appendingHeader(channel, delta))
            return []
        case .thought:
            return delta.isEmpty ? [] : [.thinking(delta)]
        case .visible:
            return delta.isEmpty ? [] : [.content(delta)]
        case .toolArguments(let name, let body):
            let grown = body + delta
            guard grown.utf8.count <= HarmonyToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            harmonyState = .toolArguments(name: name, body: grown)
            return []
        }
    }

    private func consumeHarmonyMarker(
        tokenID: Int32
    ) throws -> [StructuredAssistantEvent] {
        switch (tokenID, harmonyState) {
        case (tokenizer.harmonyStartID, .expectStart):
            harmonyState = .header(recipient: "")
            return []
        case (tokenizer.channelStartID, .header(let recipient)):
            harmonyState = .channelHeader(recipient: recipient, channel: "")
            return []
        case (tokenizer.harmonyMessageID, .channelHeader(let recipient, let channel)):
            try enterHarmonyContent(recipient: recipient, channel: channel)
            return []
        case (tokenizer.harmonyConstrainID, .header),
             (tokenizer.harmonyConstrainID, .channelHeader):
            return []
        case (tokenizer.endOfTurnID, .thought),
             (tokenizer.endOfTurnID, .visible):
            harmonyState = .expectStart
            return []
        case (tokenizer.harmonyCallID, .toolArguments(let name, let body)):
            harmonyState = .ended
            do {
                let call = try HarmonyToolCallParser().parse(
                    name: name, body: body,
                    allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        case (tokenizer.harmonyReturnID, .visible):
            harmonyState = .ended
            return []
        default:
            failed = true
            throw ToolCallParserError.malformed
        }
    }

    /// Routes a completed `{recipient}<|channel|>{channel}` header: a
    /// `to=functions.NAME` recipient on either side opens a buffered tool
    /// call regardless of channel; otherwise the channel name decides.
    private func enterHarmonyContent(recipient: String, channel: String) throws {
        let name: String?
        do {
            name = try HarmonyToolCallParser.recipientFunctionName(
                inHeader: recipient + " " + channel)
        } catch {
            failed = true
            throw error
        }
        if let name {
            harmonyState = .toolArguments(name: name, body: "")
            return
        }
        let channelName = channel.split(whereSeparator: \.isWhitespace)
            .first.map(String.init) ?? ""
        switch channelName {
        case "analysis":
            harmonyState = .thought
        case "final", "commentary":
            harmonyState = .visible
        default:
            failed = true
            throw ToolCallParserError.malformed
        }
    }

    private func appendingHeader(_ header: String, _ delta: String) throws -> String {
        let grown = header + delta
        guard grown.utf8.count <= Self.maximumHarmonyHeaderBytes else {
            failed = true
            throw ToolCallParserError.malformed
        }
        return grown
    }

    /// Kimi transitions: between structural markers only whitespace may
    /// stream (the template renders them adjacent; the reference parser
    /// tolerates `\s*`), and content after the section closes fails closed
    /// because a re-render would drop it.
    private func consumeKimi(tokenID: Int32,
                             delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.kimiToolSectionBeginID {
            guard case .content = kimiState else {
                failed = true
                throw ToolCallParserError.malformed
            }
            let prefix = try boundaryPrefix(
                delta, marker: GFTokenizer.kimiToolSectionBeginMark)
            kimiState = .section
            return visibleEvents(prefix)
        }
        if tokenID == tokenizer.kimiToolCallBeginID {
            let prefix = try boundaryPrefix(
                delta, marker: GFTokenizer.kimiToolCallBeginMark)
            guard case .section = kimiState, prefix.allSatisfy(\.isWhitespace) else {
                failed = true
                throw ToolCallParserError.malformed
            }
            kimiState = .callID("")
            return []
        }
        if tokenID == tokenizer.kimiToolArgumentBeginID {
            let prefix = try boundaryPrefix(
                delta, marker: GFTokenizer.kimiToolArgumentBeginMark)
            guard case .callID(let id) = kimiState else {
                failed = true
                throw ToolCallParserError.malformed
            }
            kimiState = .arguments(id: id + prefix, body: "")
            return []
        }
        if tokenID == tokenizer.kimiToolCallEndID {
            let prefix = try boundaryPrefix(
                delta, marker: GFTokenizer.kimiToolCallEndMark)
            guard case .arguments(let id, let body) = kimiState else {
                failed = true
                throw ToolCallParserError.malformed
            }
            kimiState = .section
            do {
                let call = try KimiToolCallParser().parse(
                    id: id, body: body + prefix, allowedTools: allowedTools)
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if tokenID == tokenizer.kimiToolSectionEndID {
            let prefix = try boundaryPrefix(
                delta, marker: GFTokenizer.kimiToolSectionEndMark)
            guard case .section = kimiState, prefix.allSatisfy(\.isWhitespace) else {
                failed = true
                throw ToolCallParserError.malformed
            }
            kimiState = .sectionEnded
            return []
        }
        return try consumeKimiText(delta)
    }

    private func consumeKimiText(_ delta: String) throws -> [StructuredAssistantEvent] {
        switch kimiState {
        case .content:
            return delta.isEmpty ? [] : [.content(delta)]
        case .section, .sectionEnded:
            guard delta.allSatisfy(\.isWhitespace) else {
                failed = true
                throw ToolCallParserError.malformed
            }
            return []
        case .callID(let id):
            let grown = id + delta
            guard grown.utf8.count <= Self.maximumKimiCallIDBytes else {
                failed = true
                throw ToolCallParserError.malformed
            }
            kimiState = .callID(grown)
            return []
        case .arguments(let id, let body):
            let grown = body + delta
            guard grown.utf8.count <= KimiToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            kimiState = .arguments(id: id, body: grown)
            return []
        }
    }

    /// ChatML transitions: `<think>`…`</think>` route thought text to
    /// thinking events, and `<tool_call>`…`</tool_call>` buffer tokens for
    /// the Qwen parser. Everything else streams as visible content.
    private func consumeChatML(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            let prefix = try boundaryPrefix(delta, marker: "<tool_call>")
            let events = visibleEvents(prefix)
            toolTokens = []
            return events
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            _ = try boundaryPrefix(delta, marker: "</tool_call>")
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try QwenToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        if tokenID == tokenizer.thinkStartID {
            let prefix = try boundaryPrefix(delta, marker: "<think>")
            let events = visibleEvents(prefix)
            channel = .thought
            return events
        }
        if tokenID == tokenizer.thinkEndID {
            _ = try boundaryPrefix(delta, marker: "</think>")
            channel = .visible
            return []
        }
        guard channel != .thought else {
            return delta.isEmpty ? [] : [.thinking(delta)]
        }
        return delta.isEmpty ? [] : [.content(delta)]
    }

    /// Routes the detokenizer's final buffered bytes through the current
    /// channel instead of allowing thought/tool tails to become visible.
    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        if tokenizer.dialect == .harmony {
            switch harmonyState {
            case .visible:
                return visibleEvents(text)
            case .thought:
                return text.isEmpty ? [] : [.thinking(text)]
            case .toolArguments(let name, let body):
                let grown = body + text
                guard grown.utf8.count <= HarmonyToolCallParser.maximumBytes else {
                    failed = true
                    throw ToolCallParserError.oversized
                }
                harmonyState = .toolArguments(name: name, body: grown)
                return []
            case .expectStart, .header, .channelHeader, .ended:
                return []
            }
        }
        if tokenizer.dialect == .kimi {
            return try consumeKimiText(text)
        }
        guard toolTokens == nil else { return [] }
        if channel == .thought {
            return text.isEmpty ? [] : [.thinking(text)]
        }
        return visibleEvents(text)
    }

    private func boundaryPrefix(_ delta: String, marker: String) throws -> String {
        guard delta.hasSuffix(marker) else {
            failed = true
            throw ToolCallParserError.malformed
        }
        return String(delta.dropLast(marker.count))
    }

    private func visibleEvents(_ text: String) -> [StructuredAssistantEvent] {
        guard channel != .thought, !text.isEmpty else { return [] }
        return [.content(text)]
    }

    /// A truncation inside content is survivable; one inside a header or a
    /// buffered tool call left the stream structurally incomplete.
    public func finish() throws {
        guard !failed, toolTokens == nil else {
            throw ToolCallParserError.malformed
        }
        if tokenizer.dialect == .kimi {
            switch kimiState {
            case .content, .sectionEnded:
                return
            case .section, .callID, .arguments:
                throw ToolCallParserError.malformed
            }
        }
        guard tokenizer.dialect == .harmony else { return }
        switch harmonyState {
        case .thought, .visible, .expectStart, .ended:
            return
        case .header(let recipient) where recipient.isEmpty:
            return
        case .header, .channelHeader, .toolArguments:
            throw ToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }

    /// Whether the running parse sits outside a thought block: a stop token can
    /// land inside an unclosed `<think>`, and the template splits on
    /// `'</think>' in content`, so without the delimiter the partial reasoning
    /// re-renders as visible text.
    public var thoughtChannelClosed: Bool {
        guard !failed else { return false }
        switch tokenizer.dialect {
        case .chatml:
            return channel != .thought
        case .harmony:
            if case .thought = harmonyState { return false }
            return true
        case .kimi:
            return true
        }
    }
}
