import Foundation
import Testing
@testable import NVMAI

/// Kimi dialect coverage against the C1 synthetic tokenizer fixture: a
/// byte-level BPE vocab plus the Kimi special tokens at their real IDs, with
/// the model's `chat_template.jinja` alongside.
///
/// Golden strings are byte-exact jinja2 renders of that template
/// (trim_blocks/lstrip_blocks as transformers sets them), with `tojson`
/// compact with sorted keys — the same normalization the Harmony goldens pin,
/// because JSON object order does not survive decoding.
@Suite("Kimi template")
struct KimiTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "KimiTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private typealias Message = GFTokenizer.Message

    private func render(_ messages: [Message],
                        tools: [GFTokenizer.FunctionDefinition] = []) throws -> String {
        try tok.kimiChatTemplate(messages, tools: tools)
    }

    private static let weatherTool = GFTokenizer.FunctionDefinition(
        name: "get_weather",
        description: "Look up weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "city": .object(["type": .string("string"),
                                 "description": .string("City name")]),
                "days": .object(["type": .string("number")]),
            ]),
            "required": .array([.string("city")]),
        ]))

    private static let pingTool = GFTokenizer.FunctionDefinition(
        name: "ping",
        description: "Health check",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([:]),
        ]))

    private static let weatherJSON = #"{"function":{"description":"Look up weather","#
        + #""name":"get_weather","parameters":{"properties":{"city":{"description":"#
        + #""City name","type":"string"},"days":{"type":"number"}},"required":["city"],"#
        + #""type":"object"}},"type":"function"}"#
    private static let pingJSON = #"{"function":{"description":"Health check","#
        + #""name":"ping","parameters":{"properties":{},"type":"object"}},"#
        + #""type":"function"}"#

    @Test("Dialect and special-token IDs match the converted-tokenizer contract")
    func specialTokenIDs() {
        #expect(tok.dialect == .kimi)
        #expect(tok.bosID == 163_584)
        #expect(tok.eosID == 163_585)
        #expect(tok.padID == 163_839)
        #expect(tok.endOfTurnID == 163_586)
        #expect(tok.channelStartID == 163_601)
        #expect(tok.channelEndID == 163_586)
        #expect(tok.kimiToolSectionBeginID == 163_595)
        #expect(tok.kimiToolSectionEndID == 163_596)
        #expect(tok.kimiToolCallBeginID == 163_597)
        #expect(tok.kimiToolArgumentBeginID == 163_598)
        #expect(tok.kimiToolCallEndID == 163_599)
        #expect(tok.toolCallStartID == nil)
        #expect(tok.toolCallEndID == nil)
        #expect(tok.toolResponseID == nil)
        #expect(tok.toolResponseEndID == nil)
        #expect(tok.thinkStartID == nil)
        #expect(tok.thinkEndID == nil)
        #expect(tok.harmonyStartID == nil)
    }

    @Test("Stop tokens are im_end and EOS")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [163_586, 163_585])
    }

    @Test("Logits vocab is the Kimi-Linear row count")
    func vocabSize() {
        #expect(tok.vocabSize == 163_840)
    }

    @Test("Single user turn renders the exact Kimi string")
    func singleUserTurn() throws {
        let p = try render([Message(role: .user, content: "Hi")])
        #expect(p == "<|im_user|>user<|im_middle|>Hi<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Public template renders text-only chat with the generation suffix")
    func publicTemplate() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(p == "<|im_user|>user<|im_middle|>Hi<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Multi-turn renders system, user, and assistant marks")
    func multiTurn() throws {
        let p = try render([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B"),
            Message(role: .user, content: "C"),
        ])
        #expect(p == "<|im_system|>system<|im_middle|>Be terse.<|im_end|>"
            + "<|im_user|>user<|im_middle|>A<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>B<|im_end|>"
            + "<|im_user|>user<|im_middle|>C<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Message content is not trimmed")
    func untrimmedContent() throws {
        let p = try render([Message(role: .user, content: "  spaced  \n")])
        #expect(p == "<|im_user|>user<|im_middle|>  spaced  \n<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("A name labels the turn; an empty name falls back to the role")
    func nameOverride() throws {
        let named = try render([Message(role: .user, content: "Hi",
                                        name: "alice")])
        #expect(named == "<|im_user|>alice<|im_middle|>Hi<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
        let empty = try render([Message(role: .user, content: "Hi", name: "")])
        #expect(empty == "<|im_user|>user<|im_middle|>Hi<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Developer role renders through the system mark")
    func developerRole() throws {
        let p = try render([
            Message(role: .developer, content: "Rules"),
            Message(role: .user, content: "Hi"),
        ])
        #expect(p == "<|im_system|>developer<|im_middle|>Rules<|im_end|>"
            + "<|im_user|>user<|im_middle|>Hi<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Tools declare as a tool_declare system turn with compact JSON")
    func toolDeclaration() throws {
        let p = try render([
            Message(role: .system, content: "Be helpful."),
            Message(role: .user, content: "Weather in Paris?"),
        ], tools: [Self.weatherTool, Self.pingTool])
        #expect(p == "<|im_system|>tool_declare<|im_middle|>"
            + "[" + Self.weatherJSON + "," + Self.pingJSON + "]<|im_end|>"
            + "<|im_system|>system<|im_middle|>Be helpful.<|im_end|>"
            + "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Tool loop renders the call section and the Return-of result body")
    func toolLoop() throws {
        let p = try render([
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "",
                    toolCalls: [.init(id: "functions.get_weather:0",
                                      name: "get_weather",
                                      arguments: #"{"city":"Paris"}"#)]),
            Message(role: .tool, content: "22C, clear",
                    toolCallID: "functions.get_weather:0"),
        ], tools: [Self.weatherTool])
        #expect(p == "<|im_system|>tool_declare<|im_middle|>"
            + "[" + Self.weatherJSON + "]<|im_end|>"
            + "<|im_user|>user<|im_middle|>Weather in Paris?<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>"
            + "<|tool_calls_section_begin|>"
            + "<|tool_call_begin|>functions.get_weather:0"
            + "<|tool_call_argument_begin|>{\"city\":\"Paris\"}<|tool_call_end|>"
            + "<|tool_calls_section_end|><|im_end|>"
            + "<|im_system|>tool<|im_middle|>"
            + "## Return of functions.get_weather:0\n22C, clear<|im_end|>"
            + "<|im_assistant|>assistant<|im_middle|>")
    }

    @Test("Call content precedes the section and calls render in order")
    func contentAndMultipleCalls() throws {
        let p = try render([
            Message(role: .user, content: "Weather?"),
            Message(role: .assistant, content: "Checking now.",
                    toolCalls: [
                        .init(id: "functions.get_weather:0", name: "get_weather",
                              arguments: #"{"city":"Paris","days":2}"#),
                        .init(id: "functions.ping:1", name: "ping",
                              arguments: "{}"),
                    ]),
        ], tools: [Self.weatherTool, Self.pingTool])
        var expected = "<|im_assistant|>assistant<|im_middle|>Checking now."
        expected += "<|tool_calls_section_begin|>"
        expected += "<|tool_call_begin|>functions.get_weather:0"
        expected += "<|tool_call_argument_begin|>{\"city\":\"Paris\",\"days\":2}"
        expected += "<|tool_call_end|>"
        expected += "<|tool_call_begin|>functions.ping:1"
        expected += "<|tool_call_argument_begin|>{}<|tool_call_end|>"
        expected += "<|tool_calls_section_end|><|im_end|>"
        #expect(p.contains(expected))
    }

    @Test("String arguments pass through bare, as the template renders them")
    func stringArguments() throws {
        let p = try render([
            Message(role: .user, content: "Go"),
            Message(role: .assistant, content: "",
                    toolCalls: [.init(id: "functions.ping:0", name: "ping",
                                      arguments: "{\"raw\": 1}")]),
        ], tools: [Self.pingTool])
        #expect(p.contains(
            "<|tool_call_begin|>functions.ping:0"
                + "<|tool_call_argument_begin|>{\"raw\": 1}<|tool_call_end|>"))
    }

    @Test("Rendered prompt round-trips through the special-token vocabulary")
    func encodesToSpecialIDs() throws {
        let p = try render([Message(role: .user, content: "Hi")])
        let ids = tok.encode(p, addBOS: false)
        #expect(ids.first == 163_587)
        #expect(ids.contains(tok.endOfTurnID))
        #expect(ids.contains(163_588))
        #expect(ids.contains(tok.channelStartID))
        #expect(tok.decode(ids, skipSpecialTokens: false) == p)
    }

    @Test("Tool markers encode as single literal tokens")
    func toolMarkersEncodeAsBarriers() throws {
        let p = try render([
            Message(role: .user, content: "Go"),
            Message(role: .assistant, content: "",
                    toolCalls: [.init(id: "functions.ping:0", name: "ping",
                                      arguments: "{}")]),
        ], tools: [Self.pingTool])
        let ids = tok.encode(p, addBOS: false)
        for id in [tok.kimiToolSectionBeginID, tok.kimiToolSectionEndID,
                   tok.kimiToolCallBeginID, tok.kimiToolArgumentBeginID,
                   tok.kimiToolCallEndID] {
            #expect(ids.contains(id ?? -1))
        }
        #expect(tok.decode(ids, skipSpecialTokens: false) == p)
    }

    @Test("Tool chat encodes the hand renderer's output")
    func encodeToolChat() throws {
        let ids = try tok.encodeToolChat(
            messages: [Message(role: .user, content: "Weather?")],
            tools: [Self.pingTool])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.hasPrefix("<|im_system|>tool_declare<|im_middle|>"))
        #expect(text.hasSuffix("<|im_end|><|im_assistant|>assistant<|im_middle|>"))
    }

    @Test("Messages without content are rejected")
    func missingContent() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([Message(role: .user, content: nil)])
        }
        #expect(throws: GFTokenizerError.self) {
            _ = try render([Message(role: .assistant, content: nil)])
        }
    }

    @Test("Tool result without a tool_call_id is rejected")
    func missingToolCallID() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .tool, content: "result"),
            ])
        }
    }

    @Test("Tool-result KV continuation stays unsupported for kimi")
    func toolResultContinuationUnsupported() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.encodeToolResultContinuation(
                cachedMessages: [Message(role: .user, content: "Hi")],
                assistant: Message(role: .assistant, content: nil, toolCalls: [
                    .init(id: "functions.lookup:0", name: "lookup",
                          arguments: "{}"),
                ]),
                incomingMessages: [Message(role: .user, content: "Hi")],
                tools: [])
        }
    }
}
