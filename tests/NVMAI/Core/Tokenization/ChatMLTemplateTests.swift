import Foundation
import Testing
@testable import NVMAI

/// ChatML (Qwen) dialect coverage against a synthetic tokenizer fixture: a
/// byte-level BPE vocab plus the nine ChatML special tokens at their real
/// Qwen3.6 IDs, with the model's `chat_template.jinja` alongside.
@Suite("ChatML template")
struct ChatMLTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private typealias Message = GFTokenizer.Message

    @Test("Special-token IDs match the Qwen3.6 contract")
    func specialTokenIDs() {
        #expect(tok.endOfTurnID == 248046)
        #expect(tok.eosID == 248044)
        #expect(tok.toolCallStartID == 248058)
        #expect(tok.toolCallEndID == 248059)
        #expect(tok.toolResponseID == 248066)
        #expect(tok.toolResponseEndID == 248067)
        #expect(tok.thinkStartID == 248068)
        #expect(tok.thinkEndID == 248069)
    }

    @Test("Stop tokens are im_end and endoftext only")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [tok.endOfTurnID, tok.eosID])
        #expect(tok.stopTokenIDs.count == 2)
    }

    @Test("Logits vocab is the model's padded row count")
    func vocabSize() {
        #expect(tok.vocabSize == 248_320)
    }

    @Test("Encode never prepends a BOS")
    func noBOS() {
        let with = tok.encode("hi", addBOS: true)
        let without = tok.encode("hi", addBOS: false)
        #expect(with == without)
    }

    @Test("Single user turn renders the exact ChatML string")
    func singleUserTurn() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(p == "<|im_start|>user\nHi<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Thinking mode leaves the generation think block open")
    func thinkingModeGenerationPrompt() async throws {
        let thinking = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .on)
        let prompt = try thinking.applyChatTemplate([
            Message(role: .user, content: "Hi"),
        ])
        #expect(thinking.thinkingMode == .on)
        #expect(prompt.hasSuffix("<|im_start|>assistant\n<think>\n"))
        #expect(!prompt.hasSuffix("<think>\n\n</think>\n\n"))
    }

    @Test("Environment compatibility resolves only the documented modes")
    func thinkingModeEnvironmentCompatibility() {
        #expect(ModelThinkingMode.resolved(environment: [:]) == .off)
        #expect(ModelThinkingMode.resolved(
            environment: ["NVMAI_THINKING_MODE": "on"]) == .on)
        #expect(ModelThinkingMode.resolved(
            environment: ["NVMAI_THINKING_MODE": "adaptive"]) == .adaptive)
        #expect(ModelThinkingMode.resolved(
            environment: ["NVMAI_THINKING_MODE": "medium"]) == .off)
    }

    @Test("Adaptive mode injects nothing after the role header")
    func adaptiveModeGenerationPrompt() async throws {
        let adaptive = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .adaptive)
        let prompt = try adaptive.applyChatTemplate([
            Message(role: .user, content: "Hi"),
        ])
        #expect(adaptive.thinkingMode == .adaptive)
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
        #expect(!prompt.hasSuffix("<think>\n"))
        #expect(!prompt.hasSuffix("</think>\n\n"))
    }

    @Test("Adaptive tool chat appends the bare header after the jinja render")
    func adaptiveToolChat() async throws {
        let adaptive = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .adaptive)
        let ids = try adaptive.encodeToolChat(
            messages: [Message(role: .user, content: "Weather?")],
            tools: [GFTokenizer.FunctionDefinition(
                name: "get_weather",
                description: "Look up weather",
                parameters: .object(["type": .string("object")]))])
        let text = adaptive.decode(ids, skipSpecialTokens: false)
        #expect(text.hasSuffix("<|im_start|>assistant\n"))
        #expect(!text.hasSuffix("<think>\n"))
    }

    @Test("Multi-turn renders roles verbatim with assistant unrenamed")
    func multiTurn() throws {
        let p = try tok.applyChatTemplate([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B"),
            Message(role: .user, content: "C"),
        ])
        #expect(p == "<|im_start|>system\nBe terse.<|im_end|>\n"
            + "<|im_start|>user\nA<|im_end|>\n"
            + "<|im_start|>assistant\nB<|im_end|>\n"
            + "<|im_start|>user\nC<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Message content is trimmed before rendering")
    func contentTrimming() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "  Hi \n")])
        #expect(p.contains("<|im_start|>user\nHi<|im_end|>\n"))
    }

    @Test("System message after a user turn is rejected")
    func misplacedSystemTurn() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.applyChatTemplate([
                Message(role: .user, content: "Hi"),
                Message(role: .system, content: "Too late"),
            ])
        }
    }

    @Test("Prompt encodes turn boundaries to the special IDs")
    func encodesToSpecialIDs() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        let ids = tok.encode(p, addBOS: false)
        #expect(ids.first == 248045, "expected <|im_start|> first, got \(String(describing: ids.first))")
        #expect(ids.contains(tok.endOfTurnID))
        #expect(ids.contains(tok.thinkStartID ?? -1))
        #expect(ids.contains(tok.thinkEndID ?? -1))
        #expect(tok.decode(ids, skipSpecialTokens: false) == p)
    }

    @Test("Text continuation bridges from im_end into the next user turn")
    func textContinuation() throws {
        let ids = tok.encodeTextContinuation(userContent: " Next \n")
        #expect(ids.first == tok.endOfTurnID)
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text == "<|im_end|>\n<|im_start|>user\nNext<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test("Tool-result KV continuation is unsupported for chatml")
    func toolResultContinuationUnsupported() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.encodeToolResultContinuation(
                cachedMessages: [Message(role: .user, content: "Hi")],
                assistant: Message(role: .assistant, content: nil, toolCalls: [
                    .init(id: "call_1", name: "lookup", arguments: "{}"),
                ]),
                incomingMessages: [Message(role: .user, content: "Hi")],
                tools: [])
        }
    }

    @Test("Tool chat renders the bundled Jinja template with thinking disabled")
    func toolChatRendersJinja() throws {
        let ids = try tok.encodeToolChat(
            messages: [
                Message(role: .system, content: "Be helpful."),
                Message(role: .user, content: "Weather in Paris?"),
            ],
            tools: [
                .init(name: "get_weather",
                      description: "Look up weather",
                      parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")]),
                        ]),
                      ])),
            ])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.hasPrefix("<|im_start|>system\n# Tools"))
        #expect(text.contains("get_weather"))
        #expect(text.contains("Be helpful."))
        #expect(text.contains("<|im_start|>user\nWeather in Paris?<|im_end|>\n"))
        let suffix = String(text.suffix(80))
        #expect(text.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
                "expected enable_thinking=false generation prompt, got suffix: \(suffix)")
    }

    @Test("Tool chat renders assistant reasoning alongside tool calls")
    func toolChatRendersReasoningWithToolCalls() throws {
        let ids = try tok.encodeToolChat(
            messages: [
                Message(role: .user, content: "Weather in Paris?"),
                Message(role: .assistant, content: "", toolCalls: [
                    .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
                ], thinking: "Need to look up the weather in Paris."),
            ],
            tools: [
                .init(name: "get_weather",
                      description: "Look up weather",
                      parameters: .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")]),
                        ]),
                      ])),
            ])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.contains("<think>\nNeed to look up the weather in Paris.\n</think>"))
        #expect(text.contains("<tool_call>\n<function=get_weather>"))
    }

    @Test("Tool chat uses the same explicit thinking mode as text chat")
    func thinkingToolChatRendersJinja() async throws {
        let thinking = try await GFTokenizer.load(
            from: Self.fixtureFolder(), thinkingMode: .on)
        let ids = try thinking.encodeToolChat(
            messages: [Message(role: .user, content: "Weather?")],
            tools: [
                .init(name: "weather", description: "Look up weather",
                      parameters: .object(["type": .string("object")])),
            ])
        let text = thinking.decode(ids, skipSpecialTokens: false)
        #expect(text.hasSuffix("<|im_start|>assistant\n<think>\n"))
    }

    // MARK: - Settled boundary

    private static let weatherTool = GFTokenizer.FunctionDefinition(
        name: "get_weather",
        description: "Look up weather",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["city": .object(["type": .string("string")])]),
        ]))

    private func promptIDs(_ messages: [Message],
                           tools: [GFTokenizer.FunctionDefinition]) throws -> [Int32] {
        try tools.isEmpty
            ? tok.encode(tok.applyChatTemplate(messages), addBOS: false)
            : tok.encodeToolChat(messages: messages, tools: tools)
    }

    private func settledText(_ messages: [Message],
                             tools: [GFTokenizer.FunctionDefinition]) throws -> String {
        let text = tok.decode(try promptIDs(messages, tools: tools),
                              skipSpecialTokens: false)
        return String(text.dropLast(tok.generationSuffix.count))
    }

    @Test("Settled boundary ends the plain multi-turn render at the last query")
    func settledBoundaryPlainMultiTurn() throws {
        let messages: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny."),
            Message(role: .user, content: "What about Berlin?"),
        ]
        let fullIDs = try promptIDs(messages, tools: [])
        let settled = try settledText(messages, tools: [])
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|im_start|>user\nWhat about Berlin?<|im_end|>\n"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: [])
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    @Test("Settled boundary skips tool_response user turns on the tools path")
    func settledBoundarySkipsToolResponseTurns() throws {
        let settledMessages: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ], thinking: "Paris first."),
            Message(role: .user, content: "<tool_response>\n{\"temp\":18}\n</tool_response>"),
            Message(role: .assistant, content: "18C."),
            Message(role: .user, content: "What about Berlin?"),
        ]
        let liveMessages: [Message] = [
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_2", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ], thinking: "Berlin now."),
            Message(role: .user, content: "<tool_response>\n{\"temp\":12}\n</tool_response>"),
        ]
        let messages = settledMessages + liveMessages
        let tools = [Self.weatherTool]
        let fullIDs = try promptIDs(messages, tools: tools)
        let settled = try settledText(settledMessages, tools: tools)
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|im_start|>user\nWhat about Berlin?<|im_end|>\n"))
        #expect(!settled.contains("Berlin now."))
        #expect(!settled.contains("{\"temp\":12}"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: tools)
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    @Test("Settled boundary rejects a list whose only user turns are tool responses")
    func settledBoundaryWithoutAQuery() {
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.settledBoundaryTokenCount(
                messages: [
                    Message(role: .system, content: "Be terse."),
                    Message(role: .user,
                            content: "<tool_response>\n{}\n</tool_response>"),
                ],
                tools: [])
        }
    }
}
