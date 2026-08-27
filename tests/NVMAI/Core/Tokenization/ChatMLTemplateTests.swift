import Foundation
import Jinja
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

    /// The call measured breaking mid-loop byte-exactness (nested cached=0/743
    /// against flat cached=486/506): a float array, a string array, and an
    /// object whose members are not in the order `tojson` would sort them into.
    private static let nestedArguments = "{\"depths\":[1.5,3.25],"
        + "\"sites\":[\"harbour\",\"quarry\"],"
        + "\"window\":{\"start\":\"2026-09-01\",\"end\":\"2026-09-07\"}}"

    private static let nestedParameterBlock =
        "<parameter=depths>\n[1.5,3.25]\n</parameter>\n"
        + "<parameter=sites>\n[\"harbour\",\"quarry\"]\n</parameter>\n"
        + "<parameter=window>\n{\"start\":\"2026-09-01\",\"end\":\"2026-09-07\"}\n</parameter>\n"

    private static let surveyTool = GFTokenizer.FunctionDefinition(
        name: "plan_survey",
        description: "Plan a survey",
        parameters: .object(["type": .string("object")]))

    @Test("Nested tool arguments render as the bytes the call carried")
    func toolChatRendersNestedArgumentsVerbatim() throws {
        let ids = try tok.encodeToolChat(
            messages: [
                Message(role: .user, content: "Survey harbour and quarry."),
                Message(role: .assistant, content: "", toolCalls: [
                    .init(id: "call_1", name: "plan_survey",
                          arguments: Self.nestedArguments),
                ], thinking: "Both sites."),
                Message(role: .tool, content: "{\"ok\":true}", toolCallID: "call_1"),
            ],
            tools: [Self.surveyTool])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.contains("<tool_call>\n<function=plan_survey>\n"
            + Self.nestedParameterBlock + "</function>\n</tool_call>"))
        #expect(text.contains("[1.5,3.25]"))
        #expect(text.contains("[\"harbour\",\"quarry\"]"))
        #expect(text.contains("{\"start\":\"2026-09-01\",\"end\":\"2026-09-07\"}"))
        #expect(!text.contains("\"end\":\"2026-09-07\",\"start\""))
    }

    /// The same call as the model writes it: parameters in non-sorted order, a
    /// nested object whose members are also non-sorted and spaced, and an array
    /// with interior spacing.
    private static let emittedParameterBlock =
        "<parameter=window>\n{\"start\":\"2026-09-01\",  \"end\":\"2026-09-07\"}\n</parameter>\n"
        + "<parameter=depths>\n[1.5, 3.25]\n</parameter>\n"
        + "<parameter=site>\nharbour\n</parameter>\n"

    @Test("A tool call survives emission, parse and re-render byte for byte")
    func toolCallRoundTripsFromEmissionToRender() throws {
        let call = try QwenToolCallParser().parse(
            "\n<function=plan_survey>\n" + Self.emittedParameterBlock + "</function>\n",
            allowedTools: ["plan_survey"],
            id: "call_1")
        let ids = try tok.encodeToolChat(
            messages: [
                Message(role: .user, content: "Survey harbour."),
                Message(role: .assistant, content: "", toolCalls: [
                    .init(id: call.id, name: call.name, arguments: call.argumentsJSON),
                ], thinking: "The first week."),
                Message(role: .tool, content: "{\"ok\":true}", toolCallID: call.id),
            ],
            tools: [Self.surveyTool])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.contains("<tool_call>\n<function=plan_survey>\n"
            + Self.emittedParameterBlock + "</function>\n</tool_call>"))
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

    /// `ServerInference.encodePrompt`, which routes on the whole request's
    /// list rather than on the messages it is handed.
    private func promptIDs(_ messages: [Message],
                           routedWith full: [Message],
                           tools: [GFTokenizer.FunctionDefinition]) throws -> [Int32] {
        try GFTokenizer.usesToolTemplate(messages: full, tools: tools)
            ? tok.encodeToolChat(messages: messages, tools: tools)
            : tok.encode(tok.applyChatTemplate(messages), addBOS: false)
    }

    private func settledText(_ messages: [Message],
                             routedWith full: [Message],
                             tools: [GFTokenizer.FunctionDefinition]) throws -> String {
        let text = tok.decode(try promptIDs(messages, routedWith: full, tools: tools),
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
        let fullIDs = try promptIDs(messages, routedWith: messages, tools: [])
        let settled = try settledText(messages, routedWith: messages, tools: [])
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
        let fullIDs = try promptIDs(messages, routedWith: messages, tools: tools)
        let settled = try settledText(settledMessages, routedWith: messages, tools: tools)
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|im_start|>user\nWhat about Berlin?<|im_end|>\n"))
        #expect(!settled.contains("Berlin now."))
        #expect(!settled.contains("{\"temp\":12}"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: tools)
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    @Test("Settled boundary follows a settled tool round-trip onto the Jinja path")
    func settledBoundaryFollowsHistoryOntoTheToolPath() throws {
        let settledMessages: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ]),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C."),
            Message(role: .user, content: "What about Berlin?"),
        ]
        let messages = settledMessages + [Message(role: .assistant, content: "Rainy.")]
        #expect(GFTokenizer.usesToolTemplate(messages: messages, tools: []))
        let fullIDs = try promptIDs(messages, routedWith: messages, tools: [])
        let settled = try settledText(settledMessages, routedWith: messages, tools: [])
        #expect(settled.contains("<|im_start|>user\n<tool_response>\n{\"temp\":18}"))
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|im_start|>user\nWhat about Berlin?<|im_end|>\n"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: [])
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    @Test("Settled boundary follows a live-only tool round-trip onto the Jinja path")
    func settledBoundaryFollowsLiveOnlyHistoryOntoTheToolPath() throws {
        let settledMessages: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant,
                    content: "<think>\nParis first.\n</think>\n\nSunny."),
            Message(role: .user, content: "What about Berlin?"),
        ]
        let messages = settledMessages + [
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ]),
            Message(role: .tool, content: "{\"temp\":12}", toolCallID: "call_1"),
        ]
        #expect(!GFTokenizer.usesToolTemplate(messages: settledMessages, tools: []))
        #expect(GFTokenizer.usesToolTemplate(messages: messages, tools: []))
        let fullIDs = try promptIDs(messages, routedWith: messages, tools: [])
        let settled = try settledText(settledMessages, routedWith: messages, tools: [])
        #expect(!settled.contains("Paris first."))
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|im_start|>user\nWhat about Berlin?<|im_end|>\n"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: [])
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    // MARK: - Settled form

    /// The byte-exactness property: the next request's own render must carry
    /// the settled live region verbatim, starting at the settled boundary.
    private func expectSettledRegionInNextRender(
        completed: [Message],
        nextQuery: Message,
        tools: [GFTokenizer.FunctionDefinition]
    ) throws -> String {
        let next = completed + [nextQuery]
        let boundary = try tok.settledBoundaryTokenCount(messages: completed, tools: tools)
        let liveIDs = try tok.settledLiveRegionTokens(messages: completed, tools: tools)
        let nextIDs = try promptIDs(next, routedWith: next, tools: tools)
        try #require(nextIDs.count >= boundary + liveIDs.count)
        #expect(Array(nextIDs[boundary ..< boundary + liveIDs.count]) == liveIDs)

        let liveText = tok.decode(liveIDs, skipSpecialTokens: false)
        let settledText = tok.decode(Array(nextIDs.prefix(boundary)), skipSpecialTokens: false)
        #expect(tok.decode(nextIDs, skipSpecialTokens: false)
            .hasPrefix(settledText + liveText))
        return liveText
    }

    @Test("Settled live region drops a plain turn's reasoning")
    func settledLiveRegionPlainMultiTurn() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "Sunny.", thinking: "Paris is warm."),
        ]
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "What about Berlin?"),
            tools: [])
        #expect(liveText == "<|im_start|>assistant\nSunny.<|im_end|>\n")
    }

    @Test("Settled live region drops a whole tool loop's reasoning")
    func settledLiveRegionToolLoop() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather",
                      arguments: "{\"units\":\"c\",\"city\":\"Paris\",\"days\":3}"),
            ], thinking: "Paris first."),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C.", thinking: "That is warm."),
        ]
        let tools = [Self.weatherTool]
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "What about Berlin?"),
            tools: tools)

        #expect(!liveText.contains("<think>"))
        #expect(!liveText.contains("Paris first."))
        #expect(!liveText.contains("That is warm."))
        #expect(liveText.hasPrefix("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n"))
        #expect(liveText.contains("<parameter=units>\nc\n</parameter>\n"
            + "<parameter=city>\nParis\n</parameter>\n"
            + "<parameter=days>\n3\n</parameter>\n"))
        #expect(liveText.contains("<|im_start|>user\n<tool_response>\n{\"temp\":18}"))
        #expect(liveText.hasSuffix("<|im_start|>assistant\n18C.<|im_end|>\n"))

        // The same turns render with their reasoning while the request is live.
        let liveForm = tok.decode(try promptIDs(completed, routedWith: completed, tools: tools),
                                  skipSpecialTokens: false)
        #expect(liveForm.contains("<think>\nParis first.\n</think>"))
        #expect(liveForm.contains("<think>\nThat is warm.\n</think>"))
    }

    @Test("Settled live region carries nested tool arguments verbatim")
    func settledLiveRegionNestedToolArguments() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Survey harbour and quarry."),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "plan_survey",
                      arguments: Self.nestedArguments),
            ], thinking: "Both sites."),
            Message(role: .tool, content: "{\"ok\":true}", toolCallID: "call_1"),
            Message(role: .assistant, content: "Planned.", thinking: "Done."),
        ]
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "And the estuary?"),
            tools: [Self.surveyTool])
        #expect(!liveText.contains("<think>"))
        #expect(liveText.contains(Self.nestedParameterBlock))
        #expect(!liveText.contains("\"end\":\"2026-09-07\",\"start\""))
    }

    @Test("Settled live region carries a settled tool round-trip's history")
    func settledLiveRegionAfterASettledToolLoop() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ], thinking: "Paris first."),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C."),
            Message(role: .user, content: "What about Berlin?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_2", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ], thinking: "Berlin now."),
            Message(role: .tool, content: "{\"temp\":12}", toolCallID: "call_2"),
            Message(role: .assistant, content: "12C.", thinking: "Cooler."),
        ]
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "And Rome?"),
            tools: [Self.weatherTool])
        #expect(!liveText.contains("Berlin now."))
        #expect(!liveText.contains("Cooler."))
        #expect(!liveText.contains("Paris"))
        #expect(!liveText.contains("{\"temp\":18}"))
        #expect(liveText.contains("{\"temp\":12}"))
    }

    @Test("Settled live region follows tool-shaped history with no tools declared")
    func settledLiveRegionFollowsHistoryOntoTheToolPath() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ]),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C.", thinking: "Warm."),
        ]
        #expect(GFTokenizer.usesToolTemplate(messages: completed, tools: []))
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "What about Berlin?"),
            tools: [])
        #expect(!liveText.contains("Warm."))
        #expect(liveText.contains("<|im_start|>user\n<tool_response>\n{\"temp\":18}"))
    }

    @Test("Settled form rejects a request still awaiting a tool result")
    func settledLiveRegionRequiresACompletedRequest() {
        let awaiting: [Message] = [
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: "", toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ], thinking: "Paris first."),
        ]
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.settledLiveRegionTokens(messages: awaiting,
                                                tools: [Self.weatherTool])
        }
        #expect(throws: GFTokenizerError.self) {
            _ = try tok.settledLiveRegionTokens(
                messages: awaiting + [Message(role: .tool, content: "{\"temp\":18}",
                                              toolCallID: "call_1")],
                tools: [Self.weatherTool])
        }
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

/// The seam the ChatML tools render path reads its `arguments` mapping from.
@Suite("Verbatim tool arguments")
struct VerbatimJinjaArgumentsTests {
    /// Each root member as the template's `args_value | string` branch would
    /// write it, in source order. Every member must reach that branch, so a
    /// value left un-stringified fails here rather than silently taking
    /// `tojson`.
    private func rendered(_ text: String) throws -> [(String, String)] {
        guard case .object(let members) = try JSONValue.verbatimJinjaObject(text) else {
            throw ToolCallParserError.malformed
        }
        return try members.map { key, value in
            guard case .string(let written) = value else {
                throw ToolCallParserError.malformed
            }
            return (key, written)
        }
    }

    @Test("A non-string member is its exact lexeme span, interior spacing kept")
    func sliceIsTheLexemeSpan() throws {
        let members = try rendered(#"{ "a" : [1,  2] , "b" : { "x" : true } }"#)
        #expect(members.map(\.0) == ["a", "b"])
        #expect(members.map(\.1) == ["[1,  2]", #"{ "x" : true }"#])
    }

    @Test("A string member stays the parsed value, without its quotes")
    func stringsAreUnchanged() throws {
        let members = try rendered(#"{"s":"harbour","e":"a\nb\/c","u":"é"}"#)
        #expect(members.map(\.1) == ["harbour", "a\nb/c", "é"])
    }

    @Test("Numbers, booleans and null carry their source lexeme")
    func lexemesAreVerbatim() throws {
        let members = try rendered(#"{"a":3.250,"b":1e2,"c":-0,"d":false,"e":null}"#)
        #expect(members.map(\.1) == ["3.250", "1e2", "-0", "false", "null"])
    }

    @Test("The validating parse still yields typed values, not slices")
    func orderedJinjaObjectIsUnchanged() throws {
        guard case .object(let members) =
                try JSONValue.orderedJinjaObject(#"{"a":[1,  2],"b":3,"c":"x"}"#) else {
            throw ToolCallParserError.malformed
        }
        #expect(members["a"] == .array([.int(1), .int(2)]))
        #expect(members["b"] == .int(3))
        #expect(members["c"] == .string("x"))
    }

    @Test("Acceptance is unchanged from the validating parse")
    func acceptanceMatchesOrderedJinjaObject() {
        let accepted = [#"{}"#, #"{"a":1}"#, #"{"a":[1,{"b":null}]}"#, #"{ "a" : "x" }"#]
        let rejected = [#"[1]"#, #""x""#, #"{"a":1"#, #"{"a":1}x"#, #"{"a":1e400}"#]
        for text in accepted {
            #expect((try? JSONValue.orderedJinjaObject(text)) != nil, "\(text)")
            #expect((try? JSONValue.verbatimJinjaObject(text)) != nil, "\(text)")
        }
        for text in rejected {
            #expect((try? JSONValue.orderedJinjaObject(text)) == nil, "\(text)")
            #expect((try? JSONValue.verbatimJinjaObject(text)) == nil, "\(text)")
        }
    }
}
