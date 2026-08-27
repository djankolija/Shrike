import Foundation
import Testing
@testable import NVMAI

/// Harmony (gpt-oss) dialect coverage against a synthetic tokenizer fixture:
/// a byte-level BPE vocab plus the Harmony special tokens at their real
/// o200k_harmony IDs, with the model's `chat_template.jinja` alongside.
///
/// Golden strings are byte-exact jinja2 renders of that template with
/// `strftime_now` pinned to 2026-08-26, `tojson` compact with sorted keys,
/// and tool properties alphabetized (the renderer iterates sorted keys
/// because JSON object order does not survive decoding).
@Suite("Harmony template")
struct HarmonyTemplateTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: Self.fixtureFolder())
    }

    static func fixtureFolder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "HarmonyTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }

    private typealias Message = GFTokenizer.Message
    private static let goldenDate = "2026-08-26"

    private func render(_ messages: [Message],
                        tools: [GFTokenizer.FunctionDefinition] = []) throws -> String {
        try tok.harmonyChatTemplate(messages, tools: tools,
                                    currentDate: Self.goldenDate)
    }

    private func systemBlock(withTools: Bool = false) -> String {
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
            + "Knowledge cutoff: 2024-06\n"
            + "Current date: 2026-08-26\n\n"
            + "Reasoning: medium\n\n"
            + "# Valid channels: analysis, commentary, final. Channel must be included for every message."
            + (withTools
                ? "\nCalls to these tools must go to the commentary channel: 'functions'."
                : "")
            + "<|end|>"
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
                "detailed": .object(["type": .string("boolean")]),
                "tags": .object(["type": .string("array"),
                                 "items": .object(["type": .string("string")])]),
                "unit": .object(["type": .string("string"),
                                 "enum": .array([.string("c"), .string("f")]),
                                 "default": .string("c")]),
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

    private static let weatherNamespace = "## functions\n\nnamespace functions {\n\n"
        + "// Look up weather\ntype get_weather = (_: {\n"
        + "// City name\ncity: string,\ndays?: number,\ndetailed?: boolean,\n"
        + "tags?: string[],\nunit?: \"c\" | \"f\", // default: c\n}) => any;\n\n"

    @Test("Dialect and special-token IDs match the o200k_harmony contract")
    func specialTokenIDs() {
        #expect(tok.dialect == .harmony)
        #expect(tok.bosID == 199998)
        #expect(tok.eosID == 200002)
        #expect(tok.padID == 199999)
        #expect(tok.endOfTurnID == 200007)
        #expect(tok.channelStartID == 200005)
        #expect(tok.channelEndID == 200007)
        #expect(tok.harmonyStartID == 200006)
        #expect(tok.harmonyMessageID == 200008)
        #expect(tok.harmonyConstrainID == 200003)
        #expect(tok.harmonyCallID == 200012)
        #expect(tok.harmonyReturnID == 200002)
        #expect(tok.toolCallStartID == nil)
        #expect(tok.toolCallEndID == nil)
        #expect(tok.toolResponseID == nil)
        #expect(tok.toolResponseEndID == nil)
        #expect(tok.thinkStartID == nil)
        #expect(tok.thinkEndID == nil)
    }

    @Test("Stop tokens are return and call only; end is not a stop")
    func stopTokens() {
        #expect(tok.stopTokenIDs == [200002, 200012])
        #expect(!tok.stopTokenIDs.contains(tok.endOfTurnID))
    }

    @Test("Logits vocab is the gpt-oss row count")
    func vocabSize() {
        #expect(tok.vocabSize == 201_088)
    }

    @Test("Single user turn renders the exact Harmony string")
    func singleUserTurn() throws {
        let p = try render([Message(role: .user, content: "Hi")])
        #expect(p == systemBlock()
            + "<|start|>user<|message|>Hi<|end|><|start|>assistant")
    }

    @Test("Public template uses today's date and the generation suffix")
    func publicTemplate() throws {
        let p = try tok.applyChatTemplate([Message(role: .user, content: "Hi")])
        #expect(p.range(of: "Current date: \\d{4}-\\d{2}-\\d{2}\n",
                        options: .regularExpression) != nil)
        #expect(p.hasSuffix("<|end|><|start|>assistant"))
    }

    @Test("Multi-turn maps system to developer and drops previous thinking")
    func multiTurn() throws {
        let p = try render([
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "A"),
            Message(role: .assistant, content: "B", thinking: "hidden"),
            Message(role: .user, content: "C"),
        ])
        #expect(p == systemBlock()
            + "<|start|>developer<|message|># Instructions\n\nBe terse.<|end|>"
            + "<|start|>user<|message|>A<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>B<|end|>"
            + "<|start|>user<|message|>C<|end|><|start|>assistant")
    }

    @Test("Message content is not trimmed")
    func untrimmedContent() throws {
        let p = try render([Message(role: .user, content: "  spaced  \n")])
        #expect(p == systemBlock()
            + "<|start|>user<|message|>  spaced  \n<|end|><|start|>assistant")
    }

    @Test("Tools render as a TypeScript namespace in the developer message")
    func toolDefinitions() throws {
        let p = try render([
            Message(role: .system, content: "Be helpful."),
            Message(role: .user, content: "Weather in Paris?"),
        ], tools: [Self.weatherTool, Self.pingTool])
        #expect(p == systemBlock(withTools: true)
            + "<|start|>developer<|message|># Instructions\n\nBe helpful.\n\n# Tools\n\n"
            + Self.weatherNamespace
            + "// Health check\ntype ping = () => any;\n\n"
            + "} // namespace functions<|end|>"
            + "<|start|>user<|message|>Weather in Paris?<|end|><|start|>assistant")
    }

    @Test("Tools without a system message still open the developer block")
    func toolsWithoutSystem() throws {
        let p = try render([Message(role: .user, content: "Weather?")],
                           tools: [Self.pingTool])
        #expect(p == systemBlock(withTools: true)
            + "<|start|>developer<|message|>\n\n# Tools\n\n"
            + "## functions\n\nnamespace functions {\n\n"
            + "// Health check\ntype ping = () => any;\n\n"
            + "} // namespace functions<|end|>"
            + "<|start|>user<|message|>Weather?<|end|><|start|>assistant")
    }

    @Test("Tool loop replays analysis, renders the call, and JSON-quotes the result")
    func toolLoop() throws {
        let p = try render([
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: nil,
                    toolCalls: [.init(id: "call_1", name: "get_weather",
                                      arguments: #"{"city":"Paris"}"#)],
                    thinking: "Need the live number."),
            Message(role: .tool, content: "22C, clear"),
        ], tools: [Self.weatherTool])
        #expect(p == systemBlock(withTools: true)
            + "<|start|>developer<|message|>\n\n# Tools\n\n"
            + Self.weatherNamespace
            + "} // namespace functions<|end|>"
            + "<|start|>user<|message|>Weather in Paris?<|end|>"
            + "<|start|>assistant<|channel|>analysis<|message|>Need the live number.<|end|>"
            + "<|start|>assistant to=functions.get_weather<|channel|>commentary json"
            + "<|message|>{\"city\":\"Paris\"}<|call|>"
            + "<|start|>functions.get_weather to=assistant<|channel|>commentary"
            + "<|message|>\"22C, clear\"<|end|><|start|>assistant")
    }

    @Test("TypeScript type branches match the template byte-for-byte")
    func kitchenSinkTypes() throws {
        let sink = GFTokenizer.FunctionDefinition(
            name: "sink",
            description: "Exercises the TS type branches",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "box": .object([
                        "type": .string("object"),
                        "properties": .object([
                            "depth": .object(["type": .string("integer")]),
                            "label": .object(["type": .string("string")]),
                        ]),
                        "required": .array([.string("label")]),
                    ]),
                    "choice": .object([
                        "oneOf": .array([.object(["type": .string("string")]),
                                         .object(["type": .string("number")])]),
                    ]),
                    "count": .object(["type": .string("integer"),
                                      "default": .integer(3)]),
                    "flexible": .object([
                        "type": .array([.string("object"), .string("object")]),
                    ]),
                    "items": .object([
                        "type": .string("array"),
                        "items": .object([
                            "type": .string("object"),
                            "properties": .object([
                                "id": .object(["type": .string("number")]),
                            ]),
                        ]),
                    ]),
                    "maybe": .object(["type": .string("string"),
                                      "nullable": .bool(true)]),
                    "mystery": .object([:]),
                    "nums": .object(["type": .string("array")]),
                ]),
                "required": .array([.string("count")]),
            ]))
        let p = try render([Message(role: .user, content: "Go")], tools: [sink])
        let expected = "type sink = (_: {\n"
            + "box?: {\ndepth?: \n                number, label: \n                string},\n"
            + "choice?: string | \n                number,\n"
            + "count: number, // default: 3,\n"
            + "flexible?: object | object,\n"
            + "items?: {\nid?: \n                number}[],\n"
            + "maybe?: string | null,\n"
            + "mystery?: any,\n"
            + "nums?: any[]\n"
            + "}) => any;\n\n"
        #expect(p.contains(expected))
    }

    @Test("Rendered prompt round-trips through the special-token vocabulary")
    func encodesToSpecialIDs() throws {
        let p = try render([Message(role: .user, content: "Hi")])
        let ids = tok.encode(p, addBOS: false)
        #expect(ids.first == tok.harmonyStartID)
        #expect(ids.contains(tok.harmonyMessageID ?? -1))
        #expect(ids.contains(tok.endOfTurnID))
        #expect(ids.last == tok.encode("assistant", addBOS: false).last)
        #expect(tok.decode(ids, skipSpecialTokens: false) == p)
    }

    @Test("Tool chat encodes the hand renderer's output")
    func encodeToolChat() throws {
        let ids = try tok.encodeToolChat(
            messages: [Message(role: .user, content: "Weather?")],
            tools: [Self.pingTool])
        let text = tok.decode(ids, skipSpecialTokens: false)
        #expect(text.contains("namespace functions"))
        #expect(text.range(of: "Current date: \\d{4}-\\d{2}-\\d{2}\n",
                           options: .regularExpression) != nil)
        #expect(text.hasSuffix("<|end|><|start|>assistant"))
    }

    @Test("System message after the first turn is rejected")
    func misplacedSystemTurn() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .system, content: "Too late"),
            ])
        }
    }

    @Test("Tool result without a preceding tool call is rejected")
    func orphanToolResult() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .tool, content: "result"),
            ])
        }
    }

    @Test("Analysis in both content and thinking on a tool call is rejected")
    func bothAnalysisFields() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .assistant, content: "a",
                        toolCalls: [.init(id: "call_1", name: "ping",
                                          arguments: "{}")],
                        thinking: "b"),
            ])
        }
    }

    @Test("Inline channel markup in assistant fields is rejected")
    func inlineChannelMarkup() {
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .assistant,
                        content: "<|channel|>final<|message|>sneaky"),
            ])
        }
    }

    @Test("More than one tool call per assistant message is rejected")
    func multipleToolCalls() {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "ping", arguments: "{}")
        #expect(throws: GFTokenizerError.self) {
            _ = try render([
                Message(role: .user, content: "Hi"),
                Message(role: .assistant, content: nil, toolCalls: [call, call]),
            ])
        }
    }

    // MARK: - Settled boundary

    private func promptIDs(_ messages: [Message],
                           tools: [GFTokenizer.FunctionDefinition]) throws -> [Int32] {
        tok.encode(try tok.harmonyChatTemplate(messages, tools: tools), addBOS: false)
    }

    private func settledText(_ messages: [Message],
                             tools: [GFTokenizer.FunctionDefinition]) throws -> String {
        let text = try tok.harmonyChatTemplate(messages, tools: tools)
        return String(text.dropLast(GFTokenizer.harmonyGenerationSuffix.count))
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
        #expect(settled.hasSuffix("<|start|>user<|message|>What about Berlin?<|end|>"))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: [])
        #expect(tok.decode(Array(fullIDs.prefix(boundary)), skipSpecialTokens: false)
            == settled)
    }

    @Test("Settled boundary ends a tool loop at the last user turn")
    func settledBoundaryToolLoop() throws {
        let settledMessages: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: nil, toolCalls: [
                .init(id: "call_1", name: "get_weather", arguments: "{\"city\":\"Paris\"}"),
            ], thinking: "Paris first."),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C."),
            Message(role: .user, content: "What about Berlin?"),
        ]
        let liveMessages: [Message] = [
            Message(role: .assistant, content: nil, toolCalls: [
                .init(id: "call_2", name: "get_weather", arguments: "{\"city\":\"Berlin\"}"),
            ], thinking: "Berlin now."),
            Message(role: .tool, content: "{\"temp\":12}", toolCallID: "call_2"),
        ]
        let messages = settledMessages + liveMessages
        let tools = [Self.weatherTool]
        let fullIDs = try promptIDs(messages, tools: tools)
        let settled = try settledText(settledMessages, tools: tools)
        #expect(tok.decode(fullIDs, skipSpecialTokens: false).hasPrefix(settled))
        #expect(settled.hasSuffix("<|start|>user<|message|>What about Berlin?<|end|>"))
        #expect(!settled.contains("Berlin now."))
        let boundary = try tok.settledBoundaryTokenCount(messages: messages, tools: tools)
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
        let boundary = try tok.settledBoundaryTokenCount(messages: completed, tools: tools)
        let liveIDs = try tok.settledLiveRegionTokens(messages: completed, tools: tools)
        let nextIDs = try promptIDs(completed + [nextQuery], tools: tools)
        try #require(nextIDs.count >= boundary + liveIDs.count)
        #expect(Array(nextIDs[boundary ..< boundary + liveIDs.count]) == liveIDs)

        let liveText = tok.decode(liveIDs, skipSpecialTokens: false)
        let settledText = tok.decode(Array(nextIDs.prefix(boundary)), skipSpecialTokens: false)
        #expect(tok.decode(nextIDs, skipSpecialTokens: false)
            .hasPrefix(settledText + liveText))
        return liveText
    }

    @Test("Settled live region drops a text turn's analysis")
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
        #expect(liveText == "<|start|>assistant<|channel|>final<|message|>Sunny.<|end|>")
    }

    @Test("Settled live region keeps a tool call's analysis and drops the answer's")
    func settledLiveRegionToolLoop() throws {
        let completed: [Message] = [
            Message(role: .system, content: "Be terse."),
            Message(role: .user, content: "Weather in Paris?"),
            Message(role: .assistant, content: nil, toolCalls: [
                .init(id: "call_1", name: "get_weather",
                      arguments: "{\"city\":\"Paris\"}"),
            ], thinking: "Paris first."),
            Message(role: .tool, content: "{\"temp\":18}", toolCallID: "call_1"),
            Message(role: .assistant, content: "18C.", thinking: "That is warm."),
        ]
        let liveText = try expectSettledRegionInNextRender(
            completed: completed,
            nextQuery: Message(role: .user, content: "What about Berlin?"),
            tools: [Self.weatherTool])

        // Retention is per message, not by position: the tool-call turn's
        // analysis survives into the settled region, the answer's does not.
        #expect(liveText.contains(
            "<|start|>assistant<|channel|>analysis<|message|>Paris first.<|end|>"))
        #expect(!liveText.contains("That is warm."))
        #expect(liveText.contains("<|channel|>commentary json<|message|>{\"city\":\"Paris\"}"))
        #expect(liveText.hasSuffix(
            "<|start|>assistant<|channel|>final<|message|>18C.<|end|>"))
    }
}
