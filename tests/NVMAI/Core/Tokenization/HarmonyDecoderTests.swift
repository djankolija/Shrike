import Foundation
import Testing
@testable import NVMAI

/// StructuredAssistantDecoder in Harmony mode: the channel-header state
/// machine over the fixture tokenizer's special IDs. Streams replicate what
/// generation sees after the `<|start|>assistant` prompt suffix — including
/// the `<|call|>` / `<|return|>` stop tokens, which the server feeds to the
/// decoder after the generation loop breaks on them.
@Suite("Harmony decoder")
struct HarmonyDecoderTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: HarmonyTemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
                                   idGenerator: { "call_fixed" })
    }

    private func feed(_ text: String,
                      into decoder: StructuredAssistantDecoder) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var detok = GFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        return events
    }

    private func visibleText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .content(let delta) = event { result += delta }
        }
    }

    private func thinkingText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .thinking(let delta) = event { result += delta }
        }
    }

    @Test("Final channel streams as content and return ends the turn")
    func finalChannel() throws {
        let d = decoder()
        let events = try feed("<|channel|>final<|message|>Hello there<|return|>", into: d)
        #expect(visibleText(events) == "Hello there")
        #expect(thinkingText(events).isEmpty)
        try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("The thought channel is open only inside an analysis message")
    func thoughtChannelTracksTheAnalysisBlock() throws {
        let d = decoder()
        #expect(d.thoughtChannelClosed)
        _ = try feed("<|channel|>analysis<|message|>still thinking", into: d)
        #expect(!d.thoughtChannelClosed)
        _ = try feed("<|end|><|start|>assistant<|channel|>final<|message|>Answer", into: d)
        #expect(d.thoughtChannelClosed)
    }

    @Test("Analysis streams as thinking, then a new block carries the answer")
    func analysisThenFinal() throws {
        let d = decoder()
        let events = try feed(
            "<|channel|>analysis<|message|>let me think<|end|>"
                + "<|start|>assistant<|channel|>final<|message|>Answer<|return|>",
            into: d)
        #expect(thinkingText(events) == "let me think")
        #expect(visibleText(events) == "Answer")
        try d.finish()
    }

    @Test("Commentary without a recipient is a visible preamble")
    func commentaryPreamble() throws {
        let d = decoder()
        let events = try feed("<|channel|>commentary<|message|>Checking now.", into: d)
        #expect(visibleText(events) == "Checking now.")
        try d.finish()
    }

    @Test("Tool call with the recipient after the channel marker")
    func recipientAfterChannel() throws {
        let d = decoder()
        let events = try feed(
            "<|channel|>commentary to=functions.get_weather json"
                + "<|message|>{\"city\":\"Paris\"}<|call|>",
            into: d)
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Tool call with the recipient before the channel marker")
    func recipientBeforeChannel() throws {
        let d = decoder()
        let events = try feed(
            " to=functions.get_weather<|channel|>commentary json<|message|>{}<|call|>",
            into: d)
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object([:]),
            argumentsJSON: "{}"))])
        try d.finish()
    }

    @Test("Constrain marker in the header is transparent")
    func constrainInHeader() throws {
        let d = decoder()
        let events = try feed(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json"
                + "<|message|>{}<|call|>",
            into: d)
        #expect(events.count == 1)
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("A functions recipient wins over the channel name")
    func recipientWinsOverChannel() throws {
        let d = decoder()
        let events = try feed(
            "<|channel|>analysis to=functions.get_weather json<|message|>{}<|call|>",
            into: d)
        #expect(d.hasToolCalls)
        #expect(thinkingText(events).isEmpty)
        try d.finish()
    }

    @Test("Unknown channel name fails closed")
    func unknownChannel() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<|channel|>weird<|message|>text", into: d)
        }
    }

    @Test("Unknown tool inside a call fails closed")
    func unknownTool() {
        let d = decoder(allowedTools: [])
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try feed(
                "<|channel|>commentary to=functions.get_weather json"
                    + "<|message|>{}<|call|>",
                into: d)
        }
    }

    @Test("Non-object argument JSON fails closed")
    func malformedArguments() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed(
                "<|channel|>commentary to=functions.get_weather json"
                    + "<|message|>not json<|call|>",
                into: d)
        }
    }

    @Test("Return from the analysis channel fails closed")
    func returnFromAnalysis() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<|channel|>analysis<|message|>hm<|return|>", into: d)
        }
    }

    @Test("Start marker inside content fails closed")
    func startInsideContent() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<|channel|>final<|message|>text<|start|>more", into: d)
        }
    }

    @Test("Content after the ended state fails closed")
    func contentAfterEnd() throws {
        let d = decoder()
        _ = try feed("<|channel|>final<|message|>done<|return|>", into: d)
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("trailing", into: d)
        }
    }

    @Test("Finish inside a header is malformed")
    func finishMidHeader() throws {
        let d = decoder()
        _ = try feed("<|channel|>final", into: d)
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Finish inside buffered tool arguments is malformed")
    func finishMidArguments() throws {
        let d = decoder()
        _ = try feed(
            "<|channel|>commentary to=functions.get_weather json<|message|>{\"a\":",
            into: d)
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Finish on an untouched decoder passes")
    func finishEmptyStream() throws {
        try decoder().finish()
    }

    @Test("Detokenizer tail follows the active channel")
    func tailRespectsChannel() throws {
        let d = decoder()
        _ = try feed("<|channel|>final<|message|>", into: d)
        #expect(try d.consumeTail("tail") == [.content("tail")])

        let thinking = decoder()
        _ = try feed("<|channel|>analysis<|message|>", into: thinking)
        #expect(try thinking.consumeTail("hidden") == [.thinking("hidden")])
    }

    @Test("Detokenizer tail inside buffered arguments joins the body")
    func tailJoinsArguments() throws {
        let d = decoder()
        _ = try feed(
            "<|channel|>commentary to=functions.get_weather json<|message|>{\"city\":",
            into: d)
        #expect(try d.consumeTail("\"Paris\"}") == [])
        let events = try d.consume(tokenID: try #require(tok.harmonyCallID), delta: "")
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        try d.finish()
    }
}
