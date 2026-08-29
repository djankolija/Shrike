import Foundation
import Testing
@testable import Shrike

/// StructuredAssistantDecoder in Kimi mode: the tool-call section state
/// machine over the fixture tokenizer. Unlike Harmony, the five section
/// markers are literal ByteLevel barriers, so their text arrives in the
/// delta and the generation loop's stop tokens (`<|im_end|>` / `[EOS]`)
/// never reach the decoder — no boundary replay is involved.
@Suite("Kimi decoder")
struct KimiDecoderTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: KimiTemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok, allowedTools: allowedTools)
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

    private static let weatherCall = "<|tool_calls_section_begin|>"
        + "<|tool_call_begin|>functions.get_weather:0"
        + "<|tool_call_argument_begin|>{\"city\":\"Paris\"}<|tool_call_end|>"
        + "<|tool_calls_section_end|>"

    private static let parsedWeatherCall = ParsedToolCall(
        id: "functions.get_weather:0",
        name: "get_weather",
        arguments: .object(["city": .string("Paris")]),
        argumentsJSON: #"{"city":"Paris"}"#)

    @Test("Plain content streams as visible text")
    func plainContent() throws {
        let d = decoder()
        let events = try feed("Hello there", into: d)
        #expect(visibleText(events) == "Hello there")
        try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Content then a tool-call section emits both")
    func contentThenCall() throws {
        let d = decoder()
        let events = try feed("Checking.\n" + Self.weatherCall, into: d)
        #expect(visibleText(events) == "Checking.\n")
        #expect(events.last == .toolCall(Self.parsedWeatherCall))
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Two calls in one section both parse, in order")
    func twoCalls() throws {
        let d = decoder(allowedTools: ["get_weather", "ping"])
        let events = try feed(
            "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|>functions.get_weather:0"
                + "<|tool_call_argument_begin|>{\"city\":\"Paris\"}<|tool_call_end|>"
                + "<|tool_call_begin|>functions.ping:1"
                + "<|tool_call_argument_begin|>{}<|tool_call_end|>"
                + "<|tool_calls_section_end|>",
            into: d)
        #expect(events == [
            .toolCall(Self.parsedWeatherCall),
            .toolCall(ParsedToolCall(id: "functions.ping:1", name: "ping",
                                     arguments: .object([:]),
                                     argumentsJSON: "{}")),
        ])
        try d.finish()
    }

    @Test("Whitespace between structural markers is tolerated")
    func whitespaceBetweenMarkers() throws {
        let d = decoder(allowedTools: ["get_weather", "ping"])
        let events = try feed(
            "<|tool_calls_section_begin|>\n"
                + "<|tool_call_begin|>functions.ping:0"
                + "<|tool_call_argument_begin|>{}<|tool_call_end|>\n"
                + "<|tool_calls_section_end|>",
            into: d)
        #expect(events.count == 1)
        #expect(visibleText(events).isEmpty)
        try d.finish()
    }

    @Test("The call id is trimmed before parsing")
    func paddedCallID() throws {
        let d = decoder()
        let events = try feed(
            "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|> functions.get_weather:0 "
                + "<|tool_call_argument_begin|>{\"city\":\"Paris\"}<|tool_call_end|>"
                + "<|tool_calls_section_end|>",
            into: d)
        #expect(events == [.toolCall(Self.parsedWeatherCall)])
        try d.finish()
    }

    @Test("Unknown tool inside a call fails closed")
    func unknownTool() {
        let d = decoder(allowedTools: [])
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try feed(Self.weatherCall, into: d)
        }
    }

    @Test("Non-object argument JSON fails closed")
    func malformedArguments() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed(
                "<|tool_calls_section_begin|>"
                    + "<|tool_call_begin|>functions.get_weather:0"
                    + "<|tool_call_argument_begin|>not json<|tool_call_end|>",
                into: d)
        }
    }

    @Test("A call id outside the functions namespace fails closed")
    func malformedCallID() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed(
                "<|tool_calls_section_begin|>"
                    + "<|tool_call_begin|>get_weather"
                    + "<|tool_call_argument_begin|>{}<|tool_call_end|>",
                into: d)
        }
    }

    @Test("Non-whitespace text between markers fails closed")
    func textBetweenMarkers() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<|tool_calls_section_begin|>oops<|tool_call_begin|>",
                         into: d)
        }
    }

    @Test("Content after the section closes fails closed")
    func contentAfterSection() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed(Self.weatherCall + "trailing", into: d)
        }
    }

    @Test("A second section begin fails closed")
    func doubleSectionBegin() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("<|tool_calls_section_begin|><|tool_calls_section_begin|>",
                         into: d)
        }
    }

    @Test("A call begin outside a section fails closed")
    func callBeginOutsideSection() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try feed("text<|tool_call_begin|>functions.get_weather:0", into: d)
        }
    }

    @Test("Finish inside buffered arguments is malformed")
    func finishMidArguments() throws {
        let d = decoder()
        _ = try feed(
            "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|>functions.get_weather:0"
                + "<|tool_call_argument_begin|>{\"city\":",
            into: d)
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Finish inside an unclosed section is malformed")
    func finishMidSection() throws {
        let d = decoder()
        _ = try feed("<|tool_calls_section_begin|>", into: d)
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Finish on an untouched decoder passes")
    func finishEmptyStream() throws {
        try decoder().finish()
    }

    @Test("Detokenizer tail streams as content outside a section")
    func tailAsContent() throws {
        let d = decoder()
        _ = try feed("partial", into: d)
        #expect(try d.consumeTail("tail") == [.content("tail")])
        try d.finish()
    }

    @Test("Detokenizer tail inside buffered arguments joins the body")
    func tailJoinsArguments() throws {
        let d = decoder()
        _ = try feed(
            "<|tool_calls_section_begin|>"
                + "<|tool_call_begin|>functions.get_weather:0"
                + "<|tool_call_argument_begin|>{\"city\":",
            into: d)
        #expect(try d.consumeTail("\"Paris\"}") == [])
        let events = try d.consume(
            tokenID: try #require(tok.kimiToolCallEndID),
            delta: GFTokenizer.kimiToolCallEndMark)
        #expect(events == [.toolCall(Self.parsedWeatherCall)])
        _ = try d.consume(
            tokenID: try #require(tok.kimiToolSectionEndID),
            delta: GFTokenizer.kimiToolSectionEndMark)
        try d.finish()
    }
}
