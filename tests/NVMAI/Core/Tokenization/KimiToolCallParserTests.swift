import Foundation
import Testing
@testable import NVMAI

@Suite("Kimi tool-call parser")
struct KimiToolCallParserTests {
    private let allowed: Set<String> = ["get_weather", "ping"]

    @Test("Parses a formatted id and JSON-object arguments")
    func parsesCall() throws {
        let call = try KimiToolCallParser().parse(
            id: "functions.get_weather:0",
            body: "{\"city\": \"Paris\", \"days\": 2}",
            allowedTools: allowed)
        #expect(call.id == "functions.get_weather:0")
        #expect(call.name == "get_weather")
        #expect(call.arguments == .object(["city": .string("Paris"),
                                           "days": .integer(2)]))
        #expect(call.argumentsJSON == #"{"city":"Paris","days":2}"#)
    }

    @Test("Whitespace around the id and body is trimmed")
    func trimsWhitespace() throws {
        let call = try KimiToolCallParser().parse(
            id: " functions.ping:12 ",
            body: "\n{}\n",
            allowedTools: allowed)
        #expect(call.id == "functions.ping:12")
        #expect(call.name == "ping")
        #expect(call.arguments == .object([:]))
    }

    @Test("Ids outside the functions namespace are malformed")
    func rejectsForeignNamespace() {
        for id in ["get_weather:0", "tools.get_weather:0",
                   "functions.get_weather", "functions.:0",
                   "functions.get weather:0", "functions.a:b:0",
                   "functions.get_weather:0extra"] {
            #expect(throws: ToolCallParserError.malformed) {
                _ = try KimiToolCallParser().parse(id: id, body: "{}",
                                                   allowedTools: allowed)
            }
        }
    }

    @Test("An unlisted tool is rejected by name")
    func unknownTool() {
        #expect(throws: ToolCallParserError.unknownTool("lookup")) {
            _ = try KimiToolCallParser().parse(id: "functions.lookup:0",
                                               body: "{}",
                                               allowedTools: allowed)
        }
    }

    @Test("Non-object argument JSON is malformed")
    func rejectsNonObjectArguments() {
        for body in ["not json", "[1,2]", "\"{}\"", "42", ""] {
            #expect(throws: ToolCallParserError.malformed) {
                _ = try KimiToolCallParser().parse(id: "functions.ping:0",
                                                   body: body,
                                                   allowedTools: allowed)
            }
        }
    }

    @Test("An oversized body is rejected before parsing")
    func oversizedBody() {
        let body = "{\"k\":\"" + String(repeating: "x", count: 256 * 1024) + "\"}"
        #expect(throws: ToolCallParserError.oversized) {
            _ = try KimiToolCallParser().parse(id: "functions.ping:0",
                                               body: body,
                                               allowedTools: allowed)
        }
    }
}
