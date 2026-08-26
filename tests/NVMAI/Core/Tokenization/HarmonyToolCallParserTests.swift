import Foundation
import Testing
@testable import NVMAI

@Suite("Harmony tool-call parser")
struct HarmonyToolCallParserTests {

    @Test("Recipient parses from either side of the channel marker")
    func recipientOrders() throws {
        #expect(try HarmonyToolCallParser.recipientFunctionName(
            inHeader: "assistant to=functions.get_weather commentary json") == "get_weather")
        #expect(try HarmonyToolCallParser.recipientFunctionName(
            inHeader: " commentary to=functions.get_weather json") == "get_weather")
        #expect(try HarmonyToolCallParser.recipientFunctionName(
            inHeader: " final") == nil)
        #expect(try HarmonyToolCallParser.recipientFunctionName(
            inHeader: "") == nil)
    }

    @Test("Recipient outside the functions namespace is malformed, not nil")
    func foreignNamespace() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try HarmonyToolCallParser.recipientFunctionName(
                inHeader: "commentary to=browser.search json")
        }
    }

    @Test("Invalid function names are malformed")
    func invalidName() {
        #expect(throws: ToolCallParserError.malformed) {
            _ = try HarmonyToolCallParser.recipientFunctionName(
                inHeader: "commentary to=functions.we!rd json")
        }
        #expect(throws: ToolCallParserError.malformed) {
            _ = try HarmonyToolCallParser.recipientFunctionName(
                inHeader: "commentary to=functions. json")
        }
    }

    @Test("Arguments parse as a JSON object with canonical re-encoding")
    func parseArguments() throws {
        let call = try HarmonyToolCallParser().parse(
            name: "get_weather",
            body: " {\"unit\":\"c\",\"city\":\"Paris\"} \n",
            allowedTools: ["get_weather"],
            id: "call_fixed")
        #expect(call == ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris"), "unit": .string("c")]),
            argumentsJSON: #"{"city":"Paris","unit":"c"}"#))
    }

    @Test("Tools outside the allowed set are rejected")
    func unknownTool() {
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try HarmonyToolCallParser().parse(
                name: "get_weather", body: "{}", allowedTools: [], id: "x")
        }
    }

    @Test("Non-object bodies are malformed")
    func nonObjectBody() {
        for body in ["[1, 2]", "\"text\"", "42", "", "{broken"] {
            #expect(throws: ToolCallParserError.malformed) {
                _ = try HarmonyToolCallParser().parse(
                    name: "get_weather", body: body,
                    allowedTools: ["get_weather"], id: "x")
            }
        }
    }

    @Test("Oversized bodies are rejected before parsing")
    func oversizedBody() {
        let body = "{\"a\":\"" + String(repeating: "x",
                                        count: HarmonyToolCallParser.maximumBytes) + "\"}"
        #expect(throws: ToolCallParserError.oversized) {
            _ = try HarmonyToolCallParser().parse(
                name: "get_weather", body: body,
                allowedTools: ["get_weather"], id: "x")
        }
    }
}
