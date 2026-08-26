import Foundation

/// Parses Kimi tool calls: the text between `<|tool_call_begin|>` and
/// `<|tool_call_argument_begin|>` is the model-emitted formatted call id
/// (`functions.NAME:INDEX`), and the text from there to `<|tool_call_end|>`
/// is the arguments as one JSON object. The id is preserved verbatim on the
/// parsed call — the chat template re-renders history by that id, so it must
/// round-trip — which is why there is no generated-id parameter here.
public struct KimiToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    /// The function name inside a Kimi formatted call id. A non-matching id
    /// is malformed rather than nil so it cannot fall through to visible
    /// content.
    public static func functionName(inCallID id: String) throws -> String {
        guard id.range(of: "^functions\\.[A-Za-z0-9_-]{1,64}:[0-9]{1,10}$",
                       options: .regularExpression) != nil,
              let colon = id.lastIndex(of: ":") else {
            throw ToolCallParserError.malformed
        }
        let start = id.index(id.startIndex, offsetBy: "functions.".count)
        return String(id[start..<colon])
    }

    public func parse(id rawID: String,
                      body: String,
                      allowedTools: Set<String>) throws -> ParsedToolCall {
        guard body.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        let id = rawID.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = try Self.functionName(inCallID: id)
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let arguments = try? JSONDecoder().decode(JSONValue.self,
                                                        from: Data(trimmed.utf8)),
              case .object = arguments else {
            throw ToolCallParserError.malformed
        }
        return ParsedToolCall(id: id,
                              name: name,
                              arguments: arguments,
                              argumentsJSON: try arguments.encoded())
    }
}
