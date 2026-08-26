import Foundation

/// Parses Harmony tool calls: the header names the recipient
/// (`to=functions.NAME`, on either side of `<|channel|>` — the decoder hands
/// over the concatenated header text), and the body between `<|message|>` and
/// `<|call|>` is the arguments as one JSON object.
public struct HarmonyToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    /// The function name a Harmony header addresses, or nil for a header with
    /// no recipient. A recipient outside the `functions.` namespace or with an
    /// invalid name is malformed rather than nil so it cannot fall through to
    /// visible content.
    public static func recipientFunctionName(inHeader header: String) throws -> String? {
        guard let range = header.range(of: "to=[^ \n\t<]+",
                                       options: .regularExpression) else {
            return nil
        }
        let recipient = header[range].dropFirst("to=".count)
        guard recipient.hasPrefix("functions.") else {
            throw ToolCallParserError.malformed
        }
        let name = String(recipient.dropFirst("functions.".count))
        guard name.range(of: "^[A-Za-z0-9_-]{1,64}$",
                         options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        return name
    }

    public func parse(name: String,
                      body: String,
                      allowedTools: Set<String>,
                      id: String) throws -> ParsedToolCall {
        guard body.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
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
