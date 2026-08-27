import Foundation
import Jinja
import OrderedCollections

public indirect enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue])
    case array([JSONValue])
    case string(String)
    case integer(Int64)
    case unsignedInteger(UInt64)
    case decimal(Decimal)
    case number(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        } else if let value = try? container.decode(UInt64.self) {
            self = .unsignedInteger(value)
        } else if let value = try? container.decode(Decimal.self) {
            self = .decimal(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: JSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .integer(let value): try container.encode(value)
        case .unsignedInteger(let value): try container.encode(value)
        case .decimal(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    public func jinjaSendableValue() throws -> any Sendable {
        switch self {
        case .object(let value):
            return try value.mapValues { try $0.jinjaSendableValue() }
        case .array(let value):
            return try value.map { try $0.jinjaSendableValue() }
        case .string(let value):
            return value
        case .integer(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .unsignedInteger(let value):
            guard let value = Int(exactly: value) else {
                throw ToolCallParserError.malformed
            }
            return value
        case .decimal(let value):
            let text = NSDecimalNumber(decimal: value).stringValue
            guard let double = Double(text),
                  double.isFinite,
                  let roundTrip = Decimal(
                    string: String(double),
                    locale: Locale(identifier: "en_US_POSIX")),
                  roundTrip == value else {
                throw ToolCallParserError.malformed
            }
            return double
        case .number(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return Optional<String>.none as String?
        }
    }

    public var objectValue: [String: JSONValue]? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public func encoded(sortedKeys: Bool = true) throws -> String {
        let encoder = JSONEncoder()
        if sortedKeys { encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes] }
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}

extension JSONValue {
    /// Serialise key/value pairs in the given order, which `[String: JSONValue]`
    /// cannot express, embedding each value's JSON text as the caller wrote it.
    ///
    /// The caller supplies that text so a value the model emitted can be
    /// carried through whole: encoding a parsed value instead sorts nested
    /// object keys, drops the source's spacing and normalises number lexemes,
    /// none of which the KV holds.
    public static func encodedObject(_ pairs: [(String, String)]) throws -> String {
        let encoder = JSONEncoder()
        var parts: [String] = []
        for (key, valueText) in pairs {
            let keyData = try encoder.encode(key)
            guard let keyText = String(data: keyData, encoding: .utf8) else {
                throw ToolCallParserError.malformed
            }
            parts.append("\(keyText):\(valueText)")
        }
        return "{" + parts.joined(separator: ",") + "}"
    }

    /// Parse a JSON object into a Jinja value that keeps the text's key order.
    ///
    /// `JSONDecoder` cannot do this: it stores members in a dictionary before any
    /// `Codable` conformance runs, so `allKeys` comes back in hash order.
    public static func orderedJinjaObject(_ text: String) throws -> Jinja.Value {
        try jinjaObject(text, verbatimNonStringMembers: false)
    }

    /// `orderedJinjaObject` with each member that is not a JSON string carried
    /// as the exact source slice its value spans, as a Jinja string.
    ///
    /// The template's other branch, `tojson`, is not the identity on the text
    /// it parsed — it sorts nested object keys, drops the source's spacing and
    /// escapes `/` — and a slice is. Strings are excluded because their slice
    /// carries the quotes the string branch drops. Acceptance is unchanged:
    /// every value is still parsed, only its rendered form differs.
    static func verbatimJinjaObject(_ text: String) throws -> Jinja.Value {
        try jinjaObject(text, verbatimNonStringMembers: true)
    }

    private static func jinjaObject(_ text: String,
                                    verbatimNonStringMembers: Bool) throws -> Jinja.Value {
        var scanner = OrderedJSONScanner(
            text, verbatimNonStringMembers: verbatimNonStringMembers)
        let value = try scanner.parseValue()
        scanner.skipWhitespace()
        guard scanner.isAtEnd, case .object = value else {
            throw ToolCallParserError.malformed
        }
        return value
    }
}

/// Minimal recursive-descent JSON reader that preserves object key order.
struct OrderedJSONScanner {
    private let scalars: [Character]
    private var index: Int = 0
    private let verbatimNonStringMembers: Bool
    private var depth: Int = 0

    init(_ text: String, verbatimNonStringMembers: Bool) {
        scalars = Array(text)
        self.verbatimNonStringMembers = verbatimNonStringMembers
    }

    var isAtEnd: Bool { index >= scalars.count }

    mutating func skipWhitespace() {
        while index < scalars.count, scalars[index].isWhitespace { index += 1 }
    }

    private mutating func expect(_ character: Character) throws {
        skipWhitespace()
        guard index < scalars.count, scalars[index] == character else {
            throw ToolCallParserError.malformed
        }
        index += 1
    }

    private mutating func peek() throws -> Character {
        skipWhitespace()
        guard index < scalars.count else { throw ToolCallParserError.malformed }
        return scalars[index]
    }

    mutating func parseValue() throws -> Jinja.Value {
        switch try peek() {
        case "{": return try parseObject()
        case "[": return try parseArray()
        case "\"": return .string(try parseString())
        case "t", "f": return .boolean(try parseLiteralBool())
        case "n": try parseNull(); return .null
        default: return try parseNumber()
        }
    }

    private mutating func parseObject() throws -> Jinja.Value {
        try expect("{")
        depth += 1
        defer { depth -= 1 }
        // Only the root object's members are what the template iterates.
        let verbatim = verbatimNonStringMembers && depth == 1
        var members: OrderedDictionary<String, Jinja.Value> = [:]
        skipWhitespace()
        if try peek() == "}" { index += 1; return .object(members) }
        while true {
            let key = try parseString()
            try expect(":")
            members[key] = verbatim ? try verbatimValue() : try parseValue()
            skipWhitespace()
            let next = try peek()
            index += 1
            if next == "}" { break }
            guard next == "," else { throw ToolCallParserError.malformed }
        }
        return .object(members)
    }

    /// The value at the cursor, or — where it is not a JSON string — the source
    /// it spans: from its first character to its last, leading whitespace
    /// skipped and interior spacing kept as written.
    private mutating func verbatimValue() throws -> Jinja.Value {
        skipWhitespace()
        let start = index
        let value = try parseValue()
        guard case .string = value else {
            return .string(String(scalars[start..<index]))
        }
        return value
    }

    private mutating func parseArray() throws -> Jinja.Value {
        try expect("[")
        var items: [Jinja.Value] = []
        skipWhitespace()
        if try peek() == "]" { index += 1; return .array(items) }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            let next = try peek()
            index += 1
            if next == "]" { break }
            guard next == "," else { throw ToolCallParserError.malformed }
        }
        return .array(items)
    }

    private mutating func parseString() throws -> String {
        try expect("\"")
        var out = ""
        while index < scalars.count {
            let character = scalars[index]
            index += 1
            if character == "\"" { return out }
            if character != "\\" { out.append(character); continue }
            guard index < scalars.count else { throw ToolCallParserError.malformed }
            let escape = scalars[index]
            index += 1
            switch escape {
            case "\"", "\\", "/": out.append(escape)
            case "b": out.append("\u{08}")
            case "f": out.append("\u{0C}")
            case "n": out.append("\n")
            case "r": out.append("\r")
            case "t": out.append("\t")
            case "u":
                guard index + 4 <= scalars.count else { throw ToolCallParserError.malformed }
                let hex = String(scalars[index..<(index + 4)])
                index += 4
                guard let code = UInt32(hex, radix: 16) else {
                    throw ToolCallParserError.malformed
                }
                if code >= 0xD800, code <= 0xDBFF,
                   index + 6 <= scalars.count,
                   scalars[index] == "\\", scalars[index + 1] == "u",
                   let low = UInt32(String(scalars[(index + 2)..<(index + 6)]), radix: 16),
                   low >= 0xDC00, low <= 0xDFFF {
                    index += 6
                    let combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    guard let scalar = Unicode.Scalar(combined) else {
                        throw ToolCallParserError.malformed
                    }
                    out.append(Character(scalar))
                } else {
                    guard let scalar = Unicode.Scalar(code) else {
                        throw ToolCallParserError.malformed
                    }
                    out.append(Character(scalar))
                }
            default: throw ToolCallParserError.malformed
            }
        }
        throw ToolCallParserError.malformed
    }

    private mutating func parseLiteralBool() throws -> Bool {
        if matches("true") { return true }
        if matches("false") { return false }
        throw ToolCallParserError.malformed
    }

    private mutating func parseNull() throws {
        guard matches("null") else { throw ToolCallParserError.malformed }
    }

    private mutating func matches(_ literal: String) -> Bool {
        let characters = Array(literal)
        guard index + characters.count <= scalars.count,
              Array(scalars[index..<(index + characters.count)]) == characters else {
            return false
        }
        index += characters.count
        return true
    }

    private mutating func parseNumber() throws -> Jinja.Value {
        skipWhitespace()
        let start = index
        while index < scalars.count,
              "0123456789+-.eE".contains(scalars[index]) {
            index += 1
        }
        let text = String(scalars[start..<index])
        guard !text.isEmpty else { throw ToolCallParserError.malformed }
        if let integer = Int(text) { return .int(integer) }
        // Value-exactness gate matching `jinjaSendableValue`: a number the
        // renderer cannot reproduce exactly (e.g. UInt64.max) is rejected,
        // never rounded.
        guard let double = Double(text), double.isFinite,
              let literal = Decimal(
                string: text, locale: Locale(identifier: "en_US_POSIX")),
              let roundTrip = Decimal(
                string: String(double),
                locale: Locale(identifier: "en_US_POSIX")),
              roundTrip == literal else {
            throw ToolCallParserError.malformed
        }
        return .double(double)
    }
}
