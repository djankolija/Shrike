import ArgumentParser
import Shrike

// Declared once here rather than in each binary's parser: two modules
// conforming the same type collide the moment anything imports both.
extension ModelThinkingMode: ExpressibleByArgument {}

extension RuntimeRoPEScalingMode: ExpressibleByArgument {}

extension ReasoningEffort: ExpressibleByArgument {}

extension KVCachePrecision: ExpressibleByArgument {
    public init?(argument: String) {
        guard let bits = Int(argument), let precision = KVCachePrecision(rawValue: bits) else {
            return nil
        }
        self = precision
    }
}

/// A seed written either as `0x5EED0019` or in decimal. Both benches write their
/// defaults as hex literals, so both accept that spelling.
public struct BenchSeed: Equatable, Sendable {
    public let value: UInt64

    public init(_ value: UInt64) { self.value = value }
}

extension BenchSeed: ExpressibleByArgument {
    public init?(argument: String) {
        let hex = argument.hasPrefix("0x") || argument.hasPrefix("0X")
        guard let parsed = UInt64(hex ? String(argument.dropFirst(2)) : argument,
                                 radix: hex ? 16 : 10) else { return nil }
        self.init(parsed)
    }

    public var defaultValueDescription: String {
        "0x" + String(value, radix: 16, uppercase: true)
    }
}

/// `--arms a,b,c` and `--positions 1024,4096`: one comma-joined value, not a
/// repeated flag. No arm name contains a comma and this is the existing surface.
public struct CommaSeparatedNames: Equatable, Sendable {
    public let names: [String]

    public init(_ names: [String]) { self.names = names }
}

extension CommaSeparatedNames: ExpressibleByArgument {
    public init?(argument: String) {
        let parts = argument.split(separator: ",").map(String.init)
        guard !parts.isEmpty else { return nil }
        self.init(parts)
    }

    public var defaultValueDescription: String { names.joined(separator: ",") }
}

public struct CommaSeparatedCounts: Equatable, Sendable {
    public let counts: [Int]

    public init(_ counts: [Int]) { self.counts = counts }
}

extension CommaSeparatedCounts: ExpressibleByArgument {
    public init?(argument: String) {
        var parsed: [Int] = []
        for piece in argument.split(separator: ",") {
            guard let count = Int(piece), count > 0 else { return nil }
            parsed.append(count)
        }
        guard !parsed.isEmpty else { return nil }
        self.init(parsed)
    }

    public var defaultValueDescription: String {
        counts.map(String.init).joined(separator: ",")
    }
}

public enum ExpertCacheBudgetArgument {
    public static var help: ArgumentHelp {
        ArgumentHelp("""
            Bytes the routed-expert cache may use, e.g. 8G, 2G, 512M. Slots are \
            derived from this and the model's expert stride, so this is the knob \
            and the slot count is the result. Default 8G, which holds the \
            measured routing working set; smaller budgets are markedly slower \
            because expert reads bypass the page cache and have no fallback.
            """,
            valueName: "size")
    }

    public static func bytes(_ value: String) throws -> Int {
        guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
            throw ValidationError(
                "--ram-budget must be a positive size such as 2G, 512M or a byte count")
        }
        return parsed
    }
}
