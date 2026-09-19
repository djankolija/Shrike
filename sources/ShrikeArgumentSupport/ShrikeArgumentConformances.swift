import ArgumentParser
import Shrike

// Declared once here rather than in each binary's parser: two modules
// conforming the same type collide the moment anything imports both.
extension ModelThinkingMode: ExpressibleByArgument {}

extension RuntimeRoPEScalingMode: ExpressibleByArgument {}

extension ReasoningEffort: ExpressibleByArgument {}

// Lenient where its sibling is not: the pre-parser form accepted any casing
// and the help never promised otherwise.
extension ReasoningRetention: ExpressibleByArgument {
    public init?(argument: String) {
        self.init(rawValue: argument.lowercased())
    }
}

extension KVCachePrecision: ExpressibleByArgument {
    public init?(argument: String) {
        guard let bits = Int(argument), let precision = KVCachePrecision(rawValue: bits) else {
            return nil
        }
        self = precision
    }
}
