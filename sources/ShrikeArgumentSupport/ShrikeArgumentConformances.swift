import ArgumentParser
import Shrike

// Declared once here rather than in each binary's parser: two modules
// conforming the same type collide the moment anything imports both.
extension ModelThinkingMode: ExpressibleByArgument {}

extension RuntimeRoPEScalingMode: ExpressibleByArgument {}

extension KVCachePrecision: ExpressibleByArgument {
    public init?(argument: String) {
        guard let bits = Int(argument), let precision = KVCachePrecision(rawValue: bits) else {
            return nil
        }
        self = precision
    }
}
