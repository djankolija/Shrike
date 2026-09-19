enum BenchError: Error, CustomStringConvertible {
    case usage(String)
    case model(String)
    case coder(String)
    case allocation(String)
    case missingResource(String)
    case commandBuffer
    case encoder
    case gpu(String)

    var description: String {
        switch self {
        case .usage(let text): return text
        case .model(let text): return "model: \(text)"
        case .coder(let text): return "coder: \(text)"
        case .allocation(let what): return "allocation failed: \(what)"
        case .missingResource(let what): return "missing resource: \(what)"
        case .commandBuffer: return "command buffer creation failed"
        case .encoder: return "compute encoder creation failed"
        case .gpu(let text): return "GPU error: \(text)"
        }
    }
}
