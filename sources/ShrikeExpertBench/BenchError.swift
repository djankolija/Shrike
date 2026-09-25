enum BenchError: Error, CustomStringConvertible {
    case model(String)
    case allocation(String)
    case commandBuffer
    case encoder
    case gpu(String)

    var description: String {
        switch self {
        case .model(let text): return "model: \(text)"
        case .allocation(let what): return "allocation failed: \(what)"
        case .commandBuffer: return "command buffer creation failed"
        case .encoder: return "compute encoder creation failed"
        case .gpu(let text): return "GPU error: \(text)"
        }
    }
}
