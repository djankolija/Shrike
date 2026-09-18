import Foundation

struct PlainExpert {
    struct Tensor {
        let offset: Int
        let size: Int
        let shape: [Int]
    }

    let index: Int
    let bytes: [UInt8]
    let tensors: [String: Tensor]

    func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw BenchError.model("expert \(index) has no tensor \(name)") }
        return t
    }
}

enum Experts {
    static let groupSize = 64

    static func load(model: String, layer: Int, count: Int) throws -> (stride: Int, experts: [PlainExpert]) {
        let root = URL(fileURLWithPath: model)
        let layoutURL = root.appendingPathComponent("packed_experts/layout.json")
        guard let data = try? Data(contentsOf: layoutURL),
              let layout = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let stride = layout["expertStride"] as? Int,
              let layers = layout["layers"] as? [[String: Any]] else {
            throw BenchError.model("cannot read \(layoutURL.path)")
        }
        guard let entry = layers.first(where: { ($0["layer"] as? Int) == layer }),
              let file = entry["file"] as? String,
              let experts = entry["experts"] as? [[String: Any]] else {
            throw BenchError.model("layer \(layer) is not in layout.json")
        }
        let handle = try FileHandle(forReadingFrom: root.appendingPathComponent("packed_experts/\(file)"))
        defer { try? handle.close() }
        var loaded: [PlainExpert] = []
        for expert in experts.prefix(count) {
            guard let index = expert["expert"] as? Int,
                  let offset = expert["offset"] as? Int,
                  let size = expert["size"] as? Int,
                  let tensors = expert["tensors"] as? [String: [String: Any]] else {
                throw BenchError.model("malformed expert entry in layout.json")
            }
            try handle.seek(toOffset: UInt64(offset))
            guard let blob = try handle.read(upToCount: size), blob.count == size else {
                throw BenchError.model("short read of expert \(index)")
            }
            var parsed: [String: PlainExpert.Tensor] = [:]
            for (name, t) in tensors {
                guard let off = t["offset"] as? Int, let sz = t["size"] as? Int,
                      let shape = t["shape"] as? [Int] else {
                    throw BenchError.model("malformed tensor \(name) of expert \(index)")
                }
                parsed[name] = PlainExpert.Tensor(offset: off, size: sz, shape: shape)
            }
            loaded.append(PlainExpert(index: index, bytes: [UInt8](blob), tensors: parsed))
        }
        guard loaded.count == count else {
            throw BenchError.model("layer \(layer) holds \(loaded.count) experts, \(count) wanted")
        }
        return (stride, loaded)
    }
}
