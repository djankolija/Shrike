import Foundation

struct PrefixCode {
    let lengths: [Int]
    let reversedCodes: [UInt32]
    let table: [UInt8]

    static func build(frequencies: [Int], maxLength: Int = 8) throws -> PrefixCode {
        precondition(frequencies.count == 16)
        var freq = frequencies.map { max($0, 1) }
        var lengths: [Int] = []
        for _ in 0..<32 {
            lengths = huffmanLengths(freq)
            if lengths.max()! <= maxLength { break }
            let total = freq.reduce(0, +)
            freq = freq.map { $0 + max(total / 64, 1) }
        }
        guard lengths.max()! <= maxLength else {
            throw BenchError.coder("could not limit the code to \(maxLength) bits")
        }
        let order = (0..<16).sorted { (lengths[$0], $0) < (lengths[$1], $1) }
        var canonical = [UInt32](repeating: 0, count: 16)
        var code: UInt32 = 0
        var previous = 0
        for sym in order {
            code <<= UInt32(lengths[sym] - previous)
            canonical[sym] = code
            previous = lengths[sym]
            code += 1
        }
        var reversed = [UInt32](repeating: 0, count: 16)
        var table = [UInt8](repeating: 0, count: 256)
        for sym in 0..<16 {
            let len = lengths[sym]
            var r: UInt32 = 0
            for bit in 0..<len where canonical[sym] & (1 << UInt32(len - 1 - bit)) != 0 {
                r |= 1 << UInt32(bit)
            }
            reversed[sym] = r
            for pad in 0..<(1 << (maxLength - len)) {
                table[Int(r) | (pad << len)] = UInt8(len << 4 | sym)
            }
        }
        return PrefixCode(lengths: lengths, reversedCodes: reversed, table: table)
    }

    private static func huffmanLengths(_ freq: [Int]) -> [Int] {
        var nodes: [(weight: Int, symbols: [Int])] = freq.enumerated().map { ($0.element, [$0.offset]) }
        var lengths = [Int](repeating: 0, count: freq.count)
        while nodes.count > 1 {
            nodes.sort { $0.weight < $1.weight }
            let a = nodes.removeFirst()
            let b = nodes.removeFirst()
            for s in a.symbols + b.symbols { lengths[s] += 1 }
            nodes.append((a.weight + b.weight, a.symbols + b.symbols))
        }
        return lengths
    }

}

/// Mirrors `ExpertOffsets` in moe.metal field for field.
struct PlainOffsets {
    var gateW: UInt32 = 0
    var gateS: UInt32 = 0
    var gateB: UInt32 = 0
    var upW: UInt32 = 0
    var upS: UInt32 = 0
    var upB: UInt32 = 0
    var downW: UInt32 = 0
    var downS: UInt32 = 0
    var downB: UInt32 = 0
    var gateAB: UInt32 = 0
    var upAB: UInt32 = 0
    var downAB: UInt32 = 0
}

/// Mirrors `CodedOffsets` in expert.metal field for field.
struct CodedOffsets {
    var gateDir: UInt32 = 0
    var gateData: UInt32 = 0
    var gateS: UInt32 = 0
    var gateB: UInt32 = 0
    var upDir: UInt32 = 0
    var upData: UInt32 = 0
    var upS: UInt32 = 0
    var upB: UInt32 = 0
    var gateSTable: UInt32 = 0
    var gateBTable: UInt32 = 0
    var upSTable: UInt32 = 0
    var upBTable: UInt32 = 0
}

struct CodedExpert {
    let bytes: [UInt8]
    let offsets: CodedOffsets
    let phase1Bytes: Int
    let streamBytes: Int
}

struct AuxTable {
    let patterns: [UInt16]
    let indexOf: [UInt16: Int]

    init(experts: [PlainExpert], tensor: String) throws {
        var seen = Set<UInt16>()
        for expert in experts {
            let t = try expert.tensor(tensor)
            expert.bytes.withUnsafeBufferPointer { buf in
                let base = UnsafeRawPointer(buf.baseAddress!) + t.offset
                for i in 0..<(t.size / 2) {
                    seen.insert(base.load(fromByteOffset: i * 2, as: UInt16.self))
                }
            }
        }
        patterns = seen.sorted()
        var map: [UInt16: Int] = [:]
        for (i, p) in patterns.enumerated() { map[p] = i }
        indexOf = map
    }
}

enum Coder {
    static let lanes = 32
    static let blockWeights = 256
    static let laneWeights = 8

    static func nibbleHistogram(experts: [PlainExpert], tensors: [String]) throws -> [Int] {
        var counts = [Int](repeating: 0, count: 16)
        for expert in experts {
            for name in tensors {
                let t = try expert.tensor(name)
                for byte in expert.bytes[t.offset..<(t.offset + t.size)] {
                    counts[Int(byte & 0x0F)] += 1
                    counts[Int(byte >> 4)] += 1
                }
            }
        }
        return counts
    }

    /// Lane l's symbols of a row: for every block of 256 weights, the eight
    /// weights the plain kernel's lane l reads, in the order it reads them.
    private static func laneSymbols(row: UnsafeBufferPointer<UInt8>, cols: Int, lane: Int) -> [UInt8] {
        var symbols: [UInt8] = []
        symbols.reserveCapacity(cols / lanes)
        for block in 0..<(cols / blockWeights) {
            for i in 0..<laneWeights {
                let w = block * blockWeights + lane * laneWeights + i
                let byte = row[w / 2]
                symbols.append(w % 2 == 0 ? byte & 0x0F : byte >> 4)
            }
        }
        return symbols
    }

    private static func emit(_ symbols: [UInt8], code: PrefixCode) -> [UInt8] {
        var out: [UInt8] = []
        var acc: UInt64 = 0
        var bits = 0
        for s in symbols {
            acc |= UInt64(code.reversedCodes[Int(s)]) << UInt64(bits)
            bits += code.lengths[Int(s)]
            while bits >= 8 {
                out.append(UInt8(acc & 0xFF))
                acc >>= 8
                bits -= 8
            }
        }
        if bits > 0 { out.append(UInt8(acc & 0xFF)) }
        return out
    }

    static func codeRole(expert: PlainExpert, tensor: String, code: PrefixCode) throws
        -> (directory: [UInt32], data: [UInt8], streamBytes: Int) {
        let t = try expert.tensor(tensor)
        let rows = t.shape[0]
        let cols = t.shape[1]
        guard cols % blockWeights == 0 else {
            throw BenchError.coder("\(tensor): \(cols) columns are not a whole number of blocks")
        }
        let rowBytes = cols / 2
        var directory: [UInt32] = []
        var data: [UInt8] = []
        var streamBytes = 0
        try expert.bytes.withUnsafeBufferPointer { buf in
            for r in 0..<rows {
                let row = UnsafeBufferPointer(start: buf.baseAddress! + t.offset + r * rowBytes, count: rowBytes)
                var streams: [[UInt8]] = []
                for lane in 0..<lanes {
                    streams.append(emit(laneSymbols(row: row, cols: cols, lane: lane), code: code))
                }
                guard streams.allSatisfy({ $0.count <= 255 }) else {
                    throw BenchError.coder("a lane stream exceeds 255 bytes")
                }
                directory.append(UInt32(data.count))
                data.append(contentsOf: streams.map { UInt8($0.count) })
                for s in streams {
                    data.append(contentsOf: s)
                    streamBytes += s.count
                }
            }
        }
        data.append(contentsOf: [UInt8](repeating: 0, count: 16))
        return (directory, data, streamBytes)
    }

    static func packAux(expert: PlainExpert, tensor: String, table: AuxTable) throws -> [UInt8] {
        let t = try expert.tensor(tensor)
        let rows = t.shape[0]
        let groups = t.shape[1]
        guard groups % 2 == 0 else { throw BenchError.coder("\(tensor): odd group count") }
        var out: [UInt8] = []
        out.reserveCapacity(rows * groups * 3 / 2)
        try expert.bytes.withUnsafeBufferPointer { buf in
            let base = UnsafeRawPointer(buf.baseAddress!) + t.offset
            for r in 0..<rows {
                var k = 0
                while k < groups {
                    let a = base.load(fromByteOffset: (r * groups + k) * 2, as: UInt16.self)
                    let b = base.load(fromByteOffset: (r * groups + k + 1) * 2, as: UInt16.self)
                    guard let ia = table.indexOf[a], let ib = table.indexOf[b], ia < 4096, ib < 4096 else {
                        throw BenchError.coder("\(tensor): a pattern is not in the table")
                    }
                    out.append(UInt8(ia & 0xFF))
                    out.append(UInt8((ia >> 8) | ((ib & 0xF) << 4)))
                    out.append(UInt8(ib >> 4))
                    k += 2
                }
            }
        }
        return out
    }

    struct AuxTables {
        let gateS: AuxTable
        let gateB: AuxTable
        let upS: AuxTable
        let upB: AuxTable

        var concatenated: [UInt16] { gateS.patterns + gateB.patterns + upS.patterns + upB.patterns }
        var elementOffsets: (gateS: Int, gateB: Int, upS: Int, upB: Int) {
            let a = 0
            let b = a + gateS.patterns.count
            let c = b + gateB.patterns.count
            let d = c + upS.patterns.count
            return (a, b, c, d)
        }
    }

    private static func aligned(_ n: Int) -> Int { (n + 3) & ~3 }

    private static func directoryBytes(_ d: [UInt32]) -> [UInt8] {
        d.withUnsafeBufferPointer { Array(UnsafeRawBufferPointer($0)) }
    }

    /// The eight sections of a coded expert's phase-1 bytes, in layout order.
    private static func sections(_ expert: PlainExpert, code: PrefixCode, aux: AuxTables?) throws
        -> (chunks: [[UInt8]], streamBytes: Int) {
        func plain(_ tensor: String) throws -> [UInt8] {
            let t = try expert.tensor(tensor)
            return Array(expert.bytes[t.offset..<(t.offset + t.size)])
        }
        func auxOrPlain(_ tensor: String, _ table: (AuxTables) -> AuxTable) throws -> [UInt8] {
            if let aux { return try packAux(expert: expert, tensor: tensor, table: table(aux)) }
            return try plain(tensor)
        }
        let gate = try codeRole(expert: expert, tensor: "gate", code: code)
        let up = try codeRole(expert: expert, tensor: "up", code: code)
        let chunks: [[UInt8]] = [
            directoryBytes(gate.directory), gate.data,
            try auxOrPlain("gate_scales", \.gateS), try auxOrPlain("gate_biases", \.gateB),
            directoryBytes(up.directory), up.data,
            try auxOrPlain("up_scales", \.upS), try auxOrPlain("up_biases", \.upB),
        ]
        return (chunks, gate.streamBytes + up.streamBytes)
    }

    /// Every expert laid out at the same section offsets, each section sized
    /// by the layer's largest, which is the per-layer stride of the design;
    /// `phase1Bytes` is what the kernel can address, `codedBytes` what each
    /// expert's rows actually hold.
    static func codeExperts(_ experts: [PlainExpert], code: PrefixCode, aux: AuxTables?) throws
        -> (coded: [CodedExpert], offsets: CodedOffsets, phase1Bytes: Int) {
        let all = try experts.map { try sections($0, code: code, aux: aux) }
        let sizes = (0..<8).map { i in all.map { $0.chunks[i].count }.max()! }
        var starts: [Int] = []
        var cursor = 0
        for size in sizes {
            starts.append(cursor)
            cursor = aligned(cursor + size)
        }
        let phase1Bytes = cursor
        var offsets = CodedOffsets(gateDir: UInt32(starts[0]), gateData: UInt32(starts[1]),
                                   gateS: UInt32(starts[2]), gateB: UInt32(starts[3]),
                                   upDir: UInt32(starts[4]), upData: UInt32(starts[5]),
                                   upS: UInt32(starts[6]), upB: UInt32(starts[7]))
        if let aux {
            let e = aux.elementOffsets
            offsets.gateSTable = UInt32(e.gateS)
            offsets.gateBTable = UInt32(e.gateB)
            offsets.upSTable = UInt32(e.upS)
            offsets.upBTable = UInt32(e.upB)
        }
        let coded = all.map { expert -> CodedExpert in
            var bytes = [UInt8](repeating: 0, count: phase1Bytes + 16)
            for (i, chunk) in expert.chunks.enumerated() {
                bytes.replaceSubrange(starts[i]..<(starts[i] + chunk.count), with: chunk)
            }
            let codedBytes = expert.chunks.reduce(0) { $0 + $1.count }
            return CodedExpert(bytes: bytes, offsets: offsets, phase1Bytes: codedBytes,
                               streamBytes: expert.streamBytes)
        }
        return (coded, offsets, phase1Bytes)
    }
}
