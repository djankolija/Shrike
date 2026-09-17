/// The ladder kernel's switches, one per function constant in Metal/ladder.metal.
struct LadderSwitches: Hashable {
    var qRegs = false
    var posBlock: UInt32 = 4
    var doubleBuffer = false
    var noSoftmax = false
    var noV = false
    var loadOnly = false
    var fullRow = false
    var loadBytes: UInt32 = 4
    var staticLoops = false
    var oForm: UInt32 = 0
    var dForm: UInt32 = 0
    var safeMath = false

    static let headDim = 256
    static let qPerKV = 8
    static let rowValues = 512

    var slice: Int { fullRow ? Self.rowValues : Self.headDim }

    var threadgroupBytes: Int {
        let qBytes = qRegs ? 0 : Self.qPerKV * Self.headDim * 4
        let staging = (doubleBuffer ? 2 : 1) * Int(posBlock) * slice * 4 * 2
        return qBytes + staging
    }

    func validate() throws {
        if fullRow && !loadOnly {
            throw BenchError.usage("fullrow is a load-only arm; add loadonly")
        }
        if ![4, 8].contains(posBlock) {
            throw BenchError.usage("the position block is 4 or 8")
        }
        if ![4, 8, 16].contains(loadBytes) {
            throw BenchError.usage("load bytes are 4, 8 or 16")
        }
    }
}

enum ArmKind {
    case production(specialized: Bool)
    case ladder(LadderSwitches)
    case stream(StreamSwitches)
}

struct Arm {
    let name: String
    let kind: ArmKind

    static let defaultLadder = [
        "prod", "copy", "qregs", "block8", "dbuf", "load8", "load16",
        "qregs+dbuf+load8", "nosoftmax", "nov", "loadonly", "loadonly+fullrow",
    ]

    static let switchHelp: [(String, String)] = [
        ("prod", "the production pipeline through the library's wrapper (partial and combine)"),
        ("prodplain", "the production pipeline without the V4.1 function-constant specialization"),
        ("copy", "the ladder kernel with every switch at its default (the shipped kernel)"),
        ("qregs", "Q in per-lane registers, the 8 KB threadgroup copy gone"),
        ("block8", "eight positions per staging block instead of four"),
        ("dbuf", "double-buffered staging: the next block loads while this one computes"),
        ("load8", "eight packed bytes per thread per row instead of four"),
        ("load16", "sixteen packed bytes per thread per row"),
        ("nosoftmax", "the online softmax removed (dot and V accumulate only)"),
        ("nov", "V never loaded or accumulated (the K scan and the softmax only)"),
        ("loadonly", "the staging loads only, no compute: the floor of this access pattern"),
        ("fullrow", "with loadonly: each threadgroup reads the whole 544-byte row"),
        ("sloops", "the per-lane loops with a static trip count, the same elements in the same order"),
        ("o1", "the V accumulate as fma(o, alpha, p * v); o2: fma(p, v, o * alpha); o3: p * v + o * alpha"),
        ("d1", "the denominator as fma(d, alpha, p)"),
        ("safemath", "the kernel compiled with contraction and reassociation off"),
        ("stream2 / stream4 / stream8", "the streaming scan with 2, 4 or 8 heads per simdgroup"),
        ("noload", "with stream: the loads replaced by arithmetic, the ALU-bound twin"),
        ("u2", "with stream: two positions per iteration, their chains overlapped"),
        ("lazy", "with stream: the rescale only when the running max moves"),
    ]

    static func parse(_ name: String) throws -> Arm {
        if name == "prod" { return Arm(name: name, kind: .production(specialized: true)) }
        if name == "prodplain" { return Arm(name: name, kind: .production(specialized: false)) }
        if name.hasPrefix("stream") { return try parseStream(name) }
        var sw = LadderSwitches()
        for token in name.split(separator: "+").map(String.init) {
            switch token {
            case "copy": break
            case "qregs": sw.qRegs = true
            case "block4": sw.posBlock = 4
            case "block8": sw.posBlock = 8
            case "dbuf": sw.doubleBuffer = true
            case "load4": sw.loadBytes = 4
            case "load8": sw.loadBytes = 8
            case "load16": sw.loadBytes = 16
            case "nosoftmax": sw.noSoftmax = true
            case "nov": sw.noV = true
            case "loadonly": sw.loadOnly = true
            case "fullrow": sw.fullRow = true
            case "sloops": sw.staticLoops = true
            case "o1": sw.oForm = 1
            case "o2": sw.oForm = 2
            case "o3": sw.oForm = 3
            case "d1": sw.dForm = 1
            case "safemath": sw.safeMath = true
            default: throw BenchError.usage("unknown arm switch \(token) in \(name)")
            }
        }
        try sw.validate()
        return Arm(name: name, kind: .ladder(sw))
    }

    private static func parseStream(_ name: String) throws -> Arm {
        var sw = StreamSwitches()
        for token in name.split(separator: "+").map(String.init) {
            switch token {
            case "stream2": sw.headsPerSimdgroup = 2
            case "stream4": sw.headsPerSimdgroup = 4
            case "stream8": sw.headsPerSimdgroup = 8
            case "noload": sw.noLoad = true
            case "u2": sw.unroll2 = true
            case "lazy": sw.lazy = true
            default: throw BenchError.usage("unknown stream switch \(token) in \(name)")
            }
        }
        try sw.validate()
        return Arm(name: name, kind: .stream(sw))
    }
}
