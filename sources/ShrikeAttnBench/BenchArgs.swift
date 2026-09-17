struct BenchArgs {
    var arms: [String] = Arm.defaultLadder
    var positions: [Int] = [1024, 4096, 8192]
    var repeats = 7
    var warmup = 2
    var seed: UInt64 = 0x5EED_0019
    var listArms = false

    static let usage = """
    ShrikeAttnBench: the decode attention scan on synthetic rows at the served shape.

      --arms <a,b,...>       arms to run (default: the step-zero ladder; --list names them)
      --positions <n,...>    context lengths (default 1024,4096,8192)
      --repeats <n>          timed command buffers per arm and length, median reported (7)
      --warmup <n>           untimed command buffers before the repeats (2)
      --seed <n>             row generator seed
      --list                 print the arm names and exit
      --help

    A composite arm joins switches with '+', e.g. qregs+dbuf+load8.
    """

    static func parse(_ argv: [String]) throws -> BenchArgs {
        var args = BenchArgs()
        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < argv.count else { throw BenchError.usage("\(flag) needs a value") }
            return argv[i]
        }
        while i < argv.count {
            let flag = argv[i]
            switch flag {
            case "--help", "-h": throw BenchError.help
            case "--list": args.listArms = true
            case "--arms": args.arms = try value(flag).split(separator: ",").map(String.init)
            case "--positions":
                args.positions = try value(flag).split(separator: ",").map { piece in
                    guard let n = Int(piece), n > 0 else {
                        throw BenchError.usage("bad position count \(piece)")
                    }
                    return n
                }
            case "--repeats": args.repeats = try Self.count(value(flag), flag)
            case "--warmup": args.warmup = try Self.count(value(flag), flag, allowZero: true)
            case "--seed":
                let text = try value(flag)
                guard let n = UInt64(text.hasPrefix("0x") ? String(text.dropFirst(2)) : text,
                                     radix: text.hasPrefix("0x") ? 16 : 10) else {
                    throw BenchError.usage("bad seed \(text)")
                }
                args.seed = n
            default: throw BenchError.usage("unknown argument \(flag)")
            }
            i += 1
        }
        return args
    }

    private static func count(_ text: String, _ flag: String,
                              allowZero: Bool = false) throws -> Int {
        guard let n = Int(text), n > 0 || (allowZero && n == 0) else {
            throw BenchError.usage("\(flag) needs a positive count")
        }
        return n
    }
}
