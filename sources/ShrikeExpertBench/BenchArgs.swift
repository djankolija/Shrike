struct BenchArgs {
    var model = ""
    var layer = 20
    var experts = 8
    var arms: [String] = ["plain", "coded", "coded+aux"]
    var repeats = 15
    var warmup = 3
    var batch = 20
    var seed: UInt64 = 0x5EED_0021

    static let usage = """
    ShrikeExpertBench: the decode phase-1 gate/up kernel over real experts, plain and coded.

      --model <dir>          the .gturbo directory (required)
      --layer <n>            the layer whose experts are read (default 20)
      --experts <n>          experts per pass, the routed top-k (default 8)
      --arms <a,b,...>       plain, coded, coded+aux (default all three)
      --repeats <n>          timed command buffers per arm, median reported (15)
      --warmup <n>           untimed command buffers before the repeats (3)
      --batch <n>            dispatches per command buffer, so the GPU holds its
                             clock; the time reported is per dispatch (20)
      --seed <n>             the activation vector's seed
      --help
    """

    static func parse(_ argv: [String]) throws -> BenchArgs {
        var args = BenchArgs()
        var i = 0
        func value(_ flag: String) throws -> String {
            i += 1
            guard i < argv.count else { throw BenchError.usage("\(flag) needs a value") }
            return argv[i]
        }
        func int(_ flag: String) throws -> Int {
            let raw = try value(flag)
            guard let n = Int(raw), n > 0 else { throw BenchError.usage("\(flag): bad count \(raw)") }
            return n
        }
        while i < argv.count {
            let flag = argv[i]
            switch flag {
            case "--help", "-h": throw BenchError.help
            case "--model": args.model = try value(flag)
            case "--layer":
                let raw = try value(flag)
                guard let n = Int(raw), n >= 0 else { throw BenchError.usage("--layer: bad index \(raw)") }
                args.layer = n
            case "--experts": args.experts = try int(flag)
            case "--arms": args.arms = try value(flag).split(separator: ",").map(String.init)
            case "--repeats": args.repeats = try int(flag)
            case "--batch": args.batch = try int(flag)
            case "--warmup":
                let raw = try value(flag)
                guard let n = Int(raw), n >= 0 else { throw BenchError.usage("--warmup: bad count \(raw)") }
                args.warmup = n
            case "--seed":
                let raw = try value(flag)
                guard let n = UInt64(raw) else { throw BenchError.usage("--seed: bad value \(raw)") }
                args.seed = n
            default: throw BenchError.usage("unknown flag \(flag)")
            }
            i += 1
        }
        guard !args.model.isEmpty else { throw BenchError.usage("--model is required") }
        guard (1...8).contains(args.experts) else { throw BenchError.usage("--experts is 1 to 8") }
        return args
    }
}
