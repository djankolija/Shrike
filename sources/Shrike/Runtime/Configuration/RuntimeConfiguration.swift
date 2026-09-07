import Foundation

public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
    case causalMatrix = "causal-matrix"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
    case agingLFU = "aging-lfu"
}

/// Where the predictive prefetch ring issues a layer's reads: `after` the
/// layer's demand batch has completed (the drive is otherwise idle and the
/// demand read is never slowed), or `beside` it, right after the demand
/// batch's submission (v14 T1's shape).
public enum RuntimePrefetchPlacement: String, Codable, Sendable {
    case after
    case beside
}

/// How the next-layer router probe is dispatched: as its own GEMV and
/// selection after the authoritative router's (`separate`), or in the same
/// two dispatches as a second grid row (`fused`).
public enum RuntimePrefetchProbe: String, Codable, Sendable {
    case separate
    case fused
}

/// The predictive routed-expert prefetch. Every value is validated whether
/// or not the ring is on, so a mistyped knob never runs as a default it is
/// not.
public struct RuntimePrefetch: Codable, Sendable, Equatable {
    public let enabled: Bool
    /// Predicted experts considered per layer; nil takes the architecture's
    /// top-k at runner initialisation, an explicit value is checked against it
    /// there.
    public let topM: Int?
    public let inFlight: Int
    public let placement: RuntimePrefetchPlacement
    public let distance: Int
    public let tracePath: String?
    /// The bound within which a wait for a prediction still in flight counts
    /// as joined rather than late; the wait itself runs to completion, since
    /// the prediction's cell is claimed, so 0 is refused rather than read as
    /// the "never wait" it once was.
    public let joinMicros: Int
    public let probe: RuntimePrefetchProbe

    public static let allowedInFlight = 1...8
    public static let allowedDistance = 1...8
    public static let allowedJoinMicros = 1...2000

    public static let off = RuntimePrefetch(enabled: false, topM: nil, inFlight: 1,
                                            placement: .after, distance: 1, tracePath: nil)
    public static let production = RuntimePrefetch(enabled: true, topM: nil, inFlight: 1,
                                                   placement: .after, distance: 1, tracePath: nil)

    public init(enabled: Bool, topM: Int?, inFlight: Int, placement: RuntimePrefetchPlacement,
                distance: Int, tracePath: String?,
                joinMicros: Int = 400, probe: RuntimePrefetchProbe = .fused) {
        self.enabled = enabled
        self.topM = topM
        self.inFlight = inFlight
        self.placement = placement
        self.distance = distance
        self.tracePath = tracePath
        self.joinMicros = joinMicros
        self.probe = probe
    }

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimePrefetch {
        let enabled: Bool
        switch environment["SHRIKE_PREDICTIVE_PREFETCH"] {
        case nil, "1": enabled = true
        case "0": enabled = false
        case let raw?:
            throw RuntimeConfigurationError.invalidPrefetch(
                "SHRIKE_PREDICTIVE_PREFETCH '\(raw)'; allowed: 0, 1")
        }
        let topM = try positiveInt(environment, "SHRIKE_PREFETCH_TOP_M", allowed: 1...Int.max)
        let inFlight = try positiveInt(environment, "SHRIKE_PREFETCH_INFLIGHT",
                                       allowed: allowedInFlight) ?? production.inFlight
        let placement: RuntimePrefetchPlacement
        if let raw = environment["SHRIKE_PREFETCH_PLACEMENT"] {
            guard let value = RuntimePrefetchPlacement(rawValue: raw) else {
                throw RuntimeConfigurationError.invalidPrefetch(
                    "SHRIKE_PREFETCH_PLACEMENT '\(raw)'; allowed: after, beside")
            }
            placement = value
        } else {
            placement = production.placement
        }
        let distance = try positiveInt(environment, "SHRIKE_PREFETCH_PROBE_DISTANCE",
                                       allowed: allowedDistance) ?? production.distance
        let trace = environment["SHRIKE_PREFETCH_TRACE"].flatMap { $0.isEmpty ? nil : $0 }
        if let raw = environment["SHRIKE_PREFETCH_ADOPT"] {
            throw RuntimeConfigurationError.invalidPrefetch(
                "SHRIKE_PREFETCH_ADOPT '\(raw)' was removed by v16's merge: a prediction lands in "
                    + "the pool's own address space and needs no copy")
        }
        if environment["SHRIKE_PREFETCH_JOIN_US"] == "0" {
            throw RuntimeConfigurationError.invalidPrefetch(
                "SHRIKE_PREFETCH_JOIN_US '0' no longer means never wait: a landed prediction's cell "
                    + "is claimed, so the wait runs to completion and the bound only classifies "
                    + "joined against late; allowed: 1...2000")
        }
        let joinMicros = try positiveInt(environment, "SHRIKE_PREFETCH_JOIN_US",
                                         allowed: allowedJoinMicros) ?? production.joinMicros
        let probe: RuntimePrefetchProbe
        if let raw = environment["SHRIKE_PREFETCH_PROBE"] {
            guard let value = RuntimePrefetchProbe(rawValue: raw) else {
                throw RuntimeConfigurationError.invalidPrefetch(
                    "SHRIKE_PREFETCH_PROBE '\(raw)'; allowed: separate, fused")
            }
            probe = value
        } else {
            probe = production.probe
        }
        return RuntimePrefetch(enabled: enabled, topM: topM, inFlight: inFlight,
                               placement: placement, distance: distance, tracePath: trace,
                               joinMicros: joinMicros, probe: probe)
    }

    private static func positiveInt(_ environment: [String: String], _ name: String,
                                    allowed: ClosedRange<Int>) throws -> Int? {
        guard let raw = environment[name] else { return nil }
        guard let value = Int(raw), allowed.contains(value) else {
            let bound = allowed.upperBound == Int.max
                ? "a positive integer" : "\(allowed.lowerBound)...\(allowed.upperBound)"
            throw RuntimeConfigurationError.invalidPrefetch("\(name) '\(raw)'; allowed: \(bound)")
        }
        return value
    }
}

/// Storage precision for the autoregressive attention key/value cache.
/// Quantized modes use affine groups of 64 values and keep their scale and
/// bias alongside each token row; model weights are unaffected.
public enum KVCachePrecision: Int, Codable, CaseIterable, Sendable {
    case int4 = 4
    case int8 = 8
    case fp16 = 16

    public var label: String { "\(rawValue)-bit" }
    public var isQuantized: Bool { self != .fp16 }
}

public enum RuntimeRoPEScalingMode: String, Codable, CaseIterable, Sendable {
    case none
    case yarn
}

public enum RuntimeConfigurationError: Error, CustomStringConvertible, Equatable {
    case invalidExpertCacheSlots(Int)
    case invalidPrefillChunkTokens(Int)
    case invalidYaRNContextTokens(Int)
    case contextRequiresYaRN(Int)
    case yaRNContextMismatch(maxContext: Int, configured: Int)
    case yaRNUnsupportedArchitecture
    case invalidPrefetch(String)

    public var description: String {
        switch self {
        case .invalidExpertCacheSlots(let value):
            return "unsupported expert-cache slot count \(value); allowed: \(RuntimeConfiguration.allowedExpertCacheSlots)"
        case .invalidPrefillChunkTokens(let value):
            return "unsupported prefill chunk size \(value); allowed: \(RuntimeConfiguration.allowedPrefillChunkTokens)"
        case .invalidYaRNContextTokens(let value):
            return "unsupported YaRN context \(value); allowed: \(RuntimeConfiguration.supportedYaRNContextTokens)"
        case .contextRequiresYaRN(let value):
            return "context \(value) exceeds the native \(RuntimeConfiguration.nativeMaximumContextTokens)-token limit; enable YaRN"
        case .yaRNContextMismatch(let maxContext, let configured):
            return "YaRN is configured for \(configured) tokens, but max context is \(maxContext)"
        case .yaRNUnsupportedArchitecture:
            return "YaRN requires the Qwen3.5-MoE NeoX sub-dimension RoPE architecture"
        case .invalidPrefetch(let detail):
            return "unsupported prefetch configuration: \(detail)"
        }
    }
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let supportedContextTokens = [
        4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,
    ]
    public static let nativeMaximumContextTokens = 262_144
    public static let supportedYaRNContextTokens = [524_288, 1_048_576]
    public static let defaultYaRNContextTokens = 1_048_576
    public static let maximumContextTokens = 1_048_576
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 64, 96, 128]

    /// Target bytes for the routed-expert slot cache when no count is given.
    ///
    /// 8 GiB, which is a third of a 24 GB machine and deliberate. The slot cache
    /// has to hold the routing working set, and a routing trace over 383 real
    /// tokens measured **131 distinct experts per layer** across a 128-token
    /// window. 128 slots is the first budget that holds it.
    ///
    /// Because expert reads bypass the page cache (see `ParallelExpertReader`),
    /// there is no second cache to fall back on: whatever the slots do not hold is
    /// fetched from SSD every token. That makes the curve a cliff rather than a
    /// slope. Measured, 4-bit, short prompt, bounded:
    ///
    ///      16 slots  1.05 GB   8.73 tok/s   io 49.4 ms
    ///      32 slots  2.11 GB   8.94         io 41.3
    ///      64 slots  4.22 GB   9.91         io 28.3
    ///     128 slots  8.44 GB  18.91         io  7.2
    ///
    /// A smaller budget does not trade throughput gently for memory -- it falls off
    /// by 2.2x while saving RAM that the OS would otherwise have to hold anyway.
    ///
    /// This inverts under the page-cache policy, where the OS holds the working set
    /// and slot memory is redundant pressure: 4-bit measured 13.61 tok/s at 16
    /// slots against 8.78 at 128. So this constant is only correct while expert
    /// reads bypass the cache. Re-tune it if that ever changes, and re-tune it at
    /// the shipped `--max-context`, never a reduced one.
    public static let defaultExpertCacheBudgetBytes = 8 << 30

    /// Parses a RAM budget such as `2G`, `512M`, `8GiB` or a plain byte count.
    ///
    /// Accepts the sizes users actually type. Returns nil for anything
    /// unparseable or non-positive, so a typo becomes an argument error rather
    /// than a silently tiny cache.
    public static func parseBudgetBytes(_ text: String) -> Int? {
        let raw = text.trimmingCharacters(in: .whitespaces).uppercased()
        guard !raw.isEmpty else { return nil }
        let multipliers: [(String, Int)] = [
            ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
            ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
            ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10),
        ]
        for (suffix, scale) in multipliers where raw.hasSuffix(suffix) {
            let number = String(raw.dropLast(suffix.count))
                .trimmingCharacters(in: .whitespaces)
            guard let value = Double(number), value > 0 else { return nil }
            let bytes = value * Double(scale)
            guard bytes.isFinite, bytes >= 1, bytes < Double(Int.max) else { return nil }
            return Int(bytes)
        }
        guard let plain = Int(raw), plain > 0 else { return nil }
        return plain
    }

    /// Slots that fit `budgetBytes`, snapped to the nearest supported count.
    ///
    /// Deriving from the stride rather than hard-coding a number per quantisation
    /// keeps 4-bit and 8-bit on the same rule: 1 GiB lands on 16 slots at a
    /// 1.688 MiB stride and 8 slots at 3.188 MiB, which are the measured optima
    /// for each.
    public static func expertCacheSlots(
        expertStrideBytes: UInt64,
        layers: Int,
        budgetBytes: Int = defaultExpertCacheBudgetBytes) -> Int {
        guard expertStrideBytes > 0, layers > 0 else {
            return allowedExpertCacheSlots.first ?? 8
        }
        let perSlot = Double(expertStrideBytes) * Double(layers)
        let wanted = Double(budgetBytes) / perSlot
        return allowedExpertCacheSlots.min {
            abs(Double($0) - wanted) < abs(Double($1) - wanted)
        } ?? 8
    }
    public static let allowedPrefillChunkTokens = [
        32, 64, 128, 256, 512, 1_024, 2_048, 4_096,
    ]
    public static let qwenLongPrefillChunkTokens = 4_096

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    public let prefetch: RuntimePrefetch
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let yarnContextTokens: Int

    public init(expertCacheSlots: Int = 64,
                expertCachePolicy: RuntimeExpertCachePolicy = .agingLFU,
                rdadvisePolicy: RDAdvicePolicyMode = .default,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .causalMatrix,
                forceLogitsHead: Bool = false,
                prefetch: RuntimePrefetch = .production,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none,
                yarnContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens) throws {
        guard Self.allowedExpertCacheSlots.contains(expertCacheSlots) else {
            throw RuntimeConfigurationError.invalidExpertCacheSlots(expertCacheSlots)
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw RuntimeConfigurationError.invalidPrefillChunkTokens(prefillChunkTokens)
        }
        guard Self.supportedYaRNContextTokens.contains(yarnContextTokens) else {
            throw RuntimeConfigurationError.invalidYaRNContextTokens(yarnContextTokens)
        }
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.prefetch = prefetch
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.yarnContextTokens = yarnContextTokens
    }

    public func validate(maxContext: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        switch ropeScalingMode {
        case .none:
            guard maxContext <= Self.nativeMaximumContextTokens else {
                throw RuntimeConfigurationError.contextRequiresYaRN(maxContext)
            }
        case .yarn:
            guard maxContext == yarnContextTokens else {
                throw RuntimeConfigurationError.yaRNContextMismatch(
                    maxContext: maxContext, configured: yarnContextTokens)
            }
        }
    }

    public static var production: RuntimeConfiguration {
        // Every default is a compile-time constant on the allowed lists, so
        // the validating init cannot throw here; RuntimeConfigurationTests
        // pins that.
        // swiftlint:disable:next force_try
        try! RuntimeConfiguration()
    }

    /// Production pins the sliding-window ring on. This is deliberately a
    /// constant and not a stored option: `KVCacheManager` takes the flag as a
    /// real parameter (tests construct it both ways to cover the non-ring
    /// path), but the shipping runtime has exactly one supported setting, and
    /// the value is part of `ServerPromptCacheDomain` — making it settable
    /// would let two processes disagree about the layout of a persisted KV
    /// snapshot. Read-only here is the guarantee, not an oversight.
    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        switch expertCachePolicy {
        case .lru: .lru
        case .lfu: .lfu
        case .agingLFU: .agingLFU
        }
    }
}
