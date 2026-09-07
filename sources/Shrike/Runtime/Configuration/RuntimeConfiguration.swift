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

/// Decode scheduling for SSD-backed routed experts.
///
/// `hitFixup` commits phase 1 for resident experts while cache misses are read,
/// then computes only the missed experts before the common reduction. `barrier`
/// preserves the former all-experts-after-I/O path as a correctness/performance
/// control for A/B measurements.
public enum RuntimeDecodeExpertExecution: String, Codable, Sendable {
    case hitFixup = "hit-fixup"
    case barrier
    case gpuResidency = "gpu-residency"
    case speculative
    case speculativeValidate = "speculative-validate"

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeDecodeExpertExecution {
        guard let raw = environment["SHRIKE_DECODE_EXPERT_EXECUTION"] else {
            return .speculative
        }
        guard let value = RuntimeDecodeExpertExecution(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidDecodeExpertExecution(raw)
        }
        return value
    }
}

public enum RuntimeExpertIOSynchronization: String, Codable, Sendable {
    case host
    case event

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSynchronization {
        guard let raw = environment["SHRIKE_EXPERT_IO_SYNC"] else { return .event }
        guard let value = RuntimeExpertIOSynchronization(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSynchronization(raw)
        }
        return value
    }
}

/// Which routed experts the speculative command computes in phase 1: only on
/// an all-hit layer (the classifier zeroes its grids otherwise), or the
/// GPU-classified hits on every layer while the fixup keeps the misses.
public enum RuntimeSpecPhase1Coverage: String, Codable, Sendable {
    case allHit = "all-hit"
    case hits

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeSpecPhase1Coverage {
        guard let raw = environment["SHRIKE_SPEC_PHASE1"] else { return .allHit }
        guard let value = RuntimeSpecPhase1Coverage(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidSpecPhase1Coverage(raw)
        }
        return value
    }
}

/// How the host learns that a routed layer's router has run: the residency
/// classifier's tagged host readback polled directly (the default), or the
/// command's completion mark, which the driver publishes later. The poll needs
/// the spin host wait; under `SHRIKE_HOST_WAIT=wait` the status path's parked
/// wait is taken instead.
public enum RuntimeRouterWake: String, Codable, Sendable {
    case status
    case word

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeRouterWake {
        guard let raw = environment["SHRIKE_ROUTER_WAKE"] else { return .word }
        guard let value = RuntimeRouterWake(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidRouterWake(raw)
        }
        return value
    }
}

/// Where the predictive prefetch ring issues a layer's reads: `after` the
/// layer's demand batch has completed (the drive is otherwise idle and the
/// demand read is never slowed), or `beside` it, right after the demand
/// batch's submission (v14 T1's shape).
public enum RuntimePrefetchPlacement: String, Codable, Sendable {
    case after
    case beside
}

/// How an adopted prediction reaches its cache slot: a host `memcpy` at plan
/// time (`copy`), or a GPU blit at the head of the fixup command that computes
/// it (`blit`, only where the fixup computes adopted experts).
public enum RuntimePrefetchAdoption: String, Codable, Sendable {
    case copy
    case blit
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
    public let adoption: RuntimePrefetchAdoption
    /// How long a plan waits for a prediction still in flight before reading
    /// the expert itself; 0 never waits.
    public let joinMicros: Int
    public let probe: RuntimePrefetchProbe

    public static let allowedInFlight = 1...8
    public static let allowedDistance = 1...8
    public static let allowedJoinMicros = 0...2000

    public static let off = RuntimePrefetch(enabled: false, topM: nil, inFlight: 1,
                                            placement: .after, distance: 1, tracePath: nil)
    public static let production = RuntimePrefetch(enabled: true, topM: nil, inFlight: 1,
                                                   placement: .after, distance: 1, tracePath: nil)

    public init(enabled: Bool, topM: Int?, inFlight: Int, placement: RuntimePrefetchPlacement,
                distance: Int, tracePath: String?, adoption: RuntimePrefetchAdoption = .blit,
                joinMicros: Int = 400, probe: RuntimePrefetchProbe = .fused) {
        self.enabled = enabled
        self.topM = topM
        self.inFlight = inFlight
        self.placement = placement
        self.distance = distance
        self.tracePath = tracePath
        self.adoption = adoption
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
        let adoption: RuntimePrefetchAdoption
        if let raw = environment["SHRIKE_PREFETCH_ADOPT"] {
            guard let value = RuntimePrefetchAdoption(rawValue: raw) else {
                throw RuntimeConfigurationError.invalidPrefetch(
                    "SHRIKE_PREFETCH_ADOPT '\(raw)'; allowed: copy, blit")
            }
            adoption = value
        } else {
            adoption = production.adoption
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
                               adoption: adoption, joinMicros: joinMicros, probe: probe)
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

public enum RuntimeExpertIOSubmission: String, Codable, Sendable {
    case deferred
    case immediate

    public static func environmentValue(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RuntimeExpertIOSubmission {
        guard let raw = environment["SHRIKE_EXPERT_IO_SUBMISSION"] else { return .immediate }
        guard let value = RuntimeExpertIOSubmission(rawValue: raw) else {
            throw RuntimeConfigurationError.invalidExpertIOSubmission(raw)
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
    case invalidDecodeExpertExecution(String)
    case invalidExpertIOSynchronization(String)
    case invalidExpertIOSubmission(String)
    case invalidSpecPhase1Coverage(String)
    case invalidRouterWake(String)
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
        case .invalidDecodeExpertExecution(let value):
            return "unsupported decode expert execution '\(value)'; allowed: hit-fixup, barrier, gpu-residency"
        case .invalidExpertIOSynchronization(let value):
            return "unsupported expert I/O synchronization '\(value)'; allowed: host, event"
        case .invalidExpertIOSubmission(let value):
            return "unsupported expert I/O submission '\(value)'; allowed: deferred, immediate"
        case .invalidSpecPhase1Coverage(let value):
            return "unsupported spec phase-1 coverage '\(value)'; allowed: all-hit, hits"
        case .invalidRouterWake(let value):
            return "unsupported router wake '\(value)'; allowed: status, word"
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
    public let decodeExpertExecution: RuntimeDecodeExpertExecution
    public let expertIOSynchronization: RuntimeExpertIOSynchronization
    public let expertIOSubmission: RuntimeExpertIOSubmission
    public let specPhase1Coverage: RuntimeSpecPhase1Coverage
    public let routerWake: RuntimeRouterWake
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
                decodeExpertExecution: RuntimeDecodeExpertExecution = .speculative,
                expertIOSynchronization: RuntimeExpertIOSynchronization = .event,
                expertIOSubmission: RuntimeExpertIOSubmission = .immediate,
                specPhase1Coverage: RuntimeSpecPhase1Coverage = .allHit,
                routerWake: RuntimeRouterWake = .word,
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
        self.decodeExpertExecution = decodeExpertExecution
        self.expertIOSynchronization = expertIOSynchronization
        self.expertIOSubmission = expertIOSubmission
        self.specPhase1Coverage = specPhase1Coverage
        self.routerWake = routerWake
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
