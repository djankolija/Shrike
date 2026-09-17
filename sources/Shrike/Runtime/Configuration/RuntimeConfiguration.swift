import Foundation

public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
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
    case unknownEnvironment([String])
    case invalidExpertSlotTable(String)
    case invalidExpertPolicy(String)

    public var description: String {
        switch self {
        case .invalidExpertSlotTable(let detail):
            return "SHRIKE_EXPERT_SLOT_TABLE refused: \(detail)"
        case .invalidExpertPolicy(let detail):
            return "SHRIKE_EXPERT_POLICY refused: \(detail); allowed: aging-lfu, slru, slru:<share in (0, 1)>"
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
            return "the prefetch trace could not be opened: \(detail)"
        case .unknownEnvironment(let names):
            return names.joined(separator: ", ")
                + ": not read by this build (removed in v17; docs/v17-consolidation.md is the record)"
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
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let headPath: RuntimeHeadPath
    public let prefetchTracePath: String?
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let yarnContextTokens: Int
    /// Off in production: a Qwen-family model outside the streaming scan's shape is refused, not served slower.
    public let attentionFallbackAllowed: Bool

    public init(expertCacheSlots: Int = 64,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                forceLogitsHead: Bool = false,
                prefetchTracePath: String? = nil,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none,
                yarnContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens,
                attentionFallbackAllowed: Bool = false) throws {
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
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.prefetchTracePath = prefetchTracePath
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
        self.yarnContextTokens = yarnContextTokens
        self.attentionFallbackAllowed = attentionFallbackAllowed
    }

    public static func environmentPrefetchTracePath(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        environment["SHRIKE_PREFETCH_TRACE"].flatMap { $0.isEmpty ? nil : $0 }
    }

    /// The per-layer slot table from `SHRIKE_EXPERT_SLOT_TABLE`, one count per
    /// layer separated by commas or the path of a JSON object keyed by layer
    /// index; a malformed table is refused, never replaced by the uniform count.
    public static func environmentExpertSlotTable(
        _ environment: [String: String] = ProcessInfo.processInfo.environment,
        layers: Int, uniformSlots: Int, leadingDenseLayers: Int
    ) throws -> [Int]? {
        guard let raw = environment["SHRIKE_EXPERT_SLOT_TABLE"], !raw.isEmpty else { return nil }
        let counts: [Int]
        if raw.contains("/") || raw.hasSuffix(".json") {
            guard let data = FileManager.default.contents(atPath: raw),
                  let map = try? JSONDecoder().decode([String: Int].self, from: data) else {
                throw RuntimeConfigurationError.invalidExpertSlotTable(
                    "\(raw) is not a readable JSON object of layer index to slot count")
            }
            guard map.count == layers, (0..<layers).allSatisfy({ map[String($0)] != nil }) else {
                throw RuntimeConfigurationError.invalidExpertSlotTable(
                    "\(raw) has \(map.count) entries for \(layers) layers")
            }
            counts = (0..<layers).map { map[String($0)] ?? 0 }
        } else {
            let fields = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            let parsed = fields.compactMap { Int($0) }
            guard parsed.count == fields.count else {
                throw RuntimeConfigurationError.invalidExpertSlotTable("not every entry is an integer")
            }
            counts = parsed
        }
        try validateExpertSlotTable(counts, layers: layers, uniformSlots: uniformSlots,
                                    leadingDenseLayers: leadingDenseLayers)
        return counts
    }

    /// The pool's eviction policy from `SHRIKE_EXPERT_POLICY`: `aging-lfu`
    /// (the default when unset), `slru` at a protected share of 0.5, or
    /// `slru:<share>`; anything else is refused.
    public static func environmentExpertPolicy(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> ExpertEvictionPolicy {
        guard let raw = environment["SHRIKE_EXPERT_POLICY"]?.lowercased(), !raw.isEmpty else {
            return .agingLFU
        }
        if raw == "aging-lfu" { return .agingLFU }
        if raw == "slru" { return .slru(protectedShare: 0.5) }
        if raw.hasPrefix("slru:"),
           let share = Double(raw.dropFirst("slru:".count)), share > 0, share < 1 {
            return .slru(protectedShare: share)
        }
        throw RuntimeConfigurationError.invalidExpertPolicy(raw)
    }

    public static let minimumExpertSlotsPerLayer = 8

    static func validateExpertSlotTable(_ counts: [Int], layers: Int, uniformSlots: Int,
                                        leadingDenseLayers: Int) throws {
        guard counts.count == layers else {
            throw RuntimeConfigurationError.invalidExpertSlotTable(
                "\(counts.count) entries for \(layers) layers")
        }
        for (layer, count) in counts.enumerated() {
            if layer < leadingDenseLayers {
                guard count == 0 else {
                    throw RuntimeConfigurationError.invalidExpertSlotTable(
                        "dense layer \(layer) must have 0 slots, has \(count)")
                }
            } else if count < minimumExpertSlotsPerLayer {
                throw RuntimeConfigurationError.invalidExpertSlotTable(
                    "layer \(layer) has \(count) slots, fewer than \(minimumExpertSlotsPerLayer)")
            }
        }
        let budget = uniformSlots * (layers - leadingDenseLayers)
        let total = counts.reduce(0, +)
        guard total == budget else {
            throw RuntimeConfigurationError.invalidExpertSlotTable(
                "the table totals \(total) slots, the budget affords \(budget) "
                + "(\(uniformSlots) per routed layer)")
        }
    }

    public static let knownEnvironmentNames: Set<String> = [
        "SHRIKE_THINKING_MODE", "SHRIKE_REASONING_EFFORT", "SHRIKE_REASONING_RETENTION",
        "SHRIKE_STRIP_CLI_PROMPT", "SHRIKE_STRIP_TAGS", "SHRIKE_CONCISE_MODE",
        "SHRIKE_TOKENIZER_DIR", "SHRIKE_MODEL", "SHRIKE_PREFILL_ANE",
        "SHRIKE_RUNNER_STATS", "SHRIKE_KERNEL_STATS", "SHRIKE_ROUTE_TRACE",
        "SHRIKE_PREFETCH_TRACE", "SHRIKE_EXPERT_SLOT_TABLE", "SHRIKE_EXPERT_POLICY",
    ]

    /// Fails the launch by name on any `SHRIKE_*` variable this build does not read.
    public static func refuseUnknownEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws {
        let unknown = environment.keys.filter {
            $0.hasPrefix("SHRIKE_") && !knownEnvironmentNames.contains($0)
        }.sorted()
        guard unknown.isEmpty else {
            throw RuntimeConfigurationError.unknownEnvironment(unknown)
        }
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
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
}
