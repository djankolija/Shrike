import Foundation
import Shrike

public enum AppModelVerification: String, CaseIterable, Sendable, Identifiable {
    case fullSha256 = "full-sha256"
    case trustedInstall = "trusted-install"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fullSha256: return "Full SHA-256"
        case .trustedInstall: return "Trust verified install"
        }
    }

    var runtimeValue: ModelIntegrityPolicy {
        switch self {
        case .fullSha256: return .fullSha256
        case .trustedInstall: return .sizeCheckTrustedReceipt
        }
    }
}

public struct AppRuntimeOptions: Equatable, Sendable {
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    public static let allowedPrefillChunkTokens =
        RuntimeConfiguration.allowedPrefillChunkTokens

    public var expertCacheSlots: Int
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var modelVerification: AppModelVerification
    public var conciseMode: Bool
    public var thinkingMode: ModelThinkingMode
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode

    public init(expertCacheSlots: Int = 64,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = RuntimeConfiguration.qwenLongPrefillChunkTokens,
                modelVerification: AppModelVerification = .fullSha256,
                conciseMode: Bool = false,
                thinkingMode: ModelThinkingMode = .off,
                kvCachePrecision: KVCachePrecision = .int8,
                ropeScalingMode: RuntimeRoPEScalingMode = .none) {
        self.expertCacheSlots = expertCacheSlots
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.modelVerification = modelVerification
        self.conciseMode = conciseMode
        self.thinkingMode = thinkingMode
        self.kvCachePrecision = kvCachePrecision
        self.ropeScalingMode = ropeScalingMode
    }

    public func validate() throws {
        guard Self.allowedSlotCounts.contains(expertCacheSlots) else {
            throw AppInferenceError.invalidRequest(
                "expert cache slots must be one of \(Self.allowedSlotCounts)")
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw AppInferenceError.invalidRequest(
                "prefill chunk size must be one of \(Self.allowedPrefillChunkTokens)")
        }
    }

    public var prefillConfig: PrefillRuntimeConfig {
        prefillEnabled ? .production(chunkTokens: prefillChunkTokens) : .off
    }

    public var resultSummary: String {
        let prefill = prefillEnabled ? "prefill \(prefillChunkTokens)" : "prefill off"
        let verification = modelVerification == .fullSha256 ? "full SHA-256" : "trusted receipt"
        let scaling = ropeScalingMode == .yarn ? "YaRN" : "native RoPE"
        return "Cache \(expertCacheSlots), \(prefill), \(kvCachePrecision.label) KV, \(scaling), thinking \(thinkingMode.rawValue), \(verification)"
    }

    public static func slotsLabel(for slots: Int) -> String {
        // D21: the per-slot memory deltas were hardcoded and drifted from the
        // real pack layout, so the picker shows names only; the memory
        // implications are described in the UI text next to the picker.
        "\(slots)"
    }

    public func resolvedRuntimeConfiguration(
        forceLogitsHead: Bool,
        maxContextTokens: Int = RuntimeConfiguration.defaultYaRNContextTokens
    ) throws -> RuntimeConfiguration {
        try validate()
        return try RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead,
            kvCachePrecision: kvCachePrecision,
            ropeScalingMode: ropeScalingMode,
            yarnContextTokens: ropeScalingMode == .yarn
                ? maxContextTokens : RuntimeConfiguration.defaultYaRNContextTokens)
    }
}

public struct AppLoadedRuntimeKey: Equatable, Sendable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var expertCacheSlots: Int
    public var modelVerification: AppModelVerification
    public var forceLogitsHead: Bool
    public var kvCachePrecision: KVCachePrecision
    public var ropeScalingMode: RuntimeRoPEScalingMode
    public var thinkingMode: ModelThinkingMode

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                options: AppRuntimeOptions,
                forceLogitsHead: Bool = false) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.expertCacheSlots = options.expertCacheSlots
        self.modelVerification = options.modelVerification
        self.forceLogitsHead = forceLogitsHead
        self.kvCachePrecision = options.kvCachePrecision
        self.ropeScalingMode = options.ropeScalingMode
        self.thinkingMode = options.thinkingMode
    }
}
