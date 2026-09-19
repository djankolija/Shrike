import ArgumentParser
import Foundation
import Shrike

/// Thrown for a launch that parsed cleanly and then failed: ArgumentParser exits
/// 1 on a plain error, where a `ValidationError` would exit 64 and print the usage.
struct ServerLaunchError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) { self.description = description }
}

extension ShrikeServerCommand {
    struct ResolvedRoster {
        let config: ServerConfig
        let roster: ModelRoster
        let skipped: [ModelRoster.SkippedBundle]
    }

    public func run() async throws {
        try RuntimeConfiguration.refuseUnknownEnvironment()
        try Self.validateEnvironment(ProcessInfo.processInfo.environment)
        let signals = ServerTerminationSignals()
        let resolved = try resolveRoster()
        let effective = try merging(configDefaults: resolved.config.defaults)
        let registry = try makeRegistry(roster: resolved.roster, effective: effective)
        report(skipped: resolved.skipped, roster: resolved.roster)
        if effective.preload { try await preloadDefault(in: registry) }
        try await serve(registry: registry, effective: effective,
                        roster: resolved.roster, signals: signals)
    }

    private func resolveRoster() throws -> ResolvedRoster {
        // --model serves exactly this one; no config file is read at all.
        if let modelPath = model {
            return ResolvedRoster(
                config: ServerConfig(),
                roster: try ModelRoster.single(
                    directory: URL(fileURLWithPath: modelPath).standardizedFileURL,
                    overrideID: modelIDOverride),
                skipped: [])
        }
        let config: ServerConfig
        if let path = configPath {
            config = try ServerConfig.load(path: path)
        } else {
            let defaultPath = ("~/.shrike/server.json" as NSString).expandingTildeInPath
            config = FileManager.default.fileExists(atPath: defaultPath)
                ? try ServerConfig.load(path: defaultPath)
                : ServerConfig()
        }
        let directory = URL(fileURLWithPath:
            ((modelsDir ?? config.modelsDir ?? "~/shrike-runtime/models") as NSString)
                .expandingTildeInPath).standardizedFileURL
        let scan = try ModelRoster.scanBundles(in: directory)
        return ResolvedRoster(
            config: config,
            roster: try ModelRoster.resolve(candidates: scan.candidates,
                                            overrides: config.models),
            skipped: scan.skipped)
    }

    private func makeRegistry(roster: ModelRoster,
                              effective: ShrikeServerCommand) throws -> ModelRegistry {
        // Reads each manifest.json only; a broken bundle fails here at launch
        // rather than on its first request.
        let models = try ModelRegistry.models(for: roster, arguments: effective)
        return ModelRegistry(
            models: models,
            roster: roster,
            idleTimeout: effective.idleUnloadSeconds > 0
                ? .seconds(effective.idleUnloadSeconds) : nil)
    }

    private func report(skipped: [ModelRoster.SkippedBundle], roster: ModelRoster) {
        for bundle in skipped {
            FileHandle.standardError.write(
                Data("warning: skipping \(bundle.bundleName).gturbo: \(bundle.reason)\n".utf8))
        }
        for name in roster.droppedBundles {
            FileHandle.standardError.write(
                Data("notice: \(name).gturbo is an MTP sidecar the runtime no longer consumes; excluded from the roster\n".utf8))
        }
    }

    private func preloadDefault(in registry: ModelRegistry) async throws {
        guard let defaultModel = registry.model(for: nil) else {
            throw ServerLaunchError(
                "--preload needs a default model; mark one with \"default\": true in the config")
        }
        try await registry.preload(defaultModel)
    }

    private func serve(registry: ModelRegistry,
                       effective: ShrikeServerCommand,
                       roster: ModelRoster,
                       signals: ServerTerminationSignals) async throws {
        let server = ShrikeHTTPServer(registry: registry, queueLimit: effective.queueLimit)
        _ = try await server.start(port: effective.port)
        announce(registry: registry, effective: effective, roster: roster)
        _ = await signals.wait()
        try await server.shutdown()
        // After the server, so the reaper cannot outlive it.
        await registry.shutdown()
        await signals.cancel()
    }

    private func announce(registry: ModelRegistry,
                          effective: ShrikeServerCommand,
                          roster: ModelRoster) {
        let diskCache = effective.promptCacheMode == .off
            ? "off" : effective.promptCacheDiskDirectory ?? "off"
        let cacheMemoryMiB = effective.promptCacheMode == .off
            ? 0 : effective.promptCacheMemoryMiB
        let idle = effective.idleUnloadSeconds > 0 ? "\(effective.idleUnloadSeconds)s" : "off"
        print("ShrikeServer ready at http://127.0.0.1:\(effective.port) models=\(registry.ids.joined(separator: ",")) default=\(roster.defaultID ?? "none") context=\(effective.maxContext) prompt_cache=\(effective.promptCacheMode.rawValue) prompt_cache_memory_mib=\(cacheMemoryMiB) prompt_cache_disk=\(diskCache) thinking=\(effective.thinkingMode.rawValue) reasoning_effort=\(effective.reasoningEffort?.rawValue ?? "auto") reasoning_retention=\(effective.reasoningRetention?.rawValue ?? "as-generated") idle_unload=\(idle) preload=\(effective.preload ? "on" : "off")")
        if effective.unloadDiscardsWarmCache {
            FileHandle.standardError.write(Data(
                ("warning: --idle-unload-seconds drops the in-memory prompt cache with "
                    + "the model; add --prompt-cache-disk <dir> so entries survive an "
                    + "unload, or the first request after each unload pays a full "
                    + "cold prefill\n").utf8))
        }
    }
}
