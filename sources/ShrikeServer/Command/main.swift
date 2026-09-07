import Darwin
import Foundation
import Shrike
import ShrikeServerCore

let arguments: ServerArguments
do {
    arguments = try ServerArguments.parse(Array(CommandLine.arguments.dropFirst()))
} catch ServerArgumentError.help {
    print(ServerArguments.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(ServerArguments.usage)\n".utf8))
    exit(2)
}

do {
    try RuntimeConfiguration.refuseUnknownEnvironment()
    let signals = ServerTerminationSignals()

    let config: ServerConfig
    let roster: ModelRoster
    var skippedBundles: [ModelRoster.SkippedBundle] = []
    if let modelPath = arguments.model {
        // --model serves exactly this one; no config file is read at all.
        config = ServerConfig()
        roster = try ModelRoster.single(
            directory: URL(fileURLWithPath: modelPath).standardizedFileURL,
            overrideID: arguments.modelIDOverride)
    } else {
        if let path = arguments.configPath {
            config = try ServerConfig.load(path: path)
        } else {
            let defaultPath = ("~/.shrike/server.json" as NSString).expandingTildeInPath
            config = FileManager.default.fileExists(atPath: defaultPath)
                ? try ServerConfig.load(path: defaultPath)
                : ServerConfig()
        }
        let modelsDir = arguments.modelsDir ?? config.modelsDir ?? "~/shrike-runtime/models"
        let directory = URL(
            fileURLWithPath: (modelsDir as NSString).expandingTildeInPath).standardizedFileURL
        let scan = try ModelRoster.scanBundles(in: directory)
        skippedBundles = scan.skipped
        roster = try ModelRoster.resolve(candidates: scan.candidates, overrides: config.models)
    }

    let effective = try arguments.merging(configDefaults: config.defaults)
    // Reads each manifest.json only; a broken bundle fails here at launch
    // rather than on its first request.
    let models = try ModelRegistry.models(for: roster, arguments: effective)
    let registry = ModelRegistry(
        models: models,
        roster: roster,
        idleTimeout: effective.idleUnloadSeconds > 0
            ? .seconds(effective.idleUnloadSeconds) : nil)

    for bundle in skippedBundles {
        FileHandle.standardError.write(
            Data("warning: skipping \(bundle.bundleName).gturbo: \(bundle.reason)\n".utf8))
    }
    for name in roster.droppedBundles {
        FileHandle.standardError.write(
            Data("notice: \(name).gturbo is an MTP sidecar the runtime no longer consumes; excluded from the roster\n".utf8))
    }

    if effective.preload {
        guard let defaultModel = registry.model(for: nil) else {
            throw ServerArgumentError.invalid(
                "--preload needs a default model; mark one with \"default\": true in the config")
        }
        try await registry.preload(defaultModel)
    }

    let server = ShrikeHTTPServer(
        registry: registry,
        queueLimit: effective.queueLimit)
    _ = try await server.start(port: effective.port)

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

    _ = await signals.wait()
    try await server.shutdown()
    // After the server, so the reaper cannot outlive it.
    await registry.shutdown()
    await signals.cancel()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
