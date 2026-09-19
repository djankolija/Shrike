import ArgumentParser
import Foundation
import Shrike
import ShrikeCatalog

extension ShrikeServerCommand {
    struct ResolvedRoster {
        let config: ShrikeConfig
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
        try await serve(registry: registry, effective: effective,
                        roster: resolved.roster, signals: signals)
    }

    private func resolveRoster() throws -> ResolvedRoster {
        // --model serves exactly this one; no config file is read at all.
        if let modelPath = model {
            return ResolvedRoster(
                config: ShrikeConfig(),
                roster: try ModelRoster.single(
                    directory: URL(fileURLWithPath: modelPath).standardizedFileURL),
                skipped: [])
        }
        let config: ShrikeConfig
        if let path = configPath {
            config = try ShrikeConfig.load(path: path)
        } else {
            let defaultPath = ("~/.shrike/config.json" as NSString).expandingTildeInPath
            config = FileManager.default.fileExists(atPath: defaultPath)
                ? try ShrikeConfig.load(path: defaultPath)
                : ShrikeConfig()
        }
        let directory = URL(fileURLWithPath:
            ((config.modelsDir ?? "~/shrike-runtime/models") as NSString)
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
        return ModelRegistry(models: models, roster: roster)
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

    private func serve(registry: ModelRegistry,
                       effective: ShrikeServerCommand,
                       roster: ModelRoster,
                       signals: ServerTerminationSignals) async throws {
        let server = ShrikeHTTPServer(registry: registry)
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
        print("ShrikeServer ready at http://127.0.0.1:\(effective.port) models=\(registry.ids.joined(separator: ",")) default=\(roster.defaultID ?? "none") context=\(effective.maxContext) thinking=\(effective.thinkingMode.rawValue) reasoning_effort=\(effective.reasoningEffort?.rawValue ?? "auto")")
    }
}
