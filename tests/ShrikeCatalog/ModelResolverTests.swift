import Foundation
import Testing

import Shrike
import ShrikeCatalog

private func candidate(_ bundleName: String,
                       family: ModelFamily = .qwen36) -> RosterCandidate {
    RosterCandidate(bundleName: bundleName,
                    directory: URL(fileURLWithPath: "/models/\(bundleName).gturbo"),
                    manifestModelID: "unknown/snapshot",
                    family: family)
}

@Suite struct ModelResolverTests {
    @Test func anExplicitPathWinsOverEverything() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("resolver-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let resolved = try ModelResolver.resolve(requested: directory.path)
        #expect(resolved == directory.standardizedFileURL)
    }

    @Test func anIdResolvesToItsBundleDirectory() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("ornith15"), candidate("qwen36")], overrides: [])
        let resolved = try ModelResolver.resolve(requested: "ornith15", in: roster)
        #expect(resolved.lastPathComponent == "ornith15.gturbo")
    }

    @Test func aConfiguredDefaultIsUsedWhenNoModelIsNamed() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("ornith15"), candidate("qwen36")],
            overrides: [.init(dir: "qwen36.gturbo", isDefault: true)])
        let resolved = try ModelResolver.resolve(requested: nil, in: roster)
        #expect(resolved.lastPathComponent == "qwen36.gturbo")
    }

    @Test func theSoleBundleIsTheDefaultWithoutAnyConfiguration() throws {
        let roster = try ModelRoster.resolve(candidates: [candidate("ornith15")], overrides: [])
        let resolved = try ModelResolver.resolve(requested: nil, in: roster)
        #expect(resolved.lastPathComponent == "ornith15.gturbo")
    }

    @Test func severalBundlesAndNoDefaultIsAnErrorNamingThem() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("ornith15"), candidate("qwen36")], overrides: [])
        #expect(throws: ModelResolutionError.noDefault(available: ["ornith15", "qwen36"])) {
            try ModelResolver.resolve(requested: nil, in: roster)
        }
    }

    @Test func anUnknownIdIsAnErrorNamingWhatIsInstalled() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("ornith15"), candidate("qwen36")], overrides: [])
        #expect(throws: ModelResolutionError.unknownModel(name: "nope",
                                                          available: ["ornith15", "qwen36"])) {
            try ModelResolver.resolve(requested: "nope", in: roster)
        }
    }

    @Test func aBundleNameStaysAnAliasForAConfiguredId() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("ornith15")],
            overrides: [.init(dir: "ornith15.gturbo", id: "ornith")])
        #expect(try ModelResolver.resolve(requested: "ornith", in: roster)
            .lastPathComponent == "ornith15.gturbo")
        #expect(try ModelResolver.resolve(requested: "ornith15", in: roster)
            .lastPathComponent == "ornith15.gturbo")
    }

    @Test func theDefaultConfigPathIsTheRenamedOne() {
        #expect(ModelResolver.defaultConfigPath == "~/.shrike/config.json")
    }

    @Test func anEmptyModelsDirectoryNamesTheDirectoryItScanned() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("resolver-empty-\(UUID().uuidString)")
        let config = directory.appendingPathExtension("json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(at: config)
        }
        try Data(#"{"models_dir": "\#(directory.path)"}"#.utf8).write(to: config)

        #expect(throws: ModelResolutionError.emptyCatalog(
            directory: directory.standardizedFileURL.path)) {
            try ModelResolver.resolve(requested: nil, configPath: config.path)
        }
    }

    @Test func theModelsDirectoryPrefersTheConfigThenTheDefault() throws {
        #expect(ModelResolver.modelsDirectory(config: ShrikeConfig(modelsDir: "/from-config")).path
            == "/from-config")
        #expect(ModelResolver.modelsDirectory(config: ShrikeConfig()).path
            == (ModelResolver.defaultModelsDirectory as NSString).expandingTildeInPath)
    }

    @Test func anExplicitConfigPathIsRead() throws {
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("resolver-config-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: path) }
        try Data(#"{"models_dir": "/configured", "models": [{"dir": "a.gturbo", "default": true}]}"#
            .utf8).write(to: path)

        let config = try ModelResolver.loadConfig(path: path.path)
        #expect(config.modelsDir == "/configured")
        #expect(config.models.first?.isDefault == true)
        #expect(ModelResolver.modelsDirectory(config: config).path == "/configured")
    }
}
