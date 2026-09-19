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
}
