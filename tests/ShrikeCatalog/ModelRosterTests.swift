import Foundation
import Testing

import Shrike
import ShrikeCatalog

private func candidate(_ bundleName: String,
                       manifest: String = "unknown/snapshot",
                       family: ModelFamily = .qwen36) -> RosterCandidate {
    RosterCandidate(bundleName: bundleName,
                    directory: URL(fileURLWithPath: "/models/\(bundleName).gturbo"),
                    manifestModelID: manifest,
                    family: family)
}

@Suite struct ModelRosterTests {
    @Test func configuredIDWinsAndBundleNameRemainsAnAlias() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("kimi-linear-48b-a3b-4bit",
                                   manifest: "mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit",
                                   family: .kimiLinear48b)],
            overrides: [.init(dir: "kimi-linear-48b-a3b-4bit.gturbo", id: "kimi-linear-48b-a3b")])
        #expect(roster.ids == ["kimi-linear-48b-a3b"])
        #expect(roster.canonicalID(for: "kimi-linear-48b-a3b") == "kimi-linear-48b-a3b")
        #expect(roster.canonicalID(for: "kimi-linear-48b-a3b-4bit") == "kimi-linear-48b-a3b")
        #expect(roster.entry(for: "kimi-linear-48b-a3b-4bit")?.bundleName == "kimi-linear-48b-a3b-4bit")
    }

    @Test func unconfiguredBundleServesUnderItsOwnName() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("qwen36"), candidate("ornith15")],
            overrides: [])
        #expect(roster.ids == ["ornith15", "qwen36"])
        #expect(roster.canonicalID(for: "qwen36") == "qwen36")
        #expect(roster.canonicalID(for: "nonexistent") == nil)
    }

    @Test func overrideDirMatchesWithOrWithoutExtension() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("qwen36")],
            overrides: [.init(dir: "qwen36", id: "qwen3.6-35b-a3b")])
        #expect(roster.ids == ["qwen3.6-35b-a3b"])
    }

    @Test func duplicateIDNamesBothClaimants() throws {
        do {
            _ = try ModelRoster.resolve(
                candidates: [candidate("ornith15"), candidate("qwen36")],
                overrides: [.init(dir: "ornith15.gturbo", id: "qwen36")])
            Issue.record("expected a duplicate id failure")
        } catch let error as ModelRosterError {
            #expect(error == .duplicateID(id: "qwen36", claimants: ["ornith15", "qwen36"]))
        }
    }

    @Test func sameConfiguredIDTwiceIsADuplicate() throws {
        #expect(throws: ModelRosterError.self) {
            try ModelRoster.resolve(
                candidates: [candidate("qwen36"), candidate("ornith15")],
                overrides: [.init(dir: "qwen36.gturbo", id: "shared"),
                            .init(dir: "ornith15.gturbo", id: "shared")])
        }
    }

    @Test func mtpFamiliesAreDroppedByRule() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("qwen36"),
                         candidate("ornith15"),
                         candidate("kimi-linear-48b-a3b-4bit", family: .kimiLinear48b),
                         candidate("gpt-oss-20b-mlx-4bit", family: .gptOss20b),
                         candidate("qwen36-mtp", family: .qwen36MTP),
                         candidate("ornith15-mtp", family: .qwen36MTP)],
            overrides: [])
        #expect(roster.entries.count == 4)
        #expect(roster.droppedBundles == ["ornith15-mtp", "qwen36-mtp"])
        #expect(roster.canonicalID(for: "qwen36-mtp") == nil)
    }

    @Test func overrideOnUnknownDirFails() throws {
        do {
            _ = try ModelRoster.resolve(
                candidates: [candidate("qwen36")],
                overrides: [.init(dir: "qwen37.gturbo", id: "typo")])
            Issue.record("expected an unknown dir failure")
        } catch let error as ModelRosterError {
            #expect(error == .unknownOverrideDir("qwen37.gturbo"))
        }
    }

    @Test func overrideOnDroppedBundleFails() throws {
        do {
            _ = try ModelRoster.resolve(
                candidates: [candidate("qwen36-mtp", family: .qwen36MTP)],
                overrides: [.init(dir: "qwen36-mtp.gturbo", id: "mtp")])
            Issue.record("expected a not-servable failure")
        } catch let error as ModelRosterError {
            #expect(error == .overrideNotServable(dir: "qwen36-mtp.gturbo", family: "qwen36_mtp"))
        }
    }

    @Test func singleModelIsTheImplicitDefault() throws {
        let roster = try ModelRoster.resolve(candidates: [candidate("qwen36")], overrides: [])
        #expect(roster.defaultID == "qwen36")
    }

    @Test func multipleModelsWithoutExplicitDefaultHaveNone() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("qwen36"), candidate("ornith15")],
            overrides: [])
        #expect(roster.defaultID == nil)
    }

    @Test func explicitDefaultWins() throws {
        let roster = try ModelRoster.resolve(
            candidates: [candidate("qwen36"), candidate("gpt-oss-20b-mlx-4bit", family: .gptOss20b)],
            overrides: [.init(dir: "gpt-oss-20b-mlx-4bit.gturbo", id: "gpt-oss-20b", isDefault: true)])
        #expect(roster.defaultID == "gpt-oss-20b")
    }

    @Test func emptyRosterFails() throws {
        do {
            _ = try ModelRoster.resolve(
                candidates: [candidate("qwen36-mtp", family: .qwen36MTP)],
                overrides: [])
            Issue.record("expected an empty roster failure")
        } catch let error as ModelRosterError {
            #expect(error == .emptyRoster)
        }
    }
}

@Suite struct ModelRosterScanTests {
    @Test func scanIdentifiesBundlesAndSkipsUnreadableOnes() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("roster-scan-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }

        let good = root.appendingPathComponent("good.gturbo")
        try FileManager.default.createDirectory(at: good, withIntermediateDirectories: true)
        let manifest = """
        {
          "magic": "GTURBO", "versionMajor": 1, "versionMinor": 1, "flags": {},
          "modelID": "vendor/good-4bit",
          "arch": {
            "hiddenSize": 8, "ffnIntermediate": 8, "moeIntermediateSize": 8,
            "numHeads": 2, "numKVHeads": 1, "numFullKVHeads": 1,
            "headDim": 4, "fullHeadDim": 4, "vocabSize": 16,
            "slidingWindow": 0, "finalLogitSoftcap": 0,
            "ropeTheta": 10000, "fullRopeTheta": 10000, "partialRotaryFactor": 1,
            "numLayers": 1, "numExperts": 2, "topKExperts": 1,
            "tieWordEmbeddings": false, "attentionKEqV": false,
            "hiddenActivation": "silu", "fullAttentionLayerMask": [0],
            "family": "qwen36"
          },
          "files": {},
          "expertsPerLayer": 2, "numLayers": 1, "expertStride": 16384
        }
        """
        try Data(manifest.utf8).write(to: good.appendingPathComponent("manifest.json"))

        let broken = root.appendingPathComponent("broken.gturbo")
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)

        let notABundle = root.appendingPathComponent("notes.txt")
        try Data("ignored".utf8).write(to: notABundle)

        let scan = try ModelRoster.scanBundles(in: root)
        #expect(scan.candidates.map(\.bundleName) == ["good"])
        #expect(scan.candidates.first?.manifestModelID == "vendor/good-4bit")
        #expect(scan.candidates.first?.family == .qwen36)
        #expect(scan.skipped.map(\.bundleName) == ["broken"])
    }

    @Test func scanningAMissingDirectoryFails() throws {
        #expect(throws: ModelRosterError.self) {
            try ModelRoster.scanBundles(
                in: URL(fileURLWithPath: "/nonexistent/models-\(UUID().uuidString)"))
        }
    }
}
