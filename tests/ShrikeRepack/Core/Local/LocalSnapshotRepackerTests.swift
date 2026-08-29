import Foundation
import Testing
@testable import ShrikeRepackCore

@Suite struct LocalSnapshotRepackerTests {
    @Test func importsMTPWithExplicitModelIdentityAndReceipt() async throws {
        let root = temporaryRoot("local-mtp-import")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)

        let result = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "ornith-1.5-35b-a3b-mtp-4bit",
                minFreeReserveBytes: 0))

        #expect(result.outputDir == output)
        #expect(result.rangeRequestCount == 0)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifestObject = try JSONSerialization.jsonObject(with: manifestData)
        let manifest = try #require(manifestObject as? [String: Any])
        #expect(manifest["modelID"] as? String
            == "ornith-1.5-35b-a3b-mtp-4bit")
        #expect((manifest["arch"] as? [String: Any])?["family"] as? String
            == "qwen36_mtp")
        #expect(try Posix.entryKind((output as NSString)
            .appendingPathComponent("verified-install.json")) == .regular)
        #expect(try Posix.entryKind((output as NSString)
            .appendingPathComponent("packed_experts/layer_00.bin")) == .regular)
        for sidecar in ["tokenizer/config.json", "tokenizer/tokenizer.json",
                        "tokenizer/tokenizer_config.json"] {
            #expect(try Posix.entryKind((output as NSString)
                .appendingPathComponent(sidecar)) == .regular)
        }
        let receiptData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("verified-install.json")))
        #expect(String(decoding: receiptData, as: UTF8.self)
            .contains("tokenizer/tokenizer.json"))
    }

    @Test func rejectsSnapshotWithoutTokenizerSidecar() async throws {
        let root = temporaryRoot("local-no-tokenizer")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)
        try FileManager.default.removeItem(atPath:
            (snapshot as NSString).appendingPathComponent("tokenizer.json"))

        await #expect(throws: RepackError.self) {
            _ = try await RemoteStreamingRepacker.runLocalSnapshot(
                options: LocalSnapshotRepackOptions(
                    inputSnapshotDir: snapshot,
                    outputDir: output,
                    modelID: "ornith-1.5-35b-a3b-mtp-4bit",
                    minFreeReserveBytes: 0))
        }
    }

    @Test func importsGptOssSnapshotWithBiasSlicesAndNormalizedNames() async throws {
        let root = temporaryRoot("local-gptoss-import")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildGptOss(at: snapshot)

        let result = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "gpt-oss-20b-toy-4bit",
                minFreeReserveBytes: 0))
        #expect(result.outputDir == output)

        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try #require(
            try JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["family"] as? String == "gpt_oss_20b")
        #expect(arch["slidingWindow"] as? Int == 8)
        #expect(arch["topKExperts"] as? Int == 2)
        #expect(arch["fullAttentionLayerMask"] as? [Int] == [0, 1, 0, 1])

        let layoutData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("packed_experts/layout.json")))
        let layout = try #require(
            try JSONSerialization.jsonObject(with: layoutData) as? [String: Any])
        let layers = try #require(layout["layers"] as? [[String: Any]])
        let firstExpert = try #require(
            (layers[0]["experts"] as? [[String: Any]])?.first)
        let tensorKeys = Set(try #require(
            firstExpert["tensors"] as? [String: Any]).keys)
        #expect(tensorKeys == ["gate", "gate_scales", "gate_biases", "gate_bias",
                               "up", "up_scales", "up_biases", "up_bias",
                               "down", "down_scales", "down_biases", "down_bias"])
    }

    @Test func rejectsUnsafeShardPathBeforeCopying() throws {
        let root = temporaryRoot("local-mtp-unsafe")
        let snapshot = (root as NSString).appendingPathComponent("snapshot")
        defer { try? FileManager.default.removeItem(atPath: root) }
        _ = try SyntheticSnapshot.buildQwenMTP(at: snapshot)
        let indexPath = (snapshot as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
        let indexObject = try JSONSerialization.jsonObject(with: data)
        var index = try #require(indexObject as? [String: Any])
        var weightMap = try #require(index["weight_map"] as? [String: String])
        for name in weightMap.keys { weightMap[name] = "../outside.safetensors" }
        index["weight_map"] = weightMap
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: indexPath))

        #expect(throws: RepackError.self) {
            _ = try LocalSnapshotLoader.load(directory: snapshot)
        }
    }

    private func temporaryRoot(_ tag: String) -> String {
        let base = (FileManager.default.currentDirectoryPath as NSString)
            .appendingPathComponent(".build/test-artifacts")
        try? FileManager.default.createDirectory(
            atPath: base, withIntermediateDirectories: true)
        let path = (base as NSString)
            .appendingPathComponent("\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path,
                                                 withIntermediateDirectories: true)
        return path
    }
}
