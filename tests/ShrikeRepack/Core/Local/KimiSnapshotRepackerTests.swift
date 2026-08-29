import Foundation
import Testing
@testable import ShrikeRepackCore

/// Kimi-Linear local import: KDA q/k/v (+conv) fusion, the `kv_b_proj` split
/// into `embed_q` (computed, 8-bit) and `unembed_out` (sliced), the dense
/// layer 0, and the 8-bit router overrides — verified down to the bytes the
/// import writes.
@Suite struct KimiSnapshotRepackerTests {

    @Test func importsKimiSnapshotWithFusionsAndSplit() async throws {
        let (root, output, plan) = try await importedKimi("kimi-import")
        defer { try? FileManager.default.removeItem(atPath: root) }

        let entries = Dictionary(uniqueKeysWithValues:
            plan.resident.entries.map { ($0.name, $0) })
        let l0 = "language_model.model.layers.0."
        let l3 = "language_model.model.layers.3."

        let fused = try #require(entries[l0 + "linear_attn.in_proj_qkv.weight"])
        #expect(fused.logicalShape4 == [192, 64, 0, 0])
        #expect(fused.quantSpec?.bits == 4)
        #expect(fused.copies.count == 9)

        let conv = try #require(entries[l0 + "linear_attn.conv1d.weight"])
        #expect(conv.logicalShape4 == [192, 4, 1, 0])
        #expect(conv.dtype == 1)

        #expect(entries[l0 + "linear_attn.in_proj_b.weight"] != nil)
        for gone in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                     "self_attn.v_proj.weight", "self_attn.q_conv.conv.weight",
                     "self_attn.b_proj.weight"] {
            #expect(entries[l0 + gone] == nil)
        }

        let embed = try #require(entries[l3 + "self_attn.embed_q.weight"])
        #expect(embed.logicalShape4 == [128, 64, 0, 0])
        #expect(embed.quantSpec?.bits == 8)
        #expect(embed.computed != nil)
        #expect(embed.copies.isEmpty)

        let unembed = try #require(entries[l3 + "self_attn.unembed_out.weight"])
        #expect(unembed.logicalShape4 == [128, 64, 0, 0])
        #expect(unembed.quantSpec?.bits == 4)
        #expect(unembed.copies.count == 6)

        #expect(entries[l3 + "self_attn.kv_b_proj.weight"] == nil)
        #expect(entries[l3 + "self_attn.q_proj.weight"] != nil)

        #expect(plan.layers.map(\.expertsPerLayer) == [0, 4, 4, 4, 4])
        let gate = try #require(entries[
            "language_model.model.layers.1.mlp.gate.weight"])
        #expect(gate.quantSpec?.bits == 8)

        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try #require(
            try JSONSerialization.jsonObject(with: manifestData) as? [String: Any])
        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["family"] as? String == "kimi_linear_48b")
        #expect(arch["numLeadingDenseLayers"] as? Int == 1)
        #expect(arch["fullAttentionLayerMask"] as? [Int] == [2, 2, 2, 3, 2])
        #expect(manifest["expertsPerLayer"] as? Int == 4)
        #expect(manifest["bitWidthOverridesHonored"] as? Int == 4)
    }

    @Test func verifyInstallAcceptsTheZeroExpertDenseLayer() async throws {
        let (root, output, _) = try await importedKimi("kimi-verify")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let result = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(result.unexpectedEntries.isEmpty)
    }

    @Test func fusedAndSlicedBytesMatchTheSource() async throws {
        let (root, _, plan) = try await importedKimi("kimi-bytes")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let shard = try ShardReader(directory: snapshotDir(root))

        let l0 = "language_model.model.layers.0."
        let fused = try #require(plan.resident.entries.first {
            $0.name == l0 + "linear_attn.in_proj_qkv.weight"
        })
        let resident = try Data(contentsOf: URL(
            fileURLWithPath: plan.resident.path))
        var expected = Data()
        for proj in ["q_proj", "k_proj", "v_proj"] {
            expected += try shard.bytes("model.layers.0.self_attn.\(proj).weight")
        }
        #expect(resident.subdata(
            in: Int(fused.fileOffset)..<Int(fused.fileOffset + fused.sizeBytes))
            == expected)
        var expectedScales = Data()
        for proj in ["q_proj", "k_proj", "v_proj"] {
            expectedScales += try shard.bytes("model.layers.0.self_attn.\(proj).scales")
        }
        #expect(resident.subdata(
            in: Int(fused.scaleOffset)..<Int(fused.scaleOffset + fused.scaleSize))
            == expectedScales)

        let arch = SyntheticSnapshot.KimiArch()
        let unembed = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.3.self_attn.unembed_out.weight"
        })
        let kvb = try shard.bytes("model.layers.3.self_attn.kv_b_proj.weight")
        let rowBytes = arch.kvLoraRank / 2
        var expectedUnembed = Data()
        for head in 0..<arch.numHeads {
            let first = (head * (arch.qkNope + arch.vHeadDim) + arch.qkNope) * rowBytes
            expectedUnembed += kvb.subdata(
                in: first..<(first + arch.vHeadDim * rowBytes))
        }
        #expect(resident.subdata(
            in: Int(unembed.fileOffset)..<Int(unembed.fileOffset + unembed.sizeBytes))
            == expectedUnembed)
    }

    @Test func embedQIsTheDequantizedTransposeWithinRequantError() async throws {
        let (root, _, plan) = try await importedKimi("kimi-embedq")
        defer { try? FileManager.default.removeItem(atPath: root) }
        let shard = try ShardReader(directory: snapshotDir(root))
        let arch = SyntheticSnapshot.KimiArch()

        let source = QuantizedTensor(
            weight: try shard.bytes("model.layers.3.self_attn.kv_b_proj.weight"),
            scales: try shard.bytes("model.layers.3.self_attn.kv_b_proj.scales"),
            biases: try shard.bytes("model.layers.3.self_attn.kv_b_proj.biases"),
            columns: arch.kvLoraRank, bits: 4, groupSize: arch.groupSize)

        let entry = try #require(plan.resident.entries.first {
            $0.name == "language_model.model.layers.3.self_attn.embed_q.weight"
        })
        let resident = try Data(contentsOf: URL(fileURLWithPath: plan.resident.path))
        let output = QuantizedTensor(
            weight: resident.subdata(in:
                Int(entry.fileOffset)..<Int(entry.fileOffset + entry.sizeBytes)),
            scales: resident.subdata(in:
                Int(entry.scaleOffset)..<Int(entry.scaleOffset + entry.scaleSize)),
            biases: resident.subdata(in:
                Int(entry.biasOffset)..<Int(entry.biasOffset + entry.biasSize)),
            columns: arch.qkNope, bits: 8, groupSize: arch.groupSize)

        var maxError: Float = 0
        var maxScale: Float = 0
        for head in 0..<arch.numHeads {
            for l in 0..<arch.kvLoraRank {
                let outRow = head * arch.kvLoraRank + l
                for n in 0..<arch.qkNope {
                    let sourceRow = head * (arch.qkNope + arch.vHeadDim) + n
                    let expected = source.value(row: sourceRow, column: l)
                    let actual = output.value(row: outRow, column: n)
                    maxError = max(maxError, abs(actual - expected))
                    maxScale = max(maxScale, output.scale(row: outRow, column: n))
                }
            }
        }
        #expect(maxError <= maxScale + 1e-4)
        #expect(maxScale > 0)
    }

    // MARK: - Helpers

    private func importedKimi(_ tag: String) async throws
        -> (root: String, output: String, plan: RepackPlan) {
        let root = temporaryRoot(tag)
        let snapshot = snapshotDir(root)
        let output = (root as NSString).appendingPathComponent("model.gturbo")
        _ = try SyntheticSnapshot.buildKimiLinear(at: snapshot)
        let result = try await RemoteStreamingRepacker.runLocalSnapshot(
            options: LocalSnapshotRepackOptions(
                inputSnapshotDir: snapshot,
                outputDir: output,
                modelID: "kimi-linear-toy-4bit",
                minFreeReserveBytes: 0))
        var plan = result.plan
        plan = RepackPlan(arch: plan.arch, baseMode: plan.baseMode,
                          baseGroupSize: plan.baseGroupSize,
                          bitsOverrideCount: plan.bitsOverrideCount,
                          resident: ResidentFilePlan(
                              path: (output as NSString)
                                  .appendingPathComponent("model_weights.bin"),
                              entries: plan.resident.entries,
                              stringTable: plan.resident.stringTable,
                              stringTableOffsets: plan.resident.stringTableOffsets,
                              indexSize: plan.resident.indexSize,
                              residentSize: plan.resident.residentSize),
                          layers: plan.layers,
                          matchedModelID: plan.matchedModelID,
                          excludedMultimodalTensorNames: plan.excludedMultimodalTensorNames)
        return (root, output, plan)
    }

    private func snapshotDir(_ root: String) -> String {
        (root as NSString).appendingPathComponent("snapshot")
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

/// Reads tensor payloads straight out of a single-shard synthetic snapshot,
/// independent of the repack code under test.
private struct ShardReader {
    private let data: Data
    private let offsets: [String: (Int, Int)]

    init(directory: String) throws {
        let shard = (directory as NSString)
            .appendingPathComponent("model-00001-of-00001.safetensors")
        data = try Data(contentsOf: URL(fileURLWithPath: shard))
        let headerLength = data.subdata(in: 0..<8).withUnsafeBytes {
            Int(UInt64(littleEndian: $0.load(as: UInt64.self)))
        }
        let header = try JSONSerialization.jsonObject(
            with: data.subdata(in: 8..<(8 + headerLength))) as? [String: Any] ?? [:]
        var offsets: [String: (Int, Int)] = [:]
        for (name, value) in header where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let range = entry["data_offsets"] as? [Int] else { continue }
            offsets[name] = (8 + headerLength + range[0], 8 + headerLength + range[1])
        }
        self.offsets = offsets
    }

    func bytes(_ name: String) throws -> Data {
        let range = try #require(offsets[name])
        return data.subdata(in: range.0..<range.1)
    }
}

/// Dequantizes MLX-affine packed tensors for numeric comparisons.
private struct QuantizedTensor {
    let weight: Data
    let scales: Data
    let biases: Data
    let columns: Int
    let bits: Int
    let groupSize: Int

    func value(row: Int, column: Int) -> Float {
        let valuesPerWord = 32 / bits
        let wordsPerRow = columns / valuesPerWord
        let word = weight.withUnsafeBytes {
            $0.bindMemory(to: UInt32.self)[row * wordsPerRow + column / valuesPerWord]
        }
        let q = Float((word >> UInt32((column % valuesPerWord) * bits))
            & UInt32((1 << bits) - 1))
        return scale(row: row, column: column) * q + bias(row: row, column: column)
    }

    func scale(row: Int, column: Int) -> Float {
        companion(scales, row: row, column: column)
    }

    func bias(row: Int, column: Int) -> Float {
        companion(biases, row: row, column: column)
    }

    private func companion(_ data: Data, row: Int, column: Int) -> Float {
        let groupsPerRow = columns / groupSize
        let half = data.withUnsafeBytes {
            $0.bindMemory(to: UInt16.self)[row * groupsPerRow + column / groupSize]
        }
        return ComputedResidentMaterializer.bf16ToFloat(half)
    }
}
