import Foundation
import NVMAIFormat
@testable import NVMAI
@testable import NVMAIRepackCore

/// Synthetic gpt-oss toy fixture: a tiny runnable `.gturbo/` directory with
/// the gpt-oss tensor-name contract (plain biased q/k/v/o self_attn with
/// per-Q-head sinks, no QK norms, `mlp.router` + additive bias, no shared
/// expert, 12-slice routed-expert blobs with additive BF16 biases, `family`
/// in the manifest). Mirrors `QwenToySynthetic.write` for the gpt-oss toy
/// config.
enum GptOssToySynthetic {

    /// Build the toy directory in a temp dir and return its URL.
    static func write(weightBits: Int = 4) throws -> URL {
        precondition([4, 8].contains(weightBits))
        let toy = ArchConfig.gptOssToy()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-gptoss-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let u16 = MemoryLayout<UInt16>.stride

        func affineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            let usedBits = weightBits <= 4 ? 4 : 8
            let groups = cols / Quantization.groupSize
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name,
                                dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols * usedBits / 8),
                                scaleBytes: auxBytes,
                                biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name,
                         dtype: 1,
                         shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0,
                         biasBytes: 0)
        }

        // 1. Resident specs — the gpt-oss per-layer contract.
        var specs: [ResidentSpec] = [
            affineSpec("language_model.model.embed_tokens.weight",
                       rows: toy.vocabSize, cols: d),
            affineSpec("language_model.lm_head.weight",
                       rows: toy.vocabSize, cols: d),
            bf16Spec("language_model.model.norm.weight", count: d),
        ]
        let qDim = toy.numHeads * toy.fullHeadDim
        let kvDim = toy.numFullKVHeads * toy.fullHeadDim
        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            specs.append(bf16Spec("\(prefix).input_layernorm.weight", count: d))
            specs.append(bf16Spec("\(prefix).post_attention_layernorm.weight", count: d))
            specs.append(affineSpec("\(prefix).mlp.router.weight",
                                    rows: toy.numExperts, cols: d))
            specs.append(bf16Spec("\(prefix).mlp.router.bias", count: toy.numExperts))
            specs.append(affineSpec("\(prefix).self_attn.q_proj.weight",
                                    rows: qDim, cols: d))
            specs.append(affineSpec("\(prefix).self_attn.k_proj.weight",
                                    rows: kvDim, cols: d))
            specs.append(affineSpec("\(prefix).self_attn.v_proj.weight",
                                    rows: kvDim, cols: d))
            specs.append(affineSpec("\(prefix).self_attn.o_proj.weight",
                                    rows: d, cols: qDim))
            specs.append(bf16Spec("\(prefix).self_attn.q_proj.bias", count: qDim))
            specs.append(bf16Spec("\(prefix).self_attn.k_proj.bias", count: kvDim))
            specs.append(bf16Spec("\(prefix).self_attn.v_proj.bias", count: kvDim))
            specs.append(bf16Spec("\(prefix).self_attn.o_proj.bias", count: d))
            specs.append(bf16Spec("\(prefix).self_attn.sinks", count: toy.numHeads))
        }

        // 2. Serialize the resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes  = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let indexBytes = UInt64(stringTableBase + stringTable.count)
        let alignedIndexBytes = ((indexBytes + GTurboFormatV1.alignmentBytes - 1) &
                                 ~(GTurboFormatV1.alignmentBytes - 1))

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = alignedIndexBytes
        let align: UInt64 = UInt64(MemoryLayout<UInt32>.alignment)
        func alignedCursor(_ cursor: UInt64) -> UInt64 {
            ((cursor + align - 1) & ~(align - 1))
        }
        for spec in specs {
            let weightOffset = alignedCursor(payloadCursor)
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset,
                sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: spec.scaleBytes,
                biasOffset: biasOffset,
                biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
            let tensorSize = spec.weightBytes + spec.scaleBytes + spec.biasBytes
            payloadCursor = weightOffset + tensorSize
        }
        let residentSize = alignedCursor(payloadCursor) - alignedIndexBytes

        let totalBytes = Int(alignedIndexBytes + residentSize)
        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: alignedIndexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                let dst = base.advanced(by: entriesBase + i * entryBytes)
                GTurboBinary.writeIndexEntry(into: dst, entry: e,
                                             nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase), sb.baseAddress!, stringTable.count)
            }
            // Quantized tensors: weight bytes 0x11, scales 0.01, biases zero.
            // BF16 tensors: 0.25 — small enough that the additive attention
            // biases and sinks perturb rather than swamp the softmax.
            for entry in entries where entry.dtype == 0 {
                memset(base.advanced(by: Int(entry.fileOffset)), 0x11, Int(entry.sizeBytes))
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(0.25)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Packed experts: 12-slice blobs — int4/8 gate/up/down plus their
        // quant scales/biases and the three additive BF16 bias rows.
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001
                        + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8], tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols,
                                                   expert: expert, role: role)
                let quantized = projectionRows.map { Quantization.quantizeInt4Affine($0) }
                let packedOffset = bytes.count
                let usedBits = weightBits <= 4 ? 4 : 8
                if usedBits == 4 {
                    for row in quantized { bytes.append(contentsOf: row.packed) }
                } else {
                    bytes += [UInt8](repeating: 0x11,
                                     count: rows * cols * usedBits / 8)
                }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols],
                    "bits": usedBits,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": bytes.count - biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / Quantization.groupSize],
                ]
                let additiveOffset = bytes.count
                let additive = (0..<rows).map { row in
                    Quantization.bf16Bits(Float(expert + 1) * 0.01
                        + Float(role + 1) * 0.02
                        + Float(row % 5) * 0.005)
                }
                appendU16(additive, to: &bytes)
                tensors["\(prefix)_bias"] = [
                    "offset": additiveOffset, "size": bytes.count - additiveOffset,
                    "dtype": "BF16", "shape": [rows],
                ]
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize, cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize, cols: d, role: 1)
            addProjection(prefix: "down", rows: d, cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        let sampleExpertBytes = toyExpertBlob(expert: 0).bytes.count
        let expertStride = UInt64(((sampleExpertBytes + 16_383) / 16_384) * 16_384)
        let layerBytes = Int(expertStride) * toy.numExperts
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E).bytes
                let baseB = E * Int(expertStride)
                precondition(blob.count <= Int(expertStride),
                             "toy expert blob exceeds stride")
                for (i, byte) in blob.enumerated() {
                    payload[baseB + i] = byte
                }
            }
            let url = exp.appendingPathComponent(String(format: "layer_%02d.bin", L))
            try payload.write(to: url)
        }
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            let basename = String(format: "layer_%02d.bin", L)
            let url = exp.appendingPathComponent(basename)
            layerShaByName["packed_experts/\(basename)"] = try Sha256Verifier.hashFile(at: url)
        }

        // 4. layout.json
        var layersArr: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E)
                experts.append([
                    "expert": E,
                    "offset": UInt64(E) * expertStride,
                    "size":   expertStride,
                    "tensors": blob.tensors,
                ])
            }
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": experts,
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 5. manifest.json (arch with the gpt-oss family field)
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": Int(totalBytes), "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count, "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "family": toy.family.rawValue,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
        ]
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false, "aneSharedExpert": false],
            "modelID": "gptoss-toy-\(weightBits)bit",
            "arch": archDict,
            "quant": [
                "embedding": quantSlot(weightBits),
                "attention": quantSlot(weightBits),
                "router": quantSlot(weightBits),
                "sharedExpert": quantSlot(weightBits),
                "routedExpert": quantSlot(weightBits),
            ],
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(withJSONObject: manifestRoot,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }

    private static func quantSlot(_ bits: Int) -> [String: Any] {
        let usedBits = bits <= 4 ? 4 : 8
        return ["weightBits": usedBits, "scheme": "affine", "scaleType": "bf16",
         "biasType": "bf16", "groupSize": Quantization.groupSize]
    }
}

