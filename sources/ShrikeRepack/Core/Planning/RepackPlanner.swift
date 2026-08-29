import Foundation

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes: UInt64 = 16_384
}

// MARK: - Plan data types

/// One resolved byte range from a source shard into the resident file.
struct ResidentCopy: Sendable {
    let shardPath: String
    let sourceOffset: UInt64
    let destinationOffset: UInt64
    let size: UInt64
}

/// A byte range within one source tensor's payload.
struct SourceSlice: Sendable {
    let tensor: SourceTensor
    let offset: UInt64
    let size: UInt64

    init(whole tensor: SourceTensor) {
        self.tensor = tensor
        self.offset = 0
        self.size = tensor.sizeBytes
    }

    init(tensor: SourceTensor, offset: UInt64, size: UInt64) {
        self.tensor = tensor
        self.offset = offset
        self.size = size
    }
}

/// Resident tensors whose bytes are synthesized at write time instead of
/// range-copied. Only local snapshot imports may carry these.
enum ComputedResident: Sendable {
    /// Kimi `kv_b_proj` K-halves (`W_UK`), dequantized per head, transposed,
    /// and requantized 8-bit g64 as [heads * latent, nope] so the absorbed
    /// q-embedding GEMV reads rows of `W_UK^T`. 8-bit because the transpose
    /// crosses the source's quantization grouping axis, so these values eat a
    /// second quantization the straight-copied tensors never see.
    case kimiEmbedQ(weight: SourceTensor, scales: SourceTensor,
                    biases: SourceTensor, heads: Int, nopeDim: Int,
                    vDim: Int, latentDim: Int, groupSize: Int)
}

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Resolved source ranges that fill this entry's regions; empty for a
    /// computed entry.
    let copies: [ResidentCopy]
    /// Set when the writer must synthesize this entry's bytes.
    let computed: ComputedResident?
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    let dtype: UInt8                   // 0=U32, 1=BF16
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// For each expert e (0..<expertsPerLayer): source byte offset & size.
    let sourceOffsetPerExpert: UInt64  // stride per expert in source
    let sourceTensor: SourceTensor
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    /// gate/up/down × {weights, scales, biases}; families with additive expert
    /// biases append a "bias" component per role (9 or 12 entries).
    let subTensors: [PerExpertTensorSlice]
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: ArchInfo
    let baseMode: String                  // "affine"
    let baseGroupSize: Int                // 64
    let bitsOverrideCount: Int
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int,
                         family: RepackModelFamily) -> Bucket {
        if family == .qwen36MTP {
            if name.hasPrefix("layers.") {
                if let role = routedExpertRole(in: name, family: family),
                   let layer = layerIndex(in: name),
                   layer >= 0 && layer < numLayers {
                    return .routedExpert(role: role, layer: layer)
                }
                return .lmResident
            }
            if name == "norm.weight" || name.hasPrefix("fc.")
                || name.hasPrefix("pre_fc_norm_") {
                return .lmResident
            }
            return .unknown
        }
        if family == .gptOss20b || family == .kimiLinear48b {
            if name.hasPrefix("model.") || name.hasPrefix("lm_head.") {
                if let role = routedExpertRole(in: name, family: family),
                   let layer = layerIndex(in: name),
                   layer >= 0 && layer < numLayers {
                    return .routedExpert(role: role, layer: layer)
                }
                return .lmResident
            }
            return .unknown
        }
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name, family: family),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    /// gpt-oss experts carry an additive `.bias` alongside the quantization
    /// companions; it travels with the expert blob, never the resident file.
    static func isRoutedExpertAdditiveBias(_ name: String,
                                           family: RepackModelFamily) -> Bool {
        family == .gptOss20b && name.contains(".mlp.experts.") && name.hasSuffix(".bias")
    }

    private static func routedExpertRole(in name: String,
                                         family: RepackModelFamily) -> String? {
        let container = family == .gptOss20b ? ".mlp.experts." : ".mlp.switch_mlp."
        guard name.contains(container) else { return nil }
        if family == .gptOss20b && !name.hasSuffix(".weight") { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        let marker = name.hasPrefix("layers.") ? "layers." : ".layers."
        guard let r = name.range(of: marker) else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {

        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        for (name, _) in registry {
            // Companion tensors (.scales/.biases) are dropped together with
            // their primary tensor below; listing them separately would make
            // `excludedMultimodalTensorNames` mention tensors that are never
            // planned on their own.
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            if isRoutedExpertAdditiveBias(name, family: arch.family) { continue }
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            let b = classify(name, numLayers: arch.numLayers, family: arch.family)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering(family: arch.family))
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            arch: arch,
                                            registry: registry, meta: meta)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: arch,
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames)
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.")
    }

    // MARK: - Resident planning

    /// A resident destination before file offsets exist: straight per-tensor
    /// copies, a concatenation of source slices, or a computed tensor.
    private struct ResidentBlueprint {
        let name: String
        let dtype: UInt8
        let logicalShape4: [UInt32]
        let quantSpec: QuantSpec?
        let weightSlices: [SourceSlice]
        let scaleSlices: [SourceSlice]
        let biasSlices: [SourceSlice]
        let computed: ComputedResident?
        let computedSizes: (weight: UInt64, scale: UInt64, bias: UInt64)?
    }

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         arch: ArchInfo,
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata) throws
                                        -> ResidentFilePlan {
        let blueprints: [ResidentBlueprint]
        if arch.family == .kimiLinear48b {
            blueprints = try kimiResidentBlueprints(sortedNames: baseNames,
                                                    registry: registry,
                                                    meta: meta, arch: arch)
        } else {
            blueprints = try baseNames.map {
                try identityBlueprint(
                    sourceName: $0,
                    destinationName: residentDestinationName($0, family: arch.family),
                    registry: registry, meta: meta)
            }
        }

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(blueprints.count)
        for bp in blueprints {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: bp.name.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + blueprints.count * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(blueprints.count)

        for bp in blueprints {
            let wSize = bp.computedSizes?.weight
                ?? bp.weightSlices.reduce(UInt64(0)) { $0 + $1.size }
            let sSize = bp.computedSizes?.scale
                ?? bp.scaleSlices.reduce(UInt64(0)) { $0 + $1.size }
            let bSize = bp.computedSizes?.bias
                ?? bp.biasSlices.reduce(UInt64(0)) { $0 + $1.size }
            let wOff = fileCursor
            let sOff = wOff + wSize
            let bOff = sOff + sSize
            fileCursor = bOff + bSize

            var copies: [ResidentCopy] = []
            copies.reserveCapacity(bp.weightSlices.count
                + bp.scaleSlices.count + bp.biasSlices.count)
            var destination = wOff
            for slice in bp.weightSlices + bp.scaleSlices + bp.biasSlices {
                copies.append(ResidentCopy(
                    shardPath: slice.tensor.shardPath,
                    sourceOffset: slice.tensor.absoluteOffset + slice.offset,
                    destinationOffset: destination,
                    size: slice.size))
                destination += slice.size
            }

            entries.append(ResidentEntry(
                name: bp.name, dtype: bp.dtype,
                logicalShape4: bp.logicalShape4,
                fileOffset: wOff, sizeBytes: wSize,
                scaleOffset: sSize > 0 ? sOff : 0, scaleSize: sSize,
                biasOffset: bSize > 0 ? bOff : 0, biasSize: bSize,
                quantSpec: bp.quantSpec,
                copies: copies,
                computed: bp.computed))
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    private static func identityBlueprint(
        sourceName: String,
        destinationName: String,
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata
    ) throws -> ResidentBlueprint {
        guard let weight = registry[sourceName] else {
            throw RepackError.missingTensor(name: sourceName)
        }
        let isQuantizedPacked = (weight.dtype == .u32) && sourceName.hasSuffix(".weight")
        guard isQuantizedPacked else {
            // Unquantized (BF16/FP32 norm / scalar) — no companions.
            return ResidentBlueprint(
                name: destinationName,
                dtype: ietnyDtype(weight.dtype),
                logicalShape4: try padTo4(weight.shape),
                quantSpec: nil,
                weightSlices: [SourceSlice(whole: weight)],
                scaleSlices: [], biasSlices: [],
                computed: nil, computedSizes: nil)
        }
        let (scales, biases) = try quantCompanions(
            of: sourceName, reportedAs: destinationName, registry: registry)
        let spec = IndexLoader.quantSpec(forTensor: sourceName, meta: meta)
        let logical = try logicalShape(forPackedSource: weight.shape,
                                       scalesShape: scales.shape)
        return ResidentBlueprint(
            name: destinationName,
            dtype: 0,
            logicalShape4: try padTo4(logical),
            quantSpec: spec,
            weightSlices: [SourceSlice(whole: weight)],
            scaleSlices: [SourceSlice(whole: scales)],
            biasSlices: [SourceSlice(whole: biases)],
            computed: nil, computedSizes: nil)
    }

    private static func quantCompanions(
        of sourceName: String,
        reportedAs name: String,
        registry: [String: SourceTensor]
    ) throws -> (scales: SourceTensor, biases: SourceTensor) {
        let base = String(sourceName.dropLast(".weight".count))
        guard let scales = registry[base + ".scales"] else {
            throw RepackError.missingScalesCompanion(name: name)
        }
        guard let biases = registry[base + ".biases"] else {
            throw RepackError.missingBiasesCompanion(name: name)
        }
        if scales.dtype != .bf16 || biases.dtype != .bf16 {
            throw RepackError.dtypeMismatch(name: name,
                detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
        }
        return (scales, biases)
    }

    // MARK: - Kimi resident fusions

    /// Rewrites the sorted Kimi source-name list into destination blueprints:
    /// KDA q/k/v and their convs row-concatenate into the engine's fused
    /// `linear_attn.in_proj_qkv` / `linear_attn.conv1d` layouts (q, k, v
    /// order), `b_proj` renames to `linear_attn.in_proj_b`, and MLA
    /// `kv_b_proj` splits into `embed_q` (computed) + `unembed_out` (sliced).
    /// Everything else passes through under its own prefixed name.
    private static func kimiResidentBlueprints(
        sortedNames: [String],
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo
    ) throws -> [ResidentBlueprint] {
        var consumed: Set<String> = []
        var blueprints: [ResidentBlueprint] = []
        blueprints.reserveCapacity(sortedNames.count)

        func layerKind(_ name: String) -> UInt8? {
            guard let layer = layerIndex(in: name),
                  arch.fullAttentionLayerMask.indices.contains(layer) else {
                return nil
            }
            return arch.fullAttentionLayerMask[layer]
        }

        for name in sortedNames {
            if consumed.contains(name) { continue }
            let kind = layerKind(name)
            if kind == 2, name.hasSuffix(".self_attn.q_proj.weight") {
                blueprints.append(try kimiFusedQKVBlueprint(
                    qName: name, registry: registry, meta: meta,
                    consumed: &consumed))
                continue
            }
            if kind == 2, name.hasSuffix(".self_attn.q_conv.conv.weight") {
                blueprints.append(try kimiFusedConvBlueprint(
                    qName: name, registry: registry, consumed: &consumed))
                continue
            }
            if kind == 2, name.hasSuffix(".self_attn.b_proj.weight") {
                blueprints.append(try identityBlueprint(
                    sourceName: name,
                    destinationName: residentDestinationName(
                        name.replacingOccurrences(
                            of: ".self_attn.b_proj.",
                            with: ".linear_attn.in_proj_b."),
                        family: arch.family),
                    registry: registry, meta: meta))
                continue
            }
            if kind == 3, name.hasSuffix(".self_attn.kv_b_proj.weight") {
                blueprints.append(contentsOf: try kimiKVBSplitBlueprints(
                    kvbName: name, registry: registry, meta: meta, arch: arch))
                continue
            }
            blueprints.append(try identityBlueprint(
                sourceName: name,
                destinationName: residentDestinationName(name, family: arch.family),
                registry: registry, meta: meta))
        }
        return blueprints
    }

    private static func kimiFusedQKVBlueprint(
        qName: String,
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata,
        consumed: inout Set<String>
    ) throws -> ResidentBlueprint {
        let destinationName = residentDestinationName(
            qName.replacingOccurrences(of: ".self_attn.q_proj.",
                                       with: ".linear_attn.in_proj_qkv."),
            family: .kimiLinear48b)
        var weightSlices: [SourceSlice] = []
        var scaleSlices: [SourceSlice] = []
        var biasSlices: [SourceSlice] = []
        var totalRows: UInt64 = 0
        var columns: UInt64 = 0
        var spec: QuantSpec?
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let sourceName = qName.replacingOccurrences(
                of: ".self_attn.q_proj.", with: ".self_attn.\(proj).")
            guard let weight = registry[sourceName], weight.dtype == .u32,
                  weight.shape.count == 2 else {
                throw RepackError.shapeMismatch(name: sourceName,
                    detail: "expected a quantized rank-2 KDA \(proj) to fuse")
            }
            let (scales, biases) = try quantCompanions(
                of: sourceName, reportedAs: destinationName, registry: registry)
            let logical = try logicalShape(forPackedSource: weight.shape,
                                           scalesShape: scales.shape)
            let projSpec = IndexLoader.quantSpec(forTensor: sourceName, meta: meta)
            if columns == 0 { columns = logical[1] }
            if let spec, (spec != projSpec || columns != logical[1]) {
                throw RepackError.shapeMismatch(name: sourceName,
                    detail: "KDA q/k/v disagree on quantization or width; cannot fuse")
            }
            spec = projSpec
            totalRows += logical[0]
            weightSlices.append(SourceSlice(whole: weight))
            scaleSlices.append(SourceSlice(whole: scales))
            biasSlices.append(SourceSlice(whole: biases))
            consumed.insert(sourceName)
        }
        return ResidentBlueprint(
            name: destinationName,
            dtype: 0,
            logicalShape4: try padTo4([totalRows, columns]),
            quantSpec: spec,
            weightSlices: weightSlices,
            scaleSlices: scaleSlices,
            biasSlices: biasSlices,
            computed: nil, computedSizes: nil)
    }

    private static func kimiFusedConvBlueprint(
        qName: String,
        registry: [String: SourceTensor],
        consumed: inout Set<String>
    ) throws -> ResidentBlueprint {
        let destinationName = residentDestinationName(
            qName.replacingOccurrences(of: ".self_attn.q_conv.conv.",
                                       with: ".linear_attn.conv1d."),
            family: .kimiLinear48b)
        var slices: [SourceSlice] = []
        var channels: UInt64 = 0
        var tail: [UInt64] = []
        for conv in ["q_conv", "k_conv", "v_conv"] {
            let sourceName = qName.replacingOccurrences(
                of: ".self_attn.q_conv.", with: ".self_attn.\(conv).")
            guard let weight = registry[sourceName], weight.dtype == .bf16,
                  weight.shape.count == 3 else {
                throw RepackError.shapeMismatch(name: sourceName,
                    detail: "expected a BF16 rank-3 KDA \(conv) to fuse")
            }
            let weightTail = Array(weight.shape.dropFirst())
            if tail.isEmpty { tail = weightTail }
            guard weightTail == tail else {
                throw RepackError.shapeMismatch(name: sourceName,
                    detail: "KDA convs disagree on kernel shape; cannot fuse")
            }
            channels += weight.shape[0]
            slices.append(SourceSlice(whole: weight))
            consumed.insert(sourceName)
        }
        return ResidentBlueprint(
            name: destinationName,
            dtype: 1,
            logicalShape4: try padTo4([channels] + tail),
            quantSpec: nil,
            weightSlices: slices,
            scaleSlices: [], biasSlices: [],
            computed: nil, computedSizes: nil)
    }

    static let kimiEmbedQBits = 8

    private static func kimiKVBSplitBlueprints(
        kvbName: String,
        registry: [String: SourceTensor],
        meta: IndexLoader.SourceMetadata,
        arch: ArchInfo
    ) throws -> [ResidentBlueprint] {
        guard let weight = registry[kvbName], weight.dtype == .u32,
              weight.shape.count == 2 else {
            throw RepackError.shapeMismatch(name: kvbName,
                detail: "expected a quantized rank-2 kv_b_proj to split")
        }
        let (scales, biases) = try quantCompanions(
            of: kvbName, reportedAs: kvbName, registry: registry)
        let spec = IndexLoader.quantSpec(forTensor: kvbName, meta: meta)
        let heads = UInt64(arch.numHeads)
        let nope = UInt64(arch.mlaQKNopeDim)
        let vDim = UInt64(arch.mlaVHeadDim)
        let latent = UInt64(arch.mlaKVLoraRank)
        let group = UInt64(meta.baseGroupSize)
        let logical = try logicalShape(forPackedSource: weight.shape,
                                       scalesShape: scales.shape)
        guard logical == [heads * (nope + vDim), latent],
              latent.isMultiple(of: group),
              nope.isMultiple(of: group),
              latent.isMultiple(of: UInt64(32 / spec.bits)) else {
            throw RepackError.shapeMismatch(name: kvbName,
                detail: "kv_b_proj \(logical) does not split at "
                    + "heads \(heads), nope \(nope), v \(vDim), latent \(latent)")
        }

        let weightRowBytes = latent * UInt64(spec.bits) / 8
        let companionRowBytes = latent / group * 2
        var unembedWeights: [SourceSlice] = []
        var unembedScales: [SourceSlice] = []
        var unembedBiases: [SourceSlice] = []
        for head in 0..<heads {
            let firstVRow = head * (nope + vDim) + nope
            unembedWeights.append(SourceSlice(
                tensor: weight,
                offset: firstVRow * weightRowBytes,
                size: vDim * weightRowBytes))
            unembedScales.append(SourceSlice(
                tensor: scales,
                offset: firstVRow * companionRowBytes,
                size: vDim * companionRowBytes))
            unembedBiases.append(SourceSlice(
                tensor: biases,
                offset: firstVRow * companionRowBytes,
                size: vDim * companionRowBytes))
        }
        let unembed = ResidentBlueprint(
            name: residentDestinationName(
                kvbName.replacingOccurrences(of: ".kv_b_proj.",
                                             with: ".unembed_out."),
                family: .kimiLinear48b),
            dtype: 0,
            logicalShape4: try padTo4([heads * vDim, latent]),
            quantSpec: spec,
            weightSlices: unembedWeights,
            scaleSlices: unembedScales,
            biasSlices: unembedBiases,
            computed: nil, computedSizes: nil)

        let embedRows = heads * latent
        let embedBits = UInt64(Self.kimiEmbedQBits)
        let embed = ResidentBlueprint(
            name: residentDestinationName(
                kvbName.replacingOccurrences(of: ".kv_b_proj.",
                                             with: ".embed_q."),
                family: .kimiLinear48b),
            dtype: 0,
            logicalShape4: try padTo4([embedRows, nope]),
            quantSpec: QuantSpec(bits: Self.kimiEmbedQBits),
            weightSlices: [], scaleSlices: [], biasSlices: [],
            computed: .kimiEmbedQ(weight: weight, scales: scales,
                                  biases: biases, heads: Int(heads),
                                  nopeDim: Int(nope), vDim: Int(vDim),
                                  latentDim: Int(latent),
                                  groupSize: Int(group)),
            computedSizes: (
                weight: embedRows * nope * embedBits / 8,
                scale: embedRows * (nope / group) * 2,
                bias: embedRows * (nope / group) * 2))
        return [embed, unembed]
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        guard expertCount > 0 else {
            throw RepackError.configurationInvalid(
                detail: "layer \(layer) has routed experts but numExperts is zero")
        }
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())
            let logicalPerExpert = try logicalShape(forPackedSource: perExpertSourceShape,
                                                    scalesShape: scalesLogical)

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: 0,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: 1,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: 1,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)

            if arch.family == .gptOss20b {
                guard let ab = registry[base + ".bias"] else {
                    throw RepackError.missingTensor(name: base + ".bias")
                }
                if ab.dtype != .bf16 || ab.shape.count != 2
                    || Int(ab.shape[0]) != expertCount {
                    throw RepackError.shapeMismatch(name: base + ".bias",
                        detail: "expected BF16 rank-2 with leading \(expertCount), "
                            + "got \(ab.dtype) \(ab.shape)")
                }
                let perExpertABSize = ab.sizeBytes / UInt64(expertCount)
                guard perExpertABSize * UInt64(expertCount) == ab.sizeBytes else {
                    throw RepackError.shapeMismatch(name: base + ".bias",
                        detail: "source bytes not evenly divisible by \(expertCount) experts")
                }
                subs.append(PerExpertTensorSlice(
                    role: role, component: "bias", dtype: 1,
                    logicalShape: Array(ab.shape.dropFirst()),
                    offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertABSize,
                    sourceOffsetPerExpert: perExpertABSize, sourceTensor: ab,
                    bitsForWeights: nil))
                blobCursor += perExpertABSize
            }
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d { case .u32: 0; case .bf16: 1; case .fp16: 2; case .fp32: 3 }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) throws -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) {
            guard v <= UInt64(UInt32.max) else {
                throw RepackError.shapeMismatch(
                    name: "shape",
                    detail: "dimension \(v) does not fit a UInt32 index")
            }
            out.append(UInt32(v))
        }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of an MLX packed quantized tensor. The scale grid is
    /// authoritative because six-bit values do not have an integral U32
    /// packing factor.
    private static func logicalShape(forPackedSource source: [UInt64],
                                     scalesShape: [UInt64]) throws -> [UInt64] {
        guard !source.isEmpty else { return source }
        guard let lastScale = scalesShape.last else {
            throw RepackError.shapeMismatch(
                name: "scales",
                detail: "packed tensor has an empty scales shape")
        }
        let (scaled, overflow) = lastScale.multipliedReportingOverflow(by: 64)
        guard !overflow else {
            throw RepackError.shapeMismatch(
                name: "scales",
                detail: "scales dimension \(lastScale) overflows when scaled by 64")
        }
        var out = source
        out[out.count - 1] = scaled
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm (and, for
    /// families with an untied head, `lm_head` last).
    private static func lmResidentOrdering(family: RepackModelFamily)
        -> (String, String) -> Bool {
        let flatNames = family == .gptOss20b || family == .kimiLinear48b
        let embedName = flatNames
            ? "model.embed_tokens.weight" : "language_model.model.embed_tokens.weight"
        let normName = flatNames
            ? "model.norm.weight" : "language_model.model.norm.weight"
        let headName = flatNames
            ? "lm_head.weight" : "language_model.lm_head.weight"
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == embedName { return (0, 0, 0, n) }
            if n == normName  { return (3, 0, 0, n) }
            if n == headName  { return (4, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot: Int
                switch family {
                case .gptOss20b:     slot = gptOssSlotRank(in: n)
                case .kimiLinear48b: slot = kimiSlotRank(in: n)
                case .qwen36, .qwen36MTP: slot = qwenSlotRank(in: n)
                }
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order for the Qwen 3.6 family: full-attention
    /// projections/norms, then the gated-DeltaNet linear-attention bundle,
    /// then router, shared-expert gate and MLP, then the two layer norms.
    private static func qwenSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.k_proj.weight")   { return 1 }
        if n.contains(".self_attn.v_proj.weight")   { return 2 }
        if n.contains(".self_attn.o_proj.weight")   { return 3 }
        if n.contains(".self_attn.q_norm.weight")   { return 4 }
        if n.contains(".self_attn.k_norm.weight")   { return 5 }
        if n.contains(".linear_attn.in_proj_qkv.weight") { return 6 }
        if n.contains(".linear_attn.in_proj_z.weight")   { return 7 }
        if n.contains(".linear_attn.in_proj_a.weight")   { return 8 }
        if n.contains(".linear_attn.in_proj_b.weight")   { return 9 }
        if n.contains(".linear_attn.conv1d.weight")      { return 10 }
        if n.hasSuffix(".linear_attn.A_log")             { return 11 }
        if n.hasSuffix(".linear_attn.dt_bias")           { return 12 }
        if n.contains(".linear_attn.norm.weight")        { return 13 }
        if n.contains(".linear_attn.out_proj.weight")    { return 14 }
        if n.contains(".mlp.gate.weight")                { return 15 }
        if n.contains(".mlp.shared_expert_gate.weight")  { return 16 }
        if n.contains(".mlp.shared_expert.gate_proj.weight") { return 17 }
        if n.contains(".mlp.shared_expert.up_proj.weight")   { return 18 }
        if n.contains(".mlp.shared_expert.down_proj.weight") { return 19 }
        if n.hasSuffix(".input_layernorm.weight")        { return 20 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 21 }
        return 100
    }

    /// Normalize source names to the runtime tensor-name contract
    /// (`language_model.model.…`). MTP prefixes its bare layer names; the flat
    /// families prefix their `model.` / `lm_head.` names; Qwen names already
    /// carry the contract prefix.
    static func residentDestinationName(_ source: String,
                                        family: RepackModelFamily) -> String {
        switch family {
        case .qwen36:
            return source
        case .qwen36MTP:
            if source.hasPrefix("layers.") {
                return "language_model.model." + source
            }
            if source == "norm.weight" {
                return "language_model.model.norm.weight"
            }
            return source
        case .gptOss20b, .kimiLinear48b:
            if source.hasPrefix("model.") || source.hasPrefix("lm_head.") {
                return "language_model." + source
            }
            return source
        }
    }

    /// Within-layer slot order for gpt-oss: attention projections with their
    /// additive biases, sinks, router, then the two layer norms.
    private static func gptOssSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")   { return 0 }
        if n.contains(".self_attn.q_proj.bias")     { return 1 }
        if n.contains(".self_attn.k_proj.weight")   { return 2 }
        if n.contains(".self_attn.k_proj.bias")     { return 3 }
        if n.contains(".self_attn.v_proj.weight")   { return 4 }
        if n.contains(".self_attn.v_proj.bias")     { return 5 }
        if n.contains(".self_attn.o_proj.weight")   { return 6 }
        if n.contains(".self_attn.o_proj.bias")     { return 7 }
        if n.hasSuffix(".self_attn.sinks")          { return 8 }
        if n.contains(".mlp.router.weight")         { return 9 }
        if n.contains(".mlp.router.bias")           { return 10 }
        if n.hasSuffix(".input_layernorm.weight")   { return 11 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 12 }
        return 100
    }

    /// Within-layer slot order for Kimi-Linear: the KDA bundle, the MLA
    /// bundle, router + correction bias, shared-expert MLP, the dense-layer
    /// MLP, then the two layer norms.
    private static func kimiSlotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight")        { return 0 }
        if n.contains(".self_attn.k_proj.weight")        { return 1 }
        if n.contains(".self_attn.v_proj.weight")        { return 2 }
        if n.contains(".self_attn.q_conv.conv.weight")   { return 3 }
        if n.contains(".self_attn.k_conv.conv.weight")   { return 4 }
        if n.contains(".self_attn.v_conv.conv.weight")   { return 5 }
        if n.contains(".self_attn.f_a_proj.weight")      { return 6 }
        if n.contains(".self_attn.f_b_proj.weight")      { return 7 }
        if n.contains(".self_attn.g_a_proj.weight")      { return 8 }
        if n.contains(".self_attn.g_b_proj.weight")      { return 9 }
        if n.contains(".self_attn.b_proj.weight")        { return 10 }
        if n.hasSuffix(".self_attn.A_log")               { return 11 }
        if n.hasSuffix(".self_attn.dt_bias")             { return 12 }
        if n.contains(".self_attn.o_norm.weight")        { return 13 }
        if n.contains(".self_attn.kv_a_proj_with_mqa.weight") { return 14 }
        if n.contains(".self_attn.kv_a_layernorm.weight") { return 15 }
        if n.contains(".self_attn.kv_b_proj.weight")     { return 16 }
        if n.contains(".self_attn.o_proj.weight")        { return 17 }
        if n.contains(".mlp.gate.weight")                { return 18 }
        if n.hasSuffix(".mlp.e_score_correction_bias")   { return 19 }
        if n.contains(".mlp.shared_experts.gate_proj.weight") { return 20 }
        if n.contains(".mlp.shared_experts.up_proj.weight")   { return 21 }
        if n.contains(".mlp.shared_experts.down_proj.weight") { return 22 }
        if n.contains(".mlp.gate_proj.weight")           { return 23 }
        if n.contains(".mlp.up_proj.weight")             { return 24 }
        if n.contains(".mlp.down_proj.weight")           { return 25 }
        if n.hasSuffix(".input_layernorm.weight")        { return 26 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 27 }
        return 100
    }
}
