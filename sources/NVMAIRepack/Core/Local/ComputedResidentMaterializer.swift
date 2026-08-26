import Foundation

/// Writes the plan's computed resident tensors after the range copies land.
/// Only local snapshot imports produce computed entries; the source bytes are
/// read straight from the snapshot's shards.
enum ComputedResidentMaterializer {

    static func materialize(plan: RepackPlan,
                            snapshotDirectory: String,
                            residentPath: String) throws {
        let computed = plan.resident.entries.filter { $0.computed != nil }
        guard !computed.isEmpty else { return }
        let fd = open(residentPath, O_WRONLY)
        guard fd >= 0 else {
            throw RepackError.fileOpenFailed(path: residentPath, errno: errno)
        }
        defer { close(fd) }
        for entry in computed {
            switch entry.computed {
            case .kimiEmbedQ(let weight, let scales, let biases, let heads,
                             let nope, let vDim, let latent, let group):
                try materializeKimiEmbedQ(
                    entry: entry, weight: weight, scales: scales, biases: biases,
                    heads: heads, nopeDim: nope, vDim: vDim, latentDim: latent,
                    groupSize: group,
                    snapshotDirectory: snapshotDirectory,
                    residentFD: fd, residentPath: residentPath)
            case nil:
                continue
            }
        }
    }

    // MARK: - Kimi embed_q

    private static func materializeKimiEmbedQ(
        entry: ResidentEntry,
        weight: SourceTensor, scales: SourceTensor, biases: SourceTensor,
        heads: Int, nopeDim: Int, vDim: Int, latentDim: Int, groupSize: Int,
        snapshotDirectory: String,
        residentFD: Int32, residentPath: String
    ) throws {
        let sourceBits = weight.sizeBytes * 8
            / (UInt64(heads * (nopeDim + vDim)) * UInt64(latentDim))
        let bits = Int(sourceBits)
        let packed = try readTensor(weight, snapshotDirectory: snapshotDirectory)
            .withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let scaleHalves = try readTensor(scales, snapshotDirectory: snapshotDirectory)
            .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        let biasHalves = try readTensor(biases, snapshotDirectory: snapshotDirectory)
            .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }

        let valuesPerWord = 32 / bits
        let wordsPerRow = latentDim / valuesPerWord
        let groupsPerRow = latentDim / groupSize
        let mask = UInt32((1 << bits) - 1)

        func sourceValue(row: Int, column: Int) -> Float {
            let word = packed[row * wordsPerRow + column / valuesPerWord]
            let shift = UInt32((column % valuesPerWord) * bits)
            let q = Float((word >> shift) & mask)
            let groupIndex = row * groupsPerRow + column / groupSize
            return bf16ToFloat(scaleHalves[groupIndex]) * q
                + bf16ToFloat(biasHalves[groupIndex])
        }

        let outRows = heads * latentDim
        let outGroupsPerRow = nopeDim / groupSize
        let outWordsPerRow = nopeDim / 4
        var outWords = [UInt32](repeating: 0, count: outRows * outWordsPerRow)
        var outScales = [UInt16](repeating: 0, count: outRows * outGroupsPerRow)
        var outBiases = [UInt16](repeating: 0, count: outRows * outGroupsPerRow)
        var row = [Float](repeating: 0, count: nopeDim)

        for head in 0..<heads {
            let kBase = head * (nopeDim + vDim)
            for l in 0..<latentDim {
                for n in 0..<nopeDim {
                    row[n] = sourceValue(row: kBase + n, column: l)
                }
                let outRow = head * latentDim + l
                for g in 0..<outGroupsPerRow {
                    let start = g * groupSize
                    var low = row[start]
                    var high = row[start]
                    for n in start..<(start + groupSize) {
                        low = min(low, row[n])
                        high = max(high, row[n])
                    }
                    // Quantize against the BF16-rounded scale/bias the runtime
                    // will read back, not the exact f32 pair.
                    let scale = bf16ToFloat(floatToBF16((high - low) / 255))
                    let bias = bf16ToFloat(floatToBF16(low))
                    outScales[outRow * outGroupsPerRow + g] = floatToBF16(scale)
                    outBiases[outRow * outGroupsPerRow + g] = floatToBF16(bias)
                    for n in start..<(start + groupSize) {
                        let q: UInt32
                        if scale > 0 {
                            q = UInt32(max(0, min(255,
                                (row[n] - bias) / scale + 0.5)))
                        } else {
                            q = 0
                        }
                        let word = outRow * outWordsPerRow + n / 4
                        outWords[word] |= (q & 0xFF) << UInt32((n % 4) * 8)
                    }
                }
            }
        }

        try pwriteAll(fd: residentFD, path: residentPath,
                      bytes: outWords.withUnsafeBytes { Data($0) },
                      expectedSize: entry.sizeBytes, offset: entry.fileOffset)
        try pwriteAll(fd: residentFD, path: residentPath,
                      bytes: outScales.withUnsafeBytes { Data($0) },
                      expectedSize: entry.scaleSize, offset: entry.scaleOffset)
        try pwriteAll(fd: residentFD, path: residentPath,
                      bytes: outBiases.withUnsafeBytes { Data($0) },
                      expectedSize: entry.biasSize, offset: entry.biasOffset)
    }

    // MARK: - IO + format helpers

    private static func readTensor(_ tensor: SourceTensor,
                                   snapshotDirectory: String) throws -> Data {
        let path = (snapshotDirectory as NSString)
            .appendingPathComponent(tensor.shardPath)
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        var data = Data(count: Int(tensor.sizeBytes))
        try data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            try Posix.preadAll(fd: fd, path: path, buf: base,
                               count: Int(tensor.sizeBytes),
                               offset: tensor.absoluteOffset)
        }
        return data
    }

    private static func pwriteAll(fd: Int32, path: String, bytes: Data,
                                  expectedSize: UInt64, offset: UInt64) throws {
        guard UInt64(bytes.count) == expectedSize else {
            throw RepackError.configurationInvalid(
                detail: "computed tensor produced \(bytes.count) bytes; "
                    + "the plan reserved \(expectedSize)")
        }
        try bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            var written = 0
            while written < raw.count {
                let n = pwrite(fd, base + written, raw.count - written,
                               off_t(offset) + off_t(written))
                guard n > 0 else {
                    throw RepackError.pwriteShort(path: path,
                                                  expected: raw.count,
                                                  wrote: written, errno: errno)
                }
                written += n
            }
        }
    }

    static func bf16ToFloat(_ half: UInt16) -> Float {
        Float(bitPattern: UInt32(half) << 16)
    }

    static func floatToBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }
}
