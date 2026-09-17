import Foundation
import Metal

/// Fixed-size recurrent state for gated-DeltaNet linear-attention layers
/// (`LayerKind.linear`). Unlike KV rows, this state does not grow with
/// context: each linear layer owns
///
///  - a delta-rule state `S`, FP32 `[numVHeads, valueHeadDim, keyHeadDim]`
///    (2 MiB per layer at Qwen 3.6's 32x128x128), and
///  - a causal-conv tail of the last `convKernelSize - 1` pre-activation
///    `mixed_qkv` rows, FP16 `[convKernelSize - 1, convDim]`,
///
/// in two parities (v20 T3.3): a decode step reads the parity holding the
/// state at the cursor and writes the other, so a pass committed ahead of a
/// stop and then drained leaves the parity the stop saw untouched. Prefill,
/// the snapshot and the restore work in place on one parity, the caller's.
///
/// Buffers are allocated once in `init`; the decode hot path never allocates.
/// `reset()` zero-fills both parities explicitly — the recurrence and the
/// conv both define the empty-context state as zeros, and zeroing 120-odd
/// MiB per generation start is cheap next to a prefill.
public final class GDNStateManager {
    public let config: ArchConfig

    /// `[parity][layer]`, non-nil only at indices whose layer mask is 2.
    private let stateBuffers: [[MTLBuffer?]]
    private let convTailBuffers: [[MTLBuffer?]]

    public let stateBytesPerLayer: Int
    public let convTailBytesPerLayer: Int

    public static let parityCount = 2

    private static let fp32Size = 4
    private static let fp16Size = 2

    public init(device: MTLDevice, config: ArchConfig) throws {
        self.config = config
        let la = config.linearAttention
        let stateBytes = la.numVHeads * la.valueHeadDim * la.keyHeadDim * Self.fp32Size
        let convTailBytes = max(0, la.convKernelSize - 1) * la.qkvDim * Self.fp16Size
        self.stateBytesPerLayer = stateBytes
        self.convTailBytesPerLayer = convTailBytes

        var states: [[MTLBuffer?]] = []
        var tails: [[MTLBuffer?]] = []
        for parity in 0..<Self.parityCount {
            var parityStates: [MTLBuffer?] = []
            var parityTails: [MTLBuffer?] = []
            parityStates.reserveCapacity(config.numLayers)
            parityTails.reserveCapacity(config.numLayers)
            for layer in 0..<config.numLayers {
                guard config.layerIsLinear(layer) else {
                    parityStates.append(nil)
                    parityTails.append(nil)
                    continue
                }
                guard stateBytes > 0, convTailBytes > 0 else {
                    throw ModelError.internalInconsistency(
                        detail: "linear layer \(layer) present but linearAttention config is empty")
                }
                guard let state = device.makeBuffer(length: stateBytes,
                                                    options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                state.label = "gdn.state.layer\(layer).p\(parity)"
                guard let tail = device.makeBuffer(length: convTailBytes,
                                                   options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                tail.label = "gdn.convtail.layer\(layer).p\(parity)"
                parityStates.append(state)
                parityTails.append(tail)
            }
            states.append(parityStates)
            tails.append(parityTails)
        }
        self.stateBuffers = states
        self.convTailBuffers = tails
        zeroAll()
    }

    /// Delta-rule state `S` for a linear layer at `parity`.
    public func stateBuffer(layer: Int, parity: Int) -> MTLBuffer {
        guard let buffer = stateBuffers[parity][layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    /// Rolling window of the last `convKernelSize - 1` mixed_qkv rows at `parity`.
    public func convTailBuffer(layer: Int, parity: Int) -> MTLBuffer {
        guard let buffer = convTailBuffers[parity][layer] else {
            preconditionFailure("layer \(layer) is not a linear-attention layer")
        }
        return buffer
    }

    public func isLinear(layer: Int) -> Bool { stateBuffers[0][layer] != nil }

    /// Reset all recurrent state, both parities, to the empty-context value (zeros).
    public func reset() {
        zeroAll()
    }

    func snapshotSegmentLengths() -> [Int] {
        var lengths: [Int] = []
        lengths.reserveCapacity(config.numLayers * 2)
        for layer in 0..<config.numLayers where isLinear(layer: layer) {
            lengths.append(stateBytesPerLayer)
            lengths.append(convTailBytesPerLayer)
        }
        return lengths
    }

    func appendSnapshotPayload(to payload: inout Data,
                               segmentLengths: [Int],
                               parity: Int) throws {
        guard segmentLengths == snapshotSegmentLengths() else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        var segment = 0
        for layer in 0..<config.numLayers {
            guard let state = stateBuffers[parity][layer],
                  let tail = convTailBuffers[parity][layer] else { continue }
            let stateLength = segmentLengths[segment]
            payload.append(state.contents().assumingMemoryBound(to: UInt8.self),
                           count: stateLength)
            segment += 1
            let tailLength = segmentLengths[segment]
            payload.append(tail.contents().assumingMemoryBound(to: UInt8.self),
                           count: tailLength)
            segment += 1
        }
    }

    func restoreSnapshot(segmentLengths: [Int],
                         bytes: UnsafeRawBufferPointer,
                         offset: inout Int,
                         parity: Int) throws {
        guard segmentLengths == snapshotSegmentLengths() else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        reset()
        var segment = 0
        for layer in 0..<config.numLayers {
            guard let state = stateBuffers[parity][layer],
                  let tail = convTailBuffers[parity][layer] else { continue }
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: segmentLengths[segment],
                                    destination: state)
            segment += 1
            try copySnapshotSegment(bytes: bytes,
                                    offset: &offset,
                                    length: segmentLengths[segment],
                                    destination: tail)
            segment += 1
        }
    }

    private func copySnapshotSegment(bytes: UnsafeRawBufferPointer,
                                     offset: inout Int,
                                     length: Int,
                                     destination: MTLBuffer) throws {
        guard length <= destination.length,
              offset >= 0,
              length >= 0,
              offset <= bytes.count - length,
              let source = bytes.baseAddress?.advanced(by: offset) else {
            throw InferenceStateSnapshotError.invalidLayout
        }
        memcpy(destination.contents(), source, length)
        offset += length
    }

    private func zeroAll() {
        for parity in stateBuffers {
            for buffer in parity {
                if let buffer { memset(buffer.contents(), 0, buffer.length) }
            }
        }
        for parity in convTailBuffers {
            for buffer in parity {
                if let buffer { memset(buffer.contents(), 0, buffer.length) }
            }
        }
    }
}
