import Foundation

/// FP32 reference for MLA absorbed-form attention (Kimi-Linear): MQA over
/// fused rows `[latent | k_pe]` of `qkDim` elements where V is each row's
/// `vDim`-prefix. Materializes the full score row per Q head, softmaxes,
/// then reduces against the row prefixes — no online merging.
public enum MLAAttentionRef {
    /// Q layout: `[numQHeads, qkDim]`. Rows: `[seqLen, qkDim]`.
    /// Output: `[numQHeads, vDim]`.
    public static func apply(
        q: [Float],
        rows: [Float],
        qkDim: Int,
        vDim: Int,
        numQHeads: Int,
        seqLen: Int,
        scale: Float
    ) -> [Float] {
        precondition(q.count == numQHeads * qkDim)
        precondition(rows.count == seqLen * qkDim)
        precondition(vDim <= qkDim)

        var out = [Float](repeating: 0, count: numQHeads * vDim)
        guard seqLen > 0 else { return out }
        for qh in 0..<numQHeads {
            let qBase = qh * qkDim
            var scores = [Float](repeating: 0, count: seqLen)
            for p in 0..<seqLen {
                var dot: Float = 0
                for i in 0..<qkDim { dot += q[qBase + i] * rows[p * qkDim + i] }
                scores[p] = dot * scale
            }
            var mx = -Float.infinity
            for s in scores { mx = max(mx, s) }
            var sum: Float = 0
            for p in 0..<seqLen {
                scores[p] = expf(scores[p] - mx)
                sum += scores[p]
            }
            let inv = 1 / sum
            for d in 0..<vDim {
                var acc: Float = 0
                for p in 0..<seqLen { acc += scores[p] * rows[p * qkDim + d] }
                out[qh * vDim + d] = acc * inv
            }
        }
        return out
    }
}
