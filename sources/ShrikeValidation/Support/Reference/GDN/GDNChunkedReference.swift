import Foundation
import Shrike

/// CPU model of the chunked gated delta rule the `gdn_chunk_*` kernels run
/// (v12 P4): `GDNReference.deltaRule` rewritten over 64-row chunks as matrix
/// products. The derivation and the names used here are in
/// docs/v12-implementation-plan.md, Task 4. Scalar per-head decay only.
public struct GDNChunkedReference {
    public static let chunk = 64

    public let cfg: LinearAttentionConfig
    public let aLog: [Float]           // [Hv]
    public let dtBias: [Float]         // [Hv]

    public init(cfg: LinearAttentionConfig, aLog: [Float], dtBias: [Float]) {
        self.cfg = cfg
        self.aLog = aLog
        self.dtBias = dtBias
    }

    /// `normed[t]` are the `[q][k][v]` rows `GDNReference.normalize` returns,
    /// `a[t]`/`b[t]` are `[Hv]` rows; `state` is `[Hv, Dv, Dk]` and is replaced
    /// by the state after the last row. Returns the fp32 outputs
    /// `[T][Hv * Dv]` (not fp16-rounded) and the state after row 0.
    public func run(normed: [[Float]], a: [[Float]], b: [[Float]],
                    state: inout [Float]) -> (y: [[Float]], checkpoint: [Float]) {
        let T = normed.count
        let Hv = cfg.numVHeads
        let Dv = cfg.valueHeadDim
        var y = [[Float]](repeating: [Float](repeating: 0, count: Hv * Dv), count: T)
        var checkpoint = state
        let chunkCount = (T + Self.chunk - 1) / Self.chunk
        for h in 0..<Hv {
            for c in 0..<chunkCount {
                let f = factors(head: h, chunkIndex: c, normed: normed, a: a, b: b)
                scan(head: h, chunkIndex: c, factors: f, normed: normed,
                     state: &state, y: &y, checkpoint: &checkpoint)
            }
        }
        return (y, checkpoint)
    }

    private struct ChunkFactors {
        let tinv: [Float]        // [C, C], (I + A)⁻¹
        let m: [Float]           // [C, C]
        let beta: [Float]        // [C]
        let gamma: [Float]       // [C]
        let lambda: [Float]      // [C]
        let gammaChunk: Float
        let valid: Int
    }

    private func factors(head h: Int, chunkIndex c: Int,
                         normed: [[Float]], a: [[Float]], b: [[Float]]) -> ChunkFactors {
        let C = Self.chunk
        let Hk = cfg.numKHeads
        let Hv = cfg.numVHeads
        let Dk = cfg.keyHeadDim
        let hk = h / (Hv / Hk)
        let qBase = hk * Dk
        let kBase = Hk * Dk + hk * Dk
        let r0 = c * C
        let valid = min(C, normed.count - r0)
        let expA = expf(aLog[h])

        var beta = [Float](repeating: 0, count: C)
        var ell = [Float](repeating: 0, count: C)
        var running: Float = 0
        for t in 0..<C {
            if t < valid {
                running += -expA * GDNReference.softplus(a[r0 + t][h] + dtBias[h])
                beta[t] = GDNReference.sigmoid(b[r0 + t][h])
            }
            ell[t] = running
        }

        var A = [Float](repeating: 0, count: C * C)
        var M = [Float](repeating: 0, count: C * C)
        for t in 0..<valid {
            for i in 0...t {
                var kk: Float = 0
                var qk: Float = 0
                for dk in 0..<Dk {
                    kk += normed[r0 + t][kBase + dk] * normed[r0 + i][kBase + dk]
                    qk += normed[r0 + t][qBase + dk] * normed[r0 + i][kBase + dk]
                }
                let decay = expf(ell[t] - ell[i])
                if i < t { A[t * C + i] = beta[t] * decay * kk }
                M[t * C + i] = decay * qk
            }
        }

        var tinv = [Float](repeating: 0, count: C * C)
        for j in 0..<C {
            var x = [Float](repeating: 0, count: C)
            x[j] = 1
            var t = j + 1
            while t < C {
                var acc: Float = 0
                for i in j..<t { acc += A[t * C + i] * x[i] }
                x[t] = -acc
                t += 1
            }
            for row in 0..<C { tinv[row * C + j] = x[row] }
        }

        return ChunkFactors(tinv: tinv, m: M, beta: beta,
                            gamma: ell.map { expf($0) },
                            lambda: ell.map { expf(ell[C - 1] - $0) },
                            gammaChunk: expf(ell[C - 1]),
                            valid: valid)
    }

    private func scan(head h: Int, chunkIndex c: Int, factors f: ChunkFactors,
                      normed: [[Float]], state: inout [Float], y: inout [[Float]],
                      checkpoint: inout [Float]) {
        let C = Self.chunk
        let Hk = cfg.numKHeads
        let Hv = cfg.numVHeads
        let Dk = cfg.keyHeadDim
        let Dv = cfg.valueHeadDim
        let hk = h / (Hv / Hk)
        let qBase = hk * Dk
        let kBase = Hk * Dk + hk * Dk
        let vBase = 2 * Hk * Dk + h * Dv
        let r0 = c * C
        let valid = f.valid
        func s(_ dv: Int, _ dk: Int) -> Float { state[(h * Dv + dv) * Dk + dk] }

        var X = [Float](repeating: 0, count: C * Dv)
        for t in 0..<valid {
            for dv in 0..<Dv {
                var ks: Float = 0
                for dk in 0..<Dk { ks += normed[r0 + t][kBase + dk] * s(dv, dk) }
                X[t * Dv + dv] = f.beta[t] * (normed[r0 + t][vBase + dv] - f.gamma[t] * ks)
            }
        }
        var U = [Float](repeating: 0, count: C * Dv)
        for t in 0..<C {
            for dv in 0..<Dv {
                var acc: Float = 0
                for i in 0...t { acc += f.tinv[t * C + i] * X[i * Dv + dv] }
                U[t * Dv + dv] = acc
            }
        }

        if c == 0 {
            for dv in 0..<Dv {
                for dk in 0..<Dk {
                    checkpoint[(h * Dv + dv) * Dk + dk] =
                        f.gamma[0] * s(dv, dk) + U[dv] * normed[r0][kBase + dk]
                }
            }
        }

        for t in 0..<valid {
            for dv in 0..<Dv {
                var qs: Float = 0
                for dk in 0..<Dk { qs += normed[r0 + t][qBase + dk] * s(dv, dk) }
                var mu: Float = 0
                for i in 0...t { mu += f.m[t * C + i] * U[i * Dv + dv] }
                y[r0 + t][h * Dv + dv] = f.gamma[t] * qs + mu
            }
        }

        var next = [Float](repeating: 0, count: Dv * Dk)
        for dv in 0..<Dv {
            for dk in 0..<Dk {
                var acc: Float = f.gammaChunk * s(dv, dk)
                for t in 0..<valid {
                    acc += f.lambda[t] * U[t * Dv + dv] * normed[r0 + t][kBase + dk]
                }
                next[dv * Dk + dk] = acc
            }
        }
        for dv in 0..<Dv {
            for dk in 0..<Dk {
                state[(h * Dv + dv) * Dk + dk] = next[dv * Dk + dk]
            }
        }
    }
}
