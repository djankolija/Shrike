#include <metal_stdlib>
using namespace metal;

// ============================================================================
// mla.metal — Kimi-Linear MLA absorbed-form projections (layer-mask value 3).
//
// Per token the absorbed form needs, for every query head h:
//   embed:   mlaQ[h] = [ W_UKᵀ[h] · q_nope[h]  |  q_pe[h] ]      (576 = 512+64)
//   unembed: out[h]  = W_UV[h] · attn[h]                          (128 from 512)
//
// Both are 32 tiny per-head GEMVs per token; these kernels batch every
// (token, head, output-row) into one dispatch, one SIMD per output row,
// 8 rows per threadgroup — the same shape as the dequant GEMV kernels.
//
// Weights come from the repacked kv_b_proj split:
//   embed_q     [H * latent, nope]  8-bit affine g64 (requantized transpose)
//   unembed_out [H * vDim,  latent] source-precision affine (4-bit) g64
//
// K24 applies: the affine row-dot bodies are duplicated from
// dequant_int8.metal / dequant_int4.metal so this module stays self-contained
// in the concatenated library; keep the math in lockstep with those.
// ============================================================================

constant constexpr uint kMLAGroupSize = 64;
constant constexpr uint kMLARowsPerTG = 8;

static inline float mla_int8_row_dot(
    device const uint8_t* W_row,
    device const bfloat*  s_row,
    device const bfloat*  b_row,
    device const half*    x,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMLAGroupSize;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint i0 = g * kMLAGroupSize + lane * 2u;
        const uint i1 = i0 + 1u;
        const float q0 = float(uint(W_row[i0]));
        const float q1 = float(uint(W_row[i1]));
        const float x0 = float(x[i0]);
        const float x1 = float(x[i1]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float mla_int4_row_dot(
    device const uint8_t* W_row,
    device const bfloat*  s_row,
    device const bfloat*  b_row,
    device const half*    x,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMLAGroupSize;
    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Sub-tensor weight offsets are only 2-byte aligned; read the 4-byte
        // chunk as two ushorts (same constraint as the other int4 bodies).
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g  = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 =  w4        & 0xFFu;
        const uint b1 = (w4 >> 8)  & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y), e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y), e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMLAGroupSize / 2) + lane];
        const float x0 = float(x[g * kMLAGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMLAGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// mlaQ[t, h, 0..latent)      = W_UKᵀ[h] · qRaw[t, h, 0..nope)   (int8 dot)
// mlaQ[t, h, latent..+rope)  = qRaw[t, h, nope..+rope)          (copy)
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mla_embed_q(
    device const uint8_t* W        [[buffer(0)]],   // [H * latent, nope] int8
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const half*    qRaw     [[buffer(3)]],   // [T, H * (nope + rope)]
    device half*          y        [[buffer(4)]],   // [T, H * (latent + rope)]
    constant uint&        numHeads [[buffer(5)]],
    constant uint&        nopeDim  [[buffer(6)]],
    constant uint&        ropeDim  [[buffer(7)]],
    constant uint&        latentDim [[buffer(8)]],
    constant uint&        tokens   [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint H = numHeads;
    const uint rowsPerHead = latentDim + ropeDim;
    const uint flat = tg_idx * kMLARowsPerTG + sg_idx;
    if (flat >= tokens * H * rowsPerHead) return;

    const uint t = flat / (H * rowsPerHead);
    const uint rem = flat - t * H * rowsPerHead;
    const uint h = rem / rowsPerHead;
    const uint r = rem - h * rowsPerHead;

    const uint qStride = H * (nopeDim + ropeDim);
    device const half* x = qRaw + t * qStride + h * (nopeDim + ropeDim);
    device half* out = y + t * H * rowsPerHead + h * rowsPerHead + r;

    if (r >= latentDim) {
        if (lane == 0) { *out = x[nopeDim + (r - latentDim)]; }
        return;
    }
    const uint row = h * latentDim + r;
    const uint n_groups = nopeDim / kMLAGroupSize;
    const float acc = mla_int8_row_dot(W + uint(row) * nopeDim,
                                       scales + uint(row) * n_groups,
                                       biases + uint(row) * n_groups,
                                       x, nopeDim, lane);
    if (lane == 0) { *out = half(acc); }
}

// out[t, h, 0..vDim) = W_UV[h] · attn[t, h, 0..latent)  (int4 dot)
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void mla_unembed(
    device const uint8_t* W        [[buffer(0)]],   // [H * vDim, latent] int4
    device const bfloat*  scales   [[buffer(1)]],
    device const bfloat*  biases   [[buffer(2)]],
    device const half*    attn     [[buffer(3)]],   // [T, H * latent]
    device half*          y        [[buffer(4)]],   // [T, H * vDim]
    constant uint&        numHeads [[buffer(5)]],
    constant uint&        latentDim [[buffer(6)]],
    constant uint&        vDim     [[buffer(7)]],
    constant uint&        tokens   [[buffer(8)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    const uint H = numHeads;
    const uint flat = tg_idx * kMLARowsPerTG + sg_idx;
    if (flat >= tokens * H * vDim) return;

    const uint t = flat / (H * vDim);
    const uint rem = flat - t * H * vDim;
    const uint h = rem / vDim;
    const uint r = rem - h * vDim;

    device const half* x = attn + t * H * latentDim + h * latentDim;
    const uint row = h * vDim + r;
    const uint n_groups = latentDim / kMLAGroupSize;
    const float acc = mla_int4_row_dot(W + uint(row) * (latentDim / 2),
                                       scales + uint(row) * n_groups,
                                       biases + uint(row) * n_groups,
                                       x, latentDim, lane);
    if (lane == 0) { y[t * H * vDim + h * vDim + r] = half(acc); }
}
