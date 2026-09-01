#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16 or affine INT8/4.
//   V   : [seq_len, num_kv_heads, head_dim]             Same format as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
// gpt-oss attention sinks: a learned per-Q-head logit that joins the combine's
// softmax max and denominator as one extra term and contributes no value row.
constant bool FC_ATTN_HAS_SINKS [[function_constant(66)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];

static inline bool attn_fc_has_sinks() {
    return is_function_constant_defined(FC_ATTN_HAS_SINKS) && FC_ATTN_HAS_SINKS;
}

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

static inline float attn_load_kv(
    device const uchar* cache,
    uint physical_position,
    uint flat_element,
    uint elements_per_row,
    uint bits,
    uint row_stride,
    uint values_bytes,
    uint group_size
) {
    if (bits == 16u) {
        device const half* fp16 = reinterpret_cast<device const half*>(cache);
        return float(fp16[physical_position * elements_per_row + flat_element]);
    }
    device const uchar* row = cache + physical_position * row_stride;
    uint quantized;
    if (bits == 8u) {
        quantized = uint(row[flat_element]);
    } else {
        const uchar packed = row[flat_element / 2u];
        quantized = (flat_element & 1u) == 0u
            ? uint(packed & 0x0fu) : uint(packed >> 4u);
    }
    const uint groups = (elements_per_row + group_size - 1u) / group_size;
    device const half* scales = reinterpret_cast<device const half*>(row + values_bytes);
    device const half* biases = scales + groups;
    const uint group = flat_element / group_size;
    return float(quantized) * float(scales[group]) + float(biases[group]);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
                              uint simd_lane_id,
                              uint simd_group_id,
                              uint simdgroups,
                              threadgroup float* scratch,
                              threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            const uint flat = kv_head * HD + i;
            const float kval = attn_load_kv(K, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            partial = fma(q_smem[i], kval, partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            const uint flat = kv_head * HD + i;
            const float vval = attn_load_kv(V, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            o_local[slot] = o_local[slot] * alpha + p_exp * vval;
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// v11 inner-loop variant: one SIMD group per position — the 256-dim dot is
// 32 lanes × 8 strided elements reduced with simd_sum alone, so the position
// loop runs with NO threadgroup barrier (the default kernel pays a full
// block_reduce_sum with two barriers per position). Each simdgroup keeps its
// own online-softmax state over its strided position subset; the chunk-end
// merge applies the combine's rescale algebra one level down. Reorders the
// softmax summation relative to attention_decode_partial (a264b22-class).
[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial_sg(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float m_scratch[kAttnMaxSimdGroups];
    threadgroup float d_scratch[kAttnMaxSimdGroups];
    threadgroup float o_scratch[kAttnMaxSimdGroups * kAttnMaxHeadDim];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerLane = (kAttnMaxHeadDim + 31u) / 32u;
    float o_local[kPerLane];
    for (uint k = 0; k < kPerLane; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start + simd_group_id; p < p_end; p += simdgroups) {
        const uint phys_p = attn_ring_slot(p);
        float partial = 0.0f;
        for (uint i = simd_lane_id; i < HD; i += 32u) {
            const uint flat = kv_head * HD + i;
            const float kval = attn_load_kv(K, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            partial = fma(q_smem[i], kval, partial);
        }
        const float s = simd_sum(partial) * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = simd_lane_id; i < HD; i += 32u) {
            const uint flat = kv_head * HD + i;
            const float vval = attn_load_kv(V, phys_p, flat, NKV * HD,
                                             kv_bits, kv_stride, kv_value_bytes,
                                             kv_group_size);
            o_local[slot] = o_local[slot] * alpha + p_exp * vval;
            slot += 1;
        }
        m_run = m_new;
    }

    if (simd_lane_id == 0) {
        m_scratch[simd_group_id] = m_run;
        d_scratch[simd_group_id] = d_run;
    }
    uint slot = 0;
    for (uint i = simd_lane_id; i < HD; i += 32u) {
        o_scratch[simd_group_id * kAttnMaxHeadDim + i] = o_local[slot];
        slot += 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_glob = -INFINITY;
    for (uint g = 0; g < simdgroups; ++g) { m_glob = max(m_glob, m_scratch[g]); }
    // An all-empty chunk must publish (-inf, 0, 0) exactly like the default
    // kernel: with m_glob still -inf, the rescale exponent is (-inf) - (-inf)
    // = NaN, so the merge below is guarded rather than computed.
    const bool empty = (m_glob == -INFINITY);
    float d_glob = 0.0f;
    if (!empty) {
        for (uint g = 0; g < simdgroups; ++g) {
            d_glob += d_scratch[g] * attn_softmax_exp(m_scratch[g] - m_glob);
        }
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_glob; d_out[base] = d_glob; }
    device float* o_row = o_out + base * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        if (!empty) {
            for (uint g = 0; g < simdgroups; ++g) {
                acc += o_scratch[g * kAttnMaxHeadDim + i]
                    * attn_softmax_exp(m_scratch[g] - m_glob);
            }
        }
        o_row[i] = acc;
    }
}

// v11 V4: KV-head-shared partial. One TG owns a (kv_head, chunk) pair and
// stages each K/V row into threadgroup memory ONCE; the qPerKV simdgroups
// each dot their own Q head against the staged row — device traffic drops by
// the sharing degree (8× at ornith's 16/2). Own head-dim cap (256) so the
// static threadgroup arrays stay at ~10 KB and occupancy survives (the T2
// lesson). Reduction order differs from attention_decode_partial in the
// intra-dot tree (lane-strided chain + simd_sum) — a264b22-class, signed off
// 2026-09-01.
constant constexpr uint kAttnSharedMaxHeadDim = 256;
// Positions staged per barrier round: divides the two TG barriers per
// position by the block size; 4 keeps k+v staging at 8 KB.
constant constexpr uint kAttnSharedPosBlock = 4;

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial_shared(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],
    device       float* d_out         [[buffer(4)]],
    device       float* o_out         [[buffer(5)]],
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxSimdGroups * kAttnSharedMaxHeadDim];
    threadgroup float k_smem[kAttnSharedPosBlock * kAttnSharedMaxHeadDim];
    threadgroup float v_smem[kAttnSharedPosBlock * kAttnSharedMaxHeadDim];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint qPerKV = NQ / NKV;

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    for (uint i = lid; i < qPerKV * HD; i += lsize) {
        const uint head = i / HD;
        q_smem[i] = float(Q[(kv_head * qPerKV + head) * HD + (i - head * HD)]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerLane = (kAttnSharedMaxHeadDim + 31u) / 32u;
    float o_local[kPerLane];
    for (uint s = 0; s < kPerLane; ++s) { o_local[s] = 0.0f; }
    float m_run = -INFINITY;
    float d_run = 0.0f;
    const bool liveHead = simd_group_id < qPerKV;
    threadgroup const float* q_mine = q_smem + simd_group_id * HD;

    for (uint pb = p_start; pb < p_end; pb += kAttnSharedPosBlock) {
        const uint blockCount = min(uint(kAttnSharedPosBlock), p_end - pb);
        for (uint e = lid; e < blockCount * HD; e += lsize) {
            const uint j = e / HD;
            const uint i = e - j * HD;
            const uint phys_p = attn_ring_slot(pb + j);
            const uint flat = kv_head * HD + i;
            k_smem[e] = attn_load_kv(K, phys_p, flat, NKV * HD,
                                      kv_bits, kv_stride, kv_value_bytes,
                                      kv_group_size);
            v_smem[e] = attn_load_kv(V, phys_p, flat, NKV * HD,
                                      kv_bits, kv_stride, kv_value_bytes,
                                      kv_group_size);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (liveHead) {
            for (uint j = 0; j < blockCount; ++j) {
                threadgroup const float* k_row = k_smem + j * HD;
                threadgroup const float* v_row = v_smem + j * HD;
                float partial = 0.0f;
                for (uint i = simd_lane_id; i < HD; i += 32u) {
                    partial = fma(q_mine[i], k_row[i], partial);
                }
                const float s = simd_sum(partial) * attn_fc_scale(scale);
                const float m_new = max(m_run, s);
                const float alpha = attn_softmax_exp(m_run - m_new);
                const float p_exp = attn_softmax_exp(s     - m_new);
                d_run = d_run * alpha + p_exp;
                uint slot = 0;
                for (uint i = simd_lane_id; i < HD; i += 32u) {
                    o_local[slot] = o_local[slot] * alpha + p_exp * v_row[i];
                    slot += 1;
                }
                m_run = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (liveHead) {
        const uint q_head = kv_head * qPerKV + simd_group_id;
        const uint base = q_head * NC + chunk;
        if (simd_lane_id == 0) { m_out[base] = m_run; d_out[base] = d_run; }
        device float* o_row = o_out + base * HD;
        uint slot = 0;
        for (uint i = simd_lane_id; i < HD; i += 32u) {
            o_row[i] = o_local[slot];
            slot += 1;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const uchar* K             [[buffer(1)]],
    device const uchar* V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    constant     uint&  kv_bits       [[buffer(14)]],
    constant     uint&  kv_stride     [[buffer(15)]],
    constant     uint&  kv_value_bytes [[buffer(16)]],
    constant     uint&  kv_group_size [[buffer(17)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const uint flat = kv_head * HD + i;
            const float k_val = attn_load_kv(K, phys_p, flat, NKV * HD,
                                              kv_bits, kv_stride, kv_value_bytes,
                                              kv_group_size);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const uint flat = kv_head * HD + i;
            const float v_val = attn_load_kv(V, phys_p, flat, NKV * HD,
                                              kv_bits, kv_stride, kv_value_bytes,
                                              kv_group_size);
            o_local[slot] += p_exp * v_val;
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// MLA (Kimi-Linear) split-KV partial: MQA over fused FP16 cache rows
// [latent | k_pe] of qk_dim elements, where V is the row's v_dim-prefix.
// Per-head Q rows are qk_dim wide ([W_UKᵀ·q_nope | q_pe]); the partial's
// o rows are v_dim wide, so the generic attention_decode_combine finishes
// the merge when called with head_dim = v_dim. Always one KV head, always
// FP16 rows, no sinks, no ring — runtime params only, no FC specialization.
// ============================================================================

constant constexpr uint kAttnMLAMaxQKDim = 576;

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_mla_partial(
    device const half*  Q             [[buffer(0)]],   // [num_q_heads, qk_dim]
    device const half*  KV            [[buffer(1)]],   // [seq_len, qk_dim] FP16
    device       float* m_out         [[buffer(2)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(4)]],   // [num_q_heads * num_chunks * v_dim]
    constant     uint&  qk_dim        [[buffer(5)]],
    constant     uint&  v_dim         [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  seq_len       [[buffer(8)]],
    constant     uint&  chunk_len     [[buffer(9)]],
    constant     uint&  num_chunks    [[buffer(10)]],
    constant     float& scale         [[buffer(11)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMLAMaxQKDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint QK = qk_dim;
    const uint VD = v_dim;
    const uint NC = num_chunks;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    if (q_head >= num_q_heads) return;
    const uint p_start = chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    device const half* Q_row = Q + uint(q_head) * QK;
    for (uint i = lid; i < QK; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        device const half* row = KV + uint(p) * QK;
        float partial = 0.0f;
        for (uint i = lid; i < QK; i += lsize) {
            partial = fma(q_smem[i], float(row[i]), partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= scale;

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < VD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha + p_exp * float(row[i]);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * VD;
    uint slot = 0;
    for (uint i = lid; i < VD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    device const bfloat* sinks       [[buffer(6)]],    // [num_q_heads], read iff FC_ATTN_HAS_SINKS
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float sink = 0.0f;
    if (attn_fc_has_sinks()) {
        sink = float(sinks[q_head]);
        m_glob = max(m_glob, sink);
    }
    if (m_glob == -INFINITY) {
        // All chunks empty (e.g. seq_len == kv_start): zero the row rather
        // than producing NaN from exp(-inf - -inf). Unreachable with sinks
        // (the sink is finite); there the normal path yields D=1, out=0.
        device half* out_row = out + uint(q_head) * HD;
        for (uint i = lid; i < HD; i += lsize) { out_row[i] = 0.0h; }
        return;
    }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    if (attn_fc_has_sinks()) { D += attn_softmax_exp(sink - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}
