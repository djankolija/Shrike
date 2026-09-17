#include <metal_stdlib>
using namespace metal;

// The streaming scan (docs/v19-scan-rewrite.md, Approach A): rows from device to
// registers, no threadgroup memory and no barrier in the loop. A threadgroup owns
// one KV head and one chunk; its eight simdgroups are S_HPT position streams by
// (8 / S_HPT) head sets, so every row is read by the head sets of one stream and
// each stream writes its own partial: 64 / S_HPT chunks dispatched, 64 partials
// per query head as today.
constant uint S_HPT     [[function_constant(0)]];
constant bool S_NO_LOAD [[function_constant(1)]];
constant bool S_UNROLL2 [[function_constant(2)]];
constant bool S_LAZY    [[function_constant(3)]];

constant constexpr uint HD = 256;
constant constexpr uint QPKV = 8;
constant constexpr uint ROW_STRIDE = 544;
constant constexpr uint VALUES = 512;
constant constexpr uint GROUP = 64;
constant constexpr uint GROUPS = 8;
constant constexpr uint PER_LANE = HD / 32;
constant constexpr uint HPT_MAX = 8;
constant constexpr uint OUT_CHUNKS = 64;

struct SState {
    float q[HPT_MAX][PER_LANE];
    float o[HPT_MAX][PER_LANE];
    float m[HPT_MAX];
    float d[HPT_MAX];
};

static inline uint2 s_load8(device const uchar* cache, uint pos, uint offset) {
    if (S_NO_LOAD) {
        const uint a = pos * 2654435761u ^ (offset * 40503u);
        return uint2(a, a * 1664525u + 1013904223u);
    }
    return *reinterpret_cast<device const uint2*>(cache + pos * ROW_STRIDE + offset);
}

static inline float2 s_scale_bias(device const uchar* cache, uint pos, uint group) {
    if (S_NO_LOAD) { return float2(0.001f, -0.1f); }
    device const half* scales = reinterpret_cast<device const half*>(cache + pos * ROW_STRIDE + VALUES);
    return float2(float(scales[group]), float(scales[GROUPS + group]));
}

static inline void s_dequant8(uint2 w, float2 sb, thread float* dst) {
    const uchar4 lo = as_type<uchar4>(w.x);
    const uchar4 hi = as_type<uchar4>(w.y);
    dst[0] = float(uint(lo.x)) * sb.x + sb.y;
    dst[1] = float(uint(lo.y)) * sb.x + sb.y;
    dst[2] = float(uint(lo.z)) * sb.x + sb.y;
    dst[3] = float(uint(lo.w)) * sb.x + sb.y;
    dst[4] = float(uint(hi.x)) * sb.x + sb.y;
    dst[5] = float(uint(hi.y)) * sb.x + sb.y;
    dst[6] = float(uint(hi.z)) * sb.x + sb.y;
    dst[7] = float(uint(hi.w)) * sb.x + sb.y;
}

static inline void s_dots(thread const SState& st, thread const float* k, float scale,
                          thread float* s) {
    for (uint h = 0; h < S_HPT; ++h) {
        float partial = 0.0f;
        for (uint e = 0; e < PER_LANE; ++e) { partial = fma(st.q[h][e], k[e], partial); }
        s[h] = simd_sum(partial) * scale;
    }
}

// The scores are uniform across the simdgroup after simd_sum, so the lazy
// branch is uniform: the exp and the eight-wide rescale run only when the
// running max moves.
static inline void s_softmax(thread SState& st, thread const float* s,
                             thread float* p, thread float* alpha) {
    for (uint h = 0; h < S_HPT; ++h) {
        if (S_LAZY) {
            if (s[h] > st.m[h]) {
                const float a = fast::exp(st.m[h] - s[h]);
                st.d[h] *= a;
                for (uint e = 0; e < PER_LANE; ++e) { st.o[h][e] *= a; }
                st.m[h] = s[h];
            }
            p[h] = fast::exp(s[h] - st.m[h]);
            st.d[h] += p[h];
            alpha[h] = 1.0f;
        } else {
            const float m_new = max(st.m[h], s[h]);
            alpha[h] = fast::exp(st.m[h] - m_new);
            p[h] = fast::exp(s[h] - m_new);
            st.d[h] = st.d[h] * alpha[h] + p[h];
            st.m[h] = m_new;
        }
    }
}

static inline void s_accumulate(thread SState& st, thread const float* v,
                                thread const float* p, thread const float* alpha) {
    for (uint h = 0; h < S_HPT; ++h) {
        for (uint e = 0; e < PER_LANE; ++e) {
            st.o[h][e] = S_LAZY ? fma(p[h], v[e], st.o[h][e])
                                : fma(st.o[h][e], alpha[h], p[h] * v[e]);
        }
    }
}

static inline void s_one(thread SState& st, device const uchar* K, device const uchar* V,
                         uint pos, uint offset, uint group, float scale) {
    const uint2 wk = s_load8(K, pos, offset);
    const uint2 wv = s_load8(V, pos, offset);
    float k[PER_LANE];
    s_dequant8(wk, s_scale_bias(K, pos, group), k);
    float s[HPT_MAX], p[HPT_MAX], alpha[HPT_MAX];
    s_dots(st, k, scale, s);
    s_softmax(st, s, p, alpha);
    float v[PER_LANE];
    s_dequant8(wv, s_scale_bias(V, pos, group), v);
    s_accumulate(st, v, p, alpha);
}

static inline void s_two(thread SState& st, device const uchar* K, device const uchar* V,
                         uint pos, uint offset, uint group, float scale) {
    const uint2 wk0 = s_load8(K, pos, offset);
    const uint2 wk1 = s_load8(K, pos + 1u, offset);
    const uint2 wv0 = s_load8(V, pos, offset);
    const uint2 wv1 = s_load8(V, pos + 1u, offset);
    float k0[PER_LANE], k1[PER_LANE];
    s_dequant8(wk0, s_scale_bias(K, pos, group), k0);
    s_dequant8(wk1, s_scale_bias(K, pos + 1u, group), k1);
    float s0[HPT_MAX], s1[HPT_MAX];
    s_dots(st, k0, scale, s0);
    s_dots(st, k1, scale, s1);
    float p0[HPT_MAX], a0[HPT_MAX], p1[HPT_MAX], a1[HPT_MAX];
    s_softmax(st, s0, p0, a0);
    float v0[PER_LANE];
    s_dequant8(wv0, s_scale_bias(V, pos, group), v0);
    s_accumulate(st, v0, p0, a0);
    s_softmax(st, s1, p1, a1);
    float v1[PER_LANE];
    s_dequant8(wv1, s_scale_bias(V, pos + 1u, group), v1);
    s_accumulate(st, v1, p1, a1);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void stream_partial(
    device const half*  Q          [[buffer(0)]],
    device const uchar* K          [[buffer(1)]],
    device const uchar* V          [[buffer(2)]],
    device float*       m_out      [[buffer(3)]],
    device float*       d_out      [[buffer(4)]],
    device float*       o_out      [[buffer(5)]],
    constant uint&      seq_len    [[buffer(6)]],
    constant uint&      chunk_len  [[buffer(7)]],
    constant uint&      num_chunks [[buffer(8)]],
    constant float&     scale      [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]]
) {
    const uint head_sets = QPKV / S_HPT;
    const uint streams = S_HPT;
    const uint kv_head = tg_id / num_chunks;
    const uint chunk = tg_id % num_chunks;
    const uint head_set = sg % head_sets;
    const uint stream = sg / head_sets;
    const uint p_start = chunk * chunk_len;
    const uint p_end = min(p_start + chunk_len, seq_len);
    const uint run = (p_end > p_start) ? (p_end - p_start + streams - 1u) / streams : 0u;
    const uint s_start = p_start + stream * run;
    const uint s_end = min(s_start + run, p_end);
    const uint head_base = kv_head * HD;
    const uint offset = head_base + PER_LANE * lane;
    const uint group = offset / GROUP;
    const uint q_head0 = kv_head * QPKV + head_set * S_HPT;

    SState st;
    for (uint h = 0; h < S_HPT; ++h) {
        device const half* qrow = Q + (q_head0 + h) * HD + PER_LANE * lane;
        for (uint e = 0; e < PER_LANE; ++e) {
            st.q[h][e] = float(qrow[e]);
            st.o[h][e] = 0.0f;
        }
        st.m[h] = -INFINITY;
        st.d[h] = 0.0f;
    }

    uint pos = s_start;
    if (S_UNROLL2) {
        for (; pos + 1u < s_end; pos += 2u) { s_two(st, K, V, pos, offset, group, scale); }
    }
    for (; pos < s_end; ++pos) { s_one(st, K, V, pos, offset, group, scale); }

    const uint out_chunk = chunk * streams + stream;
    for (uint h = 0; h < S_HPT; ++h) {
        const uint base = (q_head0 + h) * OUT_CHUNKS + out_chunk;
        if (lane == 0) { m_out[base] = st.m[h]; d_out[base] = st.d[h]; }
        device float* o_row = o_out + base * HD + PER_LANE * lane;
        for (uint e = 0; e < PER_LANE; ++e) { o_row[e] = st.o[h][e]; }
    }
}
