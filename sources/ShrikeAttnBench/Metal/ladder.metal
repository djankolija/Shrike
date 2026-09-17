#include <metal_stdlib>
using namespace metal;

// attention_decode_partial_shared hardcoded to the served shape, one switch per
// function constant; at the defaults it must time within 10 % of the production
// pipeline or the ladder is void (docs/v19-scan-rewrite.md, S0.3).
constant bool L_Q_REGS     [[function_constant(0)]];
constant uint L_POS_BLOCK  [[function_constant(1)]];
constant bool L_DOUBLE_BUF [[function_constant(2)]];
constant bool L_NO_SOFTMAX [[function_constant(3)]];
constant bool L_NO_V       [[function_constant(4)]];
constant bool L_LOAD_ONLY  [[function_constant(5)]];
constant bool L_FULL_ROW   [[function_constant(6)]];
constant uint L_LOAD_BYTES [[function_constant(7)]];
constant bool L_STATIC_LOOPS [[function_constant(8)]];
constant uint L_O_FORM [[function_constant(9)]];
constant uint L_D_FORM [[function_constant(10)]];

static inline float l_o_update(float o, float alpha, float p, float v) {
    if (L_O_FORM == 1u) { return fma(o, alpha, p * v); }
    if (L_O_FORM == 2u) { return fma(p, v, o * alpha); }
    if (L_O_FORM == 3u) { return p * v + o * alpha; }
    return o * alpha + p * v;
}

static inline float l_d_update(float d, float alpha, float p) {
    if (L_D_FORM == 1u) { return fma(d, alpha, p); }
    return d * alpha + p;
}

constant constexpr uint HD = 256;
constant constexpr uint QPKV = 8;
constant constexpr uint ROW_STRIDE = 544;
constant constexpr uint VALUES = 512;
constant constexpr uint GROUP = 64;
constant constexpr uint GROUPS = 8;
constant constexpr uint PER_LANE = HD / 32;

struct LState {
    float o[PER_LANE];
    float m;
    float d;
};

static inline void l_dequant4(uchar4 raw, float s, float b, threadgroup float* dst) {
    dst[0] = float(uint(raw.x)) * s + b;
    dst[1] = float(uint(raw.y)) * s + b;
    dst[2] = float(uint(raw.z)) * s + b;
    dst[3] = float(uint(raw.w)) * s + b;
}

// One load unit of L_LOAD_BYTES packed values at element `flat` of row `pos`,
// dequantised into dst. A unit never crosses a group (4, 8 and 16 divide 64).
static inline void l_stage_unit(device const uchar* cache, uint pos, uint flat,
                                threadgroup float* dst) {
    device const uchar* row = cache + pos * ROW_STRIDE;
    device const half* scales = reinterpret_cast<device const half*>(row + VALUES);
    const uint group = flat / GROUP;
    const float s = float(scales[group]);
    const float b = float(scales[GROUPS + group]);
    device const uchar* src = row + flat;
    if (L_LOAD_BYTES == 16u) {
        const uint4 w = *reinterpret_cast<device const uint4*>(src);
        l_dequant4(as_type<uchar4>(w.x), s, b, dst);
        l_dequant4(as_type<uchar4>(w.y), s, b, dst + 4);
        l_dequant4(as_type<uchar4>(w.z), s, b, dst + 8);
        l_dequant4(as_type<uchar4>(w.w), s, b, dst + 12);
    } else if (L_LOAD_BYTES == 8u) {
        const uint2 w = *reinterpret_cast<device const uint2*>(src);
        l_dequant4(as_type<uchar4>(w.x), s, b, dst);
        l_dequant4(as_type<uchar4>(w.y), s, b, dst + 4);
    } else {
        l_dequant4(*reinterpret_cast<device const uchar4*>(src), s, b, dst);
    }
}

static inline void l_stage_block(device const uchar* K, device const uchar* V,
                                 uint pb, uint block_count, uint slice, uint head_base,
                                 threadgroup float* kbuf, threadgroup float* vbuf,
                                 uint lid, uint lsize) {
    const uint units_per_row = slice / L_LOAD_BYTES;
    const uint total = block_count * units_per_row;
    for (uint u = lid; u < total; u += lsize) {
        const uint j = u / units_per_row;
        const uint base = (u - j * units_per_row) * L_LOAD_BYTES;
        const uint pos = pb + j;
        l_stage_unit(K, pos, head_base + base, kbuf + j * slice + base);
        if (!L_NO_V) {
            l_stage_unit(V, pos, head_base + base, vbuf + j * slice + base);
        }
    }
}

// The shipped loops walk i = lane, lane + 32, ... with a slot counter the
// compiler cannot bound; the static form walks s = 0 ..< PER_LANE, the same
// elements in the same order, so the two must hash identical.
static inline void l_compute_block(threadgroup const float* kbuf,
                                   threadgroup const float* vbuf,
                                   uint block_count,
                                   thread const float* q_reg,
                                   threadgroup const float* q_mine,
                                   float scale, thread LState& st, uint lane) {
    for (uint j = 0; j < block_count; ++j) {
        threadgroup const float* k_row = kbuf + j * HD;
        threadgroup const float* v_row = vbuf + j * HD;
        float partial = 0.0f;
        if (L_STATIC_LOOPS) {
            for (uint s = 0; s < PER_LANE; ++s) {
                const uint i = lane + 32u * s;
                const float qv = L_Q_REGS ? q_reg[s] : q_mine[i];
                partial = fma(qv, k_row[i], partial);
            }
        } else {
            uint slot = 0;
            for (uint i = lane; i < HD; i += 32u) {
                const float qv = L_Q_REGS ? q_reg[slot] : q_mine[i];
                partial = fma(qv, k_row[i], partial);
                slot += 1;
            }
        }
        const float s = simd_sum(partial) * scale;
        if (L_NO_SOFTMAX) {
            st.d += s;
            if (!L_NO_V) {
                if (L_STATIC_LOOPS) {
                    for (uint t = 0; t < PER_LANE; ++t) {
                        st.o[t] = fma(s, v_row[lane + 32u * t], st.o[t]);
                    }
                } else {
                    uint slot = 0;
                    for (uint i = lane; i < HD; i += 32u) {
                        st.o[slot] = fma(s, v_row[i], st.o[slot]);
                        slot += 1;
                    }
                }
            }
        } else {
            const float m_new = max(st.m, s);
            const float alpha = fast::exp(st.m - m_new);
            const float p_exp = fast::exp(s - m_new);
            st.d = l_d_update(st.d, alpha, p_exp);
            if (!L_NO_V) {
                if (L_STATIC_LOOPS) {
                    for (uint t = 0; t < PER_LANE; ++t) {
                        st.o[t] = l_o_update(st.o[t], alpha, p_exp, v_row[lane + 32u * t]);
                    }
                } else {
                    uint slot = 0;
                    for (uint i = lane; i < HD; i += 32u) {
                        st.o[slot] = st.o[slot] * alpha + p_exp * v_row[i];
                        slot += 1;
                    }
                }
            }
            st.m = m_new;
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void ladder_partial(
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
    threadgroup float*  smem       [[threadgroup(0)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]]
) {
    const uint kv_head = L_FULL_ROW ? 0u : tg_id / num_chunks;
    const uint chunk = L_FULL_ROW ? tg_id : tg_id % num_chunks;
    const uint p_start = chunk * chunk_len;
    const uint p_end = min(p_start + chunk_len, seq_len);
    const uint slice = L_FULL_ROW ? VALUES : HD;
    const uint head_base = L_FULL_ROW ? 0u : kv_head * HD;
    const uint block_floats = L_POS_BLOCK * slice;
    threadgroup float* q_smem = smem;
    threadgroup float* k_smem = smem + (L_Q_REGS ? 0u : QPKV * HD);
    threadgroup float* v_smem = k_smem + (L_DOUBLE_BUF ? 2u : 1u) * block_floats;
    const bool live = sg < QPKV;

    float q_reg[PER_LANE];
    if (L_Q_REGS) {
        for (uint s = 0; s < PER_LANE; ++s) {
            q_reg[s] = live ? float(Q[(kv_head * QPKV + sg) * HD + lane + 32u * s]) : 0.0f;
        }
    } else if (!L_LOAD_ONLY) {
        for (uint i = lid; i < QPKV * HD; i += lsize) {
            const uint head = i / HD;
            q_smem[i] = float(Q[(kv_head * QPKV + head) * HD + (i - head * HD)]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup const float* q_mine = q_smem + sg * HD;

    LState st;
    for (uint s = 0; s < PER_LANE; ++s) { st.o[s] = 0.0f; }
    st.m = -INFINITY;
    st.d = 0.0f;
    float acc = 0.0f;

    uint parity = 0;
    if (L_DOUBLE_BUF && p_start < p_end) {
        l_stage_block(K, V, p_start, min(L_POS_BLOCK, p_end - p_start), slice, head_base,
                      k_smem, v_smem, lid, lsize);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint pb = p_start; pb < p_end; pb += L_POS_BLOCK) {
        const uint block_count = min(L_POS_BLOCK, p_end - pb);
        threadgroup float* kbuf = k_smem;
        threadgroup float* vbuf = v_smem;
        if (L_DOUBLE_BUF) {
            kbuf = k_smem + parity * block_floats;
            vbuf = v_smem + parity * block_floats;
            const uint nb = pb + L_POS_BLOCK;
            if (nb < p_end) {
                l_stage_block(K, V, nb, min(L_POS_BLOCK, p_end - nb), slice, head_base,
                              k_smem + (parity ^ 1u) * block_floats,
                              v_smem + (parity ^ 1u) * block_floats, lid, lsize);
            }
        } else {
            l_stage_block(K, V, pb, block_count, slice, head_base, kbuf, vbuf, lid, lsize);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (L_LOAD_ONLY) {
            acc += kbuf[lid] + (L_NO_V ? 0.0f : vbuf[lid]);
        } else if (live) {
            l_compute_block(kbuf, vbuf, block_count, q_reg, q_mine, scale, st, lane);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        parity ^= 1u;
    }

    if (L_LOAD_ONLY) {
        o_out[tg_id * lsize + lid] = acc;
        return;
    }
    if (live) {
        const uint q_head = kv_head * QPKV + sg;
        const uint base = q_head * num_chunks + chunk;
        if (lane == 0) { m_out[base] = st.m; d_out[base] = st.d; }
        device float* o_row = o_out + base * HD;
        if (L_STATIC_LOOPS) {
            for (uint t = 0; t < PER_LANE; ++t) { o_row[lane + 32u * t] = st.o[t]; }
        } else {
            uint slot = 0;
            for (uint i = lane; i < HD; i += 32u) {
                o_row[i] = st.o[slot];
                slot += 1;
            }
        }
    }
}
