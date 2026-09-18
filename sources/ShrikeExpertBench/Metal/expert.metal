// v21 S0.4: the decode phase-1 gate/up kernel over coded experts. Compiled behind
// moe.metal's source in one library, so the helpers and the RoutedBlobs struct
// are the production ones, and the arithmetic below is the production kernel's
// in the production kernel's order: lane l of a row holds, for every block of
// 256 weights, the eight weights the plain lane reads from its four bytes, so
// the fma chain and the reductions see the same operands in the same order.
//
// The coded role: a row directory (uint32 per row), then per row 32 stream
// lengths in bytes and 32 byte-aligned streams, one per lane, each a prefix
// code over the sixteen levels read LSB-first, decoded through a 256-entry
// table of (length << 4) | symbol. With FC_CODED_AUX the scales and biases are
// 12-bit indices, two per three bytes, into per-role tables of bf16 patterns.

constant bool FC_CODED_AUX [[function_constant(20)]];

struct CodedOffsets {
    uint gate_dir;
    uint gate_data;
    uint gate_s;
    uint gate_b;
    uint up_dir;
    uint up_data;
    uint up_s;
    uint up_b;
    uint gate_s_table;
    uint gate_b_table;
    uint up_s_table;
    uint up_b_table;
};

struct LaneStream {
    device const uchar* p;
    uint buf;
    uint have;
};

static inline LaneStream coded_open_stream(device const uchar* row, uint lane) {
    const uint len = uint(row[lane]);
    const uint start = 32u + simd_prefix_exclusive_sum(len);
    LaneStream s;
    s.p = row + start;
    s.buf = 0u;
    s.have = 0u;
    return s;
}

static inline uint coded_symbol(thread LaneStream& s, constant uchar* table) {
    if (s.have < 8u) {
        s.buf |= uint(*s.p) << s.have;
        s.p += 1;
        s.have += 8u;
    }
    const uint e = uint(table[s.buf & 0xFFu]);
    const uint len = e >> 4;
    s.buf >>= len;
    s.have -= len;
    return e & 0xFu;
}

static inline uint coded_aux_index(device const uchar* packed, uint g) {
    const uint k = g >> 1;
    const uint b0 = uint(packed[3u * k]);
    const uint b1 = uint(packed[3u * k + 1u]);
    const uint b2 = uint(packed[3u * k + 2u]);
    return (g & 1u) ? ((b1 >> 4) | (b2 << 4)) : (b0 | ((b1 & 0xFu) << 8));
}

static inline float2 coded_gate_up_rows(
    threadgroup const half* x,
    threadgroup const half* xsums,
    device const uchar* base,
    constant CodedOffsets& off,
    device const bfloat* aux,
    constant uchar* table,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    device const uint* g_dir = (device const uint*)(base + off.gate_dir);
    device const uint* u_dir = (device const uint*)(base + off.up_dir);
    LaneStream gstream = coded_open_stream(base + off.gate_data + g_dir[row], lane);
    LaneStream ustream = coded_open_stream(base + off.up_data + u_dir[row], lane);
    device const bfloat* gS_row = (device const bfloat*)(base + off.gate_s) + row * n_groups;
    device const bfloat* gB_row = (device const bfloat*)(base + off.gate_b) + row * n_groups;
    device const bfloat* uS_row = (device const bfloat*)(base + off.up_s) + row * n_groups;
    device const bfloat* uB_row = (device const bfloat*)(base + off.up_b) + row * n_groups;
    const uint aux_row_bytes = (n_groups * 3u + 1u) / 2u;
    device const uchar* gS_idx = base + off.gate_s + row * aux_row_bytes;
    device const uchar* gB_idx = base + off.gate_b + row * aux_row_bytes;
    device const uchar* uS_idx = base + off.up_s + row * aux_row_bytes;
    device const uchar* uB_idx = base + off.up_b + row * aux_row_bytes;
    device const bfloat* gS_tab = aux + off.gate_s_table;
    device const bfloat* gB_tab = aux + off.gate_b_table;
    device const bfloat* uS_tab = aux + off.up_s_table;
    device const bfloat* uB_tab = aux + off.up_b_table;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint g = blk * 4u + (lane >> 3);
        float gs, gb, us, ub;
        if (FC_CODED_AUX) {
            gs = float(gS_tab[coded_aux_index(gS_idx, g)]);
            gb = float(gB_tab[coded_aux_index(gB_idx, g)]);
            us = float(uS_tab[coded_aux_index(uS_idx, g)]);
            ub = float(uB_tab[coded_aux_index(uB_idx, g)]);
        } else {
            gs = float(gS_row[g]);
            gb = float(gB_row[g]);
            us = float(uS_row[g]);
            ub = float(uB_row[g]);
        }
        const uint elem = byte_base * 2u;
        const float e0 = float(x[elem]), e1 = float(x[elem + 1u]);
        const float e2 = float(x[elem + 2u]), e3 = float(x[elem + 3u]);
        const float e4 = float(x[elem + 4u]), e5 = float(x[elem + 5u]);
        const float e6 = float(x[elem + 6u]), e7 = float(x[elem + 7u]);
        const float sum = float(xsums[elem >> 3]);

        const uint g0 = coded_symbol(gstream, table), g1 = coded_symbol(gstream, table);
        const uint g2 = coded_symbol(gstream, table), g3 = coded_symbol(gstream, table);
        const uint g4 = coded_symbol(gstream, table), g5 = coded_symbol(gstream, table);
        const uint g6 = coded_symbol(gstream, table), g7 = coded_symbol(gstream, table);
        float g_dot = 0.0f;
        g_dot = fma(float(g0), e0, g_dot); g_dot = fma(float(g1), e1, g_dot);
        g_dot = fma(float(g2), e2, g_dot); g_dot = fma(float(g3), e3, g_dot);
        g_dot = fma(float(g4), e4, g_dot); g_dot = fma(float(g5), e5, g_dot);
        g_dot = fma(float(g6), e6, g_dot); g_dot = fma(float(g7), e7, g_dot);

        const uint u0 = coded_symbol(ustream, table), u1 = coded_symbol(ustream, table);
        const uint u2 = coded_symbol(ustream, table), u3 = coded_symbol(ustream, table);
        const uint u4 = coded_symbol(ustream, table), u5 = coded_symbol(ustream, table);
        const uint u6 = coded_symbol(ustream, table), u7 = coded_symbol(ustream, table);
        float u_dot = 0.0f;
        u_dot = fma(float(u0), e0, u_dot); u_dot = fma(float(u1), e1, u_dot);
        u_dot = fma(float(u2), e2, u_dot); u_dot = fma(float(u3), e3, u_dot);
        u_dot = fma(float(u4), e4, u_dot); u_dot = fma(float(u5), e5, u_dot);
        u_dot = fma(float(u6), e6, u_dot); u_dot = fma(float(u7), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

kernel void moe_phase1_coded_gate_up_act(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant CodedOffsets& coded_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    constant uchar* table [[buffer(8)]],
    device const bfloat* aux [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    threadgroup half xt[kMoEXMaxD];
    threadgroup half xsum[kMoEXSumMax];
    const uint DD = moe_fc_d(D);
    const uint tid = sg_idx * 32u + lane;
    for (uint i = tid; i < DD; i += rows_per_tg * 32u) {
        xt[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < (DD >> 3); i += rows_per_tg * 32u) {
        const uint b = i << 3;
        xsum[i] = ((xt[b] + xt[b + 1u]) + (xt[b + 2u] + xt[b + 3u]))
                + ((xt[b + 4u] + xt[b + 5u]) + (xt[b + 6u] + xt[b + 7u]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * moe_fc_f(F)) return;
    const uint slot = rowg / moe_fc_f(F);
    const uint f = rowg % moe_fc_f(F);

    device const uint8_t* base = routed.blob[slot];
    const float2 gu = coded_gate_up_rows(xt, xsum, base, coded_offsets, aux, table, f, DD, lane);
    if (lane == 0) acts[slot * moe_fc_f(F) + f] = half(moe_glu(gu));
}
