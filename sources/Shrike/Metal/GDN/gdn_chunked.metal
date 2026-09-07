// Chunked gated delta rule for prefill (v12 P4), scalar per-head decay,
// Dk = Dv = 128. The serial recurrence `gdn_delta_step_prefill` runs is
// rewritten over 64-row chunks as matrix products; the derivation and the
// symbol names are in docs/v12-implementation-plan.md, Task 4.
//
// `gdn_chunk_factors` (one threadgroup per chunk and head, all chunks in
// parallel) builds what every value column shares: the decay-weighted
// `T⁻¹ = (I + A)⁻¹` and `M`, plus the per-row scalars. `gdn_chunk_scan`
// (one threadgroup per head and 32-column block of the state) then walks the
// chunks in order, carrying its state block in threadgroup memory. Pad rows
// of a partial last chunk must be zero in `conv_out` (the encoder blits
// them) and get `β = 0`, `log α = 0` here.

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr int kGDNChunk = 64;
constant constexpr int kGDNChunkHeadDim = 128;
constant constexpr int kGDNChunkValueBlock = 32;
constant constexpr uint kGDNChunkThreads = 128;
constant constexpr uint kGDNFactorsBytes = 17408;
constant constexpr uint kGDNFactorsMOffset = 8192;
constant constexpr uint kGDNFactorsScalarsOffset = 16384;

struct GDNChunkParams {
    uint kHeads;
    uint vHeads;
    uint keyDim;
    uint valueDim;
    uint rows;
    uint rowStride;
    uint chunkCount;
};

using gdn_device_half_tensor =
    tensor<device half, dextents<int32_t, 2>, tensor_inline>;
using gdn_threadgroup_float_tensor =
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

kernel void gdn_chunk_factors(
    device const half*   conv_out [[buffer(0)]],   // [T, C]
    device const half*   a_proj   [[buffer(1)]],   // [T, Hv]
    device const half*   b_proj   [[buffer(2)]],   // [T, Hv]
    device const bfloat* A_log    [[buffer(3)]],   // [Hv]
    device const bfloat* dt_bias  [[buffer(4)]],   // [Hv]
    device uchar*        factors  [[buffer(5)]],
    constant GDNChunkParams& p    [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    threadgroup float a_tile[kGDNChunk * kGDNChunk];
    threadgroup float ell[kGDNChunk];
    threadgroup float beta[kGDNChunk];

    const uint c = tg.x;
    const uint h = tg.y;
    const uint Hk = p.kHeads;
    const uint Hv = p.vHeads;
    const uint Dk = p.keyDim;
    const uint C = p.rowStride;
    const uint hk = h / (Hv / Hk);
    const uint r0 = c * uint(kGDNChunk);
    const uint valid = min(uint(kGDNChunk), p.rows - r0);

    if (lid < uint(kGDNChunk)) {
        const uint t = lid;
        float logAlpha = 0.0f;
        float b = 0.0f;
        if (t < valid) {
            const float expA = exp(float(A_log[h]));
            logAlpha = -expA * gdn_softplus(float(a_proj[(r0 + t) * Hv + h])
                                            + float(dt_bias[h]));
            b = 1.0f / (1.0f + exp(-float(b_proj[(r0 + t) * Hv + h])));
        }
        ell[t] = logAlpha;
        beta[t] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float running = 0.0f;
        for (int t = 0; t < kGDNChunk; ++t) {
            running += ell[t];
            ell[t] = running;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto kk_desc = matmul2d_descriptor(
        kGDNChunk, kGDNChunk, kGDNChunkHeadDim, false, true, false);
    matmul2d<kk_desc, execution_simdgroups<4>> kk_op;

    device half* rows = const_cast<device half*>(conv_out) + r0 * C;
    gdn_device_half_tensor q_tensor(
        rows + hk * Dk,
        dextents<int32_t, 2>(kGDNChunkHeadDim, kGDNChunk),
        array<int32_t, 2>({1, int32_t(C)}));
    gdn_device_half_tensor k_tensor(
        rows + Hk * Dk + hk * Dk,
        dextents<int32_t, 2>(kGDNChunkHeadDim, kGDNChunk),
        array<int32_t, 2>({1, int32_t(C)}));

    device uchar* base = factors + (h * p.chunkCount + c) * kGDNFactorsBytes;
    device half* tinv = reinterpret_cast<device half*>(base);
    device half* m = reinterpret_cast<device half*>(base + kGDNFactorsMOffset);
    device float* scalars = reinterpret_cast<device float*>(base + kGDNFactorsScalarsOffset);

    auto kk = kk_op.get_destination_cooperative_tensor<
        decltype(k_tensor), decltype(k_tensor), float>();
    for (int e = 0; e < kk.get_capacity(); ++e) kk[e] = 0.0f;
    kk_op.run(k_tensor, k_tensor, kk);
    for (int e = 0; e < kk.get_capacity(); ++e) {
        if (!kk.is_valid_element(e)) continue;
        const auto pos = kk.get_multidimensional_index(e);
        const uint i = uint(pos[0]);
        const uint t = uint(pos[1]);
        const bool inside = i < t && t < valid;
        a_tile[t * uint(kGDNChunk) + i] = inside
            ? beta[t] * exp(ell[t] - ell[i]) * kk[e]
            : 0.0f;
    }

    auto qk = kk_op.get_destination_cooperative_tensor<
        decltype(q_tensor), decltype(k_tensor), float>();
    for (int e = 0; e < qk.get_capacity(); ++e) qk[e] = 0.0f;
    kk_op.run(q_tensor, k_tensor, qk);
    for (int e = 0; e < qk.get_capacity(); ++e) {
        if (!qk.is_valid_element(e)) continue;
        const auto pos = qk.get_multidimensional_index(e);
        const uint i = uint(pos[0]);
        const uint t = uint(pos[1]);
        const bool inside = i <= t && t < valid;
        m[t * uint(kGDNChunk) + i] = half(inside ? exp(ell[t] - ell[i]) * qk[e] : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < uint(kGDNChunk)) {
        const uint j = lid;
        float x[kGDNChunk];
        for (int t = 0; t < kGDNChunk; ++t) x[t] = 0.0f;
        x[j] = 1.0f;
        for (uint t = j + 1u; t < uint(kGDNChunk); ++t) {
            float acc = 0.0f;
            for (uint i = j; i < t; ++i) {
                acc = fma(a_tile[t * uint(kGDNChunk) + i], x[i], acc);
            }
            x[t] = -acc;
        }
        for (uint t = 0; t < uint(kGDNChunk); ++t) {
            tinv[t * uint(kGDNChunk) + j] = half(x[t]);
        }
        const float last = ell[kGDNChunk - 1];
        scalars[j] = beta[j];
        scalars[uint(kGDNChunk) + j] = exp(ell[j]);
        scalars[2u * uint(kGDNChunk) + j] = exp(last - ell[j]);
        if (j == 0u) scalars[3u * uint(kGDNChunk)] = exp(last);
    }
}

kernel void gdn_chunk_scan(
    device const half*   conv_out [[buffer(0)]],   // [T, C]
    device const uchar*  factors  [[buffer(1)]],
    device float*        state    [[buffer(2)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(3)]],   // [T, Hv * Dv]
    constant GDNChunkParams& p    [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    threadgroup float s_tile[kGDNChunkValueBlock * kGDNChunkHeadDim];
    threadgroup float u_tile[kGDNChunk * kGDNChunkValueBlock];

    const uint block = tg.x;
    const uint h = tg.y;
    const uint Hk = p.kHeads;
    const uint Hv = p.vHeads;
    const uint Dk = p.keyDim;
    const uint Dv = p.valueDim;
    const uint C = p.rowStride;
    const uint hk = h / (Hv / Hk);
    const uint dv0 = block * uint(kGDNChunkValueBlock);
    constexpr uint stateElements = uint(kGDNChunkValueBlock * kGDNChunkHeadDim);
    constexpr uint tileElements = uint(kGDNChunk * kGDNChunkValueBlock);

    device float* stateBlock = state + (h * Dv + dv0) * Dk;
    for (uint i = lid; i < stateElements; i += kGDNChunkThreads) {
        s_tile[i] = stateBlock[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto ks_desc = matmul2d_descriptor(
        kGDNChunk, kGDNChunkValueBlock, kGDNChunkHeadDim, false, true, false);
    constexpr auto tu_desc = matmul2d_descriptor(
        kGDNChunk, kGDNChunkValueBlock, kGDNChunk, false, false, false);
    constexpr auto su_desc = matmul2d_descriptor(
        kGDNChunkValueBlock, kGDNChunkHeadDim, kGDNChunk, true, false, false);
    matmul2d<ks_desc, execution_simdgroups<4>> ks_op;
    matmul2d<tu_desc, execution_simdgroups<4>> tu_op;
    matmul2d<su_desc, execution_simdgroups<4>> su_op;

    gdn_threadgroup_float_tensor s_tensor(
        s_tile,
        dextents<int32_t, 2>(kGDNChunkHeadDim, kGDNChunkValueBlock),
        array<int32_t, 2>({1, kGDNChunkHeadDim}));
    gdn_threadgroup_float_tensor u_tensor(
        u_tile,
        dextents<int32_t, 2>(kGDNChunkValueBlock, kGDNChunk),
        array<int32_t, 2>({1, kGDNChunkValueBlock}));

    for (uint c = 0; c < p.chunkCount; ++c) {
        const uint r0 = c * uint(kGDNChunk);
        const uint valid = min(uint(kGDNChunk), p.rows - r0);
        device const uchar* base = factors + (h * p.chunkCount + c) * kGDNFactorsBytes;
        device const float* scalars =
            reinterpret_cast<device const float*>(base + kGDNFactorsScalarsOffset);
        device const float* beta = scalars;
        device const float* gamma = scalars + kGDNChunk;
        device const float* lambda = scalars + 2 * kGDNChunk;
        const float gammaChunk = scalars[3 * kGDNChunk];

        device half* rows = const_cast<device half*>(conv_out) + r0 * C;
        gdn_device_half_tensor q_tensor(
            rows + hk * Dk,
            dextents<int32_t, 2>(kGDNChunkHeadDim, kGDNChunk),
            array<int32_t, 2>({1, int32_t(C)}));
        gdn_device_half_tensor k_tensor(
            rows + Hk * Dk + hk * Dk,
            dextents<int32_t, 2>(kGDNChunkHeadDim, kGDNChunk),
            array<int32_t, 2>({1, int32_t(C)}));
        gdn_device_half_tensor tinv_tensor(
            const_cast<device half*>(reinterpret_cast<device const half*>(base)),
            dextents<int32_t, 2>(kGDNChunk, kGDNChunk),
            array<int32_t, 2>({1, kGDNChunk}));
        gdn_device_half_tensor m_tensor(
            const_cast<device half*>(reinterpret_cast<device const half*>(base + kGDNFactorsMOffset)),
            dextents<int32_t, 2>(kGDNChunk, kGDNChunk),
            array<int32_t, 2>({1, kGDNChunk}));
        device const half* v_rows = rows + 2u * Hk * Dk + h * Dv + dv0;

        auto ks = ks_op.get_destination_cooperative_tensor<
            decltype(k_tensor), decltype(s_tensor), float>();
        for (int e = 0; e < ks.get_capacity(); ++e) ks[e] = 0.0f;
        ks_op.run(k_tensor, s_tensor, ks);
        for (int e = 0; e < ks.get_capacity(); ++e) {
            if (!ks.is_valid_element(e)) continue;
            const auto pos = ks.get_multidimensional_index(e);
            const uint dv = uint(pos[0]);
            const uint t = uint(pos[1]);
            u_tile[t * uint(kGDNChunkValueBlock) + dv] = t < valid
                ? beta[t] * (float(v_rows[t * C + dv]) - gamma[t] * ks[e])
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto u = tu_op.get_destination_cooperative_tensor<
            decltype(tinv_tensor), decltype(u_tensor), float>();
        for (int e = 0; e < u.get_capacity(); ++e) u[e] = 0.0f;
        tu_op.run(tinv_tensor, u_tensor, u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = 0; e < u.get_capacity(); ++e) {
            if (!u.is_valid_element(e)) continue;
            const auto pos = u.get_multidimensional_index(e);
            u_tile[uint(pos[1]) * uint(kGDNChunkValueBlock) + uint(pos[0])] = u[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto o1 = ks_op.get_destination_cooperative_tensor<
            decltype(q_tensor), decltype(s_tensor), float>();
        for (int e = 0; e < o1.get_capacity(); ++e) o1[e] = 0.0f;
        ks_op.run(q_tensor, s_tensor, o1);
        auto o2 = tu_op.get_destination_cooperative_tensor<
            decltype(m_tensor), decltype(u_tensor), float>();
        for (int e = 0; e < o2.get_capacity(); ++e) o2[e] = 0.0f;
        tu_op.run(m_tensor, u_tensor, o2);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < tileElements; i += kGDNChunkThreads) {
            u_tile[i] *= lambda[i / uint(kGDNChunkValueBlock)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto su = su_op.get_destination_cooperative_tensor<
            decltype(u_tensor), decltype(k_tensor), float>();
        for (int e = 0; e < su.get_capacity(); ++e) su[e] = 0.0f;
        su_op.run(u_tensor, k_tensor, su);
        for (int e = 0; e < su.get_capacity(); ++e) {
            if (!su.is_valid_element(e)) continue;
            const auto pos = su.get_multidimensional_index(e);
            const uint idx = uint(pos[1]) * uint(kGDNChunkHeadDim) + uint(pos[0]);
            s_tile[idx] = fma(gammaChunk, s_tile[idx], su[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int e = 0; e < o1.get_capacity(); ++e) {
            if (!o1.is_valid_element(e)) continue;
            const auto pos = o1.get_multidimensional_index(e);
            const uint t = uint(pos[1]);
            u_tile[t * uint(kGDNChunkValueBlock) + uint(pos[0])] = gamma[t] * o1[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = 0; e < o2.get_capacity(); ++e) {
            if (!o2.is_valid_element(e)) continue;
            const auto pos = o2.get_multidimensional_index(e);
            u_tile[uint(pos[1]) * uint(kGDNChunkValueBlock) + uint(pos[0])] += o2[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < tileElements; i += kGDNChunkThreads) {
            const uint t = i / uint(kGDNChunkValueBlock);
            const uint dv = i % uint(kGDNChunkValueBlock);
            if (t < valid) {
                y[(r0 + t) * Hv * Dv + h * Dv + dv0 + dv] = half(u_tile[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < stateElements; i += kGDNChunkThreads) {
        stateBlock[i] = s_tile[i];
    }
}

#endif // __HAVE_TENSOR__
