// Matrix-path causal prefill attention for the 256/16/2 shape (v12 P2).
//
// Two kernels. `attention_prefill_kv_dequant` expands the cache rows
// [0, kvValidCount) into an fp16 shadow ([token][kvHead * headDim]) with the
// same affine dequant `prefill_load_kv` applies: `matmul2d` reads its operands
// from device or threadgroup memory (never the packed cache rows), and a
// dequantized 64-key × 256 fp16 tile alone would fill the 32 KB threadgroup
// budget, so the shadow lives in device memory. `attention_prefill_causal_matrix_*`
// then owns a block of R queries of one query head: QKᵀ and PV through
// `matmul2d` against 64-key tiles of the shadow, an fp32 running max/sum per
// row (online softmax), the causal mask applied only where a tile crosses the
// query's own position. Only the score tile lives in threadgroup memory, and
// the softmax weights overwrite it in place (each lane rewrites exactly the
// slots it read).

#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr int kAttnMatrixKeys = 64;
constant constexpr int kAttnMatrixHeadDim = 256;

kernel void attention_prefill_kv_dequant(
    device const uchar* K [[buffer(0)]],
    device const uchar* V [[buffer(1)]],
    device half* shadowK [[buffer(2)]],
    device half* shadowV [[buffer(3)]],
    constant PrefillAttentionParams& p [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint elements = p.numKVHeads * p.headDim;
    if (gid.x >= elements || gid.y >= p.kvValidCount) return;
    const uint phys = prefill_kv_slot(gid.y);
    const uint out = gid.y * elements + gid.x;
    shadowK[out] = half(prefill_load_kv(K, phys, gid.x, p));
    shadowV[out] = half(prefill_load_kv(V, phys, gid.x, p));
}

template <int R, int S>
static inline void attention_prefill_causal_matrix_body(
    device half* Q,
    device half* shadowK,
    device half* shadowV,
    device half* O,
    constant PrefillAttentionParams& p,
    uint3 tg,
    uint lid,
    threadgroup float* score_tile,
    threadgroup float* row_max,
    threadgroup float* row_sum,
    threadgroup float* row_old_scale
) {
    static_assert((32 * S) % R == 0, "every row needs the same number of lanes");
    static_assert(32 % ((32 * S) / R) == 0, "a row's lanes must share one simdgroup");
    static_assert(kAttnMatrixKeys % ((32 * S) / R) == 0, "keys must split evenly across a row's lanes");
    constexpr auto qk_desc = matmul2d_descriptor(
        R, kAttnMatrixKeys, kAttnMatrixHeadDim, false, true, false);
    constexpr auto pv_desc = matmul2d_descriptor(
        R, kAttnMatrixHeadDim, kAttnMatrixKeys, false, false, false);
    matmul2d<qk_desc, execution_simdgroups<S>> qk_op;
    matmul2d<pv_desc, execution_simdgroups<S>> pv_op;

    using device_half_tensor =
        tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint q0 = tg.x * uint(R);
    const uint qh = tg.y;
    if (q0 >= p.queryCount) return;
    const uint rows_valid = min(uint(R), p.queryCount - q0);
    const uint kvh = qh / (p.numQHeads / p.numKVHeads);
    const uint elements = p.numKVHeads * p.headDim;

    if (lid < uint(R)) {
        row_max[lid] = -INFINITY;
        row_sum[lid] = 0.0f;
        row_old_scale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device_half_tensor query_tensor(
        Q + qh * p.headDim,
        dextents<int32_t, 2>(kAttnMatrixHeadDim, int32_t(p.queryCount)),
        array<int32_t, 2>({1, int32_t(p.qTokenStrideElements)}));
    device_half_tensor key_tensor(
        shadowK + kvh * p.headDim,
        dextents<int32_t, 2>(kAttnMatrixHeadDim, int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(elements)}));
    device_half_tensor value_tensor(
        shadowV + kvh * p.headDim,
        dextents<int32_t, 2>(kAttnMatrixHeadDim, int32_t(p.kvValidCount)),
        array<int32_t, 2>({1, int32_t(elements)}));
    threadgroup_float_tensor weight_tensor(
        score_tile,
        dextents<int32_t, 2>(kAttnMatrixKeys, R),
        array<int32_t, 2>({1, kAttnMatrixKeys}));

    auto query_slice = query_tensor.slice(0, int32_t(q0));
    auto first_value_slice = value_tensor.slice(0, 0);
    auto output_accumulator = pv_op.template get_destination_cooperative_tensor<
        decltype(weight_tensor), decltype(first_value_slice), float>();
    for (int element = 0; element < output_accumulator.get_capacity(); ++element) {
        output_accumulator[element] = 0.0f;
    }

    const uint last = min(p.kvValidCount, p.startPosition + q0 + rows_valid);
    for (uint key_start = 0u; key_start < last; key_start += uint(kAttnMatrixKeys)) {
        auto key_slice = key_tensor.slice(0, int32_t(key_start));
        auto score_product = qk_op.template get_destination_cooperative_tensor<
            decltype(query_slice), decltype(key_slice), float>();
        for (int element = 0; element < score_product.get_capacity(); ++element) {
            score_product[element] = 0.0f;
        }
        qk_op.run(query_slice, key_slice, score_product);
        for (int element = 0; element < score_product.get_capacity(); ++element) {
            if (!score_product.is_valid_element(element)) continue;
            const auto position = score_product.get_multidimensional_index(element);
            score_tile[uint(position[1]) * uint(kAttnMatrixKeys) + uint(position[0])] =
                score_product[element] * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            constexpr uint threads_per_row = (32u * uint(S)) / uint(R);
            constexpr uint keys_per_thread = uint(kAttnMatrixKeys) / threads_per_row;
            const uint row = lid / threads_per_row;
            const uint part = lid % threads_per_row;
            uint visible = 0u;
            if (row < rows_valid) {
                const uint causal_last = min(p.kvValidCount,
                                             p.startPosition + q0 + row + 1u);
                visible = causal_last > key_start
                    ? min(uint(kAttnMatrixKeys), causal_last - key_start)
                    : 0u;
            }
            threadgroup float* scores = score_tile + row * uint(kAttnMatrixKeys);
            threadgroup float* weights = scores;
            const uint key0 = part * keys_per_thread;
            float tile_max = -INFINITY;
            for (uint i = 0u; i < keys_per_thread; ++i) {
                const uint key = key0 + i;
                if (key < visible) tile_max = max(tile_max, scores[key]);
            }
            for (uint offset = 1u; offset < threads_per_row; offset <<= 1u) {
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, offset));
            }
            const float previous_max = row_max[row];
            const float next_max = max(previous_max, tile_max);
            const float old_scale = row_sum[row] > 0.0f
                ? fast::exp(previous_max - next_max)
                : 0.0f;
            float tile_sum = 0.0f;
            for (uint i = 0u; i < keys_per_thread; ++i) {
                const uint key = key0 + i;
                const float weight = key < visible
                    ? fast::exp(scores[key] - next_max)
                    : 0.0f;
                weights[key] = weight;
                tile_sum += weight;
            }
            for (uint offset = 1u; offset < threads_per_row; offset <<= 1u) {
                tile_sum += simd_shuffle_xor(tile_sum, offset);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (part == 0u) {
                row_old_scale[row] = old_scale;
                row_sum[row] = row_sum[row] * old_scale + tile_sum;
                row_max[row] = next_max;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto value_slice = value_tensor.slice(0, int32_t(key_start));
        auto output_product = pv_op.template get_destination_cooperative_tensor<
            decltype(weight_tensor), decltype(value_slice), float>();
        for (int element = 0; element < output_product.get_capacity(); ++element) {
            output_product[element] = 0.0f;
        }
        pv_op.run(weight_tensor, value_slice, output_product);
        for (int element = 0; element < output_accumulator.get_capacity(); ++element) {
            if (!output_accumulator.is_valid_element(element)
                || !output_product.is_valid_element(element)) {
                continue;
            }
            const auto position = output_accumulator.get_multidimensional_index(element);
            const uint row = uint(position[1]);
            output_accumulator[element] = fma(
                1.0f, output_product[element],
                output_accumulator[element] * row_old_scale[row]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < output_accumulator.get_capacity(); ++element) {
        if (!output_accumulator.is_valid_element(element)) continue;
        const auto position = output_accumulator.get_multidimensional_index(element);
        const uint d = uint(position[0]);
        const uint row = uint(position[1]);
        if (row >= rows_valid) continue;
        const float denominator = row_sum[row];
        O[(q0 + row) * p.oTokenStrideElements + qh * p.headDim + d] =
            denominator > 0.0f
                ? half(output_accumulator[element] / denominator)
                : half(0.0f);
    }
}

#define ATTN_MATRIX_KERNEL(NAME, R, S)                                          \
[[kernel, max_total_threads_per_threadgroup(32 * S)]]                           \
kernel void NAME(                                                               \
    device half* Q [[buffer(0)]],                                               \
    device half* shadowK [[buffer(1)]],                                         \
    device half* shadowV [[buffer(2)]],                                         \
    device half* O [[buffer(3)]],                                               \
    constant PrefillAttentionParams& p [[buffer(4)]],                           \
    uint3 tg [[threadgroup_position_in_grid]],                                  \
    uint lid [[thread_index_in_threadgroup]]                                    \
) {                                                                             \
    threadgroup float score_tile[R * kAttnMatrixKeys];                          \
    threadgroup float row_max[R];                                               \
    threadgroup float row_sum[R];                                               \
    threadgroup float row_old_scale[R];                                         \
    attention_prefill_causal_matrix_body<R, S>(                                 \
        Q, shadowK, shadowV, O, p, tg, lid,                                     \
        score_tile, row_max, row_sum, row_old_scale);                           \
}

ATTN_MATRIX_KERNEL(attention_prefill_causal_matrix_r32s4, 32, 4)
ATTN_MATRIX_KERNEL(attention_prefill_causal_matrix_r64s8, 64, 8)

#undef ATTN_MATRIX_KERNEL

#endif
