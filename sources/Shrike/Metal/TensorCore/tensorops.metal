#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kW4A8GroupSize = 64;
constant constexpr int kMPPAffineTileM = 64;
constant constexpr int kMPPAffineTileN = 32;
constant constexpr int kMPPAffineTileK = 64;
constant uint FC_MPP_AFFINE_BITS [[function_constant(78)]];

static inline uint mpp_affine_value(
    device const uint8_t* packed,
    uint element,
    uint bits
) {
    const uint bit_offset = element * bits;
    const uint byte_offset = bit_offset >> 3;
    const uint shift = bit_offset & 7u;
    uint word = uint(packed[byte_offset]);
    if (shift + bits > 8u) {
        word |= uint(packed[byte_offset + 1u]) << 8u;
    }
    return (word >> shift) & ((1u << bits) - 1u);
}

kernel void mpp_prefill_affine_threadgroup_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint bits = is_function_constant_defined(FC_MPP_AFFINE_BITS)
        ? FC_MPP_AFFINE_BITS : 4u;
    const uint rowBytes = K * bits / 8u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                device const uint8_t* rowWeights = packedWeights + globalN * rowBytes;
                const uint q = mpp_affine_value(rowWeights, globalK, bits);
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

/// The tile's expert blobs, laid out exactly like the prefill module's
/// streamed argument buffer so one encoded buffer serves both kernels.
struct MPPGroupedExpertBlobsMSL {
    device const uint8_t* blob[16];
};

/// One expert's slice of the staging block, 64-row aligned; `row_tile_start`
/// is `staging_row / 64`, the first grid row tile that belongs to it.
struct MPPGroupedBlockMSL {
    uint slot;
    uint pair_start;
    uint rows;
    uint staging_row;
    uint row_tile_start;
};

/// One dispatch over every expert's rows of a wave: the row tile's block
/// picks the weight pointer, the rest is `mpp_prefill_affine_threadgroup_f16`.
kernel void mpp_prefill_affine_grouped_f16(
    device const MPPGroupedExpertBlobsMSL& experts [[buffer(0)]],
    constant MPPGroupedBlockMSL* blocks            [[buffer(1)]],
    constant uint* rowTileBlock                    [[buffer(2)]],
    device half* activations                       [[buffer(3)]],
    device half* output                            [[buffer(4)]],
    constant uint& N                               [[buffer(5)]],
    constant uint& K                               [[buffer(6)]],
    constant uint& wOff                            [[buffer(7)]],
    constant uint& sOff                            [[buffer(8)]],
    constant uint& bOff                            [[buffer(9)]],
    constant uint& M                               [[buffer(10)]],
    uint3 tgid                                     [[threadgroup_position_in_grid]],
    uint3 lid3                                     [[thread_position_in_threadgroup]],
    uint3 threads3                                 [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    const MPPGroupedBlockMSL b = blocks[rowTileBlock[tgid.y]];
    device const uint8_t* packedWeights = experts.blob[b.slot] + wOff;
    device const bfloat* scales =
        reinterpret_cast<device const bfloat*>(experts.blob[b.slot] + sOff);
    device const bfloat* biases =
        reinterpret_cast<device const bfloat*>(experts.blob[b.slot] + bOff);
    const int32_t rowOrigin =
        int32_t(b.staging_row + (tgid.y - b.row_tile_start) * uint(kMPPAffineTileM));

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, rowOrigin);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint bits = is_function_constant_defined(FC_MPP_AFFINE_BITS)
        ? FC_MPP_AFFINE_BITS : 4u;
    const uint rowBytes = K * bits / 8u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                device const uint8_t* rowWeights = packedWeights + globalN * rowBytes;
                const uint q = mpp_affine_value(rowWeights, globalK, bits);
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(0, rowOrigin);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint rowEnd = b.staging_row + b.rows;
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = uint(rowOrigin) + uint(position[1]);
        if (globalM < rowEnd && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

#endif
