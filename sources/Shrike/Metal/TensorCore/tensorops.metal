#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

constant constexpr uint kW4A8GroupSize = 64;
constant constexpr int kMPPAffineTileM = 64;
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

static inline void mpp_affine_load_words(
    thread uint* words,
    device const uint8_t* src,
    uint chunkBytes
) {
    if (chunkBytes == 8u) {
        const uint2 v = *reinterpret_cast<device const uint2*>(src);
        words[0] = v.x; words[1] = v.y;
    } else if (chunkBytes == 16u) {
        const uint4 v = *reinterpret_cast<device const uint4*>(src);
        words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
    } else if (chunkBytes == 32u) {
        const uint4 v = *reinterpret_cast<device const uint4*>(src);
        const uint4 w = *reinterpret_cast<device const uint4*>(src + 16);
        words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
        words[4] = w.x; words[5] = w.y; words[6] = w.z; words[7] = w.w;
    } else {
        const uint4 a = *reinterpret_cast<device const uint4*>(src);
        const uint4 b = *reinterpret_cast<device const uint4*>(src + 16);
        const uint4 c = *reinterpret_cast<device const uint4*>(src + 32);
        const uint4 d = *reinterpret_cast<device const uint4*>(src + 48);
        words[0] = a.x; words[1] = a.y; words[2] = a.z; words[3] = a.w;
        words[4] = b.x; words[5] = b.y; words[6] = b.z; words[7] = b.w;
        words[8] = c.x; words[9] = c.y; words[10] = c.z; words[11] = c.w;
        words[12] = d.x; words[13] = d.y; words[14] = d.z; words[15] = d.w;
    }
}

// One chunk per thread: E = TILE_K / 4 consecutive elements of one row, one
// vector load (8 to 64 bytes — never straddling a row or a quant group),
// the same q / scale / bias / fma / slot as the byte path, so the two bodies
// are bit-identical. `vectorLoads` is per dispatch: the host sets it only when
// the weight base is 16-byte aligned (the row stride already is, from the
// k % 64 guard).
template <int TILE_N, int TILE_K>
static inline void mpp_affine_dequant_tile(
    threadgroup half* tile,
    device const uint8_t* packedWeights,
    device const bfloat* scales,
    device const bfloat* biases,
    uint tileIndex,
    uint columnTile,
    uint N,
    uint rowBytes,
    uint groupsPerRow,
    uint bits,
    bool vectorLoads,
    uint lid,
    uint threads
) {
    if (vectorLoads) {
        constexpr uint E = uint(TILE_K) / 4u;
        constexpr uint chunksPerRow = uint(TILE_K) / E;
        constexpr uint chunksPerTile = uint(TILE_N) * chunksPerRow;
        static_assert(kW4A8GroupSize % E == 0, "a chunk never straddles a quant group");
        const uint chunkBytes = E * bits / 8u;
        const uint perWord = 32u / bits;
        const uint mask = (1u << bits) - 1u;
        for (uint c = lid; c < chunksPerTile; c += threads) {
            const uint localN = c / chunksPerRow;
            const uint localK0 = (c % chunksPerRow) * E;
            const uint globalN = columnTile * uint(TILE_N) + localN;
            threadgroup half4* dst =
                reinterpret_cast<threadgroup half4*>(tile + localN * uint(TILE_K) + localK0);
            if (globalN < N) {
                const uint globalK0 = tileIndex * uint(TILE_K) + localK0;
                const uint group = globalK0 / kW4A8GroupSize;
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                uint words[16];
                mpp_affine_load_words(
                    words, packedWeights + globalN * rowBytes + (globalK0 * bits) / 8u, chunkBytes);
#pragma clang loop unroll(full)
                for (uint e = 0u; e < E; e += 4u) {
                    half4 v;
#pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                        const uint idx = e + j;
                        const uint q = (words[idx / perWord] >> ((idx % perWord) * bits)) & mask;
                        v[j] = half(fma(float(q), scale, bias));
                    }
                    dst[e / 4u] = v;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint e = 0u; e < E; e += 4u) {
                    dst[e / 4u] = half4(0.0h);
                }
            }
        }
        return;
    }
    for (uint linear = lid;
         linear < uint(TILE_N * TILE_K);
         linear += threads) {
        const uint localN = linear / uint(TILE_K);
        const uint localK = linear % uint(TILE_K);
        const uint globalN = columnTile * uint(TILE_N) + localN;
        if (globalN < N) {
            const uint globalK = tileIndex * uint(TILE_K) + localK;
            const uint group = globalK / kW4A8GroupSize;
            device const uint8_t* rowWeights = packedWeights + globalN * rowBytes;
            const uint q = mpp_affine_value(rowWeights, globalK, bits);
            const float scale = float(scales[globalN * groupsPerRow + group]);
            const float bias = float(biases[globalN * groupsPerRow + group]);
            tile[linear] = half(fma(float(q), scale, bias));
        } else {
            tile[linear] = half(0.0f);
        }
    }
}

// One K-tile loop for both kernels: `rowOrigin` is the A/output row this
// threadgroup starts at, `rowEnd` the store bound (`M` for the plain kernel,
// the expert block's last staging row for the grouped one). With two weight
// tiles the dequant of tile t+1 is issued before the matmul of tile t and
// the barrier after the accumulate carries both edges: it publishes t+1's
// tile and orders t's reads before that buffer is refilled at t+2.
template <int TILE_N, int TILE_K, int BUFFERS>
static inline void mpp_prefill_affine_body(
    device const uint8_t* packedWeights,
    device const bfloat* scales,
    device const bfloat* biases,
    device half* activations,
    device half* output,
    uint M,
    uint N,
    uint K,
    int32_t rowOrigin,
    uint rowEnd,
    uint columnTile,
    bool vectorLoads,
    uint lid,
    uint threads,
    threadgroup half* weightTile
) {
    static_assert(BUFFERS == 1 || BUFFERS == 2, "one or two weight tiles");
    static_assert(TILE_K % int(kW4A8GroupSize) == 0, "a K tile is whole quant groups");
    static_assert(TILE_N * TILE_K * BUFFERS * 2 <= 32768, "threadgroup budget");
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, TILE_N, TILE_K,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    constexpr int tileElements = TILE_N * TILE_K;
    threadgroup_half_tensor tileB0(
        weightTile,
        dextents<int32_t, 2>(TILE_K, TILE_N),
        array<int32_t, 2>({1, TILE_K}));
    threadgroup_half_tensor tileB1(
        weightTile + (BUFFERS == 2 ? tileElements : 0),
        dextents<int32_t, 2>(TILE_K, TILE_N),
        array<int32_t, 2>({1, TILE_K}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(TILE_K, int32_t(M)),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, rowOrigin);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB0), float>();
    auto groupProduct = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB0), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint bits = is_function_constant_defined(FC_MPP_AFFINE_BITS)
        ? FC_MPP_AFFINE_BITS : 4u;
    const uint rowBytes = K * bits / 8u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint tilesPerRow = K / uint(TILE_K);
    mpp_affine_dequant_tile<TILE_N, TILE_K>(
        weightTile, packedWeights, scales, biases,
        0u, columnTile, N, rowBytes, groupsPerRow, bits, vectorLoads, lid, threads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = 0; tile < tilesPerRow; ++tile) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        if (BUFFERS == 1) {
            if (tile > 0u) {
                mpp_affine_dequant_tile<TILE_N, TILE_K>(
                    weightTile, packedWeights, scales, biases,
                    tile, columnTile, N, rowBytes, groupsPerRow, bits, vectorLoads, lid, threads);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        } else if (tile + 1u < tilesPerRow) {
            mpp_affine_dequant_tile<TILE_N, TILE_K>(
                weightTile + ((tile + 1u) & 1u) * uint(tileElements),
                packedWeights, scales, biases,
                tile + 1u, columnTile, N, rowBytes, groupsPerRow, bits, vectorLoads, lid, threads);
        }

        device_half_tensor tileA_source(
            activations + tile * uint(TILE_K),
            dextents<int32_t, 2>(TILE_K, int32_t(M)),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = tileA_source.slice(0, rowOrigin);
        if (BUFFERS == 2 && (tile & 1u)) {
            operation.run(tileA, tileB1, groupProduct);
        } else {
            operation.run(tileA, tileB0, groupProduct);
        }
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = columnTile * uint(TILE_N) + uint(position[0]);
        const uint globalM = uint(rowOrigin) + uint(position[1]);
        if (globalM < rowEnd && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

#define MPP_AFFINE_KERNEL(NAME, TILE_N, TILE_K, BUFFERS)                        \
kernel void NAME(                                                              \
    device const uint8_t* packedWeights [[buffer(0)]],                          \
    device const bfloat* scales         [[buffer(1)]],                          \
    device const bfloat* biases         [[buffer(2)]],                          \
    device half* activations            [[buffer(3)]],                          \
    device half* output                 [[buffer(4)]],                          \
    constant uint& M                    [[buffer(5)]],                          \
    constant uint& N                    [[buffer(6)]],                          \
    constant uint& K                    [[buffer(7)]],                          \
    constant uint& vectorLoads          [[buffer(8)]],                          \
    uint3 tgid                          [[threadgroup_position_in_grid]],       \
    uint3 lid3                          [[thread_position_in_threadgroup]],     \
    uint3 threads3                      [[threads_per_threadgroup]]) {          \
    threadgroup half4 weightTile[TILE_N * TILE_K * BUFFERS / 4];                \
    mpp_prefill_affine_body<TILE_N, TILE_K, BUFFERS>(                           \
        packedWeights, scales, biases, activations, output, M, N, K,            \
        int32_t(tgid.y) * kMPPAffineTileM, M, tgid.x, vectorLoads != 0u,        \
        lid3.x, threads3.x, reinterpret_cast<threadgroup half*>(weightTile));   \
}

MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16, 32, 64, 1)
MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n32b2, 32, 64, 2)
MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n64b1, 64, 64, 1)
MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n64b2, 64, 64, 2)
MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n32k128b1, 32, 128, 1)
MPP_AFFINE_KERNEL(mpp_prefill_affine_threadgroup_f16_n32k256b1, 32, 256, 1)

#undef MPP_AFFINE_KERNEL

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
/// picks the weight pointer, the rest is the plain kernel's body.
#define MPP_GROUPED_KERNEL(NAME, TILE_N, TILE_K, BUFFERS)                       \
kernel void NAME(                                                              \
    device const MPPGroupedExpertBlobsMSL& experts [[buffer(0)]],               \
    constant MPPGroupedBlockMSL* blocks            [[buffer(1)]],               \
    constant uint* rowTileBlock                    [[buffer(2)]],               \
    device half* activations                       [[buffer(3)]],               \
    device half* output                            [[buffer(4)]],               \
    constant uint& N                               [[buffer(5)]],               \
    constant uint& K                               [[buffer(6)]],               \
    constant uint& wOff                            [[buffer(7)]],               \
    constant uint& sOff                            [[buffer(8)]],               \
    constant uint& bOff                            [[buffer(9)]],               \
    constant uint& M                               [[buffer(10)]],              \
    constant uint& vectorLoads                     [[buffer(11)]],              \
    uint3 tgid                                     [[threadgroup_position_in_grid]],   \
    uint3 lid3                                     [[thread_position_in_threadgroup]], \
    uint3 threads3                                 [[threads_per_threadgroup]]) {      \
    const MPPGroupedBlockMSL b = blocks[rowTileBlock[tgid.y]];                  \
    device const uint8_t* packedWeights = experts.blob[b.slot] + wOff;          \
    device const bfloat* scales =                                               \
        reinterpret_cast<device const bfloat*>(experts.blob[b.slot] + sOff);   \
    device const bfloat* biases =                                               \
        reinterpret_cast<device const bfloat*>(experts.blob[b.slot] + bOff);   \
    const int32_t rowOrigin = int32_t(                                          \
        b.staging_row + (tgid.y - b.row_tile_start) * uint(kMPPAffineTileM));   \
    threadgroup half4 weightTile[TILE_N * TILE_K * BUFFERS / 4];                \
    mpp_prefill_affine_body<TILE_N, TILE_K, BUFFERS>(                           \
        packedWeights, scales, biases, activations, output, M, N, K,            \
        rowOrigin, b.staging_row + b.rows, tgid.x, vectorLoads != 0u,           \
        lid3.x, threads3.x, reinterpret_cast<threadgroup half*>(weightTile));   \
}

MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16, 32, 64, 1)
MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n32b2, 32, 64, 2)
MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n64b1, 64, 64, 1)
MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n64b2, 64, 64, 2)
MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n32k128b1, 32, 128, 1)
MPP_GROUPED_KERNEL(mpp_prefill_affine_grouped_f16_n32k256b1, 32, 256, 1)

#undef MPP_GROUPED_KERNEL

#endif
