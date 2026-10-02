#pragma once

// Q5G64 RowSplit K-split MMA for small column extents. Eight warps cooperate on
// a 16-row output tile, each owning one 64-element quantization group per K slab.
// Packed Q5 is decoded to exact BF16 integers, accumulated in FP32, then scaled
// once per group. The final K-split reduction is private implementation detail.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Q5KSplitMmaSchedule {
    static constexpr int kKWarps = 8;
    static constexpr int kThreads = kKWarps * 32;
    static constexpr int kTileKPerWarp = Q5RowSplitStorage::kGroupK;
    static constexpr int kGroupK = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kCodeBytesPerGroup = Q5RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytesPerGroup = Q5RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kStaticSharedBudget = 48 * 1024;
};

template <int TileCols>
struct alignas(16) Q5KSplitStage {
    std::uint8_t codes[Q5KSplitMmaSchedule::kRowsPerCta]
                      [Q5KSplitMmaSchedule::kKWarps * Q5KSplitMmaSchedule::kCodeBytesPerGroup];
    std::uint8_t high[Q5KSplitMmaSchedule::kRowsPerCta]
                     [Q5KSplitMmaSchedule::kKWarps * Q5KSplitMmaSchedule::kHighBytesPerGroup];
    std::uint16_t scales[Q5KSplitMmaSchedule::kRowsPerCta][Q5KSplitMmaSchedule::kKWarps];
    __nv_bfloat16 activations[Q5KSplitMmaSchedule::kKWarps]
                             [TileCols * Q5KSplitMmaSchedule::kTileKPerWarp];
};

template <int TileCols>
__host__ __device__ constexpr int q5_ksplit_stages() {
    constexpr int stage = static_cast<int>(sizeof(Q5KSplitStage<TileCols>));
    constexpr int budget = Q5KSplitMmaSchedule::kStaticSharedBudget;
    return 3 * stage <= budget ? 3 : (2 * stage <= budget ? 2 : 1);
}

template <int TileCols>
__host__ __device__ constexpr int q5_ksplit_min_blocks_per_sm() {
    constexpr int fit = (100 * 1024) /
                        (q5_ksplit_stages<TileCols>() *
                         static_cast<int>(sizeof(Q5KSplitStage<TileCols>)));
    return fit < 1 ? 1 : (fit > 4 ? 4 : fit);
}

__device__ __forceinline__ int q5_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int q5_ksplit_code_offset(int row, int byte) {
    return (((byte >> 4) ^ (row & 7)) << 4) | (byte & 15);
}

__device__ __forceinline__ int q5_ksplit_high_offset(int row, int byte) {
    return (((byte >> 4) ^ ((row >> 1) & 3)) << 4) | (byte & 15);
}

union Q5KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q5_ksplit_bf16_pair(std::uint8_t packed,
                                                        std::uint8_t high, int shift) {
    const int q0 = ((static_cast<int>(packed & 0x0fu) | (((high >> shift) & 1) << 4)) ^ 0x10) - 0x10;
    const int q1 = ((static_cast<int>(packed >> 4) | (((high >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10;
    Q5KSplitBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <int OutputRows, int InputRows, int TileCols, int ActiveCols, bool AddResidual = false,
          int SplitRows = 0>
__launch_bounds__(Q5KSplitMmaSchedule::kThreads,
                  q5_ksplit_min_blocks_per_sm<TileCols>()) __global__
void q5_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                          const std::uint8_t* __restrict__ codes,
                          const std::uint8_t* __restrict__ high,
                          const std::uint8_t* __restrict__ scales,
                          __nv_bfloat16* __restrict__ out,
                          __nv_bfloat16* __restrict__ split_out, int out_ld,
                          int split_out_ld, int columns) {
    using Schedule = Q5KSplitMmaSchedule;
    constexpr int hidden = InputRows;
    constexpr int tile_k = Schedule::kTileKPerWarp;
    constexpr int warps = Schedule::kKWarps;
    constexpr int rows_per_cta = Schedule::kRowsPerCta;
    constexpr int group_k = Schedule::kGroupK;
    constexpr int groups = hidden / group_k;
    constexpr int groups_per_row = hidden / Q5RowSplitStorage::kGroupK;
    constexpr int code_row_bytes = groups_per_row * Schedule::kCodeBytesPerGroup;
    constexpr int high_row_bytes = groups_per_row * Schedule::kHighBytesPerGroup;
    constexpr int scale_row_bytes = groups_per_row * 2;
    constexpr int tile_cols = TileCols;
    static_assert(SplitRows == 0 || (SplitRows > 0 && SplitRows < OutputRows));
    constexpr int nt = tile_cols / 8;
    constexpr int stages = q5_ksplit_stages<TileCols>();
    constexpr int prefetch = stages - 1;
    static_assert(tile_cols >= 8 && tile_cols <= 32 && (tile_cols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= tile_cols && ActiveCols > tile_cols - 8);
    static_assert((hidden % group_k) == 0 && groups >= 1);
    static_assert(OutputRows % rows_per_cta == 0);
    static_assert(code_row_bytes % 16 == 0 && high_row_bytes % 16 == 0 &&
                  scale_row_bytes % 16 == 0);

    using Stage = Q5KSplitStage<tile_cols>;
    static_assert(sizeof(Stage) % 16 == 0);
    union SharedStorage {
        Stage staging[stages];
        float partial[warps * nt * 32 * 4];
    };
    static_assert(sizeof(SharedStorage) <= Schedule::kStaticSharedBudget);
    __shared__ __align__(16) SharedStorage shared;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int k_split = warp;
    const int row0 = static_cast<int>(blockIdx.x) * rows_per_cta;
    const int column_offset = static_cast<int>(blockIdx.y) * tile_cols;
    const int live_columns = min(tile_cols, columns - column_offset);
    x += static_cast<std::int64_t>(column_offset) * hidden;
    out += static_cast<std::int64_t>(column_offset) * OutputRows;

    const auto stage_x = [&](int group_k0, Stage& stage) {
        constexpr int items_per_split = ActiveCols * (tile_k / 8);
        for (int item = lane; item < items_per_split; item += 32) {
            const int col = item / (tile_k / 8);
            const int k8 = item - col * (tile_k / 8);
            auto* dst = &stage.activations[warp][col * tile_k + q5_ksplit_swizzle_64(col, k8 * 8)];
            const int source = col < live_columns ? col : 0;
            cp_async_zfill<16>(dst,
                               &x[static_cast<std::int64_t>(source) * hidden + group_k0 +
                                  warp * tile_k + k8 * 8],
                               col < live_columns ? 16 : 0);
        }
    };

    const auto stage_weight = [&](int group_k0, Stage& stage) {
        constexpr int code_chunks = warps * Schedule::kCodeBytesPerGroup / 16;
        constexpr int high_chunks = warps * Schedule::kHighBytesPerGroup / 16;
        static_assert(code_chunks + high_chunks + 1 <= 32);
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row0 + row;
            if (lane < code_chunks) {
                cp_async<16, Cache::cg>(&stage.codes[row][q5_ksplit_code_offset(row, lane * 16)],
                                        codes + static_cast<std::int64_t>(weight_row) * code_row_bytes +
                                            group_k0 / 2 + lane * 16);
            } else if (lane < code_chunks + high_chunks) {
                const int chunk = lane - code_chunks;
                cp_async<16, Cache::cg>(&stage.high[row][q5_ksplit_high_offset(row, chunk * 16)],
                                        high + static_cast<std::int64_t>(weight_row) * high_row_bytes +
                                            group_k0 / 8 + chunk * 16);
            } else if (lane == code_chunks + high_chunks) {
                cp_async<16>(&stage.scales[row][0],
                             scales + static_cast<std::int64_t>(weight_row) * scale_row_bytes +
                                 (group_k0 / Q5RowSplitStorage::kGroupK) * 2);
            }
        }
    };

    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const int code_off = k_split * Schedule::kCodeBytesPerGroup;
    const int high_off = k_split * Schedule::kHighBytesPerGroup;
    const int hshift = 2 * lid;
    float acc[nt][4] = {};

    if constexpr (stages == 1) {
        stage_weight(0, shared.staging[0]);
        stage_x(0, shared.staging[0]);
        cp_commit();
    } else {
#pragma unroll
        for (int s = 0; s < prefetch; ++s) {
            if (s < groups) {
                stage_weight(s * group_k, shared.staging[s]);
                stage_x(s * group_k, shared.staging[s]);
            }
            cp_commit();
        }
    }

    constexpr int group_unroll = groups <= 12 ? groups : 6;
#pragma unroll group_unroll
    for (int group_index = 0; group_index < groups; ++group_index) {
        const int group_k0 = group_index * group_k;
        const bool has_next = group_index + 1 < groups;
        if constexpr (stages >= 2) {
            const int fetch = group_index + prefetch;
            if (fetch < groups) {
                Stage& next = shared.staging[fetch % stages];
                stage_weight(fetch * group_k, next);
                stage_x(fetch * group_k, next);
            }
            cp_commit();
            cp_wait<prefetch>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const Stage& current = shared.staging[group_index % stages];
        float group_acc[nt][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte_col = code_off + ks * 8 + lid;
            const int high_col = high_off + ks * 2;
            const auto code_at = [&](int row, int byte) {
                return current.codes[row][q5_ksplit_code_offset(row, byte)];
            };
            const auto high_at = [&](int row, int byte) {
                return current.high[row][q5_ksplit_high_offset(row, byte)];
            };
            const unsigned af0 = q5_ksplit_bf16_pair(code_at(gid, byte_col),
                                                     high_at(gid, high_col), hshift);
            const unsigned af1 = q5_ksplit_bf16_pair(code_at(gid + 8, byte_col),
                                                     high_at(gid + 8, high_col), hshift);
            const unsigned af2 = q5_ksplit_bf16_pair(code_at(gid, byte_col + 4),
                                                     high_at(gid, high_col + 1), hshift);
            const unsigned af3 = q5_ksplit_bf16_pair(code_at(gid + 8, byte_col + 4),
                                                     high_at(gid + 8, high_col + 1), hshift);
#pragma unroll
            for (int n = 0; n < nt; ++n) {
                unsigned bf0, bf1;
                const int br = n * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&current.activations[k_split]
                                                          [br * tile_k + q5_ksplit_swizzle_64(
                                                                             br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[n][0], group_acc[n][1], group_acc[n][2], group_acc[n][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(current.scales[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(current.scales[gid + 8][k_split]));
#pragma unroll
        for (int n = 0; n < nt; ++n) {
            acc[n][0] = fmaf(group_acc[n][0], top_scale, acc[n][0]);
            acc[n][1] = fmaf(group_acc[n][1], top_scale, acc[n][1]);
            acc[n][2] = fmaf(group_acc[n][2], bot_scale, acc[n][2]);
            acc[n][3] = fmaf(group_acc[n][3], bot_scale, acc[n][3]);
        }
        if (has_next) {
            __syncthreads();
            if constexpr (stages == 1) {
                stage_weight(group_k0 + group_k, shared.staging[0]);
                stage_x(group_k0 + group_k, shared.staging[0]);
                cp_commit();
            }
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int n = 0; n < nt; ++n) {
            store_vec(partial + ((k_split * nt + n) * 32 + lane) * 4,
                      make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]));
        }
    }
    __syncthreads();
    if ((k_split & 1) == 0) {
#pragma unroll
        for (int n = 0; n < nt; ++n) {
            const float4 partner = load_vec<float4>(partial + (((k_split + 1) * nt + n) * 32 + lane) * 4);
            acc[n][0] += partner.x;
            acc[n][1] += partner.y;
            acc[n][2] += partner.z;
            acc[n][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * nt + n) * 32 + lane) * 4,
                          make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]));
            }
        }
    }
    __syncthreads();
    if (k_split == 0) {
#pragma unroll
        for (int n = 0; n < nt; ++n) {
            float4 sum = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
#pragma unroll
            for (int split = 2; split < warps; split += 2) {
                const float4 value = load_vec<float4>(partial + ((split * nt + n) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = n * 8 + 2 * lid;
            const auto store = [&](int col, int row, float value) {
                __nv_bfloat16* dst = nullptr;
                if constexpr (SplitRows == 0) {
                    dst = out + static_cast<std::int64_t>(col) * out_ld + row;
                } else if (row < SplitRows) {
                    dst = out + static_cast<std::int64_t>(col) * out_ld + row;
                } else {
                    dst = split_out + static_cast<std::int64_t>(col) * split_out_ld +
                          row - SplitRows;
                }
                if constexpr (AddResidual) { value += __bfloat162float(*dst); }
                *dst = __float2bfloat16_rn(value);
            };
            if (col0 < live_columns) {
                store(col0, row0 + gid, sum.x);
                store(col0, row0 + gid + 8, sum.z);
            }
            if (col0 + 1 < live_columns) {
                store(col0 + 1, row0 + gid, sum.y);
                store(col0 + 1, row0 + gid + 8, sum.w);
            }
        }
    }
}

template <int OutputRows, int InputRows, int Capacity, int MaxColumns = Capacity,
          bool AddResidual = false>
void launch_q5_ksplit_mma(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    constexpr int tile_cols = (Capacity + 7) / 8 * 8;
    static_assert(MaxColumns >= Capacity);
    static_assert(MaxColumns == Capacity || Capacity == tile_cols);
    if (weight.n != OutputRows || weight.padded_shape[1] != InputRows) {
        throw std::invalid_argument("q5 K-split MMA: weight geometry differs from instance");
    }
    const std::int32_t columns = x.ne[1];
    if (columns < 1 || columns > MaxColumns) {
        throw std::invalid_argument("q5 K-split MMA: column count exceeds instance capacity");
    }
    const dim3 grid(OutputRows / Q5KSplitMmaSchedule::kRowsPerCta,
                    static_cast<unsigned>((columns + tile_cols - 1) / tile_cols));
    q5_ksplit_mma_kernel<OutputRows, InputRows, tile_cols, Capacity, AddResidual>
        <<<grid, Q5KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), nullptr, OutputRows, 0, columns);
    CUDA_CHECK(cudaGetLastError());
}

template <int OutputRows, int SplitRows, int InputRows, int Capacity>
void launch_q5_ksplit_mma_split(const Tensor& x, const Weight& weight, Tensor& out,
                                Tensor& split_out, cudaStream_t stream) {
    constexpr int tile_cols = (Capacity + 7) / 8 * 8;
    if (weight.n != OutputRows || weight.padded_shape[1] != InputRows ||
        out.ne[0] != SplitRows || split_out.ne[0] != OutputRows - SplitRows ||
        out.ne[1] != x.ne[1] || split_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("q5 K-split MMA split output: tensor geometry differs from instance");
    }
    const std::int32_t columns = x.ne[1];
    if (columns < 1 || columns > Capacity) {
        throw std::invalid_argument("q5 K-split MMA split output: column count exceeds instance");
    }
    const dim3 grid(OutputRows / Q5KSplitMmaSchedule::kRowsPerCta,
                    static_cast<unsigned>((columns + tile_cols - 1) / tile_cols));
    q5_ksplit_mma_kernel<OutputRows, InputRows, tile_cols, Capacity, false, SplitRows>
        <<<grid, Q5KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), static_cast<__nv_bfloat16*>(split_out.data),
            static_cast<int>(out.nb[1] / sizeof(__nv_bfloat16)),
            static_cast<int>(split_out.nb[1] / sizeof(__nv_bfloat16)), columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
