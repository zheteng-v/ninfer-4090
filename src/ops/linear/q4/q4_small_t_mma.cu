#include "core/weight.h"
#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 8;
constexpr int kLastOptimizedT = 20;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

} // namespace

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

void launch_q4_small_t_n4096_k5120_t8(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule16;
    using Geometry = Q4LinearSmallTGeometry<4096, 5120>;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    static_assert(Q4SmallTMmaIdentityRows::kOutputRowsPerCta <= Schedule::kRowsPerCta);
    if (x.ne[1] != 8 || weight.n != Geometry::kOutputRows ||
        weight.k != Geometry::kInputRows || weight.padded_shape[1] != Geometry::kInputRows ||
        out.ne[0] != Geometry::kOutputRows || out.ne[1] != x.ne[1] ||
        (out.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat16))) != 0) {
        throw std::invalid_argument("Q4 GDN QK small-T route: tensor geometry differs from instance");
    }
    // out is a sliced view of the parent QKV tensor, so the real leading stride (10,240 here) must be
    // used instead of Geometry::kOutputRows; a shorter stride would overwrite neighbouring columns.
    const std::int32_t output_stride =
        static_cast<std::int32_t>(out.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
    if (output_stride < Geometry::kOutputRows) {
        throw std::invalid_argument("Q4 GDN QK small-T route: output leading stride is too small");
    }
    constexpr int kBlocks = Geometry::kOutputRows / Q4SmallTMmaIdentityRows::kOutputRowsPerCta;
    q4_small_t_mma_kernel<Geometry, 8, 8, Q4SmallTMmaStoreEpilogue, Q4SmallTMmaIdentityRows, false,
                          Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<__nv_bfloat16*>(out.data), Q4SmallTMmaStoreEpilogue{},
        Q4SmallTMmaIdentityRows{}, 8, output_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
