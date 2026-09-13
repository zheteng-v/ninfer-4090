#include "ops/linear/q8/q8_feature.h"

#include "core/device.h"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"
#include "ops/linear/q8/q8_small_t_mma.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int Capacity>
void launch_small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Q8N5120K25600Geometry;
    using Schedule =
        Q8SmallTMmaSchedule<Capacity <= 32 ? 8 : 4, Capacity, 2, Q8SmallTMmaScaleAccess::Shared>;
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_small_t_mma_kernel<Geometry, Capacity, Schedule, Q8ContiguousOutput,
                          Q8SmallTMmaStoreEpilogue, Q8SmallTMmaIdentityRows, false, true>
        <<<Geometry::kOutputRows / 16, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8SmallTMmaStoreEpilogue{},
            Q8SmallTMmaIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

using SmallLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <std::size_t... I>
constexpr auto small_launchers(std::index_sequence<I...>) {
    return std::array<SmallLaunch, sizeof...(I)>{&launch_small<8 * (static_cast<int>(I) + 1)>...};
}

constexpr auto kSmallLaunchers = small_launchers(std::make_index_sequence<7>{});

template <int Rows>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Keep the predicated profile even on full tiles: the Full specialization regresses T=64.
    // A single activation stage makes K128 fit while retaining the common code/scale pipeline.
    using Schedule = Q8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    q8_rowsplit_gemm_mma_kernel<Schedule, false><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q8_feature_small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    kSmallLaunchers[(x.ne[1] - 1) / 8](x, w, out, stream);
}

void launch_q8_feature_r16_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<16>(x, w, out, stream);
}

void launch_q8_feature_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<32>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
