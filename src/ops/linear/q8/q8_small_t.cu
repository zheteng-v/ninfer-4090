#include "core/weight.h"
#include "ops/linear/q8/q8_launch.h"

#include "core/device.h"
#include "ops/linear/q8/q8_config.h"
#include "ops/linear/q8/q8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Q8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

template <int Capacity>
void launch_vocabulary_tile(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    using Geometry = Q8VocabularyProjectionGeometry;
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

template <std::size_t... I>
constexpr auto vocabulary_launchers(std::index_sequence<I...>) {
    return std::array<Q8Launch, sizeof...(I)>{
        &launch_vocabulary_tile<8 * (static_cast<int>(I) + 1)>...};
}

constexpr auto kVocabularyLaunchers = vocabulary_launchers(std::make_index_sequence<5>{});
constexpr auto kMtpInputLaunchers =
    make_launchers<Q8MtpInputProjectionGeometry, kQ8MtpInputFirstSmallT>(
        std::make_index_sequence<kQ8MtpInputLastSmallT - kQ8MtpInputFirstSmallT + 1>{});
constexpr auto kMtpAttentionLaunchers =
    make_launchers<Q8MtpAttentionProjectionGeometry, kQ8MtpAttentionFirstSmallT>(
        std::make_index_sequence<kQ8MtpAttentionLastSmallT - kQ8MtpAttentionFirstSmallT + 1>{});
constexpr auto kMtpAttentionOutputLaunchers =
    make_launchers<Q8MtpAttentionOutputGeometry, kQ8MtpAttentionOutputFirstSmallT>(
        std::make_index_sequence<kQ8MtpAttentionOutputLastSmallT -
                                 kQ8MtpAttentionOutputFirstSmallT + 1>{});
constexpr auto kMtpGateUpLaunchers =
    make_launchers<Q8MtpGateUpProjectionGeometry, kQ8MtpGateUpFirstSmallT>(
        std::make_index_sequence<kQ8MtpGateUpLastSmallT - kQ8MtpGateUpFirstSmallT + 1>{});
constexpr auto kMtpDownLaunchers =
    make_launchers<Q8MtpDownProjectionGeometry, kQ8MtpDownFirstSmallT>(
        std::make_index_sequence<kQ8MtpDownLastSmallT - kQ8MtpDownFirstSmallT + 1>{});
constexpr auto k35bMtpProjectionLaunchers = make_launchers<Q835bMtpProjectionGeometry,
                                                           kQ835bMtpProjectionFirstSmallT>(
    std::make_index_sequence<kQ835bMtpProjectionLastSmallT - kQ835bMtpProjectionFirstSmallT + 1>{});
constexpr auto kDFlash2AttentionLaunchers = make_launchers<Q8DFlash2AttentionProjectionGeometry,
                                                           kQ8DFlash2AttentionFirstSmallT>(
    std::make_index_sequence<kQ8DFlash2AttentionLastSmallT - kQ8DFlash2AttentionFirstSmallT + 1>{});

} // namespace

void launch_q8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == Q8VocabularyProjectionGeometry::kOutputRows &&
        weight.k == Q8VocabularyProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8VocabularyProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8VocabularyFirstSmallT && x.ne[1] <= kQ8VocabularyLastSmallT) {
        const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
        kVocabularyLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8MtpInputProjectionGeometry::kOutputRows &&
        weight.k == Q8MtpInputProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8MtpInputProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8MtpInputFirstSmallT && x.ne[1] <= kQ8MtpInputLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kQ8MtpInputFirstSmallT);
        kMtpInputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8MtpAttentionProjectionGeometry::kOutputRows &&
        weight.k == Q8MtpAttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8MtpAttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8MtpAttentionFirstSmallT && x.ne[1] <= kQ8MtpAttentionLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kQ8MtpAttentionFirstSmallT);
        kMtpAttentionLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8MtpAttentionOutputGeometry::kOutputRows &&
        weight.k == Q8MtpAttentionOutputGeometry::kInputRows &&
        weight.padded_shape[1] == Q8MtpAttentionOutputGeometry::kInputRows &&
        x.ne[1] >= kQ8MtpAttentionOutputFirstSmallT && x.ne[1] <= kQ8MtpAttentionOutputLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kQ8MtpAttentionOutputFirstSmallT);
        kMtpAttentionOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8MtpGateUpProjectionGeometry::kOutputRows &&
        weight.k == Q8MtpGateUpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8MtpGateUpProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8MtpGateUpFirstSmallT && x.ne[1] <= kQ8MtpGateUpLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kQ8MtpGateUpFirstSmallT);
        kMtpGateUpLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8MtpDownProjectionGeometry::kOutputRows &&
        weight.k == Q8MtpDownProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8MtpDownProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8MtpDownFirstSmallT && x.ne[1] <= kQ8MtpDownLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kQ8MtpDownFirstSmallT);
        kMtpDownLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q835bMtpProjectionGeometry::kOutputRows &&
        weight.k == Q835bMtpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q835bMtpProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ835bMtpProjectionFirstSmallT && x.ne[1] <= kQ835bMtpProjectionLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kQ835bMtpProjectionFirstSmallT);
        k35bMtpProjectionLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == Q8DFlash2AttentionProjectionGeometry::kOutputRows &&
        weight.k == Q8DFlash2AttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == Q8DFlash2AttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kQ8DFlash2AttentionFirstSmallT && x.ne[1] <= kQ8DFlash2AttentionLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kQ8DFlash2AttentionFirstSmallT);
        kDFlash2AttentionLaunchers[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q8 Linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
