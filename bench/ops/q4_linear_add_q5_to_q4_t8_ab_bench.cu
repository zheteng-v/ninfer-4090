// Isolated T=8 screening A/B for the two Q5 LinearAdd projections used by the 27B model.
//
// Baseline: production Q5 K-split MMA LinearAdd (Q5_G64, N=5120, K=6144/17408).
// Candidate: existing Q4 small-T MMA plus a bench-local FP32 residual epilogue. The candidate
// uses a custom epilogue so the projection accumulator and BF16 residual are added before the one
// observable BF16 round, matching LinearAdd's numerical boundary without changing production.
// Both formats are independently quantized from the same deterministic logical source with the
// repository's grouped-absmax scale rule. The Q4 RowSplit oracle decodes stored codes/scales and
// retains its reference in FP32. Candidate-vs-Q5 difference is reported only as a quality warning.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear/q5/q5_ksplit_mma.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kRows = 5120;
constexpr int kTokens = 8;
constexpr int kQ4OracleRows = 128;
constexpr int kWarmup = 3;
constexpr int kRepeat = 15;
constexpr std::uint64_t kFlushBytes = 256ULL << 20;
constexpr std::uint32_t kWeightSeed = 0x6d2b79f5u;
constexpr std::uint32_t kOrderSeed = 0x51a7c3u;
constexpr double kOracleRelL2Max = 1.0e-2;
constexpr double kRoundSavingGateUs = 730.0;

struct Options {
    int device = 0;
    int warmup = kWarmup;
    int repeat = kRepeat;
    std::uint64_t flush_bytes = kFlushBytes;
};

struct DiffStats {
    double rel_l2 = std::numeric_limits<double>::quiet_NaN();
    double max_abs = 0.0;
    double ref_norm = 0.0;
    double actual_norm = 0.0;
    std::size_t ref_nonzero = 0;
    std::size_t actual_nonzero = 0;

    [[nodiscard]] bool within(double limit) const noexcept {
        return ref_nonzero != 0 && actual_nonzero != 0 && std::isfinite(rel_l2) &&
               rel_l2 <= limit;
    }
};

struct Timing {
    double median_us = 0.0;
    double p95_us = 0.0;
};

struct HostPlanes {
    std::vector<std::uint8_t> q4_low;
    std::vector<std::uint8_t> q5_low;
    std::vector<std::uint8_t> q5_high;
    std::vector<std::uint16_t> q4_scales;
    std::vector<std::uint16_t> q5_scales;
};

std::uint64_t parse_u64(std::string_view text, const char* label) {
    const std::string value(text);
    std::size_t parsed = 0;
    unsigned long long result = 0;
    try {
        result = std::stoull(value, &parsed, 10);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    if (parsed != value.size()) { throw std::invalid_argument(std::string("invalid ") + label); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return std::string_view(argv[i]);
        };
        if (arg == "--device") {
            options.device = static_cast<int>(parse_u64(next("device"), "device"));
        } else if (arg == "--warmup") {
            options.warmup = static_cast<int>(parse_u64(next("warmup"), "warmup"));
        } else if (arg == "--repeat") {
            options.repeat = static_cast<int>(parse_u64(next("repeat"), "repeat"));
        } else if (arg == "--flush-mib") {
            options.flush_bytes = parse_u64(next("flush-mib"), "flush-mib") << 20;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--device N] [--warmup N] [--repeat N] [--flush-mib N]\n"
                        "Q5 production LinearAdd versus Q4 small-T MMA + fused residual; T=8.\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0 || options.flush_bytes == 0) {
        throw std::invalid_argument("warmup must be nonnegative; repeat and flush-mib positive");
    }
    return options;
}

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float logical_weight(int row, int group, int local) {
    const std::uint32_t group_key = kWeightSeed ^ (static_cast<std::uint32_t>(row) * 0x9e3779b9u) ^
                                    (static_cast<std::uint32_t>(group) * 0x85ebca6bu);
    const std::uint32_t bits = mix32(group_key ^ (static_cast<std::uint32_t>(local) * 0xc2b2ae35u));
    const float centered = static_cast<float>((bits >> 8) & 0x00ffffffu) *
                               (2.0f / 16777215.0f) -
                           1.0f;
    const float envelope = 0.025f + static_cast<float>(mix32(group_key ^ 0xa511e9b3u) & 0xffffu) /
                                        65535.0f * 0.075f;
    return centered * envelope;
}

std::int8_t quant_code(float value, float reciprocal, int qmin, int qmax) {
    const long rounded = std::lrintf(value * reciprocal);
    return static_cast<std::int8_t>(std::clamp<long>(rounded, qmin, qmax));
}

void set_nibble(std::uint8_t* low, std::size_t group_index, int local, int code) {
    const std::size_t byte_index = group_index * 32 + static_cast<std::size_t>(local / 2);
    const std::uint8_t nibble = static_cast<std::uint8_t>(code) & 0x0fu;
    if ((local & 1) == 0) {
        low[byte_index] = static_cast<std::uint8_t>((low[byte_index] & 0xf0u) | nibble);
    } else {
        low[byte_index] = static_cast<std::uint8_t>((low[byte_index] & 0x0fu) | (nibble << 4));
    }
}

void set_q5_high(std::uint8_t* high, std::size_t group_index, int local, int code) {
    if ((code & 0x10) == 0) { return; }
    const std::size_t byte_index = group_index * 8 + static_cast<std::size_t>(local / 8);
    high[byte_index] |= static_cast<std::uint8_t>(1u << (local & 7));
}

HostPlanes make_source_and_quantized_planes(int rows, int k) {
    if (k % 64 != 0) { throw std::invalid_argument("both Q4/Q5 formats require K divisible by 64"); }
    const int groups_per_row = k / 64;
    const std::size_t groups = static_cast<std::size_t>(rows) * groups_per_row;
    HostPlanes planes{
        std::vector<std::uint8_t>(groups * 32, 0),
        std::vector<std::uint8_t>(groups * 32, 0),
        std::vector<std::uint8_t>(groups * 8, 0),
        std::vector<std::uint16_t>(groups),
        std::vector<std::uint16_t>(groups),
    };

    // Match tools/convert/quantization/groupwise.py: max(abs(group))/qmax, round scale to
    // binary16 before code selection, then torch.round-equivalent nearest-even quantization.
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups_per_row; ++group) {
            float logical[64];
            float max_abs = 0.0f;
            for (int local = 0; local < 64; ++local) {
                logical[local] = logical_weight(row, group, local);
                max_abs = std::max(max_abs, std::abs(logical[local]));
            }
            const std::size_t group_index = static_cast<std::size_t>(row) * groups_per_row + group;
            const float raw_q4_scale = static_cast<float>(static_cast<double>(max_abs) / 7.0);
            const float raw_q5_scale = static_cast<float>(static_cast<double>(max_abs) / 15.0);
            __half q4_half = __float2half_rn(raw_q4_scale);
            __half q5_half = __float2half_rn(raw_q5_scale);
            if (__half2float(q4_half) == 0.0f && max_abs > 0.0f) { q4_half = __ushort_as_half(1); }
            if (__half2float(q5_half) == 0.0f && max_abs > 0.0f) { q5_half = __ushort_as_half(1); }
            const float q4_scale = __half2float(q4_half);
            const float q5_scale = __half2float(q5_half);
            const float q4_recip = static_cast<float>(1.0 / static_cast<double>(q4_scale));
            const float q5_recip = static_cast<float>(1.0 / static_cast<double>(q5_scale));
            std::memcpy(&planes.q4_scales[group_index], &q4_half, sizeof(std::uint16_t));
            std::memcpy(&planes.q5_scales[group_index], &q5_half, sizeof(std::uint16_t));
            for (int local = 0; local < 64; ++local) {
                const int q4 = quant_code(logical[local], q4_recip, -8, 7);
                const int q5 = quant_code(logical[local], q5_recip, -16, 15);
                set_nibble(planes.q4_low.data(), group_index, local, q4);
                set_nibble(planes.q5_low.data(), group_index, local, q5);
                set_q5_high(planes.q5_high.data(), group_index, local, q5);
            }
        }
    }
    return planes;
}

float half_scale(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

float q4_rowseplit_value(const HostPlanes& planes, int groups_per_row, int row, int k) {
    const int group = k / 64;
    const int local = k % 64;
    const std::size_t group_index = static_cast<std::size_t>(row) * groups_per_row + group;
    const std::uint8_t byte = planes.q4_low[group_index * 32 + static_cast<std::size_t>(local / 2)];
    const int nibble = ((local & 1) == 0) ? (byte & 0x0f) : (byte >> 4);
    return static_cast<float>((nibble ^ 0x08) - 0x08) * half_scale(planes.q4_scales[group_index]);
}

float q5_rowseplit_value(const HostPlanes& planes, int groups_per_row, int row, int k) {
    const int group = k / 64;
    const int local = k % 64;
    const int lane = local / 2;
    const std::size_t group_index = static_cast<std::size_t>(row) * groups_per_row + group;
    const std::uint8_t low = planes.q5_low[group_index * 32 + static_cast<std::size_t>(lane)];
    const std::uint8_t high = planes.q5_high[group_index * 8 + static_cast<std::size_t>(local / 8)];
    const int nibble = (local & 1) == 0 ? (low & 0x0f) : (low >> 4);
    const int high_bit = (high >> (local & 7)) & 1;
    const int code = ((nibble | (high_bit << 4)) ^ 0x10) - 0x10;
    return static_cast<float>(code) * half_scale(planes.q5_scales[group_index]);
}

float bf16_bits_to_float(std::uint16_t bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return __bfloat162float(value);
}

DiffStats compare_float_bf16(const std::vector<float>& reference,
                             const std::vector<std::uint16_t>& actual) {
    if (reference.size() != actual.size()) { throw std::invalid_argument("oracle size mismatch"); }
    double diff_sq = 0.0, ref_sq = 0.0, actual_sq = 0.0, max_abs = 0.0;
    std::size_t ref_nonzero = 0, actual_nonzero = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double ref = reference[i];
        const double got = bf16_bits_to_float(actual[i]);
        diff_sq += (got - ref) * (got - ref);
        ref_sq += ref * ref;
        actual_sq += got * got;
        ref_nonzero += ref != 0.0 ? 1u : 0u;
        actual_nonzero += got != 0.0 ? 1u : 0u;
        max_abs = std::max(max_abs, std::abs(got - ref));
    }
    DiffStats result;
    result.rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq)
                                 : std::numeric_limits<double>::quiet_NaN();
    result.max_abs = max_abs;
    result.ref_norm = std::sqrt(ref_sq);
    result.actual_norm = std::sqrt(actual_sq);
    result.ref_nonzero = ref_nonzero;
    result.actual_nonzero = actual_nonzero;
    return result;
}

DiffStats compare_bf16(const std::vector<std::uint16_t>& reference,
                       const std::vector<std::uint16_t>& actual) {
    if (reference.size() != actual.size()) { throw std::invalid_argument("screen size mismatch"); }
    std::vector<float> fp32_reference(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        fp32_reference[i] = bf16_bits_to_float(reference[i]);
    }
    return compare_float_bf16(fp32_reference, actual);
}

Timing summarize(std::vector<double> values) {
    if (values.empty()) { throw std::invalid_argument("empty timing samples"); }
    std::sort(values.begin(), values.end());
    const auto at = [&](double fraction) {
        const std::size_t index = static_cast<std::size_t>(fraction * (values.size() - 1));
        return values[std::min(index, values.size() - 1)];
    };
    return {at(0.50), at(0.95)};
}

struct Q4ResidualEpilogue {
    const __nv_bfloat16* residual = nullptr;
    __nv_bfloat16* output = nullptr;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int col0, float4 sum) const {
        if (col0 < ActiveCols) {
            const std::int64_t index = static_cast<std::int64_t>(col0) * kRows + row;
            output[index] = __float2bfloat16_rn(sum.x + __bfloat162float(residual[index]));
            const std::int64_t bottom = index + 8;
            output[bottom] = __float2bfloat16_rn(sum.z + __bfloat162float(residual[bottom]));
        }
        if (col0 + 1 < ActiveCols) {
            const std::int64_t index = static_cast<std::int64_t>(col0 + 1) * kRows + row;
            output[index] = __float2bfloat16_rn(sum.y + __bfloat162float(residual[index]));
            const std::int64_t bottom = index + 8;
            output[bottom] = __float2bfloat16_rn(sum.w + __bfloat162float(residual[bottom]));
        }
    }
};

template <int K>
void launch_q4_candidate(const Tensor& x, const Weight& weight, Tensor& out,
                         const Tensor& residual, cudaStream_t stream) {
    using Schedule = ops::detail::Q4DraftSmallTSchedule16;
    using Problem = ops::detail::Q4LinearSmallTGeometry<kRows, K>;
    static_assert(K % Schedule::kGroupK == 0, "Q4 small-T MMA K must divide its 1024-value group");
    static_assert(kRows % Schedule::kRowsPerCta == 0);
    if (x.ne[0] != K || x.ne[1] != kTokens || weight.n != kRows || weight.k != K ||
        weight.padded_shape[1] != K || out.ne[0] != kRows || out.ne[1] != kTokens ||
        residual.ne[0] != kRows || residual.ne[1] != kTokens) {
        throw std::invalid_argument("Q4 LinearAdd candidate geometry mismatch");
    }
    constexpr unsigned blocks = kRows / ops::detail::Q4SmallTMmaIdentityRows::kOutputRowsPerCta;
    ops::detail::q4_small_t_mma_kernel<Problem, 8, 8, Q4ResidualEpilogue,
                                       ops::detail::Q4SmallTMmaIdentityRows, false, Schedule>
        <<<blocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data),
            Q4ResidualEpilogue{static_cast<const __nv_bfloat16*>(residual.data),
                               static_cast<__nv_bfloat16*>(out.data)},
            ops::detail::Q4SmallTMmaIdentityRows{}, kTokens);
    CUDA_CHECK(cudaGetLastError());
}

template <int K>
void launch_q5_baseline(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    ops::detail::launch_q5_ksplit_mma<kRows, K, 8, 8, true>(x, weight, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_and_upload(int k, bench::PackedQuantizedWeight& q4,
                         bench::PackedQuantizedWeight& q5, HostPlanes& planes) {
    const int groups_per_row = k / 64;
    planes = make_source_and_quantized_planes(kRows, k);
    if (q4.low_bytes != planes.q4_low.size() || q4.high_bytes != 0 ||
        q4.scale_bytes != planes.q4_scales.size() * sizeof(std::uint16_t) ||
        q5.low_bytes != planes.q5_low.size() || q5.high_bytes != planes.q5_high.size() ||
        q5.scale_bytes != planes.q5_scales.size() * sizeof(std::uint16_t)) {
        throw std::logic_error("packed benchmark weight geometry differs from RowSplit format");
    }
    (void)groups_per_row;
    q4.storage.copy_from_host(planes.q4_low.data(), planes.q4_low.size(), 0);
    q4.storage.copy_from_host(planes.q4_scales.data(), planes.q4_scales.size() * 2,
                              q4.scale_offset);
    q5.storage.copy_from_host(planes.q5_low.data(), planes.q5_low.size(), 0);
    q5.storage.copy_from_host(planes.q5_high.data(), planes.q5_high.size(), q5.high_offset);
    q5.storage.copy_from_host(planes.q5_scales.data(), planes.q5_scales.size() * 2,
                              q5.scale_offset);
}

std::vector<std::uint16_t> device_bf16_to_host(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<std::uint16_t> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(std::uint16_t));
    return values;
}

std::vector<std::int32_t> oracle_rows() {
    std::vector<std::int32_t> rows;
    rows.reserve(kQ4OracleRows);
    for (int i = 0; i < kQ4OracleRows; ++i) {
        rows.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(i) * kRows) /
                                                 kQ4OracleRows));
    }
    return rows;
}

std::vector<float> build_oracle(const HostPlanes& planes, int k,
                                const std::vector<std::uint16_t>& input,
                                const std::vector<std::uint16_t>& residual,
                                const std::vector<std::int32_t>& sampled_rows, bool q4) {
    const int groups_per_row = k / 64;
    std::vector<float> result(static_cast<std::size_t>(kTokens) * sampled_rows.size());
    for (int token = 0; token < kTokens; ++token) {
        for (std::size_t sample = 0; sample < sampled_rows.size(); ++sample) {
            const int row = sampled_rows[sample];
            double accumulator = 0.0;
            for (int kk = 0; kk < k; ++kk) {
                const float w = q4 ? q4_rowseplit_value(planes, groups_per_row, row, kk)
                                   : q5_rowseplit_value(planes, groups_per_row, row, kk);
                const float x = bf16_bits_to_float(input[static_cast<std::size_t>(token) * k + kk]);
                accumulator += static_cast<double>(w) * static_cast<double>(x);
            }
            const float base = bf16_bits_to_float(
                residual[static_cast<std::size_t>(token) * kRows + row]);
            result[static_cast<std::size_t>(token) * sampled_rows.size() + sample] =
                static_cast<float>(accumulator) + base;
        }
    }
    return result;
}

std::vector<std::uint16_t> select_rows(const std::vector<std::uint16_t>& output,
                                       const std::vector<std::int32_t>& rows) {
    std::vector<std::uint16_t> selected(static_cast<std::size_t>(kTokens) * rows.size());
    for (int token = 0; token < kTokens; ++token) {
        for (std::size_t sample = 0; sample < rows.size(); ++sample) {
            selected[static_cast<std::size_t>(token) * rows.size() + sample] =
                output[static_cast<std::size_t>(token) * kRows + rows[sample]];
        }
    }
    return selected;
}

std::pair<Timing, Timing> measure_pair(const std::function<void(cudaStream_t)>& baseline_prepare,
                                       const std::function<void(cudaStream_t)>& baseline,
                                       const std::function<void(cudaStream_t)>& candidate_prepare,
                                       const std::function<void(cudaStream_t)>& candidate,
                                       DeviceBuffer& flush, cudaStream_t stream,
                                       const Options& options, std::mt19937& order_rng) {
    for (int i = 0; i < options.warmup; ++i) {
        baseline_prepare(stream);
        bench::flush_l2(flush, stream);
        baseline(stream);
        candidate_prepare(stream);
        bench::flush_l2(flush, stream);
        candidate(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> baseline_samples, candidate_samples;
    baseline_samples.reserve(static_cast<std::size_t>(options.repeat));
    candidate_samples.reserve(static_cast<std::size_t>(options.repeat));
    std::bernoulli_distribution coin(0.5);
    for (int trial = 0; trial < options.repeat; ++trial) {
        const bool baseline_first = coin(order_rng);
        for (int slot = 0; slot < 2; ++slot) {
            const bool is_baseline = (slot == 0) == baseline_first;
            (is_baseline ? baseline_prepare : candidate_prepare)(stream);
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaEventRecord(start, stream));
            if (is_baseline) {
                baseline(stream);
            } else {
                candidate(stream);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            (is_baseline ? baseline_samples : candidate_samples)
                .push_back(static_cast<double>(milliseconds) * 1000.0);
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {summarize(std::move(baseline_samples)), summarize(std::move(candidate_samples))};
}

template <int K>
double run_case(DeviceBuffer& input, DeviceBuffer& residual, DeviceBuffer& flush,
                cudaStream_t stream, const Options& options, std::mt19937& order_rng) {
    auto q4 = bench::make_row_split_weight(QType::Q4_G64_FP16, kRows, K, K);
    auto q5 = bench::make_row_split_weight(QType::Q5_G64_FP16, kRows, K, K);
    HostPlanes planes;
    quantize_and_upload(K, q4, q5, planes);

    const std::size_t matrix_elements = static_cast<std::size_t>(kRows) * kTokens;
    DeviceBuffer baseline_out = bench::make_zeros(matrix_elements * sizeof(std::uint16_t));
    DeviceBuffer candidate_out = bench::make_zeros(matrix_elements * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(baseline_out.p, residual.p, matrix_elements * 2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(candidate_out.p, residual.p, matrix_elements * 2, cudaMemcpyDeviceToDevice));

    Tensor x(input.p, DType::BF16, {K, kTokens});
    Tensor base(residual.p, DType::BF16, {kRows, kTokens});
    Tensor baseline_tensor(baseline_out.p, DType::BF16, {kRows, kTokens});
    Tensor candidate_tensor(candidate_out.p, DType::BF16, {kRows, kTokens});

    const auto baseline_prepare = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(baseline_out.p, residual.p, matrix_elements * 2,
                                   cudaMemcpyDeviceToDevice, s));
    };
    const auto candidate_prepare = [&](cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(candidate_out.p, residual.p, matrix_elements * 2,
                                   cudaMemcpyDeviceToDevice, s));
    };
    const auto baseline = [&](cudaStream_t s) {
        launch_q5_baseline<K>(x, q5.weight, baseline_tensor, s);
    };
    const auto candidate = [&](cudaStream_t s) {
        launch_q4_candidate<K>(x, q4.weight, candidate_tensor, base, s);
    };

    baseline(stream);
    candidate(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto host_baseline = device_bf16_to_host(baseline_out, matrix_elements);
    const auto host_candidate = device_bf16_to_host(candidate_out, matrix_elements);
    const DiffStats quantization_screen = compare_bf16(host_baseline, host_candidate);

    std::vector<std::uint16_t> host_input(static_cast<std::size_t>(K) * kTokens);
    std::vector<std::uint16_t> host_residual(matrix_elements);
    input.copy_to_host(host_input.data(), host_input.size() * 2);
    residual.copy_to_host(host_residual.data(), host_residual.size() * 2);
    const auto sampled = oracle_rows();
    const auto q4_reference = build_oracle(planes, K, host_input, host_residual, sampled, true);
    const auto q5_reference = build_oracle(planes, K, host_input, host_residual, sampled, false);
    const auto q4_actual = select_rows(host_candidate, sampled);
    const auto q5_actual = select_rows(host_baseline, sampled);
    const DiffStats q4_oracle = compare_float_bf16(q4_reference, q4_actual);
    const DiffStats q5_oracle = compare_float_bf16(q5_reference, q5_actual);

    std::printf("q4_linear_add_oracle N=%d K=%d T=%d rows=%zu logical_source=seeded_groupwise_absmax "
                "q4_rel_l2=%.3e q4_max_abs=%.3e q4_pass=%d q5_rel_l2=%.3e q5_max_abs=%.3e "
                "q5_pass=%d threshold=%.1e reference=independent_fp32_rowsplit_plus_bf16_residual\n",
                kRows, K, kTokens, sampled.size(), q4_oracle.rel_l2, q4_oracle.max_abs,
                q4_oracle.within(kOracleRelL2Max) ? 1 : 0, q5_oracle.rel_l2,
                q5_oracle.max_abs, q5_oracle.within(kOracleRelL2Max) ? 1 : 0,
                kOracleRelL2Max);
    std::printf("q4_linear_add_quality_warning N=%d K=%d T=%d full_output_rel_l2=%.3e "
                "max_abs=%.3e q5_norm=%.6e q4_norm=%.6e note=candidate_vs_q5_is_not_a_correctness_gate\n",
                kRows, K, kTokens, quantization_screen.rel_l2, quantization_screen.max_abs,
                quantization_screen.ref_norm, quantization_screen.actual_norm);

    const bool oracle_ok = q4_oracle.within(kOracleRelL2Max) && q5_oracle.within(kOracleRelL2Max);
    const auto [baseline_timing, candidate_timing] =
        measure_pair(baseline_prepare, baseline, candidate_prepare, candidate, flush, stream,
                     options, order_rng);
    const double saving_us = baseline_timing.median_us - candidate_timing.median_us;
    std::printf("q4_linear_add_timing N=%d K=%d T=%d baseline=q5.production.ksplit_linear_add "
                "candidate=q4.small_t_mma.fp32_residual_fused baseline_median_us=%.3f "
                "baseline_p95_us=%.3f candidate_median_us=%.3f candidate_p95_us=%.3f "
                "ratio_candidate_over_baseline=%.4f saving_us_per_call=%.3f calls_per_round=64 "
                "weighted_saving_us_per_round=%.3f oracle_ok=%d quality_gate=not_evaluated\n",
                kRows, K, kTokens, baseline_timing.median_us, baseline_timing.p95_us,
                candidate_timing.median_us, candidate_timing.p95_us,
                candidate_timing.median_us / baseline_timing.median_us, saving_us,
                saving_us * 64.0, oracle_ok ? 1 : 0);
    return oracle_ok ? saving_us * 64.0 : std::numeric_limits<double>::quiet_NaN();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (options.device < 0 || options.device >= count) {
            throw std::invalid_argument("requested CUDA device is unavailable");
        }
        CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
        std::printf("device=%d gpu=%s sm=%d%d warmup=%d repeat=%d flush_mib=%llu "
                    "weight_seed=%u order_seed=%u kv_dtype=not_applicable\n",
                    options.device, properties.name, properties.major, properties.minor,
                    options.warmup, options.repeat,
                    static_cast<unsigned long long>(options.flush_bytes >> 20), kWeightSeed,
                    kOrderSeed);
        if (properties.major != 8 || properties.minor != 9) {
            throw std::invalid_argument("screen is qualified only for RTX 4090 sm89 execution");
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(17408) * kTokens);
        DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(kRows) * kTokens);
        DeviceBuffer flush(options.flush_bytes);
        std::mt19937 order_rng(kOrderSeed);
        const double down_saving = run_case<17408>(input, residual, flush, stream, options,
                                                   order_rng);
        const double oproj_saving = run_case<6144>(input, residual, flush, stream, options,
                                                   order_rng);
        const bool combined_valid = std::isfinite(down_saving) && std::isfinite(oproj_saving);
        const double combined_saving = combined_valid ? down_saving + oproj_saving
                                                       : std::numeric_limits<double>::quiet_NaN();
        std::printf("q4_linear_add_combined weighted_saving_us_per_round=%.3f gate_us=%.1f "
                    "calls_per_shape=64 oracle_valid=%d continue=%d "
                    "note=screening_only_not_quality_qualification\n",
                    combined_saving, kRoundSavingGateUs, combined_valid ? 1 : 0,
                    combined_valid && combined_saving >= kRoundSavingGateUs ? 1 : 0);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q4_linear_add_q5_to_q4_t8_ab_bench: %s\n", error.what());
        return 1;
    }
}
