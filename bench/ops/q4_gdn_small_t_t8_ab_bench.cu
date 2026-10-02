// Minimal faithful screening A/B for the GDN QK Q4 projection at T=8:
//   baseline  = production q4_rowsplit_gemm_simt_kernel
//               <Q4RowSplitSimtGemmSchedule<8,8,16,2,Cache::ca,1>, true, false, 0>
//               (GDN dedicated launcher shape: grid (N/8,1), 256 threads, single output)
//   candidate = existing q4_small_t_mma_kernel with a bench-local linear geometry
//               {kOutputRows=4096, kInputRows=5120}, Schedule=Q4DraftSmallTSchedule16,
//               TileCols=8, ActiveCols=8, StoreEpilogue, IdentityRows, MaskedColumns=false,
//               grid = 4096/16
// Same deterministic random RowSplit Q4_G64_FP16 weight and BF16 input. The independent FP32 Q4
// RowSplit decode/dot oracle keeps its reference in FP32; screening evidence only.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
#include <tuple>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows         = 4096;  // GDN QK output rows
constexpr std::int32_t kK            = 5120;
constexpr std::int32_t kT            = 8;
constexpr std::int32_t kGroupsPerRow = kK / 64;  // 80
constexpr std::int32_t kSimtRowsPerCta = 8;
constexpr std::int32_t kSimtThreads    = 256;
constexpr std::int32_t kOracleRows     = 256;
constexpr std::uint64_t kFlushBytes    = 256ULL << 20;
constexpr std::uint32_t kOrderSeed     = 0x51a7c3u;
constexpr std::uint32_t kWeightSeed    = 0x9e3779b9u;
constexpr double kScreenRelL2Max       = 1.0e-2;

struct GdnQkSmallTGeometry {
    static constexpr int kOutputRows   = kRows;
    static constexpr int kInputRows    = kK;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

using Q4SimtR8C8 = ops::detail::Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, ops::Cache::ca, 1>;
using Q4SmallT    = ops::detail::Q4DraftSmallTSchedule16;

struct Options {
    int device = 0;
    int warmup = 5;
    int repeat = 30;
    std::uint64_t flush_bytes = kFlushBytes;
};

struct DiffStats {
    double rel_l2           = std::numeric_limits<double>::quiet_NaN();
    double max_abs          = 0.0;
    double ref_norm         = 0.0;
    double new_norm         = 0.0;
    std::size_t ref_nonzero = 0;
    std::size_t new_nonzero = 0;

    [[nodiscard]] bool within(double limit) const noexcept {
        return ref_nonzero > 0 && new_nonzero > 0 && std::isfinite(rel_l2) && rel_l2 <= limit;
    }
};

struct Timing {
    double median_us = 0.0;
    double p95_us    = 0.0;
};

std::int32_t div_up(std::int32_t value, std::int32_t divisor) {
    return (value + divisor - 1) / divisor;
}

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
                        "GDN QK Q4 N=4096 K=5120 T=8: production SIMT baseline vs Q4 small-T MMA.\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    if (options.flush_bytes == 0) { throw std::invalid_argument("--flush-mib must be positive"); }
    return options;
}

inline float bf16_bits_to_float(std::uint16_t bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return __bfloat162float(value);
}

DiffStats compare_bf16(const std::vector<std::uint16_t>& reference,
                       const std::vector<std::uint16_t>& actual) {
    if (reference.size() != actual.size()) { throw std::invalid_argument("compare length mismatch"); }
    double diff_sq = 0.0, ref_sq = 0.0, new_sq = 0.0, max_abs = 0.0;
    std::size_t ref_nz = 0, new_nz = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double r = bf16_bits_to_float(reference[i]);
        const double a = bf16_bits_to_float(actual[i]);
        diff_sq += (a - r) * (a - r);
        ref_sq += r * r;
        new_sq += a * a;
        ref_nz += r != 0.0 ? 1U : 0U;
        new_nz += a != 0.0 ? 1U : 0U;
        max_abs = std::max(max_abs, std::abs(a - r));
    }
    DiffStats stats;
    stats.max_abs     = max_abs;
    stats.ref_norm    = std::sqrt(ref_sq);
    stats.new_norm    = std::sqrt(new_sq);
    stats.ref_nonzero = ref_nz;
    stats.new_nonzero = new_nz;
    stats.rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq)
                                : std::numeric_limits<double>::quiet_NaN();
    return stats;
}

// Oracle contract: the reference side stays FP32 and is never rounded to the output dtype.
DiffStats compare_float_bf16(const std::vector<float>& reference,
                             const std::vector<std::uint16_t>& actual) {
    if (reference.size() != actual.size()) { throw std::invalid_argument("compare length mismatch"); }
    double diff_sq = 0.0, ref_sq = 0.0, new_sq = 0.0, max_abs = 0.0;
    std::size_t ref_nz = 0, new_nz = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double r = static_cast<double>(reference[i]);
        const double a = bf16_bits_to_float(actual[i]);
        diff_sq += (a - r) * (a - r);
        ref_sq += r * r;
        new_sq += a * a;
        ref_nz += r != 0.0 ? 1U : 0U;
        new_nz += a != 0.0 ? 1U : 0U;
        max_abs = std::max(max_abs, std::abs(a - r));
    }
    DiffStats stats;
    stats.max_abs     = max_abs;
    stats.ref_norm    = std::sqrt(ref_sq);
    stats.new_norm    = std::sqrt(new_sq);
    stats.ref_nonzero = ref_nz;
    stats.new_nonzero = new_nz;
    stats.rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq)
                                : std::numeric_limits<double>::quiet_NaN();
    return stats;
}

Timing summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto at = [&](double f) {
        return values[std::min(values.size() - 1,
                               static_cast<std::size_t>(f * static_cast<double>(values.size() - 1)))];
    };
    return {at(0.50), at(0.95)};
}

// Q4_G64_FP16 RowSplit: low nibble plane + FP16 group scales, no high plane.
void fill_q4_weight(bench::PackedQuantizedWeight& packed) {
    const std::size_t groups = static_cast<std::size_t>(kRows) * kGroupsPerRow;
    if (packed.low_bytes != groups * 32 || packed.scale_bytes != groups * 2 ||
        packed.high_bytes != 0) {
        throw std::logic_error("benchmark Q4 weight planes differ from the RowSplit layout");
    }
    std::mt19937 rng(kWeightSeed);
    std::vector<std::uint8_t> low(groups * 32);
    for (std::uint8_t& byte : low) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    std::vector<std::uint16_t> scales(groups);
    for (std::size_t i = 0; i < groups; ++i) {
        const float scale    = 0.02F + static_cast<float>(rng() % 97U) * 0.004F;
        const __half encoded = __float2half_rn(scale);
        std::memcpy(&scales[i], &encoded, sizeof(scales[i]));
    }
    packed.storage.copy_from_host(low.data(), low.size(), 0);
    packed.storage.copy_from_host(scales.data(), scales.size() * 2, packed.scale_offset);
}

float q4_rowseplit_dequant(const std::uint8_t* payload, std::uint64_t scale_offset,
                           std::int32_t row, std::int32_t k) {
    const std::int32_t group = k / 64;
    const std::int32_t local = k % 64;
    const std::int64_t index = static_cast<std::int64_t>(row) * kGroupsPerRow + group;
    const std::uint8_t code  = payload[index * 32 + (local >> 1)];
    const int nibble         = (local & 1) ? (code >> 4) : (code & 0x0f);
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, payload + scale_offset + index * 2, sizeof(scale_bits));
    return static_cast<float>((nibble ^ 0x08) - 0x08) * __half2float(__ushort_as_half(scale_bits));
}

std::tuple<Timing, Timing, bool> timing_ab(const std::function<void(cudaStream_t)>& old_launch,
                                           const std::function<void(cudaStream_t)>& new_launch,
                                           DeviceBuffer& flush, cudaStream_t stream,
                                           const Options& options, std::mt19937& rng,
                                           bool candidates_clean) {
    for (int i = 0; i < options.warmup; ++i) {
        bench::flush_l2(flush, stream);
        old_launch(stream);
        bench::flush_l2(flush, stream);
        new_launch(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples_old, samples_new;
    samples_old.reserve(static_cast<std::size_t>(options.repeat));
    samples_new.reserve(static_cast<std::size_t>(options.repeat));
    std::bernoulli_distribution coin(0.5);
    for (int trial = 0; trial < options.repeat; ++trial) {
        const bool old_first = coin(rng);
        for (int slot = 0; slot < 2; ++slot) {
            const bool use_old          = (slot == 0) == old_first;
            std::vector<double>& sample = use_old ? samples_old : samples_new;
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaEventRecord(start, stream));
            if (use_old) {
                old_launch(stream);
            } else {
                new_launch(stream);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            sample.push_back(static_cast<double>(milliseconds) * 1000.0);
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {summarize(std::move(samples_old)), summarize(std::move(samples_new)), candidates_clean};
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (options.device >= device_count) { throw std::invalid_argument("device unavailable"); }
        CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
        std::printf("device=%d gpu=%s sm=%d%d warmup=%d repeat=%d order_seed=%u\n", options.device,
                    properties.name, properties.major, properties.minor, options.warmup,
                    options.repeat, kOrderSeed);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(options.flush_bytes);
        std::mt19937 rng(kOrderSeed);

        bench::PackedQuantizedWeight packed =
            bench::make_row_split_weight(QType::Q4_G64_FP16, kRows, kK, kK, {0x31, 0xa5, 0x3c00});
        fill_q4_weight(packed);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kK) * kT);
        DeviceBuffer out   = bench::make_zeros(static_cast<std::size_t>(kRows) * kT * 2);
        DeviceBuffer cand  = bench::make_zeros(static_cast<std::size_t>(kRows) * kT * 2);
        Tensor x(input.p, DType::BF16, {kK, kT});
        Tensor y(out.p, DType::BF16, {kRows, kT});
        Tensor c(cand.p, DType::BF16, {kRows, kT});

        const bool simt_full =
            (kRows % kSimtRowsPerCta) == 0 && ((kK / 64) % 16) == 0 && (kT % 8) == 0;
        if (!simt_full) { throw std::logic_error("gdn_qk production Full predicate is false"); }

        const auto baseline = [&](cudaStream_t s) {
            const dim3 grid(static_cast<unsigned>(kRows / kSimtRowsPerCta),
                            static_cast<unsigned>(div_up(kT, 8)), 1u);
            ops::detail::q4_rowsplit_gemm_simt_kernel<Q4SimtR8C8, true, false, 0>
                <<<grid, kSimtThreads, 0, s>>>(
                    static_cast<const __nv_bfloat16*>(input.p),
                    static_cast<const std::uint8_t*>(packed.weight.qdata),
                    static_cast<const std::uint8_t*>(packed.weight.scales),
                    static_cast<__nv_bfloat16*>(out.p), nullptr, kRows, 0, kRows, kK, kT,
                    packed.weight.padded_shape[1]);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto candidate = [&](cudaStream_t s) {
            constexpr int kOutputRowsPerCta = ops::detail::Q4SmallTMmaIdentityRows::kOutputRowsPerCta;
            const dim3 grid(static_cast<unsigned>(kRows / kOutputRowsPerCta), 1u, 1u);
            ops::detail::q4_small_t_mma_kernel<GdnQkSmallTGeometry, 8, 8,
                                               ops::detail::Q4SmallTMmaStoreEpilogue,
                                               ops::detail::Q4SmallTMmaIdentityRows, false, Q4SmallT>
                <<<grid, Q4SmallT::kThreads, 0, s>>>(
                    static_cast<const __nv_bfloat16*>(input.p),
                    static_cast<const std::uint8_t*>(packed.weight.qdata),
                    static_cast<const std::uint8_t*>(packed.weight.scales),
                    static_cast<__nv_bfloat16*>(cand.p), ops::detail::Q4SmallTMmaStoreEpilogue{},
                    ops::detail::Q4SmallTMmaIdentityRows{}, kT);
            CUDA_CHECK(cudaGetLastError());
        };

        baseline(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        candidate(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<std::uint16_t> host_baseline(static_cast<std::size_t>(kRows) * kT);
        std::vector<std::uint16_t> host_cand(static_cast<std::size_t>(kRows) * kT);
        CUDA_CHECK(cudaMemcpy(host_baseline.data(), out.p, host_baseline.size() * 2,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_cand.data(), cand.p, host_cand.size() * 2,
                              cudaMemcpyDeviceToHost));

        const DiffStats screen = compare_bf16(host_baseline, host_cand);
        std::printf("gdn_small_t_screen N=%d K=%d T=%d old=production.simt new=small_t.mma "
                    "old_norm=%.6e new_norm=%.6e old_nonzero=%zu new_nonzero=%zu rel_l2=%.3e "
                    "max_abs=%.3e screen_pass=%d threshold_rel_l2=%.1e screening_only=true "
                    "qualification=false\n",
                    kRows, kK, kT, screen.ref_norm, screen.new_norm, screen.ref_nonzero,
                    screen.new_nonzero, screen.rel_l2, screen.max_abs,
                    screen.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);

        // Independent FP32 RowSplit decode/dot oracle over evenly spaced rows.
        std::vector<std::uint8_t> host_weight(packed.storage.bytes);
        CUDA_CHECK(cudaMemcpy(host_weight.data(), packed.storage.p, packed.storage.bytes,
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kK) * kT);
        CUDA_CHECK(cudaMemcpy(host_x.data(), input.p, host_x.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<std::int32_t> sampled_rows;
        for (std::int32_t i = 0; i < kOracleRows; ++i) {
            sampled_rows.push_back(
                static_cast<std::int32_t>((static_cast<std::int64_t>(i) * kRows) / kOracleRows));
        }
        const std::size_t sample_count = sampled_rows.size();
        std::vector<float> oracle(sample_count * kT);
        std::vector<std::uint16_t> oracle_old(sample_count * kT), oracle_new(sample_count * kT);
        for (std::int32_t token = 0; token < kT; ++token) {
            for (std::size_t s = 0; s < sample_count; ++s) {
                const std::int32_t row = sampled_rows[s];
                double acc             = 0.0;
                for (std::int32_t kk = 0; kk < kK; ++kk) {
                    acc += static_cast<double>(q4_rowseplit_dequant(host_weight.data(),
                                                                    packed.scale_offset, row, kk)) *
                           static_cast<double>(bf16_bits_to_float(
                               host_x[static_cast<std::size_t>(token) * kK + kk]));
                }
                const std::size_t index = static_cast<std::size_t>(token) * sample_count + s;
                oracle[index]           = static_cast<float>(acc);
                oracle_old[index] = host_baseline[static_cast<std::size_t>(token) * kRows + row];
                oracle_new[index] = host_cand[static_cast<std::size_t>(token) * kRows + row];
            }
        }
        const DiffStats old_vs_oracle = compare_float_bf16(oracle, oracle_old);
        const DiffStats new_vs_oracle = compare_float_bf16(oracle, oracle_new);
        std::printf("gdn_small_t_oracle N=%d K=%d T=%d sample_rows=%zu oracle_norm=%.6e "
                    "oracle_nonzero=%zu old_rel_l2=%.3e new_rel_l2=%.3e old_max_abs=%.3e "
                    "new_max_abs=%.3e old_pass=%d new_pass=%d threshold_rel_l2=%.1e "
                    "oracle=fp32.reference screening_only=true qualification=false\n",
                    kRows, kK, kT, sample_count, old_vs_oracle.ref_norm, old_vs_oracle.ref_nonzero,
                    old_vs_oracle.rel_l2, new_vs_oracle.rel_l2, old_vs_oracle.max_abs,
                    new_vs_oracle.max_abs, old_vs_oracle.within(kScreenRelL2Max) ? 1 : 0,
                    new_vs_oracle.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);

        const bool clean = screen.within(kScreenRelL2Max) &&
                           old_vs_oracle.within(kScreenRelL2Max) &&
                           new_vs_oracle.within(kScreenRelL2Max);
        const auto [old_timing, new_timing, eligible] =
            timing_ab(baseline, candidate, flush, stream, options, rng, clean);
        std::printf("gdn_small_t_timing N=%d K=%d T=%d old=production.simt new=small_t.mma "
                    "old_median_us=%.3f old_p95_us=%.3f new_median_us=%.3f new_p95_us=%.3f "
                    "ratio_new_over_old=%.4f eligible=%d perf_recommendation=%s\n",
                    kRows, kK, kT, old_timing.median_us, old_timing.p95_us, new_timing.median_us,
                    new_timing.p95_us, new_timing.median_us / old_timing.median_us,
                    eligible ? 1 : 0,
                    eligible ? "eligible_for_review" : "withheld_screen_or_oracle_failed");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q4_gdn_small_t_t8_ab_bench: %s\n", error.what());
        return 1;
    }
}
