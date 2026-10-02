// Same-process A/B for the old and Q5 K-split Linear / LinearAdd launch routes.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows = 5120;
constexpr std::array<std::int32_t, 2> kInputRows{6144, 17408};
constexpr std::array<std::int32_t, 6> kTokenCounts{2, 4, 5, 6, 7, 8};
constexpr std::uint64_t kDefaultFlushBytes = 256ULL << 20;
constexpr std::uint32_t kOrderSeed = 0x51a7U;

// GDN value+z Q5 input projection: N=12288, K=5120, T=8, Q5_G64_FP16.
constexpr std::int32_t kGdnRows       = 12288;
constexpr std::int32_t kGdnInputRows  = 5120;
constexpr std::int32_t kGdnSplitRow   = 6144;
constexpr std::int32_t kGdnTokens     = 8;
constexpr std::int32_t kGdnFullSlabs  = 5;
constexpr std::int32_t kGdnOracleRows = 512;

// Deterministic non-constant weight content: fixed seed so every run and the CPU oracle see the
// same bytes, but row/group-varying so the L2/DRAM data pattern is not a single repeated constant.
constexpr std::uint32_t kGdnWeightSeed = 0x9e3779b9U;
// Screening gate only; never a production numerical qualification threshold.
constexpr double kScreenRelL2Max = 1.0e-2;

struct Options {
    int device = 0;
    int warmup = 5;
    int repeat = 30;
    std::uint64_t flush_bytes = kDefaultFlushBytes;
    bool gdn_k5120_ab = false;
    bool gdn_k5120_oracle = false;
};

struct Candidate {
    const char* route;
    std::function<void(cudaStream_t)> launch;
};

struct Timing {
    double median_us;
    double p95_us;
};

void restore_residual(const DeviceBuffer& source, DeviceBuffer& destination, std::size_t bytes,
                      cudaStream_t stream);
void run_ab(std::string_view op, std::int32_t output_rows, std::int32_t input_rows,
            std::int32_t tokens, const Candidate& old_route, const Candidate& new_route,
            DeviceBuffer& flush, cudaStream_t stream, const Options& options, std::mt19937& rng,
            bool restore_output, const DeviceBuffer& residual_base, DeviceBuffer& output,
            std::size_t residual_bytes);

void run_gdn_k5120_ab(DeviceBuffer& flush, cudaStream_t stream, const Options& options,
                      std::mt19937& rng);

std::uint64_t parse_u64(std::string_view value, const char* label) {
    const std::string text(value);
    std::size_t parsed = 0;
    unsigned long long result = 0;
    try {
        result = std::stoull(text, &parsed, 10);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    if (parsed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(result);
}

int parse_nonnegative(std::string_view value, const char* label) {
    const auto parsed = parse_u64(value, label);
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string(label) + " is too large");
    }
    return static_cast<int>(parsed);
}

std::uint64_t checked_mib(std::string_view value) {
    const std::uint64_t mib = parse_u64(value, "flush-mib");
    if (mib == 0 || mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
        throw std::invalid_argument("--flush-mib must be positive and fit in bytes");
    }
    return mib << 20;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[i];
        };
        if (arg == "--device") {
            options.device = parse_nonnegative(next("device"), "device");
        } else if (arg == "--warmup") {
            options.warmup = parse_nonnegative(next("warmup"), "warmup");
        } else if (arg == "--repeat") {
            options.repeat = parse_nonnegative(next("repeat"), "repeat");
        } else if (arg == "--flush-mib") {
            options.flush_bytes = checked_mib(next("flush-mib"));
        } else if (arg == "--gdn-k5120-ab") {
            options.gdn_k5120_ab = true;
        } else if (arg == "--gdn-k5120-oracle") {
            options.gdn_k5120_oracle = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--device N] [--warmup N] [--repeat N] [--flush-mib N] "
                        "[--gdn-k5120-ab] [--gdn-k5120-oracle]\n"
                        "Runs same-process randomized old/new A/B for Q5 Linear and LinearAdd; "
                        "N=5120, K=6144|17408, T=2|4|6|8. "
                        "--gdn-k5120-ab compares GDN value+z (N=12288,K=5120,T=8) split4 SIMT "
                        "against Q5 K-split MMA; --gdn-k5120-oracle adds a CPU FP32 oracle for "
                        "the first %d rows.\n",
                        argv[0], kGdnOracleRows);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.repeat <= 0) { throw std::invalid_argument("--repeat must be positive"); }
    return options;
}

Timing summarize(std::vector<double> values) {
    if (values.empty()) { throw std::invalid_argument("cannot summarize empty timings"); }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double fraction) {
        const auto index = static_cast<std::size_t>(
            fraction * static_cast<double>(values.size() - 1));
        return values[std::min(index, values.size() - 1)];
    };
    return {percentile(0.50), percentile(0.95)};
}

void restore_residual(const DeviceBuffer& source, DeviceBuffer& destination, std::size_t bytes,
                      cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.p, source.p, bytes, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void run_ab(std::string_view op, std::int32_t output_rows, std::int32_t input_rows,
            std::int32_t tokens, const Candidate& old_route, const Candidate& new_route,
            DeviceBuffer& flush, cudaStream_t stream, const Options& options, std::mt19937& rng,
            bool restore_output, const DeviceBuffer& residual_base, DeviceBuffer& output,
            std::size_t residual_bytes) {
    const Candidate* first = &old_route;
    const Candidate* second = &new_route;
    std::bernoulli_distribution coin;
    for (int i = 0; i < options.warmup; ++i) {
        if (coin(rng)) { std::swap(first, second); }
        for (const Candidate* candidate : {first, second}) {
            if (restore_output) { restore_residual(residual_base, output, residual_bytes, stream); }
            bench::flush_l2(flush, stream);
            candidate->launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::array<std::vector<double>, 2> samples;
    samples[0].reserve(static_cast<std::size_t>(options.repeat));
    samples[1].reserve(static_cast<std::size_t>(options.repeat));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int trial = 0; trial < options.repeat; ++trial) {
        if (coin(rng)) { std::swap(first, second); }
        for (const Candidate* candidate : {first, second}) {
            const int route_index = candidate == &old_route ? 0 : 1;
            if (restore_output) { restore_residual(residual_base, output, residual_bytes, stream); }
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaEventRecord(start, stream));
            candidate->launch(stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            samples[route_index].push_back(static_cast<double>(milliseconds) * 1000.0);
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const Timing old_timing = summarize(std::move(samples[0]));
    const Timing new_timing = summarize(std::move(samples[1]));
    std::printf("op=%.*s N=%d K=%d T=%d old_route=%s median_us=%.3f p95_us=%.3f "
                "new_route=%s median_us=%.3f p95_us=%.3f ratio_new_over_old=%.4f\n",
                static_cast<int>(op.size()), op.data(), output_rows, input_rows, tokens,
                old_route.route, old_timing.median_us, old_timing.p95_us, new_route.route,
                new_timing.median_us,
                new_timing.p95_us, new_timing.median_us / old_timing.median_us);
}

inline float bf16_bits_to_float(std::uint16_t bits) {
    __nv_bfloat16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return __bfloat162float(value);
}

inline std::uint16_t float_to_bf16_bits(float value) {
    const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
    std::uint16_t bits          = 0;
    std::memcpy(&bits, &rounded, sizeof(bits));
    return bits;
}

struct DiffStats {
    double rel_l2           = std::numeric_limits<double>::quiet_NaN();
    double max_abs          = 0.0;
    double ref_norm         = 0.0;
    double new_norm         = 0.0;
    std::size_t ref_nonzero = 0;
    std::size_t new_nonzero = 0;

    [[nodiscard]] bool nonzero() const noexcept {
        return ref_nonzero > 0 && new_nonzero > 0;
    }
    [[nodiscard]] bool within(double rel_l2_limit) const noexcept {
        return nonzero() && std::isfinite(rel_l2) && rel_l2 <= rel_l2_limit;
    }
};

// Norm-aware comparison. A zero reference norm yields a NaN rel_l2 (reported as a failure), never a
// silent 0.0 that could be mistaken for a pass.
DiffStats compare_bf16(const std::vector<std::uint16_t>& reference,
                       const std::vector<std::uint16_t>& actual) {
    if (reference.size() != actual.size()) {
        throw std::invalid_argument("bf16 comparison length mismatch");
    }
    double diff_sq         = 0.0;
    double ref_sq          = 0.0;
    double new_sq          = 0.0;
    double max_abs         = 0.0;
    std::size_t ref_nz     = 0;
    std::size_t new_nz     = 0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double r = bf16_bits_to_float(reference[i]);
        const double a = bf16_bits_to_float(actual[i]);
        const double d = a - r;
        diff_sq += d * d;
        ref_sq += r * r;
        new_sq += a * a;
        ref_nz += r != 0.0 ? 1U : 0U;
        new_nz += a != 0.0 ? 1U : 0U;
        max_abs = std::max(max_abs, std::abs(d));
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

// Overwrite the Q5 RowSplit planes with deterministic row/group-varying bytes and positive, varying
// FP16 group scales. Layout offsets and sizes come from the producer, so this recreates the exact
// low/high/scale planes both launch routes and the CPU oracle must agree on.
void fill_gdn_row_split_weight(bench::PackedQuantizedWeight& packed, std::int32_t rows,
                               std::int32_t groups_per_row) {
    const std::size_t groups = static_cast<std::size_t>(rows) * groups_per_row;
    if (packed.low_bytes != groups * 32 || packed.high_bytes != groups * 8 ||
        packed.scale_bytes != groups * 2) {
        throw std::logic_error("benchmark Q5 weight planes do not match the assumed RowSplit layout");
    }
    std::mt19937 rng(kGdnWeightSeed);
    std::vector<std::uint8_t> low(groups * 32);
    std::vector<std::uint8_t> high(groups * 8);
    for (std::uint8_t& byte : low) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    for (std::uint8_t& byte : high) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    std::vector<std::uint16_t> scales(groups);
    for (std::size_t i = 0; i < groups; ++i) {
        // [0.02, 0.404], always a positive normal FP16 value after rounding.
        const float scale    = 0.02F + static_cast<float>(rng() % 97U) * 0.004F;
        const __half encoded = __float2half_rn(scale);
        std::memcpy(&scales[i], &encoded, sizeof(scales[i]));
    }
    packed.storage.copy_from_host(low.data(), low.size(), 0);
    packed.storage.copy_from_host(high.data(), high.size(), packed.high_offset);
    packed.storage.copy_from_host(scales.data(), scales.size() * 2, packed.scale_offset);
}

// Independent scalar decoder for the benchmark's RowSplit Q5_G64_FP16 planes. It re-derives the
// packing from the stored low/high/scale planes instead of reusing the production device atom, so
// it can act as a cheap pre-timing consistency screen for the two launch routes.
float q5_rowseplit_dequant(const std::uint8_t* payload, std::uint64_t high_offset,
                           std::uint64_t scale_offset, std::int32_t groups_per_row,
                           std::int32_t row, std::int32_t k) {
    const std::int32_t group       = k / 64;
    const std::int32_t local       = k % 64;
    const std::int32_t lane        = local >> 1;
    const std::int64_t group_index = static_cast<std::int64_t>(row) * groups_per_row + group;
    const std::uint8_t code        = payload[group_index * 32 + lane];
    const std::uint8_t high_byte = payload[high_offset + group_index * 8 + (lane >> 2)];
    const int nibble               = (local & 1) ? (code >> 4) : (code & 0x0f);
    const int shift                = (lane & 3) * 2 + (local & 1);
    const int q = ((nibble | (static_cast<int>((high_byte >> shift) & 1u) << 4)) ^ 0x10) - 0x10;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, payload + scale_offset + group_index * 2, sizeof(scale_bits));
    return static_cast<float>(q) * __half2float(__ushort_as_half(scale_bits));
}

void run_gdn_k5120_ab(DeviceBuffer& flush, cudaStream_t stream, const Options& options,
                      std::mt19937& rng) {
    constexpr std::int32_t kRowsGdn = kGdnRows;
    constexpr std::int32_t kK       = kGdnInputRows;
    constexpr std::int32_t kT       = kGdnTokens;
    constexpr std::int32_t kSplit   = kGdnSplitRow;

    bench::PackedQuantizedWeight packed =
        bench::make_row_split_weight(QType::Q5_G64_FP16, kRowsGdn, kK, kK, {0x31, 0xa5, 0x3c00});
    fill_gdn_row_split_weight(packed, kRowsGdn, kK / 64);
    DeviceBuffer input     = bench::make_bf16(static_cast<std::size_t>(kK) * kT);
    DeviceBuffer value     = bench::make_zeros(static_cast<std::size_t>(kSplit) * kT * 2);
    DeviceBuffer z         = bench::make_zeros(static_cast<std::size_t>(kSplit) * kT * 2);
    // The production route splits value/z exactly like the baseline, so the candidate owns two
    // planes as well instead of one concatenated [T,12288] buffer.
    DeviceBuffer candidate_value = bench::make_zeros(static_cast<std::size_t>(kSplit) * kT * 2);
    DeviceBuffer candidate_z     = bench::make_zeros(static_cast<std::size_t>(kSplit) * kT * 2);
    DeviceBuffer scratch   = bench::make_zeros(2);

    Tensor x(input.p, DType::BF16, {kK, kT});
    Tensor candidate_value_out(candidate_value.p, DType::BF16, {kSplit, kT});
    Tensor candidate_z_out(candidate_z.p, DType::BF16, {kSplit, kT});

    const auto baseline = [&](cudaStream_t launch_stream) {
        const dim3 grid(static_cast<unsigned>(kRowsGdn), 1u, 1u);
        ops::detail::q5_rowsplit_gemm_simt_split4_kernel<ops::detail::Q5RowSplitSimtSchedule, kT,
                                                         kGdnFullSlabs, kK, true, kSplit>
            <<<grid, 128, 0, launch_stream>>>(
                static_cast<const __nv_bfloat16*>(input.p),
                static_cast<const std::uint8_t*>(packed.weight.qdata),
                static_cast<const std::uint8_t*>(packed.weight.qhigh),
                static_cast<const std::uint8_t*>(packed.weight.scales),
                static_cast<__nv_bfloat16*>(value.p), static_cast<__nv_bfloat16*>(z.p), kRowsGdn,
                kSplit, kK, kT, packed.weight.padded_shape[1], kGdnFullSlabs);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto candidate_launch = [&](cudaStream_t launch_stream) {
        // Production GDN value+z entry point: launch_q5_ksplit_mma_split<12288, 6144, 5120, 8>.
        ops::detail::launch_q5_ksplit_gdn_value_z_t8(x, packed.weight, candidate_value_out,
                                                     candidate_z_out, launch_stream);
    };

    baseline(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    candidate_launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::uint16_t> host_value(static_cast<std::size_t>(kSplit) * kT);
    std::vector<std::uint16_t> host_z(static_cast<std::size_t>(kSplit) * kT);
    std::vector<std::uint16_t> host_candidate_value(static_cast<std::size_t>(kSplit) * kT);
    std::vector<std::uint16_t> host_candidate_z(static_cast<std::size_t>(kSplit) * kT);
    std::vector<std::uint16_t> host_candidate(static_cast<std::size_t>(kRowsGdn) * kT);
    CUDA_CHECK(
        cudaMemcpy(host_value.data(), value.p, host_value.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_z.data(), z.p, host_z.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_candidate_value.data(), candidate_value.p,
                          host_candidate_value.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_candidate_z.data(), candidate_z.p, host_candidate_z.size() * 2,
                          cudaMemcpyDeviceToHost));

    // Both routes write value and z as two [T,6144] token-major planes; concatenation along rows is
    // the [T,12288] layout they must agree on. The comparison now uses the production entry point.
    std::vector<std::uint16_t> reference(static_cast<std::size_t>(kRowsGdn) * kT);
    for (std::int32_t token = 0; token < kT; ++token) {
        for (std::int32_t row = 0; row < kRowsGdn; ++row) {
            const std::size_t index = static_cast<std::size_t>(token) * kRowsGdn + row;
            reference[index] =
                row < kSplit ? host_value[static_cast<std::size_t>(token) * kSplit + row]
                             : host_z[static_cast<std::size_t>(token) * kSplit + (row - kSplit)];
            host_candidate[index] =
                row < kSplit
                    ? host_candidate_value[static_cast<std::size_t>(token) * kSplit + row]
                    : host_candidate_z[static_cast<std::size_t>(token) * kSplit + (row - kSplit)];
        }
    }
    const DiffStats screen = compare_bf16(reference, host_candidate);
    std::printf("gdn_k5120_screen N=%d K=%d T=%d old=rowsplit.split4 new=ksplit.mma.c8 "
                "old_norm=%.6e new_norm=%.6e old_nonzero=%zu new_nonzero=%zu rel_l2=%.3e "
                "max_abs=%.3e screen_pass=%d threshold_rel_l2=%.1e screening_only=true "
                "qualification=false\n",
                kRowsGdn, kK, kT, screen.ref_norm, screen.new_norm, screen.ref_nonzero,
                screen.new_nonzero, screen.rel_l2, screen.max_abs,
                screen.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);

    if (options.gdn_k5120_oracle) {
        std::vector<std::uint8_t> host_weight(packed.storage.bytes);
        CUDA_CHECK(cudaMemcpy(host_weight.data(), packed.storage.p, packed.storage.bytes,
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kK) * kT);
        CUDA_CHECK(cudaMemcpy(host_x.data(), input.p, host_x.size() * 2, cudaMemcpyDeviceToHost));

        // Sample half the value rows and half the z rows so the check crosses the SplitRow seam.
        const std::int32_t groups_per_row = kK / 64;
        const std::int32_t oracle_half = std::max<std::int32_t>(
            1, std::min<std::int32_t>(kGdnOracleRows / 2, kSplit));
        std::vector<std::int32_t> sampled_rows;
        sampled_rows.reserve(static_cast<std::size_t>(oracle_half) * 2);
        for (std::int32_t i = 0; i < oracle_half; ++i) { sampled_rows.push_back(i); }
        for (std::int32_t i = 0; i < oracle_half; ++i) { sampled_rows.push_back(kSplit + i); }
        const std::size_t sample_count    = sampled_rows.size();
        const std::size_t oracle_elements = sample_count * kT;
        std::vector<std::uint16_t> oracle(oracle_elements);
        std::vector<std::uint16_t> oracle_old(oracle_elements);
        std::vector<std::uint16_t> oracle_new(oracle_elements);
        for (std::int32_t token = 0; token < kT; ++token) {
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                const std::int32_t row = sampled_rows[sample];
                float acc              = 0.0F;
                for (std::int32_t kk = 0; kk < kK; ++kk) {
                    acc += q5_rowseplit_dequant(host_weight.data(), packed.high_offset,
                                                packed.scale_offset, groups_per_row, row, kk) *
                           bf16_bits_to_float(host_x[static_cast<std::size_t>(token) * kK + kk]);
                }
                const std::size_t index = static_cast<std::size_t>(token) * sample_count + sample;
                oracle[index]           = float_to_bf16_bits(acc);
                oracle_old[index]       = reference[static_cast<std::size_t>(token) * kRowsGdn + row];
                oracle_new[index] =
                    host_candidate[static_cast<std::size_t>(token) * kRowsGdn + row];
            }
        }
        const DiffStats old_vs_oracle = compare_bf16(oracle, oracle_old);
        const DiffStats new_vs_oracle = compare_bf16(oracle, oracle_new);
        std::printf("gdn_k5120_oracle N=%d K=%d T=%d sample_rows=%zu value_rows=%d z_rows=%d "
                    "oracle_norm=%.6e old_norm=%.6e new_norm=%.6e oracle_nonzero=%zu "
                    "old_vs_oracle.rel_l2=%.3e old_vs_oracle.max_abs=%.3e "
                    "new_vs_oracle.rel_l2=%.3e new_vs_oracle.max_abs=%.3e old_pass=%d "
                    "new_pass=%d threshold_rel_l2=%.1e oracle=fp32.independent_decode "
                    "screening_only=true qualification=false\n",
                    kRowsGdn, kK, kT, sample_count, oracle_half, oracle_half,
                    old_vs_oracle.ref_norm, old_vs_oracle.new_norm, new_vs_oracle.new_norm,
                    old_vs_oracle.ref_nonzero, old_vs_oracle.rel_l2, old_vs_oracle.max_abs,
                    new_vs_oracle.rel_l2, new_vs_oracle.max_abs,
                    old_vs_oracle.within(kScreenRelL2Max) ? 1 : 0,
                    new_vs_oracle.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);
    }

    const Candidate old_route{"gdn.value_z.q5.rowsplit.split4", baseline};
    const Candidate new_route{"gdn.value_z.q5.ksplit.mma.c8", candidate_launch};
    run_ab("gdn.value_z", kRowsGdn, kK, kT, old_route, new_route, flush, stream, options, rng,
           false, scratch, candidate_value, 0);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (options.device >= device_count) {
            throw std::invalid_argument("--device ordinal " + std::to_string(options.device) +
                                        " is unavailable; detected " +
                                        std::to_string(device_count) + " CUDA device(s)");
        }
        CUDA_CHECK(cudaSetDevice(options.device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
        std::printf("device=%d actual_gpu=%s sm=%d%d warmup=%d repeat=%d order_seed=%u\n",
                    options.device, properties.name, properties.major, properties.minor,
                    options.warmup, options.repeat, kOrderSeed);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(options.flush_bytes);
        std::mt19937 rng(kOrderSeed);

        if (options.gdn_k5120_ab || options.gdn_k5120_oracle) {
            run_gdn_k5120_ab(flush, stream, options, rng);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }
        for (const std::int32_t input_rows : kInputRows) {
            bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
                QType::Q5_G64_FP16, kRows, input_rows, input_rows, {0x31, 0xa5, 0x3c00});
            DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(input_rows) * 8);
            DeviceBuffer output = bench::make_zeros(static_cast<std::size_t>(kRows) * 8 * 2);
            DeviceBuffer residual_base = bench::make_bf16(static_cast<std::size_t>(kRows) * 8);

            for (const std::int32_t tokens : kTokenCounts) {
                Tensor x(input.p, DType::BF16, {input_rows, tokens});
                Tensor out(output.p, DType::BF16, {kRows, tokens});
                const auto old_linear_launch = [&](cudaStream_t launch_stream) {
                    if (tokens <= 6) {
                        ops::detail::launch_q5_simt_split2_exact(x, packed.weight, out, launch_stream);
                    } else {
                        ops::detail::launch_q5_simt_r8_c8(x, packed.weight, out, launch_stream);
                    }
                };
                const auto new_linear_launch = [&](cudaStream_t launch_stream) {
                    const bool capacity4 = tokens <= 4;
                    const ops::detail::Q5Launch launch = input_rows == 6144
                        ? (capacity4 ? ops::detail::launch_q5_ksplit_n5120_k6144_c4
                                     : ops::detail::launch_q5_ksplit_n5120_k6144_c8)
                        : (capacity4 ? ops::detail::launch_q5_ksplit_n5120_k17408_c4
                                     : ops::detail::launch_q5_ksplit_n5120_k17408_c8);
                    launch(x, packed.weight, out, launch_stream);
                };
                const Candidate old_linear{
                    tokens <= 6 ? "linear.q5.simt.split2.exact" : "linear.q5.simt.r8.c8",
                    old_linear_launch};
                const Candidate new_linear{
                    tokens <= 4 ? "linear.q5.ksplit.c4" : "linear.q5.ksplit.c8",
                    new_linear_launch};
                run_ab("linear", kRows, input_rows, tokens, old_linear, new_linear, flush, stream,
                       options, rng, false, residual_base, output, 0);

                const auto old_linear_add_launch = [&](cudaStream_t launch_stream) {
                    ops::detail::q5_linear_add_split2_exact_launch(x, packed.weight, out,
                                                                   launch_stream);
                };
                const Candidate old_linear_add{"linear_add.q5.simt.split2.exact.residual",
                                               old_linear_add_launch};
                const auto new_linear_add_launch = [&](cudaStream_t launch_stream) {
                    ops::detail::q5_linear_add_ksplit_mma_residual_launch(
                        x, packed.weight, out, launch_stream);
                };
                const Candidate new_linear_add{
                    tokens <= 4 ? "linear_add.q5.ksplit.c4.residual"
                                : "linear_add.q5.ksplit.c8.residual",
                    new_linear_add_launch};
                run_ab("linear_add", kRows, input_rows, tokens, old_linear_add, new_linear_add,
                       flush, stream, options, rng, true, residual_base, output,
                       static_cast<std::size_t>(kRows) * tokens * sizeof(std::uint16_t));
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_q5_ksplit_ab_bench: %s\n", error.what());
        return 1;
    }
}
