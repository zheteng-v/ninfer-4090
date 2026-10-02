// Faithful same-process A/B for the attention Q5 split-output projection at T=8.
//   production baseline: q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, 4, 8, 2, true, 6144>
//                        exactly as attn_input_proj::launch_q5_simt<4> launches it
//                        (grid = (resp. div_up(7168,8)=896, div_up(8,4)=2), 256 threads, out_ld=6144)
//   candidate:           launch_q5_ksplit_mma_split<7168, 6144, 5120, 8>  (split gate[6144] + v[1024])
// Deterministic row/group-varying RowSplit packing; the BF16 screen and independent FP32 packed
// decode oracle are screening evidence only, never a production numerical qualification.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
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
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kRows      = 7168;  // gate rows(6144) + v rows(1024)
constexpr std::int32_t kSplitRows = 6144;  // kSplitRow
constexpr std::int32_t kTailRows  = kRows - kSplitRows;
constexpr std::int32_t kK         = 5120;
constexpr std::int32_t kT         = 8;
constexpr std::int32_t kFullSlabs = kK / 1024;      // 5
constexpr std::int32_t kGroupsPerRow = kK / 64;     // 80
constexpr std::int32_t kSimtThreads  = 8 * 32;      // q5_rowsplit_gemm_simt kThreads
constexpr std::uint64_t kFlushBytes  = 256ULL << 20;
constexpr std::uint32_t kOrderSeed   = 0x51a7c3u;
constexpr std::uint32_t kWeightSeed  = 0x9e3779b9u;
constexpr double kScreenRelL2Max     = 1.0e-2;
constexpr std::int32_t kOracleRows   = 256;

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
                        "Faithful attention Q5 split-output T=8 A/B: SIMT split (production) vs\n"
                        "launch_q5_ksplit_mma_split<7168,6144,5120,8>.\n",
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

// Oracle contract (op-development): the reference side stays FP32 and is never rounded to the
// production output dtype; only the actual side is the stored BF16 tensor.
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

void fill_weight(bench::PackedQuantizedWeight& packed, std::int32_t rows,
                 std::int32_t groups_per_row) {
    const std::size_t groups = static_cast<std::size_t>(rows) * groups_per_row;
    if (packed.low_bytes != groups * 32 || packed.scale_bytes != groups * 2 ||
        packed.high_bytes != groups * 8) {
        throw std::logic_error("benchmark Q5 weight planes differ from the assumed RowSplit layout");
    }
    std::mt19937 rng(kWeightSeed);
    std::vector<std::uint8_t> low(groups * 32);
    std::vector<std::uint8_t> high(groups * 8);
    for (std::uint8_t& byte : low) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    for (std::uint8_t& byte : high) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    std::vector<std::uint16_t> scales(groups);
    for (std::size_t i = 0; i < groups; ++i) {
        const float scale    = 0.02F + static_cast<float>(rng() % 97U) * 0.004F;
        const __half encoded = __float2half_rn(scale);
        std::memcpy(&scales[i], &encoded, sizeof(scales[i]));
    }
    packed.storage.copy_from_host(low.data(), low.size(), 0);
    packed.storage.copy_from_host(high.data(), high.size(), packed.high_offset);
    packed.storage.copy_from_host(scales.data(), scales.size() * 2, packed.scale_offset);
}

// Q5 RowSplit G64: low nibble + high-bit plane -> 5-bit signed, times FP16 group scale.
float q5_rowseplit_dequant(const std::uint8_t* payload, std::uint64_t high_offset,
                           std::uint64_t scale_offset, std::int32_t groups_per_row,
                           std::int32_t row, std::int32_t k) {
    const std::int32_t group = k / 64;
    const std::int32_t local = k % 64;
    const std::int32_t lane  = local >> 1;
    const std::int64_t index = static_cast<std::int64_t>(row) * groups_per_row + group;
    const std::uint8_t code  = payload[index * 32 + lane];
    const std::uint8_t high  = payload[high_offset + index * 8 + (lane >> 2)];
    const int nibble         = (local & 1) ? (code >> 4) : (code & 0x0f);
    const int shift          = (lane & 3) * 2 + (local & 1);
    const int q = ((nibble | (static_cast<int>((high >> shift) & 1u) << 4)) ^ 0x10) - 0x10;
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, payload + scale_offset + index * 2, sizeof(scale_bits));
    return static_cast<float>(q) * __half2float(__ushort_as_half(scale_bits));
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
            bench::make_row_split_weight(QType::Q5_G64_FP16, kRows, kK, kK, {0x31, 0xa5, 0x3c00});
        fill_weight(packed, kRows, kGroupsPerRow);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kK) * kT);
        DeviceBuffer gate  = bench::make_zeros(static_cast<std::size_t>(kSplitRows) * kT * 2);
        DeviceBuffer v     = bench::make_zeros(static_cast<std::size_t>(kTailRows) * kT * 2);
        DeviceBuffer gate2 = bench::make_zeros(static_cast<std::size_t>(kSplitRows) * kT * 2);
        DeviceBuffer v2    = bench::make_zeros(static_cast<std::size_t>(kTailRows) * kT * 2);

        Tensor x(input.p, DType::BF16, {kK, kT});
        Tensor gate_t(gate.p, DType::BF16, {kSplitRows, kT});
        Tensor v_t(v.p, DType::BF16, {kTailRows, kT});
        Tensor gate2_t(gate2.p, DType::BF16, {kSplitRows, kT});
        Tensor v2_t(v2.p, DType::BF16, {kTailRows, kT});

        // Exact production baseline launch (attn_input_proj::launch_q5_simt<4>).
        const auto simt_launch = [&](cudaStream_t s) {
            const dim3 grid(static_cast<unsigned>(kRows / 8),
                            static_cast<unsigned>((kT + 4 - 1) / 4), 1u);
            ops::detail::q5_rowsplit_gemm_simt_kernel<ops::detail::Q5RowSplitSimtSchedule, 4, 8, 2,
                                                      true, kSplitRows>
                <<<grid, kSimtThreads, 0, s>>>(
                    static_cast<const __nv_bfloat16*>(input.p),
                    static_cast<const std::uint8_t*>(packed.weight.qdata),
                    static_cast<const std::uint8_t*>(packed.weight.qhigh),
                    static_cast<const std::uint8_t*>(packed.weight.scales),
                    static_cast<__nv_bfloat16*>(gate.p), static_cast<__nv_bfloat16*>(v.p), kRows,
                    kSplitRows, kK, kT, packed.weight.padded_shape[1], kFullSlabs);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto ksplit_launch = [&](cudaStream_t s) {
            ops::detail::launch_q5_ksplit_mma_split<kRows, kSplitRows, kK, kT>(
                x, packed.weight, gate2_t, v2_t, s);
        };

        simt_launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        ksplit_launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Assemble [T, 7168] = gate || v for both routes and compare.
        const auto assemble = [&](const DeviceBuffer& g, const DeviceBuffer& t) {
            std::vector<std::uint16_t> hg(static_cast<std::size_t>(kSplitRows) * kT);
            std::vector<std::uint16_t> ht(static_cast<std::size_t>(kTailRows) * kT);
            CUDA_CHECK(cudaMemcpy(hg.data(), g.p, hg.size() * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ht.data(), t.p, ht.size() * 2, cudaMemcpyDeviceToHost));
            std::vector<std::uint16_t> full(static_cast<std::size_t>(kRows) * kT);
            for (std::int32_t token = 0; token < kT; ++token) {
                for (std::int32_t row = 0; row < kRows; ++row) {
                    full[static_cast<std::size_t>(token) * kRows + row] =
                        row < kSplitRows
                            ? hg[static_cast<std::size_t>(token) * kSplitRows + row]
                            : ht[static_cast<std::size_t>(token) * kTailRows + (row - kSplitRows)];
                }
            }
            return full;
        };
        const std::vector<std::uint16_t> simt_full  = assemble(gate, v);
        const std::vector<std::uint16_t> split_full = assemble(gate2, v2);
        const DiffStats screen = compare_bf16(simt_full, split_full);
        std::printf("attn_q5_split_screen N=%d split=%d K=%d T=%d old=rowsplit.split.simt "
                    "new=ksplit.mma.split old_norm=%.6e new_norm=%.6e old_nonzero=%zu "
                    "new_nonzero=%zu rel_l2=%.3e max_abs=%.3e screen_pass=%d "
                    "threshold_rel_l2=%.1e screening_only=true qualification=false\n",
                    kRows, kSplitRows, kK, kT, screen.ref_norm, screen.new_norm,
                    screen.ref_nonzero, screen.new_nonzero, screen.rel_l2, screen.max_abs,
                    screen.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);

        // Independent FP32 oracle: half the sample rows from gate, half from v (crosses the seam).
        std::vector<std::uint8_t> host_weight(packed.storage.bytes);
        CUDA_CHECK(cudaMemcpy(host_weight.data(), packed.storage.p, packed.storage.bytes,
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kK) * kT);
        CUDA_CHECK(cudaMemcpy(host_x.data(), input.p, host_x.size() * 2, cudaMemcpyDeviceToHost));
        const std::int32_t half = kOracleRows / 2;
        std::vector<std::int32_t> sampled_rows;
        sampled_rows.reserve(static_cast<std::size_t>(half) * 2);
        for (std::int32_t i = 0; i < half; ++i) {
            sampled_rows.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(i) *
                                                              kSplitRows) / half));
        }
        for (std::int32_t i = 0; i < half; ++i) {
            sampled_rows.push_back(kSplitRows + static_cast<std::int32_t>(
                                                   (static_cast<std::int64_t>(i) * kTailRows) / half));
        }
        const std::size_t sample_count    = sampled_rows.size();
        const std::size_t oracle_elements = sample_count * kT;
        std::vector<float> oracle(oracle_elements);
        std::vector<std::uint16_t> oracle_old(oracle_elements), oracle_new(oracle_elements);
        for (std::int32_t token = 0; token < kT; ++token) {
            for (std::size_t sample = 0; sample < sample_count; ++sample) {
                const std::int32_t row = sampled_rows[sample];
                float acc              = 0.0F;
                for (std::int32_t kk = 0; kk < kK; ++kk) {
                    acc += q5_rowseplit_dequant(host_weight.data(), packed.high_offset,
                                                packed.scale_offset, kGroupsPerRow, row, kk) *
                           bf16_bits_to_float(host_x[static_cast<std::size_t>(token) * kK + kk]);
                }
                const std::size_t index = static_cast<std::size_t>(token) * sample_count + sample;
                oracle[index]           = acc;  // keep the FP32 reference value
                oracle_old[index]       = simt_full[static_cast<std::size_t>(token) * kRows + row];
                oracle_new[index]       = split_full[static_cast<std::size_t>(token) * kRows + row];
            }
        }
        const DiffStats old_vs_oracle = compare_float_bf16(oracle, oracle_old);
        const DiffStats new_vs_oracle = compare_float_bf16(oracle, oracle_new);
        std::printf("attn_q5_split_oracle N=%d split=%d K=%d T=%d sample_rows=%zu oracle_norm=%.6e "
                    "oracle_nonzero=%zu old_rel_l2=%.3e new_rel_l2=%.3e old_max_abs=%.3e "
                    "new_max_abs=%.3e old_pass=%d new_pass=%d threshold_rel_l2=%.1e "
                    "oracle=fp32.reference screening_only=true qualification=false\n",
                    kRows, kSplitRows, kK, kT, sample_count, old_vs_oracle.ref_norm,
                    old_vs_oracle.ref_nonzero, old_vs_oracle.rel_l2, new_vs_oracle.rel_l2,
                    old_vs_oracle.max_abs, new_vs_oracle.max_abs,
                    old_vs_oracle.within(kScreenRelL2Max) ? 1 : 0,
                    new_vs_oracle.within(kScreenRelL2Max) ? 1 : 0, kScreenRelL2Max);

        // Randomized-order timing with L2 flush between every launch.
        for (int i = 0; i < options.warmup; ++i) {
            bench::flush_l2(flush, stream);
            simt_launch(stream);
            bench::flush_l2(flush, stream);
            ksplit_launch(stream);
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
                    simt_launch(stream);
                } else {
                    ksplit_launch(stream);
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
        const Timing old_timing = summarize(std::move(samples_old));
        const Timing new_timing = summarize(std::move(samples_new));
        std::printf("attn_q5_split_timing N=%d split=%d K=%d T=%d old_median_us=%.3f "
                    "old_p95_us=%.3f new_median_us=%.3f new_p95_us=%.3f "
                    "ratio_new_over_old=%.4f\n",
                    kRows, kSplitRows, kK, kT, old_timing.median_us, old_timing.p95_us,
                    new_timing.median_us, new_timing.p95_us,
                    new_timing.median_us / old_timing.median_us);

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_attn_q5_split_t8_ab_bench: %s\n", error.what());
        return 1;
    }
}
