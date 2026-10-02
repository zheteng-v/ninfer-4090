// Public-Op benchmark for every registered GDN input-projection contract.
// Production dispatch is owned exclusively by gdn_input_proj().

#include "core/weight.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ninfer_bench_common.h"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::size_t kFlushBytes = std::size_t{256} << 20;
constexpr double kRtx5090DramGBs  = 1792.0;
constexpr double kGdnInputProjA16RelativeL2 = 3.0e-3;
constexpr double kGdnInputProjA16GrossAbs = 4.0e-3;
constexpr double kGdnInputProjA16GrossRel = 3.5e-3;

enum class Format : std::uint8_t { Q4Q5, Q8, Nvfp4, Fp8, All };
enum class CacheMode : std::uint8_t { Cold, Warm, Both };
enum class CacheState : std::uint8_t { Cold, Warm };

struct Options {
    Format format                  = Format::All;
    ops::LinearPolicy nvfp4_policy = ops::LinearPolicy::AllowA4;
    ops::LinearPolicy fp8_policy   = ops::LinearPolicy::AllowA8;
    CacheMode cache                = CacheMode::Cold;
    std::vector<std::int32_t> tokens{1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    bool route_ab = false;
    bool q5_split4_ab = false;
    bool pdl_t8_ab = false;
    std::string csv_out;
};

struct Result {
    const char* format;
    const char* policy;
    std::int32_t tokens;
    CacheState cache;
    std::size_t workspace_bytes;
    std::uint64_t logical_bytes;
    double useful_flops;
    bench::ColdTiming timing;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\n"
                 "usage: ninfer_gdn_input_proj_bench "
                 "[--format q4q5|q8|nvfp4|fp8|all] [--nvfp4-policy a16|a4] "
                 "[--fp8-policy a16|a8] "
                 "[--tokens T,...] [--cache cold|warm|both] [--warmup N] [--repeat N] "
                 "[--profile] [--route-ab] [--q5-split4-ab] [--pdl-t8-ab] [--csv-out PATH]\n",
                 message);
    std::exit(2);
}

std::int32_t parse_i32(std::string_view text, std::int32_t minimum, std::int32_t maximum,
                       const char* flag) {
    const std::string value(text);
    errno             = 0;
    char* end         = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<std::int32_t>(parsed);
}

std::vector<std::int32_t> parse_list(const char* text, const char* flag) {
    std::vector<std::int32_t> result;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const std::size_t comma     = remaining.find(',');
        const std::string_view item = remaining.substr(0, comma);
        if (item.empty()) { usage(flag); }
        result.push_back(parse_i32(item, 1, std::numeric_limits<std::int32_t>::max(), flag));
        if (comma == std::string_view::npos) { break; }
        remaining.remove_prefix(comma + 1);
    }
    if (result.empty()) { usage(flag); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) -> const char* {
            if (++index == argc) { usage(flag); }
            return argv[index];
        };
        if (argument == "--format") {
            const std::string_view value(next("--format requires a value"));
            if (value == "q4q5")
                options.format = Format::Q4Q5;
            else if (value == "q8")
                options.format = Format::Q8;
            else if (value == "nvfp4")
                options.format = Format::Nvfp4;
            else if (value == "fp8")
                options.format = Format::Fp8;
            else if (value == "all")
                options.format = Format::All;
            else
                usage("--format expects q4q5, q8, nvfp4, fp8, or all");
        } else if (argument == "--nvfp4-policy") {
            const std::string_view value(next("--nvfp4-policy requires a value"));
            if (value == "a16")
                options.nvfp4_policy = ops::LinearPolicy::A16Only;
            else if (value == "a4")
                options.nvfp4_policy = ops::LinearPolicy::AllowA4;
            else
                usage("--nvfp4-policy expects a16 or a4");
        } else if (argument == "--fp8-policy") {
            const std::string_view value(next("--fp8-policy requires a value"));
            if (value == "a16")
                options.fp8_policy = ops::LinearPolicy::A16Only;
            else if (value == "a8")
                options.fp8_policy = ops::LinearPolicy::AllowA8;
            else
                usage("--fp8-policy expects a16 or a8");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"), "--tokens");
        } else if (argument == "--cache") {
            const std::string_view value(next("--cache requires a value"));
            if (value == "cold")
                options.cache = CacheMode::Cold;
            else if (value == "warm")
                options.cache = CacheMode::Warm;
            else if (value == "both")
                options.cache = CacheMode::Both;
            else
                usage("--cache expects cold, warm, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_i32(next("--warmup requires a value"), 0, 10000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_i32(next("--repeat requires a value"), 1, 10000, "--repeat");
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--route-ab") {
            options.route_ab = true;
        } else if (argument == "--q5-split4-ab") {
            options.q5_split4_ab = true;
        } else if (argument == "--pdl-t8-ab") {
            options.pdl_t8_ab = true;
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out requires a path");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    if (options.profile && (options.format == Format::All || options.tokens.size() != 1 ||
                            options.cache == CacheMode::Both)) {
        usage("--profile requires one format, one T, and one cache state");
    }
    if ((options.route_ab || options.q5_split4_ab || options.pdl_t8_ab) &&
        (options.format != Format::Q4Q5 || options.tokens != std::vector<std::int32_t>{8} ||
         options.cache != CacheMode::Cold || options.profile || !options.csv_out.empty())) {
        usage("route A/B requires --format q4q5 --tokens 8 and cold-cache timing only");
    }
    if (static_cast<int>(options.route_ab) + static_cast<int>(options.q5_split4_ab) +
            static_cast<int>(options.pdl_t8_ab) >
        1) {
        usage("route A/B modes are mutually exclusive");
    }
    return options;
}

const char* cache_name(CacheState cache) { return cache == CacheState::Cold ? "cold" : "warm"; }

const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return "a16";
    case ops::LinearPolicy::AllowA8:
        return "a8";
    case ops::LinearPolicy::AllowA4:
        return "a4";
    }
    throw std::invalid_argument("unknown linear policy");
}

template <class Launch>
bench::ColdTiming measure_public(Launch&& launch, CacheState cache, DeviceBuffer& flush,
                                 cudaStream_t stream, int warmup, int repeat) {
    return cache == CacheState::Cold
               ? bench::measure_cold_launch(std::forward<Launch>(launch), flush, stream, warmup,
                                            repeat)
               : bench::measure_launch(std::forward<Launch>(launch), stream, warmup, repeat);
}

template <class Launch>
void profile_public(Launch&& launch, const char* format, const char* policy, CacheState cache,
                    DeviceBuffer& flush, cudaStream_t stream, int warmup) {
    for (int index = 0; index < warmup; ++index) { launch(stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (cache == CacheState::Cold) {
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::printf("PROFILE entry=gdn_input_proj format=%s policy=%s dispatch=public cache=%s\n",
                format, policy, cache_name(cache));
    std::fflush(stdout);
    CUDA_CHECK(cudaProfilerStart());
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaProfilerStop());
}

std::uint64_t tensor_bytes(std::int32_t rows, std::int32_t tokens) {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(tokens) * 2ULL;
}

void report(const Result& result) {
    const double seconds = result.timing.median_us * 1.0e-6;
    const double gbps    = static_cast<double>(result.logical_bytes) / seconds / 1.0e9;
    const double tflops  = result.useful_flops / seconds / 1.0e12;
    std::printf("entry=gdn_input_proj format=%-6s policy=%-3s cache=%-4s T=%4d "
                "workspace=%9zu median=%9.3f us min=%9.3f us p95=%9.3f us "
                "logical=%8.1f GB/s (%5.1f%% of %.0f) math=%8.2f TFLOP/s\n",
                result.format, result.policy, cache_name(result.cache), result.tokens,
                result.workspace_bytes, result.timing.median_us, result.timing.min_us,
                result.timing.p95_us, gbps, gbps / kRtx5090DramGBs * 100.0, kRtx5090DramGBs,
                tflops);
}

void append_result(std::vector<Result>& results, const char* format, const char* policy,
                   std::int32_t tokens, CacheState cache, std::size_t workspace_bytes,
                   std::uint64_t logical_bytes, double useful_flops, bench::ColdTiming timing) {
    Result result{format,          policy,        tokens,       cache,
                  workspace_bytes, logical_bytes, useful_flops, timing};
    report(result);
    results.push_back(result);
}

template <class WorkspaceCapacity, class Launch>
void measure_points(const Options& options, const char* format, const char* policy,
                    std::int32_t hidden, std::int32_t output_rows, std::uint64_t weight_bytes,
                    WorkspaceCapacity&& workspace_capacity, Launch&& make_launch,
                    DeviceBuffer& flush, cudaStream_t stream, std::vector<Result>& results) {
    const CacheState profile_cache =
        options.cache == CacheMode::Cold ? CacheState::Cold : CacheState::Warm;
    for (const std::int32_t tokens : options.tokens) {
        auto launch                       = make_launch(tokens);
        const std::size_t workspace_bytes = workspace_capacity(tokens);
        if (options.profile) {
            profile_public(launch, format, policy, profile_cache, flush, stream, options.warmup);
            continue;
        }
        const std::uint64_t logical =
            weight_bytes + tensor_bytes(hidden, tokens) + tensor_bytes(output_rows, tokens);
        const double flops = 2.0 * static_cast<double>(output_rows) * hidden * tokens;
        for (const CacheState cache : {CacheState::Cold, CacheState::Warm}) {
            if ((options.cache == CacheMode::Cold && cache != CacheState::Cold) ||
                (options.cache == CacheMode::Warm && cache != CacheState::Warm)) {
                continue;
            }
            append_result(
                results, format, policy, tokens, cache, workspace_bytes, logical, flops,
                measure_public(launch, cache, flush, stream, options.warmup, options.repeat));
        }
    }
}

void run_q4q5(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
              std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkRows + kValueRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight qk = bench::make_row_split_weight(
        QType::Q4_G64_FP16, kQkRows, kHidden, kHidden, {0x31, 0x00, 0x3c00});
    bench::PackedQuantizedWeight value_z = bench::make_row_split_weight(
        QType::Q5_G64_FP16, kValueRows + kZRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkRows + kValueRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkRows + kValueRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, qk.weight, value_z.weight, tqkv, tz, launch_stream);
        };
    };
    measure_points(
        options, "q4q5", "a16", kHidden, kOutputRows,
        qk.model_weight_bytes() + value_z.model_weight_bytes(),
        [](std::int32_t) { return std::size_t{0}; }, make_launch, flush, stream, results);
}

void run_q4q5_route_ab(const Options& options, DeviceBuffer& flush, cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows = 6144;
    constexpr std::int32_t kQkvRows = kQkRows + kValueRows;
    constexpr std::int32_t kT = 8;

    auto qk = bench::make_row_split_weight(QType::Q4_G64_FP16, kQkRows, kHidden, kHidden,
                                           {0x31, 0x00, 0x3c00});
    auto value_z = bench::make_row_split_weight(QType::Q5_G64_FP16, kValueRows + kZRows,
                                                kHidden, kHidden, {0x31, 0xa5, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kT);
    DeviceBuffer independent_qkv(static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer independent_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer grouped_qkv(static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer grouped_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    Tensor x(input.p, DType::BF16, {kHidden, kT});
    Tensor independent_qkv_tensor(independent_qkv.p, DType::BF16, {kQkvRows, kT});
    Tensor independent_z_tensor(independent_z.p, DType::BF16, {kZRows, kT});
    Tensor grouped_qkv_tensor(grouped_qkv.p, DType::BF16, {kQkvRows, kT});
    Tensor grouped_z_tensor(grouped_z.p, DType::BF16, {kZRows, kT});

    const auto independent = [&](cudaStream_t launch_stream) {
        Tensor qk_out = independent_qkv_tensor.slice(0, 0, kQkRows);
        Tensor value_out = independent_qkv_tensor.slice(0, kQkRows, kValueRows);
        ops::detail::q4_q5_gdn_input_independent_launch(
            x, qk.weight, value_z.weight, qk_out, value_out, independent_z_tensor, launch_stream);
    };
    const auto grouped = [&](cudaStream_t launch_stream) {
        ops::detail::q4_q5_gdn_input_grouped_mma_launch(
            x, qk.weight, value_z.weight, grouped_qkv_tensor, grouped_z_tensor, launch_stream);
    };

    // Compare every BF16 output against the production independent path before timing. The
    // threshold reuses the public Q4/Q5 GDN input Op's A16 reduction tolerance.
    independent(stream);
    grouped(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> independent_bits(
        static_cast<std::size_t>(kQkvRows + kZRows) * kT);
    std::vector<std::uint16_t> grouped_bits(independent_bits.size());
    independent_qkv.copy_to_host(independent_bits.data(),
                                 static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    independent_z.copy_to_host(independent_bits.data() + static_cast<std::size_t>(kQkvRows) * kT,
                               static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    grouped_qkv.copy_to_host(grouped_bits.data(),
                             static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    grouped_z.copy_to_host(grouped_bits.data() + static_cast<std::size_t>(kQkvRows) * kT,
                           static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));

    long double error_squared = 0.0L;
    long double reference_squared = 0.0L;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    const auto to_float = [](std::uint16_t bits) {
        const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
        float value = 0.0F;
        std::memcpy(&value, &word, sizeof(value));
        return static_cast<double>(value);
    };
    for (std::size_t i = 0; i < independent_bits.size(); ++i) {
        const double reference = to_float(independent_bits[i]);
        const double actual = to_float(grouped_bits[i]);
        const double error = std::abs(actual - reference);
        if (!std::isfinite(reference) || !std::isfinite(actual)) {
            throw std::runtime_error("route A/B produced non-finite output");
        }
        error_squared += static_cast<long double>(actual - reference) * (actual - reference);
        reference_squared += static_cast<long double>(reference) * reference;
        maximum_error = std::max(maximum_error, error);
        maximum_reference = std::max(maximum_reference, std::abs(reference));
    }
    const double relative_l2 = std::sqrt(static_cast<double>(error_squared / reference_squared));
    const double gross_limit = kGdnInputProjA16GrossAbs +
                               kGdnInputProjA16GrossRel * maximum_reference;
    const bool equivalent = relative_l2 <= kGdnInputProjA16RelativeL2 &&
                            maximum_error <= gross_limit;
    std::printf("route_ab_screen T=8 outputs=%zu rel_l2=%.6g max_abs=%.6g limit=%.6g %s\n",
                independent_bits.size(), relative_l2, maximum_error, gross_limit,
                equivalent ? "PASS" : "FAIL");
    if (!equivalent) { throw std::runtime_error("grouped MMA failed production-baseline screen"); }

    const bench::ColdTiming independent_timing = bench::measure_cold_launch(
        independent, flush, stream, options.warmup, options.repeat);
    const bench::ColdTiming grouped_timing =
        bench::measure_cold_launch(grouped, flush, stream, options.warmup, options.repeat);
    std::printf("route_ab entry=gdn_input_proj.q4q5 T=8 cache=cold warmup=%d repeat=%d\n",
                options.warmup, options.repeat);
    std::printf("  independent_direct median=%9.3f us min=%9.3f us p95=%9.3f us\n",
                independent_timing.median_us, independent_timing.min_us, independent_timing.p95_us);
    std::printf("  grouped_mixed_mma   median=%9.3f us min=%9.3f us p95=%9.3f us ratio=%6.3fx\n",
                grouped_timing.median_us, grouped_timing.min_us, grouped_timing.p95_us,
                grouped_timing.median_us / independent_timing.median_us);
}

void run_q5_split4_route_ab(const Options& options, DeviceBuffer& flush, cudaStream_t stream) {
    using namespace ops::detail;
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows = 6144;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    constexpr std::int32_t kT = 8;
    constexpr int kR8C8Threads = 8 * 32;
    constexpr int kSplit4Threads = 4 * 32;

    auto value_z = bench::make_row_split_weight(QType::Q5_G64_FP16, kParentRows, kHidden,
                                                kHidden, {0x31, 0xa5, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kT);
    DeviceBuffer current_value(static_cast<std::size_t>(kValueRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer current_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer split4_value(static_cast<std::size_t>(kValueRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer split4_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    Tensor x(input.p, DType::BF16, {kHidden, kT});
    Tensor current_value_tensor(current_value.p, DType::BF16, {kValueRows, kT});
    Tensor current_z_tensor(current_z.p, DType::BF16, {kZRows, kT});
    Tensor split4_value_tensor(split4_value.p, DType::BF16, {kValueRows, kT});
    Tensor split4_z_tensor(split4_z.p, DType::BF16, {kZRows, kT});

    const auto launch_current = [&](cudaStream_t launch_stream) {
        const dim3 grid(static_cast<unsigned>(ops::div_up(kParentRows, 8)), 1u, 1u);
        q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, 8, 8, 2, true, kValueRows>
            <<<grid, kR8C8Threads, 0, launch_stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z.weight.qdata),
                static_cast<const std::uint8_t*>(value_z.weight.qhigh),
                static_cast<const std::uint8_t*>(value_z.weight.scales),
                static_cast<__nv_bfloat16*>(current_value_tensor.data),
                static_cast<__nv_bfloat16*>(current_z_tensor.data), kParentRows,
                static_cast<std::int32_t>(current_value_tensor.nb[1] / sizeof(__nv_bfloat16)),
                kHidden, kT, value_z.weight.padded_shape[1], 5);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto launch_split4 = [&](cudaStream_t launch_stream) {
        const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 8, 5, kHidden, true,
                                             kValueRows>
            <<<grid, kSplit4Threads, 0, launch_stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z.weight.qdata),
                static_cast<const std::uint8_t*>(value_z.weight.qhigh),
                static_cast<const std::uint8_t*>(value_z.weight.scales),
                static_cast<__nv_bfloat16*>(split4_value_tensor.data),
                static_cast<__nv_bfloat16*>(split4_z_tensor.data), kParentRows,
                static_cast<std::int32_t>(split4_value_tensor.nb[1] / sizeof(__nv_bfloat16)),
                kHidden, kT, value_z.weight.padded_shape[1], 5);
        CUDA_CHECK(cudaGetLastError());
    };

    launch_current(stream);
    launch_split4(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    constexpr std::size_t kOutputCount = static_cast<std::size_t>(kParentRows) * kT;
    std::vector<std::uint16_t> current_bits(kOutputCount);
    std::vector<std::uint16_t> split4_bits(kOutputCount);
    current_value.copy_to_host(current_bits.data(),
                               static_cast<std::size_t>(kValueRows) * kT * sizeof(std::uint16_t));
    current_z.copy_to_host(current_bits.data() + static_cast<std::size_t>(kValueRows) * kT,
                           static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    split4_value.copy_to_host(split4_bits.data(),
                              static_cast<std::size_t>(kValueRows) * kT * sizeof(std::uint16_t));
    split4_z.copy_to_host(split4_bits.data() + static_cast<std::size_t>(kValueRows) * kT,
                          static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));

    const auto to_float = [](std::uint16_t bits) {
        const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
        float value = 0.0F;
        std::memcpy(&value, &word, sizeof(value));
        return static_cast<double>(value);
    };
    long double error_squared = 0.0L;
    long double reference_squared = 0.0L;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t i = 0; i < kOutputCount; ++i) {
        const double reference = to_float(current_bits[i]);
        const double actual = to_float(split4_bits[i]);
        if (!std::isfinite(reference) || !std::isfinite(actual)) {
            throw std::runtime_error("Q5 split4 route produced non-finite output");
        }
        const double error = std::abs(actual - reference);
        error_squared += static_cast<long double>(actual - reference) * (actual - reference);
        reference_squared += static_cast<long double>(reference) * reference;
        maximum_error = std::max(maximum_error, error);
        maximum_reference = std::max(maximum_reference, std::abs(reference));
    }
    if (!(reference_squared > 0.0L)) {
        throw std::runtime_error("Q5 split4 route comparison has zero reference norm");
    }
    const double relative_l2 = std::sqrt(static_cast<double>(error_squared / reference_squared));
    const double gross_limit = kGdnInputProjA16GrossAbs +
                               kGdnInputProjA16GrossRel * maximum_reference;
    const bool equivalent = relative_l2 <= kGdnInputProjA16RelativeL2 &&
                            maximum_error <= gross_limit;
    std::printf("q5_split4_screen shape=[%d,%d] T=8 outputs=%zu rel_l2=%.6g max_abs=%.6g "
                "limit=%.6g %s\n",
                kParentRows, kHidden, kOutputCount, relative_l2, maximum_error, gross_limit,
                equivalent ? "PASS" : "FAIL");
    if (!equivalent) {
        throw std::runtime_error("Q5 split4 failed production-r8c8 baseline screen");
    }

    const bench::ColdTiming current_timing = bench::measure_cold_launch(
        launch_current, flush, stream, options.warmup, options.repeat);
    const bench::ColdTiming split4_timing = bench::measure_cold_launch(
        launch_split4, flush, stream, options.warmup, options.repeat);
    std::printf("q5_value_z route_ab shape=[N=%d,K=%d] T=8 cache=cold warmup=%d repeat=%d\n",
                kParentRows, kHidden, options.warmup, options.repeat);
    std::printf("  rowsplit_simt.r8.c8 median=%9.3f us min=%9.3f us p95=%9.3f us\n",
                current_timing.median_us, current_timing.min_us, current_timing.p95_us);
    std::printf("  rowsplit_simt.split4  median=%9.3f us min=%9.3f us p95=%9.3f us ratio=%6.3fx\n",
                split4_timing.median_us, split4_timing.min_us, split4_timing.p95_us,
                split4_timing.median_us / current_timing.median_us);
}

void run_pdl_t8_route_ab(const Options& options, DeviceBuffer& flush, cudaStream_t stream) {
    using namespace ops::detail;
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows = 6144;
    constexpr std::int32_t kValueZRows = kValueRows + kZRows;
    constexpr std::int32_t kQkvRows = kQkRows + kValueRows;
    constexpr std::int32_t kT = 8;
    constexpr int kQ5Threads = 4 * 32;
    using Q4Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, ops::Cache::ca, 1>;

    auto qk = bench::make_row_split_weight(QType::Q4_G64_FP16, kQkRows, kHidden, kHidden,
                                           {0x31, 0x00, 0x3c00});
    auto value_z = bench::make_row_split_weight(QType::Q5_G64_FP16, kValueZRows, kHidden, kHidden,
                                                {0x31, 0xa5, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kT);
    DeviceBuffer sequential_qkv(static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer sequential_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer pdl_qkv(static_cast<std::size_t>(kQkvRows) * kT * sizeof(std::uint16_t));
    DeviceBuffer pdl_z(static_cast<std::size_t>(kZRows) * kT * sizeof(std::uint16_t));
    Tensor x(input.p, DType::BF16, {kHidden, kT});
    Tensor sequential_qkv_tensor(sequential_qkv.p, DType::BF16, {kQkvRows, kT});
    Tensor sequential_z_tensor(sequential_z.p, DType::BF16, {kZRows, kT});
    Tensor pdl_qkv_tensor(pdl_qkv.p, DType::BF16, {kQkvRows, kT});
    Tensor pdl_z_tensor(pdl_z.p, DType::BF16, {kZRows, kT});

    const auto sequential = [&](cudaStream_t launch_stream) {
        Tensor qk_out = sequential_qkv_tensor.slice(0, 0, kQkRows);
        Tensor value_out = sequential_qkv_tensor.slice(0, kQkRows, kValueRows);
        q4_q5_gdn_input_independent_launch(x, qk.weight, value_z.weight, qk_out, value_out,
                                           sequential_z_tensor, launch_stream);
    };
    const auto overlapped_pdl = [&](cudaStream_t launch_stream) {
        Tensor qk_out = pdl_qkv_tensor.slice(0, 0, kQkRows);
        Tensor value_out = pdl_qkv_tensor.slice(0, kQkRows, kValueRows);
        const std::int32_t q4_out_ld = static_cast<std::int32_t>(qk_out.nb[1] / sizeof(__nv_bfloat16));
        const std::int32_t q5_out_ld = static_cast<std::int32_t>(value_out.nb[1] / sizeof(__nv_bfloat16));
        const dim3 q5_grid(kValueZRows, 1u, 1u);
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 8, 5, kHidden, true,
                                             kValueRows, Q5Split4StoreEpilogue, true, false>
            <<<q5_grid, kQ5Threads, 0, launch_stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z.weight.qdata),
                static_cast<const std::uint8_t*>(value_z.weight.qhigh),
                static_cast<const std::uint8_t*>(value_z.weight.scales),
                static_cast<__nv_bfloat16*>(value_out.data),
                static_cast<__nv_bfloat16*>(pdl_z_tensor.data), kValueZRows, q5_out_ld,
                kHidden, kT, value_z.weight.padded_shape[1], 5);
        CUDA_CHECK(cudaGetLastError());
        const dim3 q4_grid(kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, launch_stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, true, false, 0, Q4SimtStoreEpilogue,
                                         false, true>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk.weight.qdata),
            static_cast<const std::uint8_t*>(qk.weight.scales),
            static_cast<__nv_bfloat16*>(qk_out.data), nullptr, q4_out_ld, 0, kQkRows, kHidden,
            kT, qk.weight.padded_shape[1], Q4SimtStoreEpilogue{}));
    };

    sequential(stream);
    overlapped_pdl(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::size_t qkv_count = static_cast<std::size_t>(kQkvRows) * kT;
    const std::size_t z_count = static_cast<std::size_t>(kZRows) * kT;
    std::vector<std::uint16_t> sequential_bits(qkv_count + z_count);
    std::vector<std::uint16_t> pdl_bits(qkv_count + z_count);
    sequential_qkv.copy_to_host(sequential_bits.data(), qkv_count * sizeof(std::uint16_t));
    sequential_z.copy_to_host(sequential_bits.data() + qkv_count, z_count * sizeof(std::uint16_t));
    pdl_qkv.copy_to_host(pdl_bits.data(), qkv_count * sizeof(std::uint16_t));
    pdl_z.copy_to_host(pdl_bits.data() + qkv_count, z_count * sizeof(std::uint16_t));

    long double error_squared = 0.0L;
    long double reference_squared = 0.0L;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t i = 0; i < sequential_bits.size(); ++i) {
        const std::uint32_t reference_word = static_cast<std::uint32_t>(sequential_bits[i]) << 16;
        const std::uint32_t actual_word = static_cast<std::uint32_t>(pdl_bits[i]) << 16;
        float reference_f = 0.0F;
        float actual_f = 0.0F;
        std::memcpy(&reference_f, &reference_word, sizeof(reference_f));
        std::memcpy(&actual_f, &actual_word, sizeof(actual_f));
        const double reference = reference_f;
        const double actual = actual_f;
        if (!std::isfinite(reference) || !std::isfinite(actual)) {
            throw std::runtime_error("PDL T8 route produced non-finite output");
        }
        const double error = std::abs(actual - reference);
        error_squared += static_cast<long double>(actual - reference) * (actual - reference);
        reference_squared += static_cast<long double>(reference) * reference;
        maximum_error = std::max(maximum_error, error);
        maximum_reference = std::max(maximum_reference, std::abs(reference));
    }
    if (!(reference_squared > 0.0L)) { throw std::runtime_error("PDL screen has zero reference norm"); }
    const double relative_l2 = std::sqrt(static_cast<double>(error_squared / reference_squared));
    const double gross_limit = kGdnInputProjA16GrossAbs +
                               kGdnInputProjA16GrossRel * maximum_reference;
    const bool equivalent = relative_l2 <= kGdnInputProjA16RelativeL2 &&
                            maximum_error <= gross_limit;
    std::printf("pdl_t8_screen outputs=%zu rel_l2=%.6g max_abs=%.6g limit=%.6g %s\n",
                sequential_bits.size(), relative_l2, maximum_error, gross_limit,
                equivalent ? "PASS" : "FAIL");
    if (!equivalent) { throw std::runtime_error("T8 PDL failed sequential production-baseline screen"); }

    const bench::ColdTiming sequential_timing = bench::measure_cold_launch(
        sequential, flush, stream, options.warmup, options.repeat);
    const bench::ColdTiming pdl_timing = bench::measure_cold_launch(
        overlapped_pdl, flush, stream, options.warmup, options.repeat);
    std::printf("pdl_t8_ab entry=gdn_input_proj.q4q5 T=8 cache=cold warmup=%d repeat=%d\n",
                options.warmup, options.repeat);
    std::printf("  q4_then_q5_split4 median=%9.3f us min=%9.3f us p95=%9.3f us\n",
                sequential_timing.median_us, sequential_timing.min_us, sequential_timing.p95_us);
    std::printf("  q5_split4_pdl_then_q4 median=%9.3f us min=%9.3f us p95=%9.3f us ratio=%6.3fx\n",
                pdl_timing.median_us, pdl_timing.min_us, pdl_timing.p95_us,
                pdl_timing.median_us / sequential_timing.median_us);
}

void run_q8(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
            std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 2048;
    constexpr std::int32_t kQkvRows    = 8192;
    constexpr std::int32_t kZRows      = 4096;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_row_split_weight(
        QType::Q8_G32_FP16, kOutputRows, kHidden, kHidden, {0x31, 0x00, 0x3c00});
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, launch_stream);
        };
    };
    const auto workspace_capacity = [](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::Q8_G32_FP16, kOutputRows, kHidden, ops::LinearPolicy::A16Only, tokens, tokens);
    };
    measure_points(options, "q8", "a16", kHidden, kOutputRows, parent.model_weight_bytes(),
                   workspace_capacity, make_launch, flush, stream, results);
}

void run_nvfp4(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
               std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkvRows    = 10240;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_nvfp4_weight(kOutputRows, kHidden);
    const std::size_t maximum_workspace = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kOutputRows, kHidden, options.nvfp4_policy, max_tokens, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, options.nvfp4_policy, workspace,
                                launch_stream);
        };
    };
    const auto workspace_capacity = [&](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, kOutputRows, kHidden,
                                                            options.nvfp4_policy, tokens, tokens);
    };
    measure_points(options, "nvfp4", policy_name(options.nvfp4_policy), kHidden, kOutputRows,
                   parent.model_weight_bytes(), workspace_capacity, make_launch, flush, stream,
                   results);
}

void run_fp8(const Options& options, DeviceBuffer& flush, cudaStream_t stream,
             std::vector<Result>& results) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkvRows    = 10240;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kOutputRows = kQkvRows + kZRows;
    const std::int32_t max_tokens = *std::max_element(options.tokens.begin(), options.tokens.end());
    bench::PackedQuantizedWeight parent = bench::make_fp8_weight(kOutputRows, kHidden);
    const std::size_t maximum_workspace = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kOutputRows, kHidden, options.fp8_policy, 1, max_tokens);
    WorkspaceArena workspace(std::max<std::size_t>(maximum_workspace, 256));
    DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
    DeviceBuffer qkv(static_cast<std::size_t>(kQkvRows) * max_tokens * 2);
    DeviceBuffer z(static_cast<std::size_t>(kZRows) * max_tokens * 2);
    const auto make_launch = [&](std::int32_t tokens) {
        return [&, tokens](cudaStream_t launch_stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor tqkv(qkv.p, DType::BF16, {kQkvRows, tokens});
            Tensor tz(z.p, DType::BF16, {kZRows, tokens});
            ops::gdn_input_proj(x, parent.weight, tqkv, tz, options.fp8_policy, workspace,
                                launch_stream);
        };
    };
    const auto workspace_capacity = [&](std::int32_t tokens) {
        return ops::gdn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, kOutputRows, kHidden, options.fp8_policy, tokens, tokens);
    };
    measure_points(options, "fp8", policy_name(options.fp8_policy), kHidden, kOutputRows,
                   parent.model_weight_bytes(), workspace_capacity, make_launch, flush, stream,
                   results);
}

void write_csv(const Options& options, const std::vector<Result>& results) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream output(path);
    if (!output) { throw std::runtime_error("failed to open CSV output"); }
    output << "entry,format,policy,cache,T,workspace_bytes,logical_bytes,useful_flops,"
              "median_us,min_us,p95_us\n";
    for (const Result& result : results) {
        output << "gdn_input_proj," << result.format << ',' << result.policy << ','
               << cache_name(result.cache) << ',' << result.tokens << ',' << result.workspace_bytes
               << ',' << result.logical_bytes << ',' << result.useful_flops << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << '\n';
    }
}

bool selected(Format configured, Format candidate) {
    return configured == Format::All || configured == candidate;
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        std::printf("# device=%d actual_gpu=%s sm=%d%d\n", device, properties.name,
                    properties.major, properties.minor);
        std::fflush(stdout);
        DeviceBuffer flush(kFlushBytes);
        std::vector<Result> results;

        if (options.route_ab) {
            run_q4q5_route_ab(options, flush, stream);
        } else if (options.q5_split4_ab) {
            run_q5_split4_route_ab(options, flush, stream);
        } else if (options.pdl_t8_ab) {
            run_pdl_t8_route_ab(options, flush, stream);
        } else {
            if (selected(options.format, Format::Q4Q5)) {
                run_q4q5(options, flush, stream, results);
            }
            if (selected(options.format, Format::Q8)) { run_q8(options, flush, stream, results); }
            if (selected(options.format, Format::Nvfp4)) {
                run_nvfp4(options, flush, stream, results);
            }
            if (selected(options.format, Format::Fp8)) {
                run_fp8(options, flush, stream, results);
            }
        }

        write_csv(options, results);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gdn_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
