// Internal schedule sweep for the folded Q4 LinearSwiGLU MMA kernel on sm_89.
//
// The registered large-T schedule (BM64 BN128 WN16 STAGES2) was tuned on the
// RTX 5090. This bench instantiates candidate GemmCfg variants directly -
// bypassing the route table - and times them cold-cache at prefill token
// counts, so an Ada retune can be chosen from measurements instead of
// inherited constants. Bench-only: nothing here changes product dispatch.

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;
constexpr std::uint32_t kWeightSeed = 0x7f4a7c15U;

void fill_q4_weight(bench::PackedQuantizedWeight& packed) {
    const std::size_t groups = static_cast<std::size_t>(kGateUpRows) * (kHidden / 64);
    if (packed.low_bytes != groups * 32 || packed.high_bytes != 0 ||
        packed.scale_bytes != groups * sizeof(std::uint16_t)) {
        throw std::logic_error("Q4 SwiGLU benchmark weight planes have unexpected geometry");
    }
    std::mt19937 rng(kWeightSeed);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(packed.low_bytes));
    std::vector<std::uint16_t> scales(groups);
    for (std::uint8_t& byte : codes) { byte = static_cast<std::uint8_t>(rng() >> 24); }
    for (std::uint16_t& bits : scales) {
        const __half scale = __float2half_rn(0.02F + static_cast<float>(rng() % 97U) * 0.004F);
        std::memcpy(&bits, &scale, sizeof(bits));
    }
    packed.storage.copy_from_host(codes.data(), codes.size(), 0);
    packed.storage.copy_from_host(scales.data(), scales.size() * sizeof(std::uint16_t),
                                  packed.scale_offset);
}

template <class Cfg>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        const int t          = x_slice.ne[1];
        const dim3 grid(static_cast<unsigned>(div_up(out_slice.ne[0], PM)),
                        static_cast<unsigned>(div_up(t, Cfg::BN)));
        const bool full = (t % Cfg::BN) == 0;
        if (full) {
            q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, true>
                <<<grid, Cfg::THREADS, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x_slice.data),
                    static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales),
                    static_cast<__nv_bfloat16*>(out_slice.data), out_slice.ne[0], x_slice.ne[0], t,
                    weight.padded_shape[1]);
        } else {
            q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, false>
                <<<grid, Cfg::THREADS, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x_slice.data),
                    static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const std::uint8_t*>(weight.scales),
                    static_cast<__nv_bfloat16*>(out_slice.data), out_slice.ne[0], x_slice.ne[0], t,
                    weight.padded_shape[1]);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

struct VariantResult {
    const char* name;
    double median_us;
};

template <class Cfg>
VariantResult run_variant(const char* name, std::int32_t tokens, const DeviceBuffer& input,
                          const bench::PackedQuantizedWeight& packed, DeviceBuffer& output,
                          DeviceBuffer& flush, cudaStream_t stream, int warmup, int repeat) {
    Tensor x(input.p, DType::BF16, {kHidden, tokens});
    Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
    const auto timing = bench::measure_cold_launch(
        [&](cudaStream_t launch_stream) { launch_variant<Cfg>(x, packed.weight, out, launch_stream); },
        flush, stream, warmup, repeat);
    return {name, timing.median_us};
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::int32_t> tokens{512, 1024, 2048};
        int warmup = 5;
        int repeat = 30;
        if (argc > 1) {
            tokens.clear();
            for (int i = 1; i < argc; ++i) { tokens.push_back(std::stoi(argv[i])); }
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        const std::int32_t max_t = *std::max_element(tokens.begin(), tokens.end());
        DeviceBuffer input       = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed = bench::make_row_split_weight(
            QType::Q4_G64_FP16, kGateUpRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
        fill_q4_weight(packed);

        for (const std::int32_t t : tokens) {
            std::printf("== T=%d ==\n", t);
            std::vector<VariantResult> results;
            // Registered baseline: BN128 WN16 S2, 8 warps.
            results.push_back(run_variant<GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>>(
                "c128_w16_s2 (base)", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 128, 64, 64, 8, 2, 1, false, true, true>>(
                "c128_w8_s2 (16 warps)", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>>(
                "c128_w32_s2 (4 warps)", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 96, 64, 64, 16, 2, 1, false, true, true>>(
                "c96_w16_s2", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 64, 64, 64, 16, 2, 1, false, true, true>>(
                "c64_w16_s2", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 64, 64, 64, 16, 3, 1, false, true, true>>(
                "c64_w16_s3", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 64, 64, 64, 8, 3, 1, false, true, true>>(
                "c64_w8_s3", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 128, 64, 64, 16, 2, 1, true, true, true>>(
                "c128_w16_s2_fdbuf", t, input, packed, output, flush, stream, warmup, repeat));
            results.push_back(run_variant<GemmCfg<64, 128, 64, 64, 16, 2, 2, false, true, true>>(
                "c128_w16_s2_mb2", t, input, packed, output, flush, stream, warmup, repeat));
            for (const VariantResult& r : results) {
                const double flops = 2.0 * static_cast<double>(kGateUpRows) * kHidden * t;
                std::printf("  %-22s median=%9.3f us  %7.2f TFLOP/s\n", r.name, r.median_us,
                            flops / (r.median_us * 1e-6) / 1e12);
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "q4_linear_swiglu_variants_bench: %s\n", error.what());
        return 1;
    }
}
