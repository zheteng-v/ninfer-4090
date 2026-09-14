#include "core/device.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    using ninfer::targets::qwen3_6_27b::Package;
    static_assert(ninfer::targets::qwen3_6::kMaximumVisionItemTokens == 16384);
    constexpr std::size_t kWorkspaceCeiling = 866'648'064;
    try {
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            return 77;
        }
        CUDA_CHECK(status);
        ninfer::DeviceContext device;
        const auto capacity = [&](std::uint32_t max_context) {
            ninfer::EngineOptions options;
            options.max_context         = max_context;
            options.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
            options.prefill_chunk       = 1024;
            options.kv_cache            = ninfer::KvCacheStorage::Fp8E4M3Row256;
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens         = 3;
            options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
            options.enable_vision                    = true;
            options.use_cuda_graph                   = false;
            options.context_cache.device_state_slots = 1;
            auto planner     = Package::make_sequence_planner(device, options,
                                                              Package::WeightsProfile::Qwen36Nvfp4);
            const auto pages = planner.capacity_curve().minimum_main_page_groups;
            return std::move(planner).finalize(pages).workspace_capacity_bytes();
        };
        // Increasing total context cannot exceed the single Vision item's 16K workspace.
        const auto at_item_limit    = capacity(16384);
        const auto above_item_limit = capacity(131072);
        if (at_item_limit == 0 || at_item_limit > kWorkspaceCeiling ||
            above_item_limit != at_item_limit) {
            throw std::runtime_error("Vision workspace exceeded the single-item bound");
        }
        std::cout << "Vision workspace remains bounded for long text context\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
