#pragma once

// Cumulative counters behind GET /metrics, in the flat `name value` subset of the Prometheus text
// format. The llama.cpp-compatible execution totals come from the Engine's live per-unit counters;
// NInfer-specific completion counters remain owned by this adapter.

#include "serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

enum class SlotMetricAction : std::uint8_t {
    Save,
    Restore,
    Erase,
};

class ServeMetrics {
public:
    void record(const GenerationOutcome& outcome);
    void record_slot_action(SlotMetricAction action, bool success);

    struct LastCompleted {
        int prompt_tokens = 0;
        int cached_tokens = 0;
    };

    [[nodiscard]] LastCompleted last_completed() const;

    [[nodiscard]] std::string render(std::uint32_t max_concurrency,
                                     const ninfer::RuntimeStats& live,
                                     std::size_t active_requests,
                                     const std::vector<ninfer::SlotState>& slots) const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                    = 0;
    std::uint64_t prefix_cache_hit_tokens_total_     = 0;
    std::uint64_t speculative_draft_tokens_total_    = 0;
    std::uint64_t speculative_accepted_tokens_total_ = 0;
    std::uint64_t slot_save_total_                    = 0;
    std::uint64_t slot_restore_total_                 = 0;
    std::uint64_t slot_erase_total_                   = 0;
    std::uint64_t slot_operation_failures_total_      = 0;
    LastCompleted last_completed_;
};

} // namespace ninfer::serve
