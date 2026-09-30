#pragma once

#include "ninfer/types.h"
#include "serve/request.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

enum class SlotApiFailure : std::uint8_t {
    PersistenceDisabled,
    ServiceUnavailable,
    InvalidSlot,
    InvalidRequest,
    InvalidAction,
    InvalidFilename,
    Busy,
    SessionMismatch,
    OperationFailed,
};

[[nodiscard]] ApiError make_slot_api_error(SlotApiFailure failure, std::string message,
                                           std::string_view action = {});

[[nodiscard]] std::string make_slots_body(const std::vector<ninfer::SlotState>& states,
                                          std::uint32_t slot_count,
                                          std::uint32_t max_context, bool speculative);

} // namespace ninfer::serve
