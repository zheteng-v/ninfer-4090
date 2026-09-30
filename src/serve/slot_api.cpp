#include "serve/slot_api.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace ninfer::serve {

ApiError make_slot_api_error(SlotApiFailure failure, std::string message,
                             std::string_view action) {
    ApiError error;
    error.message = std::move(message);
    switch (failure) {
    case SlotApiFailure::PersistenceDisabled:
        error.status = 501;
        error.type   = "server_error";
        error.code   = "slot_persistence_disabled";
        break;
    case SlotApiFailure::ServiceUnavailable:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    case SlotApiFailure::InvalidSlot:
        error.status = 400;
        error.code   = "invalid_slot";
        break;
    case SlotApiFailure::InvalidRequest:
        error.status = 400;
        error.code   = "invalid_request";
        break;
    case SlotApiFailure::InvalidAction:
        error.status = 400;
        error.code   = "invalid_action";
        break;
    case SlotApiFailure::InvalidFilename:
        error.status = 400;
        error.code   = "invalid_filename";
        break;
    case SlotApiFailure::Busy:
        error.status = 409;
        error.code   = "slot_busy";
        break;
    case SlotApiFailure::SessionMismatch:
        error.status = 409;
        error.code   = "slot_session_mismatch";
        break;
    case SlotApiFailure::OperationFailed:
        error.status = 400;
        error.code   = "slot_" + std::string(action) + "_failed";
        break;
    }
    return error;
}

std::string make_slots_body(const std::vector<ninfer::SlotState>& states,
                            std::uint32_t slot_count, std::uint32_t max_context,
                            bool speculative) {
    nlohmann::json slots = nlohmann::json::array();
    for (std::uint32_t index = 0; index < slot_count; ++index) {
        const ninfer::SlotState state =
            index < states.size() ? states[index] : ninfer::SlotState{};
        nlohmann::json checkpoints = nlohmann::json::array();
        for (const ninfer::SlotCheckpoint& checkpoint : state.checkpoints) {
            checkpoints.push_back({{"frontier", checkpoint.frontier},
                                   {"session_digest", checkpoint.session_digest}});
        }
        slots.push_back({{"id", index},
                         {"is_processing", state.processing},
                         {"retained", state.retained},
                         {"session_digest", state.session_digest},
                         {"checkpoints", std::move(checkpoints)},
                         {"n_ctx", max_context},
                         {"n_prompt_tokens", state.prompt_tokens},
                         {"n_prompt_tokens_cache", state.cached_tokens},
                         {"speculative", speculative}});
    }
    return slots.dump();
}

} // namespace ninfer::serve
