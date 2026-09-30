#include "serve/slot_api.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using namespace ninfer::serve;
    int failures = 0;

    std::vector<ninfer::SlotState> states(1);
    states[0].processing     = true;
    states[0].retained       = true;
    states[0].prompt_tokens  = 41;
    states[0].cached_tokens  = 37;
    states[0].session_digest = "0123456789abcdef";
    states[0].checkpoints.push_back(
        {.frontier = 17, .session_digest = "fedcba9876543210"});

    const nlohmann::json slots = nlohmann::json::parse(make_slots_body(states, 2, 200000, true));
    failures += check(slots.is_array() && slots.size() == 2, "slot list cardinality mismatch");
    failures += check(slots[0] == nlohmann::json{{"id", 0},
                                                {"is_processing", true},
                                                {"retained", true},
                                                {"session_digest", "0123456789abcdef"},
                                                {"checkpoints",
                                                 {{{"frontier", 17},
                                                   {"session_digest", "fedcba9876543210"}}}},
                                                {"n_ctx", 200000},
                                                {"n_prompt_tokens", 41},
                                                {"n_prompt_tokens_cache", 37},
                                                {"speculative", true}},
                      "occupied slot schema mismatch");
    failures += check(slots[1]["id"] == 1 && !slots[1]["is_processing"].get<bool>() &&
                          !slots[1]["retained"].get<bool>() &&
                          slots[1]["session_digest"].get<std::string>().empty(),
                      "missing Engine state was not rendered as an idle slot");

    const ApiError disabled = make_slot_api_error(
        SlotApiFailure::PersistenceDisabled, "disabled");
    const ApiError busy = make_slot_api_error(SlotApiFailure::Busy, "busy");
    const ApiError mismatch =
        make_slot_api_error(SlotApiFailure::SessionMismatch, "changed");
    const ApiError restore_failed =
        make_slot_api_error(SlotApiFailure::OperationFailed, "bad image", "restore");
    failures += check(disabled.status == 501 && disabled.code == "slot_persistence_disabled" &&
                          disabled.type == "server_error",
                      "disabled persistence error contract mismatch");
    failures += check(busy.status == 409 && busy.code == "slot_busy",
                      "busy error contract mismatch");
    failures += check(mismatch.status == 409 && mismatch.code == "slot_session_mismatch",
                      "digest mismatch error contract mismatch");
    failures += check(restore_failed.status == 400 &&
                          restore_failed.code == "slot_restore_failed",
                      "operation failure error contract mismatch");

    std::cout << (failures == 0 ? "OK\n" : "FAIL\n");
    return failures == 0 ? 0 : 1;
}
