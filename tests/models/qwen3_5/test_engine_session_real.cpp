#include "ninfer/engine.h"
#include "runtime/session_file.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.allow_prefix_reuse      = true;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    return options;
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 256;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk        = 256;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.use_cuda_graph       = false;
    options.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.context_cache.device_state_slots = 1;
    options.context_cache.host_state_slots   = 2;
    options.context_cache.host_kv_capacity_bytes               = 256ULL << 20;
    options.context_cache.max_private_continuations             = 1;
    options.context_cache.max_shared_prefixes                   = 0;
    options.context_cache.max_long_anchors_per_continuation     = 0;
    return options;
}

std::uint32_t retained_slot(const std::vector<ninfer::SlotState>& states) {
    const auto found = std::find_if(states.begin(), states.end(),
                                    [](const auto& state) { return state.retained; });
    if (found == states.end()) { throw std::runtime_error("generation retained no session"); }
    return static_cast<std::uint32_t>(std::distance(states.begin(), found));
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    char pattern[] = "/tmp/ninfer-engine-session-XXXXXX";
    char* directory = ::mkdtemp(pattern);
    if (directory == nullptr) { throw std::runtime_error("mkdtemp failed"); }
    const std::filesystem::path root(directory);
    const std::filesystem::path path = root / "slot.nsession";
    try {
        std::mutex event_mutex;
        std::condition_variable event_cv;
        std::optional<ninfer::SlotAutoSaveEvent> auto_save_event;
        ninfer::SlotSaveResult saved;
        std::string digest;
        std::uint32_t slot = 0;
        ninfer::EngineOptions options = engine_options(artifact);
        options.auto_save_evicted = true;
        options.auto_save_listener = [&](const ninfer::SlotAutoSaveEvent& event) {
            {
                std::scoped_lock lock(event_mutex);
                auto_save_event = event;
            }
            event_cv.notify_one();
        };
        {
        ninfer::Engine engine(std::move(options));

        const auto prompt = engine.tokenize_text("Durable session smoke test: count upward from one.");
        const auto generated = engine.generate(engine.prepare_tokens(prompt), request(8));
        require(generated.generated_token_ids.size() == 8, "generation did not finish");
        const auto initial_states = engine.slot_states();
        slot                      = retained_slot(initial_states);
        digest                    = initial_states[slot].session_digest;
        require(!digest.empty(), "retained session has no digest");

        saved = engine.save_slot(slot, path.string(), digest);
        require(saved.tokens != 0 && saved.bytes != 0 && saved.session_digest == digest,
                "save_slot returned an invalid summary");
        const auto image = ninfer::runtime::read_session_file(path);
        require(image.size() == saved.bytes, "saved byte count does not match the file");

        bool mismatch = false;
        try {
            (void)engine.erase_slot(slot, "0000000000000000");
        } catch (const ninfer::SlotSessionMismatch&) {
            mismatch = true;
        }
        require(mismatch && engine.slot_states()[slot].session_digest == digest,
                "digest precondition did not preserve the resident session");

        require(engine.erase_slot(slot, digest) == saved.tokens,
                "erase_slot returned the wrong depth");
        require(!engine.slot_states()[slot].retained, "erase_slot left the session resident");
        const auto restored = engine.restore_slot(slot, path.string());
        require(restored.tokens == saved.tokens && restored.session_digest == digest,
                "restore_slot changed session identity");

        auto corrupt = image;
        corrupt[corrupt.size() / 2U] ^= 0x5aU;
        ninfer::runtime::write_session_file_atomic(path, corrupt);
        bool rejected = false;
        try {
            (void)engine.restore_slot(slot, path.string());
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        const auto after_failure = engine.slot_states();
        require(rejected && after_failure[slot].retained &&
                    after_failure[slot].session_digest == digest,
                "a corrupt replacement destroyed the resident session");
        ninfer::runtime::write_session_file_atomic(path, image);

        const auto replacement_prompt =
            engine.tokenize_text("A distinct session that must evict the saved continuation.");
        const auto replacement =
            engine.generate(engine.prepare_tokens(replacement_prompt), request(8));
        require(replacement.generated_token_ids.size() == 8,
                "replacement generation did not finish");
        {
            std::unique_lock lock(event_mutex);
            require(event_cv.wait_for(lock, std::chrono::seconds(30),
                                      [&] { return auto_save_event.has_value(); }),
                    "eviction auto-save did not complete");
            require(auto_save_event->path == path && auto_save_event->error.empty() &&
                        !auto_save_event->skipped_behind_tokens &&
                        auto_save_event->tokens == saved.tokens,
                    "eviction auto-save reported an invalid outcome");
        }
        require(!ninfer::runtime::read_session_file(path).empty(),
                "eviction auto-save did not publish a readable file");
        }

        ninfer::Engine restarted(engine_options(artifact));
        const auto restarted_restore = restarted.restore_slot(slot, path.string());
        require(restarted_restore.tokens == saved.tokens &&
                    restarted_restore.session_digest == digest,
                "a fresh Engine did not restore the durable session");

        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        require(!cleanup_error, "test cleanup failed");
        std::cout << "ok\n";
        return 0;
    } catch (...) {
        std::error_code cleanup_error;
        std::filesystem::remove_all(root, cleanup_error);
        throw;
    }
}
