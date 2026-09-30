#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/session_image.h"

#include "core/device.h"
#include "core/host_kv_arena.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
namespace {

std::string ledger_prefix_digest(std::span<const TokenId> ledger) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId token : ledger) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&token);
        for (std::size_t index = 0; index < sizeof(token); ++index) {
            hash = (hash ^ bytes[index]) * 1099511628211ULL;
        }
    }
    char encoded[17]{};
    (void)std::snprintf(encoded, sizeof(encoded), "%016llx",
                        static_cast<unsigned long long>(hash));
    return encoded;
}

std::size_t checked_payload_size(std::uint32_t count, std::size_t item_bytes,
                                 std::uint64_t max_total_bytes, const char* label) {
    if (item_bytes != 0 && count > std::numeric_limits<std::size_t>::max() / item_bytes) {
        throw std::overflow_error(std::string(label) + " export payload size overflows");
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * item_bytes;
    if (bytes > max_total_bytes) {
        throw std::length_error(std::string(label) + " export payload exceeds the snapshot limit");
    }
    return bytes;
}

void include_payload_size(std::uint64_t& total, std::uint32_t count, std::size_t item_bytes,
                          std::uint64_t max_total_bytes, const char* label) {
    const std::size_t bytes = checked_payload_size(count, item_bytes, max_total_bytes, label);
    if (bytes > max_total_bytes - total) {
        throw std::length_error("session physical payload exceeds the snapshot limit");
    }
    total += bytes;
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

void zero_uncommitted_kv_tail(std::vector<std::uint8_t>& payload,
                              const HostKVPageLayout& layout, std::uint32_t frontier) {
    const std::uint32_t committed = frontier % layout.geometry.page_tokens;
    if (committed == 0 || payload.empty()) { return; }
    const std::size_t page_offset =
        static_cast<std::size_t>(pages_for_tokens(frontier) - 1U) * layout.page_stride;
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const HostKVPlaneLayout& host = layout.planes[index];
        const KVPlaneGeometry& geometry = layout.geometry.planes[index];
        const std::size_t column_bytes =
            static_cast<std::size_t>(geometry.leading_extent) * dtype_size(geometry.dtype);
        const std::size_t tail_bytes =
            static_cast<std::size_t>(layout.geometry.page_tokens - committed) * column_bytes;
        for (std::int32_t head = 0; head < geometry.head_extent; ++head) {
            auto* tail = reinterpret_cast<std::byte*>(payload.data()) + page_offset + host.offset +
                         static_cast<std::size_t>(head) * host.head_payload_bytes +
                         static_cast<std::size_t>(committed) * column_bytes;
            std::memset(tail, 0, tail_bytes);
        }
    }
}

SessionRuntimeBinding runtime_binding(const ProgramImpl& program) {
    if (!program.state_images || !program.text_kv_pages) {
        throw std::logic_error("Program session storage is unavailable");
    }
    const StateImageHostLayout& state = program.state_images->host_layout();
    const HostKVPageLayout text =
        plan_host_kv_page_layout(program.text_kv_pages->physical_pool().geometry());
    SessionRuntimeBinding binding{
        .kv_storage              = program.kv_storage,
        .speculative_backend     = program.speculative_backend,
        .proposal_head           = program.proposal_head,
        .draft_window            = program.draft_window,
        .token_domain = static_cast<std::uint32_t>(
            execution::dimension(program.parameters.model.resources().public_token_count)),
        .page_tokens              = static_cast<std::uint32_t>(kPagedKVPageSize),
        .vision_enabled           = program.vision_enabled,
        .state_image_bytes        = state.image_bytes,
        .text_page_bytes          = text.page_stride,
        .state_layout_fingerprint = session_layout_fingerprint(state),
        .text_layout_fingerprint  = session_layout_fingerprint(text),
    };
    if (program.backend_kv_pages) {
        const HostKVPageLayout backend =
            plan_host_kv_page_layout(program.backend_kv_pages->physical_pool().geometry());
        binding.backend_page_bytes         = backend.page_stride;
        binding.backend_layout_fingerprint = session_layout_fingerprint(backend);
    }
    return binding;
}

std::vector<std::uint8_t>
copy_kv_image(const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
              const HostKVExtentStore* host_extents, KVAddressSpaceHandle address,
              std::uint32_t frontier, std::uint64_t max_total_bytes, cudaStream_t stream) {
    if (!addresses.valid(address) || addresses.committed_frontier(address) < frontier) {
        throw std::logic_error("session KV address does not cover the exported frontier");
    }
    const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());
    const std::uint32_t page_count = pages_for_tokens(frontier);
    if (addresses.mapped_pages(address) < page_count) {
        throw std::logic_error("session KV address has too few mapped pages");
    }
    std::vector<std::uint8_t> payload(
        checked_payload_size(page_count, layout.page_stride, max_total_bytes, "KV"), 0);
    std::vector<DeviceKVPageHandle> device_run;
    device_run.reserve(page_count);

    const auto stage = [&] {
        std::uint32_t index = 0;
        while (index < page_count) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, index);
            const std::uint32_t required =
                index + 1U == page_count && frontier % kPagedKVPageSize != 0
                    ? frontier % static_cast<std::uint32_t>(kPagedKVPageSize)
                    : static_cast<std::uint32_t>(kPagedKVPageSize);
            if (!pages.valid(logical) || pages.committed_columns(logical) < required) {
                throw std::logic_error("session KV page does not cover its exported columns");
            }
            if (pages.host_replica_current(logical)) {
                if (host_extents == nullptr) {
                    throw std::logic_error("session KV Host replica has no extent store");
                }
                const HostKVPageReplica& replica = pages.host_replica(logical);
                const HostKVAllocationConstView source =
                    host_extents->view(replica.extent).subview(replica.page_offset, 1);
                if (source.layout() != layout) {
                    throw std::logic_error("session KV Host replica layout is inconsistent");
                }
                auto* destination = reinterpret_cast<std::byte*>(payload.data()) +
                                    static_cast<std::size_t>(index) * layout.page_stride;
                // Copy only defined plane bytes; padding stays zero and cannot expose stale arena
                // data.
                for (const HostKVPlaneLayout& plane : layout.planes) {
                    std::memcpy(destination + plane.offset, source.data() + plane.offset,
                                plane.page_payload_bytes);
                }
                ++index;
                continue;
            }

            const std::uint32_t run_begin = index;
            device_run.clear();
            while (index < page_count) {
                const LogicalKVPageHandle candidate = addresses.logical_page(address, index);
                const std::uint32_t candidate_required =
                    index + 1U == page_count && frontier % kPagedKVPageSize != 0
                        ? frontier % static_cast<std::uint32_t>(kPagedKVPageSize)
                        : static_cast<std::uint32_t>(kPagedKVPageSize);
                if (!pages.valid(candidate) ||
                    pages.committed_columns(candidate) < candidate_required) {
                    throw std::logic_error("session KV page does not cover its exported columns");
                }
                if (pages.host_replica_current(candidate)) { break; }
                if (!pages.device_resident(candidate)) {
                    throw std::logic_error("session KV page has no current readable replica");
                }
                device_run.push_back(pages.physical(candidate));
                ++index;
            }
            auto* destination = reinterpret_cast<std::byte*>(payload.data()) +
                                static_cast<std::size_t>(run_begin) * layout.page_stride;
            pages.physical_pool().copy_to_host(device_run, destination, layout, stream);
        }
    };
    try {
        stage();
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        (void)cudaStreamSynchronize(stream);
        std::rethrow_exception(failure);
    }
    return payload;
}

} // namespace

std::uint32_t
ProgramImpl::continuation_depth(const ContinuationHandle& continuation) const noexcept {
    if (!valid_continuation(continuation)) { return 0; }
    return static_cast<std::uint32_t>(
        continuation_states[ContractAccess::index(continuation)].ledger.size());
}

std::string ProgramImpl::continuation_digest(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    return ledger_prefix_digest(sequence.ledger);
}

std::vector<SlotCheckpoint>
ProgramImpl::continuation_checkpoints(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) { return {}; }
    const SequenceState& sequence = continuation_states[ContractAccess::index(continuation)];
    const std::uint32_t depth     = static_cast<std::uint32_t>(sequence.ledger.size());
    std::vector<std::uint32_t> frontiers;
    frontiers.reserve(sequence.long_anchors.size() + 2U);
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        frontiers.push_back(anchor.frontier);
    }
    if (sequence.rewrite_checkpoint.valid) {
        frontiers.push_back(sequence.rewrite_checkpoint.frontier);
    }
    if (sequence.endpoint_valid) { frontiers.push_back(sequence.execution_frontier); }
    std::sort(frontiers.begin(), frontiers.end());
    frontiers.erase(std::unique(frontiers.begin(), frontiers.end()), frontiers.end());

    std::vector<SlotCheckpoint> checkpoints;
    checkpoints.reserve(frontiers.size());
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier > depth) { continue; }
        checkpoints.push_back(SlotCheckpoint{
            .frontier = frontier,
            .session_digest = ledger_prefix_digest(
                std::span<const TokenId>(sequence.ledger.data(), frontier)),
        });
    }
    return checkpoints;
}

qwen3_5::ContinuationSummary
ProgramImpl::continuation_summary(const ContinuationHandle& continuation) const {
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("continuation holds no retained session");
    }
    return continuation_summary(continuation_states[ContractAccess::index(continuation)]);
}

std::vector<std::uint8_t>
ProgramImpl::export_continuation(const ContinuationHandle& continuation,
                                 std::string_view model_binding,
                                 std::uint64_t max_total_bytes) {
    if (model_binding.empty() || max_total_bytes == 0) {
        throw std::invalid_argument("session export binding or size limit is empty");
    }
    if (!valid_continuation(continuation)) {
        throw std::invalid_argument("session export continuation is stale");
    }
    if (has_context_transaction() || pending_transaction_ || pressure_planning_active_ ||
        has_unsettled_state_fork()) {
        throw std::logic_error("session export requires a stable Program boundary");
    }
    const std::uint32_t index = ContractAccess::index(continuation);
    if (index >= continuation_states.size() ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        throw std::logic_error("session export continuation is not catalogued");
    }
    const SequenceState& sequence = continuation_states[index];
    if (!sequence.kv || !state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("session export continuation storage is incomplete");
    }

    ContinuationSessionImage image;
    image.runtime                 = runtime_binding(*this);
    image.execution_frontier      = sequence.execution_frontier;
    image.ledger_frontier         = sequence.ledger_frontier;
    image.rope_delta              = sequence.rope_delta;
    image.dflash_context_frontier = sequence.dflash_context_frontier;
    image.mtp_drafts              = sequence.mtp_drafts;
    image.mtp_draft_count         = sequence.mtp_draft_count;
    image.tail_hidden_valid       = sequence.tail_hidden_valid;
    image.ledger                  = sequence.ledger;
    image.prefix_identity         = sequence.prefix_identity;
    image.prefix_digests          = sequence.prefix_digests;
    image.rebuild_work            = sequence.rebuild_work;
    image.rebuild_tail_begin      = sequence.rebuild_tail_begin;

    std::vector<StateImageHandle> states;
    states.reserve(2U + sequence.long_anchors.size());
    const auto state_index = [&](StateImageHandle state) {
        const auto found = std::find(states.begin(), states.end(), state);
        if (found != states.end()) {
            return static_cast<std::uint32_t>(std::distance(states.begin(), found));
        }
        if (!state_store->valid(state) ||
            state_store->role(state) != StateImageRole::CheckpointImmutable) {
            throw std::logic_error("session checkpoint StateImage is not immutable");
        }
        states.push_back(state);
        return static_cast<std::uint32_t>(states.size() - 1U);
    };
    if (sequence.endpoint_valid) {
        image.endpoint = SessionCheckpointImage{
            .kind         = runtime::CheckpointKind::SessionEndpoint,
            .frontier     = sequence.execution_frontier,
            .ordinal      = 0,
            .state_index  = state_index(sequence.state.read),
            .rebuild_work = sequence.rebuild_work,
        };
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("session rewrite checkpoint has no StateImage");
        }
        image.rewrite = SessionCheckpointImage{
            .kind         = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier     = sequence.rewrite_checkpoint.frontier,
            .ordinal      = 0,
            .state_index  = state_index(*sequence.rewrite_state),
            .rebuild_work = sequence.rewrite_checkpoint.rebuild_work,
        };
    }
    image.long_anchors.reserve(sequence.long_anchors.size());
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        image.long_anchors.push_back(SessionCheckpointImage{
            .kind         = runtime::CheckpointKind::LongAnchor,
            .frontier     = anchor.frontier,
            .ordinal      = anchor.ordinal,
            .state_index  = state_index(anchor.state),
            .rebuild_work = anchor.rebuild_work,
        });
    }
    image.state_count = static_cast<std::uint32_t>(states.size());
    const std::uint32_t text_pages = pages_for_tokens(sequence.text_kv_valid);
    const std::uint32_t backend_frontier = backend_kv_valid(sequence);
    const std::uint32_t backend_pages = pages_for_tokens(backend_frontier);
    std::uint64_t physical_payload_bytes = 0;
    include_payload_size(physical_payload_bytes, image.state_count,
                         state_images->host_layout().image_bytes, max_total_bytes, "StateImage");
    include_payload_size(physical_payload_bytes, text_pages,
                         static_cast<std::size_t>(image.runtime.text_page_bytes), max_total_bytes,
                         "Text KV");
    include_payload_size(physical_payload_bytes, backend_pages,
                         static_cast<std::size_t>(image.runtime.backend_page_bytes),
                         max_total_bytes, "Backend KV");
    try {
        image.state_payload.resize(checked_payload_size(
            image.state_count, state_images->host_layout().image_bytes, max_total_bytes,
            "StateImage"));
        for (std::uint32_t state = 0; state < image.state_count; ++state) {
            const std::size_t offset =
                static_cast<std::size_t>(state) * state_images->host_layout().image_bytes;
            state_store->copy_checkpoint_to_host(
                states[state],
                std::span<std::byte>(
                    reinterpret_cast<std::byte*>(image.state_payload.data()) + offset,
                    state_images->host_layout().image_bytes),
                device.stream);
        }

        image.text_kv.frontier = sequence.text_kv_valid;
        image.text_kv.payload = copy_kv_image(*text_kv_addresses, *text_kv_pages,
                                             host_kv_extents.get(), sequence.kv->text,
                                             image.text_kv.frontier, max_total_bytes, device.stream);
        image.backend_kv.frontier = backend_frontier;
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("session Backend KV storage is unavailable");
            }
            image.backend_kv.payload = copy_kv_image(
                *backend_kv_addresses, *backend_kv_pages, host_kv_extents.get(),
                *sequence.kv->backend, image.backend_kv.frontier, max_total_bytes, device.stream);
        } else if (image.runtime.backend_page_bytes != 0 || image.backend_kv.frontier != 0) {
            throw std::logic_error("session Backend KV metadata has no address space");
        }

        // All Device-to-Host copies use the compute stream so this wait also orders them after the
        // kernels which produced the immutable continuation.
        device.synchronize();
        zero_uncommitted_kv_tail(
            image.text_kv.payload,
            plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()),
            image.text_kv.frontier);
        if (backend_kv_pages) {
            zero_uncommitted_kv_tail(
                image.backend_kv.payload,
                plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()),
                image.backend_kv.frontier);
        }
    } catch (...) {
        // A later allocation or replica check can fail after an earlier asynchronous D2H launch.
        // Drain the stream before any staging vector is destroyed, then preserve the first error.
        const std::exception_ptr failure = std::current_exception();
        try {
            device.synchronize();
        } catch (...) {}
        std::rethrow_exception(failure);
    }
    validate_continuation_session_image(image, capacity);
    return encode_continuation_session_image(model_binding, image, max_total_bytes);
}

ContinuationHandle
ProgramImpl::import_continuation(std::span<const std::uint8_t> bytes,
                                 std::string_view expected_model_binding,
                                 std::uint64_t max_total_bytes) {
    if (expected_model_binding.empty() || max_total_bytes == 0) {
        throw std::invalid_argument("session import binding or size limit is empty");
    }
    if (has_context_transaction() || pending_transaction_ || pressure_planning_active_ ||
        has_unsettled_state_fork()) {
        throw std::logic_error("session import requires a stable Program boundary");
    }
    if (!state_store || !state_images || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("Program session storage is unavailable");
    }

    // Decode, checksum, runtime-bind, and semantically validate the complete image before any
    // Program-owned resource is reserved.
    ContinuationSessionImage image = decode_continuation_session_image(
        bytes, expected_model_binding, runtime_binding(*this), capacity, max_total_bytes);

    std::uint32_t checkpoint_frontier = 0;
    const auto include_frontier = [&](const std::optional<SessionCheckpointImage>& checkpoint) {
        if (checkpoint) {
            checkpoint_frontier = std::max(checkpoint_frontier, checkpoint->frontier);
        }
    };
    include_frontier(image.endpoint);
    include_frontier(image.rewrite);
    for (const SessionCheckpointImage& anchor : image.long_anchors) {
        checkpoint_frontier = std::max(checkpoint_frontier, anchor.frontier);
    }
    const std::uint32_t backend_checkpoint_frontier =
        speculative_backend == SpeculativeBackend::Mtp ? checkpoint_frontier - 1U
                                                       : checkpoint_frontier;

    // Complete every potentially allocating metadata operation before reserving physical state.
    SequenceState restored;
    restored.execution_frontier      = image.execution_frontier;
    restored.ledger_frontier         = image.ledger_frontier;
    restored.ledger                  = std::move(image.ledger);
    restored.prefix_identity         = std::move(image.prefix_identity);
    restored.prefix_digests          = std::move(image.prefix_digests);
    restored.rope_delta              = image.rope_delta;
    restored.text_kv_valid           = image.text_kv.frontier;
    restored.mtp_kv_valid            = speculative_backend == SpeculativeBackend::Mtp
                                           ? image.backend_kv.frontier
                                           : 0U;
    restored.dflash_context_frontier = image.dflash_context_frontier;
    restored.mtp_drafts              = image.mtp_drafts;
    restored.mtp_draft_count         = image.mtp_draft_count;
    restored.tail_hidden_valid       = image.tail_hidden_valid;
    restored.rebuild_work            = image.rebuild_work;
    restored.rebuild_tail_begin      = image.rebuild_tail_begin;
    restored.long_anchors.resize(image.long_anchors.size());

    std::vector<StateImageHandle> states(image.state_count);
    std::vector<std::uint32_t> checkpoint_references(image.state_count, 0);

    std::optional<std::uint32_t> continuation_index;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_index = index;
            break;
        }
    }
    if (!continuation_index) { throw std::bad_alloc(); }
    const std::uint32_t slot_index = *continuation_index;
    continuation_slots[slot_index].role = ContinuationSlotRole::ReservedMaterialization;

    std::optional<KVInactiveImportReservation> text_import;
    std::optional<KVInactiveImportReservation> backend_import;
    bool transfers_may_be_in_flight = false;
    try {
        for (StateImageHandle& state : states) {
            std::optional<StateImageHandle> destination = state_store->reserve_destination();
            if (!destination) { throw std::bad_alloc(); }
            state = *destination;
        }

        const auto imported_state = [&](const SessionCheckpointImage& checkpoint) {
            return states[checkpoint.state_index];
        };
        if (image.endpoint) {
            const StateImageHandle endpoint = imported_state(*image.endpoint);
            restored.state = ActiveStateBinding{.read = endpoint, .write = endpoint};
            restored.endpoint_valid = true;
        }
        if (image.rewrite) {
            const StateImageHandle rewrite = imported_state(*image.rewrite);
            restored.rewrite_state         = rewrite;
            restored.rewrite_checkpoint = RewriteCheckpoint{
                .valid = true,
                .kind  = image.rewrite->kind == runtime::CheckpointKind::TurnClosure
                             ? RewriteCheckpointKind::TurnClosure
                             : RewriteCheckpointKind::ResponseReplay,
                .frontier     = image.rewrite->frontier,
                .rebuild_work = image.rewrite->rebuild_work,
            };
            ++checkpoint_references[image.rewrite->state_index];
        }
        for (std::size_t index = 0; index < image.long_anchors.size(); ++index) {
            const SessionCheckpointImage& source = image.long_anchors[index];
            restored.long_anchors[index] = LongAnchorCheckpoint{
                .state        = imported_state(source),
                .frontier     = source.frontier,
                .ordinal      = source.ordinal,
                .rebuild_work = source.rebuild_work,
            };
            ++checkpoint_references[source.state_index];
        }

        std::optional<KVInactiveImportReservation> prepared_text =
            text_kv_addresses->prepare_inactive_import(image.text_kv.frontier);
        if (!prepared_text) { throw std::bad_alloc(); }
        text_import.emplace(std::move(*prepared_text));

        const bool has_backend = image.runtime.backend_page_bytes != 0;
        if (has_backend != (backend_kv_addresses != nullptr && backend_kv_pages != nullptr)) {
            throw std::logic_error("session Backend KV storage is inconsistent");
        }
        if (has_backend) {
            std::optional<KVInactiveImportReservation> prepared_backend =
                backend_kv_addresses->prepare_inactive_import(image.backend_kv.frontier);
            if (!prepared_backend) { throw std::bad_alloc(); }
            backend_import.emplace(std::move(*prepared_backend));
        }
        restored.kv = SequenceKVBundle{
            .text = text_import->address(),
            .backend = backend_import ? std::optional<KVAddressSpaceHandle>(backend_import->address())
                                      : std::nullopt,
        };

        // Tensor views are derived while destinations are still private; this also validates that
        // every referenced StateImage has the required Device residency.
        refresh_state_views(restored);
        for (std::uint32_t state = 0; state < image.state_count; ++state) {
            if (!state_store->checkpoint_import_publishable(states[state])) {
                throw std::logic_error("session StateImage import is not publishable");
            }
        }
        if (!text_kv_addresses->inactive_import_publishable(*text_import,
                                                             checkpoint_frontier) ||
            (backend_import &&
             !backend_kv_addresses->inactive_import_publishable(
                 *backend_import, backend_checkpoint_frontier)) ||
            continuation_slots[slot_index].role !=
                ContinuationSlotRole::ReservedMaterialization) {
            throw std::logic_error("session import reservations are not publishable");
        }

        transfers_may_be_in_flight = true;
        const StateImageHostLayout& state_layout = state_images->host_layout();
        for (std::uint32_t state = 0; state < image.state_count; ++state) {
            const std::size_t offset = static_cast<std::size_t>(state) * state_layout.image_bytes;
            state_store->enqueue_checkpoint_import(
                states[state],
                HostStateImageConstView{
                    .data = reinterpret_cast<const std::byte*>(image.state_payload.data()) + offset,
                    .layout = &state_layout,
                },
                device.stream);
        }
        text_kv_addresses->enqueue_inactive_import(
            *text_import, reinterpret_cast<const std::byte*>(image.text_kv.payload.data()),
            plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()), device.stream);
        if (backend_import) {
            backend_kv_addresses->enqueue_inactive_import(
                *backend_import,
                reinterpret_cast<const std::byte*>(image.backend_kv.payload.data()),
                plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()),
                device.stream);
        }
        device.synchronize();
        transfers_may_be_in_flight = false;

        // Publication from here to the returned handle is deliberately non-throwing.
        static_assert(std::is_nothrow_move_assignable_v<SequenceState>);
        for (std::uint32_t state = 0; state < image.state_count; ++state) {
            state_store->publish_checkpoint_import(states[state], checkpoint_references[state]);
        }
        const KVAddressSpaceHandle text = text_kv_addresses->publish_inactive_import(
            std::move(*text_import), checkpoint_frontier);
        if (text != restored.kv->text) { std::terminate(); }
        text_import.reset();
        if (backend_import) {
            const KVAddressSpaceHandle backend = backend_kv_addresses->publish_inactive_import(
                std::move(*backend_import), backend_checkpoint_frontier);
            if (!restored.kv->backend || backend != *restored.kv->backend) { std::terminate(); }
            backend_import.reset();
        }
        continuation_states[slot_index] = std::move(restored);
        continuation_slots[slot_index].role = ContinuationSlotRole::Catalogued;
        advance_resource_revision();
        return ContractAccess::make_continuation(
            this, slot_index, continuation_slots[slot_index].generation);
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (transfers_may_be_in_flight) {
            try {
                device.synchronize();
            } catch (...) {}
        }
        backend_import.reset();
        text_import.reset();
        for (auto state = states.rbegin(); state != states.rend(); ++state) {
            if (state_store->valid(*state) && !state_store->release(*state)) { std::terminate(); }
        }
        continuation_slots[slot_index].role = ContinuationSlotRole::Free;
        std::rethrow_exception(failure);
    }
}

} // namespace ninfer::models::qwen3_5::detail
