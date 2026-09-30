#include "models/qwen3_5/program/session_image.h"
#include "core/host_kv_arena.h"
#include "models/qwen3_5/state/state_image.h"
#include "runtime/session_snapshot.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

namespace qwen = ninfer::models::qwen3_5;
namespace detail = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) { return true; }
    return false;
}

ninfer::runtime::PrefillWork work(std::uint32_t frontier) {
    return ninfer::runtime::make_prefill_work(0, frontier, 0, 0, 128);
}

detail::ContinuationSessionImage make_image() {
    detail::ContinuationSessionImage image;
    image.runtime = detail::SessionRuntimeBinding{
        .kv_storage                = ninfer::KvCacheStorage::Int8Group64,
        .speculative_backend       = ninfer::SpeculativeBackend::Mtp,
        .proposal_head             = ninfer::ProposalHead::Full,
        .draft_window              = 3,
        .token_domain              = 152064,
        .page_tokens               = 64,
        .vision_enabled            = true,
        .state_image_bytes         = 48,
        .text_page_bytes           = 64,
        .backend_page_bytes        = 32,
        .state_layout_fingerprint  = 0x1111222233334444ULL,
        .text_layout_fingerprint   = 0x5555666677778888ULL,
        .backend_layout_fingerprint = 0x9999aaaabbbbccccULL,
    };
    image.execution_frontier      = 3;
    image.ledger_frontier         = 4;
    image.rope_delta              = -2;
    image.mtp_drafts[0]           = 41;
    image.mtp_drafts[1]           = 42;
    image.mtp_draft_count         = 2;
    image.tail_hidden_valid       = true;
    image.rebuild_work            = work(3);
    image.rebuild_tail_begin      = 2;
    image.ledger                  = {10, 11, 12, 13};

    qwen::PreparedPromptData prompt;
    prompt.token_ids   = image.ledger;
    prompt.token_types = {0, 1, 1, 0};
    prompt.positions   = {0, 1, 2, 3, 0, 1, 4, 5, 0, 1, 6, 7};
    prompt.identity.rewrite_execution_frontiers = {2};
    qwen::VisionItem vision;
    vision.modality      = qwen::PromptModality::Image;
    vision.grid          = {.temporal = 1, .height = 2, .width = 2};
    vision.patch_begin   = 0;
    vision.patch_count   = 4;
    vision.content_digest[0] = 0xa5;
    vision.timestamps        = {0.0};
    vision.token_spans       = {{.begin = 1, .count = 1}};
    prompt.vision_items.push_back(vision);
    image.prefix_identity.assign(prompt);
    image.prefix_digests.assign(prompt);

    image.endpoint = detail::SessionCheckpointImage{
        .kind         = ninfer::runtime::CheckpointKind::SessionEndpoint,
        .frontier     = 3,
        .ordinal      = 0,
        .state_index  = 0,
        .rebuild_work = work(3),
    };
    image.rewrite = detail::SessionCheckpointImage{
        .kind         = ninfer::runtime::CheckpointKind::TurnClosure,
        .frontier     = 2,
        .ordinal      = 0,
        .state_index  = 1,
        .rebuild_work = work(2),
    };
    image.long_anchors.push_back(detail::SessionCheckpointImage{
        .kind         = ninfer::runtime::CheckpointKind::LongAnchor,
        .frontier     = 1,
        .ordinal      = 1,
        .state_index  = 1,
        .rebuild_work = work(1),
    });
    image.state_count = 2;
    image.state_payload.resize(2 * image.runtime.state_image_bytes);
    for (std::size_t index = 0; index < image.state_payload.size(); ++index) {
        image.state_payload[index] = static_cast<std::uint8_t>(index * 17U);
    }
    image.text_kv.frontier = 3;
    image.text_kv.payload.resize(image.runtime.text_page_bytes, 0x5a);
    image.backend_kv.frontier = 2;
    image.backend_kv.payload.resize(image.runtime.backend_page_bytes, 0xc3);
    return image;
}

bool same_checkpoint(const std::optional<detail::SessionCheckpointImage>& left,
                     const std::optional<detail::SessionCheckpointImage>& right) {
    return left == right;
}

} // namespace

int main() {
    constexpr std::string_view binding = "qwen3.5-27b-v3:sm89:test";
    const auto source                  = make_image();
    const auto bytes = detail::encode_continuation_session_image(
        binding, source, ninfer::runtime::kDefaultSessionSnapshotLimit);
    const auto again = detail::encode_continuation_session_image(
        binding, source, ninfer::runtime::kDefaultSessionSnapshotLimit);
    expect(bytes == again, "Qwen continuation image encoding is not deterministic");

    qwen::StateImageHostLayout state_layout{
        .spec = {.linear = {.layers         = 2,
                            .conv_channels  = 8,
                            .conv_width     = 3,
                            .value_heads    = 2,
                            .value_head_dim = 4,
                            .key_head_dim   = 4,
                            .slot_count     = 2,
                            .conv_dtype     = ninfer::DType::BF16},
                 .hidden = 16},
        .linear_conv                  = {.offset = 0, .bytes = 64, .alignment = 64},
        .linear_conv_layer_bytes      = 32,
        .linear_recurrent             = {.offset = 64, .bytes = 128, .alignment = 64},
        .linear_recurrent_layer_bytes = 64,
        .continuation_hidden          = {.offset = 192, .bytes = 32, .alignment = 64},
        .image_bytes                  = 256,
    };
    auto wider_state_layout = state_layout;
    wider_state_layout.spec.linear.slot_count = 4;
    expect(detail::session_layout_fingerprint(state_layout) ==
               detail::session_layout_fingerprint(wider_state_layout),
           "StateImage fingerprint incorrectly binds server concurrency");
    auto changed_state_layout = state_layout;
    changed_state_layout.spec.hidden = 32;
    changed_state_layout.continuation_hidden.bytes = 64;
    changed_state_layout.image_bytes = 320;
    expect(detail::session_layout_fingerprint(state_layout) !=
               detail::session_layout_fingerprint(changed_state_layout),
           "StateImage fingerprint missed a physical geometry change");

    ninfer::KVPageGeometry kv_geometry{
        .page_tokens = 64,
        .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
        .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
    };
    const auto kv_layout = ninfer::plan_host_kv_page_layout(kv_geometry);
    kv_geometry.planes[0].head_extent = 3;
    const auto changed_kv_layout = ninfer::plan_host_kv_page_layout(kv_geometry);
    expect(detail::session_layout_fingerprint(kv_layout) !=
               detail::session_layout_fingerprint(changed_kv_layout),
           "KV fingerprint missed a physical geometry change");

    const auto restored = detail::decode_continuation_session_image(
        bytes, binding, source.runtime, 262144,
        ninfer::runtime::kDefaultSessionSnapshotLimit);
    expect(restored.runtime == source.runtime &&
               restored.execution_frontier == source.execution_frontier &&
               restored.ledger_frontier == source.ledger_frontier &&
               restored.rope_delta == source.rope_delta &&
               restored.dflash_context_frontier == source.dflash_context_frontier &&
               restored.mtp_drafts == source.mtp_drafts &&
               restored.mtp_draft_count == source.mtp_draft_count &&
               restored.tail_hidden_valid == source.tail_hidden_valid &&
               restored.rebuild_work == source.rebuild_work &&
               restored.rebuild_tail_begin == source.rebuild_tail_begin,
           "Qwen sequence metadata did not round-trip");
    expect(restored.ledger == source.ledger &&
               restored.prefix_identity.equals(source.prefix_identity) &&
               restored.prefix_digests.image() == source.prefix_digests.image(),
           "Qwen ledger or exact prefix identity did not round-trip");
    expect(same_checkpoint(restored.endpoint, source.endpoint) &&
               same_checkpoint(restored.rewrite, source.rewrite) &&
               restored.long_anchors == source.long_anchors,
           "Qwen checkpoint graph did not round-trip");
    expect(restored.state_count == source.state_count &&
               restored.state_payload == source.state_payload &&
               restored.text_kv == source.text_kv && restored.backend_kv == source.backend_kv,
           "Qwen physical payloads did not round-trip");

    auto wrong_runtime = source.runtime;
    ++wrong_runtime.text_layout_fingerprint;
    expect(throws<std::invalid_argument>([&] {
               (void)detail::decode_continuation_session_image(
                   bytes, binding, wrong_runtime, 262144,
                   ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "runtime-layout mismatch was accepted");
    expect(throws<std::runtime_error>([&] {
               (void)detail::decode_continuation_session_image(
                   bytes, "another-model", source.runtime, 262144,
                   ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "artifact-binding mismatch was accepted");
    expect(throws<std::invalid_argument>([&] {
               (void)detail::decode_continuation_session_image(
                   bytes, binding, source.runtime, 2,
                   ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "smaller server context accepted the image");

    auto unreferenced_state = source;
    unreferenced_state.state_count = 3;
    unreferenced_state.state_payload.resize(3 * source.runtime.state_image_bytes);
    expect(throws<std::invalid_argument>([&] {
               (void)detail::encode_continuation_session_image(
                   binding, unreferenced_state, ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "unreferenced StateImage payload was accepted");

    auto bad_digest = source;
    auto digest_image = bad_digest.prefix_digests.image();
    digest_image[1][0] ^= 1U;
    if (digest_image[1][0] == 0) { digest_image[1][0] = 1; }
    bad_digest.prefix_digests.restore(std::move(digest_image));
    expect(throws<std::invalid_argument>([&] {
               (void)detail::encode_continuation_session_image(
                   binding, bad_digest, ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "prefix digest inconsistent with exact identity was accepted");

    auto bad_alias = source;
    bad_alias.rewrite->state_index = bad_alias.state_count;
    expect(throws<std::invalid_argument>([&] {
               (void)detail::encode_continuation_session_image(
                   binding, bad_alias, ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "out-of-range checkpoint StateImage alias was accepted");

    auto bad_backend = source;
    bad_backend.backend_kv.frontier = 1;
    bad_backend.backend_kv.payload.resize(source.runtime.backend_page_bytes);
    expect(throws<std::invalid_argument>([&] {
               (void)detail::encode_continuation_session_image(
                   binding, bad_backend, ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "insufficient Backend KV frontier was accepted");

    auto dflash2 = source;
    dflash2.runtime.speculative_backend        = ninfer::SpeculativeBackend::DFlash2;
    dflash2.runtime.draft_window               = 5;
    dflash2.runtime.backend_page_bytes         = 0;
    dflash2.runtime.backend_layout_fingerprint = 0;
    dflash2.mtp_draft_count                    = 0;
    dflash2.dflash_context_frontier            = dflash2.execution_frontier;
    dflash2.backend_kv                         = {};
    expect(!throws<std::invalid_argument>([&] {
               const auto dflash2_bytes = detail::encode_continuation_session_image(
                   binding, dflash2, ninfer::runtime::kDefaultSessionSnapshotLimit);
               (void)detail::decode_continuation_session_image(
                   dflash2_bytes, binding, dflash2.runtime, 262144,
                   ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "DFlash2 session without paged Backend KV was rejected");

    auto dflash2_with_phantom_backend = dflash2;
    dflash2_with_phantom_backend.backend_kv.frontier = 1;
    expect(throws<std::invalid_argument>([&] {
               (void)detail::encode_continuation_session_image(
                   binding, dflash2_with_phantom_backend,
                   ninfer::runtime::kDefaultSessionSnapshotLimit);
           }),
           "DFlash2 session accepted Backend KV without a physical layout");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
