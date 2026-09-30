#pragma once

#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "runtime/contract/resources.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer {
struct HostKVPageLayout;
}

namespace ninfer::models::qwen3_5 {
struct StateImageHostLayout;
}

namespace ninfer::models::qwen3_5::detail {

inline constexpr std::uint32_t kQwen3_5SessionSchemaVersion = 1;

[[nodiscard]] std::uint64_t session_layout_fingerprint(const StateImageHostLayout& layout) noexcept;
[[nodiscard]] std::uint64_t session_layout_fingerprint(const HostKVPageLayout& layout) noexcept;

// Exact startup/layout identity needed in addition to the artifact binding carried by the outer
// snapshot. Fingerprints are computed by the Program from complete physical layouts; zero is
// reserved so an uninitialized binding cannot be accepted.
struct SessionRuntimeBinding {
    KvCacheStorage kv_storage                = KvCacheStorage::BFloat16;
    SpeculativeBackend speculative_backend   = SpeculativeBackend::None;
    ProposalHead proposal_head               = ProposalHead::Full;
    std::uint32_t draft_window                = 0;
    std::uint32_t token_domain                = 0;
    std::uint32_t page_tokens                 = 0;
    bool vision_enabled                       = false;
    std::uint64_t state_image_bytes           = 0;
    std::uint64_t text_page_bytes             = 0;
    std::uint64_t backend_page_bytes          = 0;
    std::uint64_t state_layout_fingerprint    = 0;
    std::uint64_t text_layout_fingerprint     = 0;
    std::uint64_t backend_layout_fingerprint = 0;

    [[nodiscard]] friend constexpr bool operator==(SessionRuntimeBinding,
                                                   SessionRuntimeBinding) noexcept = default;
};

struct SessionCheckpointImage {
    runtime::CheckpointKind kind = runtime::CheckpointKind::SessionEndpoint;
    std::uint32_t frontier       = 0;
    std::uint32_t ordinal        = 0;
    std::uint32_t state_index    = 0;
    runtime::PrefillWork rebuild_work;

    [[nodiscard]] friend constexpr bool operator==(SessionCheckpointImage,
                                                   SessionCheckpointImage) noexcept = default;
};

struct SessionKVImage {
    std::uint32_t frontier = 0;
    std::vector<std::uint8_t> payload;

    [[nodiscard]] friend bool operator==(const SessionKVImage&,
                                         const SessionKVImage&) noexcept = default;
};

// Fully owned, already staged continuation image. Decode constructs and validates this object
// without mutating Program stores; Program import can then reserve all destinations before publish.
struct ContinuationSessionImage {
    SessionRuntimeBinding runtime;
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;

    std::vector<TokenId> ledger;
    ResidentPrefixIdentity prefix_identity;
    PrefixShortlistDigests prefix_digests;

    std::optional<SessionCheckpointImage> endpoint;
    std::optional<SessionCheckpointImage> rewrite;
    std::vector<SessionCheckpointImage> long_anchors;
    runtime::PrefillWork rebuild_work;
    std::uint32_t rebuild_tail_begin = 0;

    std::uint32_t state_count = 0;
    std::vector<std::uint8_t> state_payload;
    SessionKVImage text_kv;
    SessionKVImage backend_kv;
};

[[nodiscard]] std::vector<std::uint8_t>
encode_continuation_session_image(std::string_view model_binding,
                                  const ContinuationSessionImage& image,
                                  std::uint64_t max_total_bytes);

[[nodiscard]] ContinuationSessionImage
decode_continuation_session_image(std::span<const std::uint8_t> bytes,
                                  std::string_view expected_model_binding,
                                  const SessionRuntimeBinding& expected_runtime,
                                  std::uint32_t maximum_context,
                                  std::uint64_t max_total_bytes);

// Shared semantic gate used by Program export and by the decoder before publication.
void validate_continuation_session_image(const ContinuationSessionImage& image,
                                         std::uint32_t maximum_context);

} // namespace ninfer::models::qwen3_5::detail
