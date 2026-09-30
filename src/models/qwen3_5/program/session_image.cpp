#include "models/qwen3_5/program/session_image.h"

#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/state/state_image.h"
#include "runtime/session_snapshot.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {
namespace {

enum class Section : std::uint32_t {
    Runtime   = 0x3501,
    Sequence  = 0x3502,
    Identity  = 0x3503,
    States    = 0x3504,
    TextKV    = 0x3505,
    BackendKV = 0x3506,
};

constexpr std::uint32_t kSectionVersion = 1;
constexpr std::uint64_t kFingerprintOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFingerprintPrime  = 1099511628211ULL;

void fingerprint_mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        hash ^= static_cast<std::uint8_t>(value >> shift);
        hash *= kFingerprintPrime;
    }
}

void fingerprint_region(std::uint64_t& hash, const LayoutRegion& region) noexcept {
    fingerprint_mix(hash, region.offset);
    fingerprint_mix(hash, region.bytes);
    fingerprint_mix(hash, region.alignment);
}

class Writer {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }

    void u32(std::uint32_t value) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void u64(std::uint64_t value) {
        for (unsigned shift = 0; shift != 64; shift += 8) {
            bytes_.push_back(static_cast<std::uint8_t>(value >> shift));
        }
    }

    void raw(const void* data, std::size_t size) {
        if (size == 0) { return; }
        const auto* begin = static_cast<const std::uint8_t*>(data);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    template <typename T>
    void vector(std::span<const T> values) {
        static_assert(std::is_integral_v<T>);
        if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("session image vector exceeds uint32");
        }
        u32(static_cast<std::uint32_t>(values.size()));
        for (const T value : values) {
            if constexpr (sizeof(T) == 1) {
                u8(static_cast<std::uint8_t>(value));
            } else if constexpr (sizeof(T) == 4) {
                u32(std::bit_cast<std::uint32_t>(value));
            } else {
                static_assert(sizeof(T) == 8);
                u64(std::bit_cast<std::uint64_t>(value));
            }
        }
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::vector<std::uint8_t> take() && noexcept { return std::move(bytes_); }

private:
    std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

    [[nodiscard]] std::uint8_t u8() {
        require(1);
        return bytes_[cursor_++];
    }

    [[nodiscard]] std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned shift = 0; shift != 32; shift += 8) {
            value |= static_cast<std::uint32_t>(bytes_[cursor_++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift != 64; shift += 8) {
            value |= static_cast<std::uint64_t>(bytes_[cursor_++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::span<const std::uint8_t> raw(std::size_t size) {
        require(size);
        const auto result = bytes_.subspan(cursor_, size);
        cursor_ += size;
        return result;
    }

    template <typename T>
    [[nodiscard]] std::vector<T> vector(std::uint32_t maximum, const char* label) {
        static_assert(std::is_integral_v<T>);
        const std::uint32_t count = u32();
        if (count > maximum) {
            throw std::invalid_argument(std::string("session image ") + label +
                                        " count exceeds its bound");
        }
        std::vector<T> result;
        result.reserve(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            if constexpr (sizeof(T) == 1) {
                result.push_back(static_cast<T>(u8()));
            } else if constexpr (sizeof(T) == 4) {
                result.push_back(std::bit_cast<T>(u32()));
            } else {
                static_assert(sizeof(T) == 8);
                result.push_back(std::bit_cast<T>(u64()));
            }
        }
        return result;
    }

    void finish() const {
        if (cursor_ != bytes_.size()) {
            throw std::invalid_argument("session image section has trailing bytes");
        }
    }

private:
    void require(std::size_t size) const {
        if (size > bytes_.size() - cursor_) {
            throw std::invalid_argument("session image section is truncated");
        }
    }

    std::span<const std::uint8_t> bytes_;
    std::size_t cursor_ = 0;
};

void write_work(Writer& writer, runtime::PrefillWork work) {
    writer.u64(work.chunks);
    writer.u64(work.tokens);
    writer.u64(work.attention_pairs);
    writer.u64(work.vision_items);
    writer.u64(work.vision_patches);
}

runtime::PrefillWork read_work(Reader& reader) {
    return runtime::PrefillWork{
        .chunks          = reader.u64(),
        .tokens          = reader.u64(),
        .attention_pairs = reader.u64(),
        .vision_items    = reader.u64(),
        .vision_patches  = reader.u64(),
    };
}

void write_checkpoint(Writer& writer, const SessionCheckpointImage& checkpoint) {
    writer.u32(static_cast<std::uint32_t>(checkpoint.kind));
    writer.u32(checkpoint.frontier);
    writer.u32(checkpoint.ordinal);
    writer.u32(checkpoint.state_index);
    write_work(writer, checkpoint.rebuild_work);
}

SessionCheckpointImage read_checkpoint(Reader& reader) {
    SessionCheckpointImage checkpoint;
    checkpoint.kind         = static_cast<runtime::CheckpointKind>(reader.u32());
    checkpoint.frontier     = reader.u32();
    checkpoint.ordinal      = reader.u32();
    checkpoint.state_index  = reader.u32();
    checkpoint.rebuild_work = read_work(reader);
    return checkpoint;
}

std::vector<std::uint8_t> encode_runtime(const SessionRuntimeBinding& runtime) {
    Writer writer;
    writer.u32(kSectionVersion);
    writer.u32(static_cast<std::uint32_t>(runtime.kv_storage));
    writer.u32(static_cast<std::uint32_t>(runtime.speculative_backend));
    writer.u32(static_cast<std::uint32_t>(runtime.proposal_head));
    writer.u32(runtime.draft_window);
    writer.u32(runtime.token_domain);
    writer.u32(runtime.page_tokens);
    writer.u32(runtime.vision_enabled ? 1U : 0U);
    writer.u64(runtime.state_image_bytes);
    writer.u64(runtime.text_page_bytes);
    writer.u64(runtime.backend_page_bytes);
    writer.u64(runtime.state_layout_fingerprint);
    writer.u64(runtime.text_layout_fingerprint);
    writer.u64(runtime.backend_layout_fingerprint);
    writer.u64(0);
    return std::move(writer).take();
}

SessionRuntimeBinding decode_runtime(std::span<const std::uint8_t> bytes) {
    Reader reader(bytes);
    if (reader.u32() != kSectionVersion) {
        throw std::invalid_argument("unsupported session runtime section version");
    }
    SessionRuntimeBinding runtime;
    runtime.kv_storage              = static_cast<KvCacheStorage>(reader.u32());
    runtime.speculative_backend     = static_cast<SpeculativeBackend>(reader.u32());
    runtime.proposal_head           = static_cast<ProposalHead>(reader.u32());
    runtime.draft_window            = reader.u32();
    runtime.token_domain            = reader.u32();
    runtime.page_tokens             = reader.u32();
    const std::uint32_t vision      = reader.u32();
    runtime.state_image_bytes       = reader.u64();
    runtime.text_page_bytes         = reader.u64();
    runtime.backend_page_bytes      = reader.u64();
    runtime.state_layout_fingerprint   = reader.u64();
    runtime.text_layout_fingerprint    = reader.u64();
    runtime.backend_layout_fingerprint = reader.u64();
    if (vision > 1 || reader.u64() != 0) {
        throw std::invalid_argument("session runtime section has invalid flags");
    }
    runtime.vision_enabled = vision != 0;
    reader.finish();
    return runtime;
}

std::vector<std::uint8_t> encode_sequence(const ContinuationSessionImage& image) {
    Writer writer;
    writer.u32(kSectionVersion);
    writer.u32(image.execution_frontier);
    writer.u32(image.ledger_frontier);
    writer.u32(std::bit_cast<std::uint32_t>(image.rope_delta));
    writer.u32(image.dflash_context_frontier);
    writer.u32(image.mtp_draft_count);
    writer.u32(image.tail_hidden_valid ? 1U : 0U);
    writer.u32(image.rebuild_tail_begin);
    writer.u32(image.state_count);
    for (const TokenId token : image.mtp_drafts) {
        writer.u32(std::bit_cast<std::uint32_t>(token));
    }
    write_work(writer, image.rebuild_work);
    writer.u32(image.endpoint.has_value() ? 1U : 0U);
    if (image.endpoint) { write_checkpoint(writer, *image.endpoint); }
    writer.u32(image.rewrite.has_value() ? 1U : 0U);
    if (image.rewrite) { write_checkpoint(writer, *image.rewrite); }
    writer.u32(static_cast<std::uint32_t>(image.long_anchors.size()));
    for (const auto& anchor : image.long_anchors) { write_checkpoint(writer, anchor); }
    return std::move(writer).take();
}

void decode_sequence(std::span<const std::uint8_t> bytes, std::uint32_t maximum_context,
                     ContinuationSessionImage& image) {
    Reader reader(bytes);
    if (reader.u32() != kSectionVersion) {
        throw std::invalid_argument("unsupported session sequence section version");
    }
    image.execution_frontier      = reader.u32();
    image.ledger_frontier         = reader.u32();
    image.rope_delta              = std::bit_cast<std::int32_t>(reader.u32());
    image.dflash_context_frontier = reader.u32();
    image.mtp_draft_count         = reader.u32();
    const std::uint32_t tail      = reader.u32();
    image.rebuild_tail_begin      = reader.u32();
    image.state_count             = reader.u32();
    if (tail > 1) { throw std::invalid_argument("session sequence has invalid flags"); }
    image.tail_hidden_valid = tail != 0;
    for (TokenId& token : image.mtp_drafts) {
        token = std::bit_cast<TokenId>(reader.u32());
    }
    image.rebuild_work = read_work(reader);
    const std::uint32_t endpoint = reader.u32();
    if (endpoint > 1) { throw std::invalid_argument("session endpoint flag is invalid"); }
    if (endpoint != 0) { image.endpoint = read_checkpoint(reader); }
    const std::uint32_t rewrite = reader.u32();
    if (rewrite > 1) { throw std::invalid_argument("session rewrite flag is invalid"); }
    if (rewrite != 0) { image.rewrite = read_checkpoint(reader); }
    const std::uint32_t anchors = reader.u32();
    if (anchors > maximum_context) {
        throw std::invalid_argument("session long-anchor count is out of range");
    }
    image.long_anchors.reserve(anchors);
    for (std::uint32_t index = 0; index < anchors; ++index) {
        image.long_anchors.push_back(read_checkpoint(reader));
    }
    reader.finish();
}

void write_vision_item(Writer& writer, const VisionItem& item) {
    writer.u32(static_cast<std::uint32_t>(item.modality));
    writer.u32(std::bit_cast<std::uint32_t>(item.grid.temporal));
    writer.u32(std::bit_cast<std::uint32_t>(item.grid.height));
    writer.u32(std::bit_cast<std::uint32_t>(item.grid.width));
    writer.u64(item.patch_begin);
    writer.u64(item.patch_count);
    writer.raw(item.content_digest.data(), item.content_digest.size());
    writer.u32(static_cast<std::uint32_t>(item.timestamps.size()));
    for (const double timestamp : item.timestamps) {
        writer.u64(std::bit_cast<std::uint64_t>(timestamp));
    }
    writer.u32(static_cast<std::uint32_t>(item.token_spans.size()));
    for (const TokenSpan span : item.token_spans) {
        writer.u64(span.begin);
        writer.u64(span.count);
    }
}

VisionItem read_vision_item(Reader& reader, std::uint32_t maximum_context) {
    VisionItem item;
    item.modality      = static_cast<PromptModality>(reader.u32());
    item.grid.temporal = std::bit_cast<std::int32_t>(reader.u32());
    item.grid.height   = std::bit_cast<std::int32_t>(reader.u32());
    item.grid.width    = std::bit_cast<std::int32_t>(reader.u32());
    const std::uint64_t patch_begin = reader.u64();
    const std::uint64_t patch_count = reader.u64();
    if (patch_begin > std::numeric_limits<std::size_t>::max() ||
        patch_count > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("session Vision patch range exceeds size_t");
    }
    item.patch_begin = static_cast<std::size_t>(patch_begin);
    item.patch_count = static_cast<std::size_t>(patch_count);
    const auto digest = reader.raw(item.content_digest.size());
    std::copy(digest.begin(), digest.end(), item.content_digest.begin());
    const std::uint32_t timestamps = reader.u32();
    if (timestamps > kMaximumPromptVisionRawPatches) {
        throw std::invalid_argument("session Vision timestamp count is out of range");
    }
    item.timestamps.reserve(timestamps);
    for (std::uint32_t index = 0; index < timestamps; ++index) {
        item.timestamps.push_back(std::bit_cast<double>(reader.u64()));
    }
    const std::uint32_t spans = reader.u32();
    if (spans > maximum_context) {
        throw std::invalid_argument("session Vision span count is out of range");
    }
    item.token_spans.reserve(spans);
    for (std::uint32_t index = 0; index < spans; ++index) {
        const std::uint64_t begin = reader.u64();
        const std::uint64_t count = reader.u64();
        if (begin > maximum_context || count > maximum_context - begin) {
            throw std::invalid_argument("session Vision token span is out of range");
        }
        item.token_spans.push_back(
            TokenSpan{static_cast<std::size_t>(begin), static_cast<std::size_t>(count)});
    }
    return item;
}

std::vector<std::uint8_t> encode_identity(const ContinuationSessionImage& image) {
    Writer writer;
    writer.u32(kSectionVersion);
    writer.vector<TokenId>(image.ledger);
    writer.vector<std::uint8_t>(image.prefix_identity.token_types());
    for (std::size_t axis = 0; axis < 3; ++axis) {
        writer.vector<std::int32_t>(image.prefix_identity.position_axis(axis));
    }
    writer.vector<std::uint32_t>(image.prefix_identity.rewrite_execution_frontiers());
    const auto& digests = image.prefix_digests.image();
    writer.u32(static_cast<std::uint32_t>(digests.size()));
    for (const auto digest : digests) {
        writer.u64(digest[0]);
        writer.u64(digest[1]);
    }
    const auto& vision = image.prefix_identity.vision_items();
    writer.u32(static_cast<std::uint32_t>(vision.size()));
    for (const VisionItem& item : vision) { write_vision_item(writer, item); }
    return std::move(writer).take();
}

void decode_identity(std::span<const std::uint8_t> bytes, std::uint32_t maximum_context,
                     ContinuationSessionImage& image) {
    Reader reader(bytes);
    if (reader.u32() != kSectionVersion) {
        throw std::invalid_argument("unsupported session identity section version");
    }
    image.ledger = reader.vector<TokenId>(maximum_context, "ledger");
    auto token_types = reader.vector<std::uint8_t>(maximum_context, "token type");
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis = reader.vector<std::int32_t>(maximum_context, "position");
    }
    auto rewrite_frontiers = reader.vector<std::uint32_t>(maximum_context, "rewrite frontier");
    const std::uint32_t digest_count = reader.u32();
    if (digest_count > maximum_context + 1ULL) {
        throw std::invalid_argument("session digest count is out of range");
    }
    std::vector<std::array<std::uint64_t, 2>> digests(digest_count);
    for (auto& digest : digests) {
        digest[0] = reader.u64();
        digest[1] = reader.u64();
    }
    const std::uint32_t vision_count = reader.u32();
    if (vision_count > maximum_context) {
        throw std::invalid_argument("session Vision item count is out of range");
    }
    std::vector<VisionItem> vision;
    vision.reserve(vision_count);
    for (std::uint32_t index = 0; index < vision_count; ++index) {
        vision.push_back(read_vision_item(reader, maximum_context));
    }
    reader.finish();
    image.prefix_identity.restore(std::move(token_types), std::move(positions), std::move(vision),
                                  std::move(rewrite_frontiers));
    image.prefix_digests.restore(std::move(digests));
}

std::uint32_t page_count(std::uint32_t frontier, std::uint32_t page_tokens) {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / page_tokens;
}

std::vector<std::uint8_t> encode_payload(std::uint32_t count, std::uint64_t item_bytes,
                                         std::span<const std::uint8_t> payload) {
    Writer writer;
    writer.u32(kSectionVersion);
    writer.u32(count);
    writer.u64(item_bytes);
    writer.raw(payload.data(), payload.size());
    return std::move(writer).take();
}

std::vector<std::uint8_t> encode_kv(const SessionKVImage& kv, std::uint32_t page_tokens,
                                    std::uint64_t page_bytes) {
    const std::uint32_t pages = page_count(kv.frontier, page_tokens);
    Writer writer;
    writer.u32(kSectionVersion);
    writer.u32(kv.frontier);
    writer.u32(pages);
    writer.u64(page_bytes);
    writer.raw(kv.payload.data(), kv.payload.size());
    return std::move(writer).take();
}

std::vector<std::uint8_t> decode_payload(std::span<const std::uint8_t> bytes,
                                         std::uint32_t expected_count,
                                         std::uint64_t expected_item_bytes,
                                         const char* label) {
    Reader reader(bytes);
    if (reader.u32() != kSectionVersion || reader.u32() != expected_count ||
        reader.u64() != expected_item_bytes) {
        throw std::invalid_argument(std::string("session ") + label +
                                    " payload header does not match metadata");
    }
    if (expected_item_bytes != 0 &&
        expected_count > std::numeric_limits<std::size_t>::max() / expected_item_bytes) {
        throw std::invalid_argument(std::string("session ") + label + " payload size overflows");
    }
    const std::size_t size = static_cast<std::size_t>(expected_count * expected_item_bytes);
    const auto payload = reader.raw(size);
    reader.finish();
    return {payload.begin(), payload.end()};
}

void validate_runtime(const SessionRuntimeBinding& runtime) {
    if (static_cast<std::uint32_t>(runtime.kv_storage) >
            static_cast<std::uint32_t>(KvCacheStorage::Fp8KeyNvfp4Value) ||
        static_cast<std::uint32_t>(runtime.speculative_backend) >
            static_cast<std::uint32_t>(SpeculativeBackend::DFlash2) ||
        static_cast<std::uint32_t>(runtime.proposal_head) >
            static_cast<std::uint32_t>(ProposalHead::Optimized) ||
        runtime.token_domain == 0 || runtime.page_tokens != kPagedKVPageSize ||
        runtime.state_image_bytes == 0 ||
        runtime.text_page_bytes == 0 || runtime.state_layout_fingerprint == 0 ||
        runtime.text_layout_fingerprint == 0) {
        throw std::invalid_argument("session runtime binding is incomplete");
    }
    const bool has_backend_layout = runtime.backend_page_bytes != 0;
    if (has_backend_layout != (runtime.backend_layout_fingerprint != 0) ||
        (runtime.speculative_backend == SpeculativeBackend::None && has_backend_layout) ||
        (runtime.speculative_backend == SpeculativeBackend::Mtp && !has_backend_layout) ||
        (runtime.speculative_backend == SpeculativeBackend::None && runtime.draft_window != 0) ||
        (runtime.speculative_backend != SpeculativeBackend::None && runtime.draft_window == 0) ||
        (runtime.speculative_backend == SpeculativeBackend::Mtp &&
         runtime.draft_window > kMtpDecodeMaximumDrafts) ||
        ((runtime.speculative_backend == SpeculativeBackend::DFlash ||
          runtime.speculative_backend == SpeculativeBackend::DFlash2) &&
         runtime.draft_window > kDFlashDecodeMaximumDrafts)) {
        throw std::invalid_argument("session backend runtime binding is inconsistent");
    }
}

void validate_checkpoint(const SessionCheckpointImage& checkpoint, runtime::CheckpointKind kind,
                         std::uint32_t state_count, std::uint32_t execution_frontier) {
    if (checkpoint.kind != kind || checkpoint.frontier == 0 ||
        checkpoint.frontier > execution_frontier || checkpoint.state_index >= state_count ||
        checkpoint.rebuild_work.tokens != checkpoint.frontier) {
        throw std::invalid_argument("session checkpoint metadata is inconsistent");
    }
}

const runtime::SessionSnapshotSectionView&
require_section(const runtime::SessionSnapshotView& snapshot, Section type) {
    const auto* section = snapshot.find_section(static_cast<std::uint32_t>(type));
    if (section == nullptr) { throw std::invalid_argument("session image is missing a section"); }
    return *section;
}

} // namespace

std::uint64_t session_layout_fingerprint(const StateImageHostLayout& layout) noexcept {
    std::uint64_t hash = kFingerprintOffset;
    fingerprint_mix(hash, 0x6e696e6673746174ULL);
    fingerprint_mix(hash, layout.spec.linear.layers);
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.linear.conv_channels));
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.linear.conv_width));
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.linear.value_heads));
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.linear.value_head_dim));
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.linear.key_head_dim));
    fingerprint_mix(hash, static_cast<std::uint32_t>(layout.spec.linear.conv_dtype));
    fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.hidden));
    fingerprint_mix(hash, layout.spec.dflash_local.has_value() ? 1U : 0U);
    if (layout.spec.dflash_local) {
        fingerprint_mix(hash, layout.spec.dflash_local->layers);
        fingerprint_mix(hash, layout.spec.dflash_local->capacity);
        fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.dflash_local->kv_heads));
        fingerprint_mix(hash, std::bit_cast<std::uint32_t>(layout.spec.dflash_local->head_dim));
    }
    fingerprint_region(hash, layout.linear_conv);
    fingerprint_mix(hash, layout.linear_conv_layer_bytes);
    fingerprint_region(hash, layout.linear_recurrent);
    fingerprint_mix(hash, layout.linear_recurrent_layer_bytes);
    fingerprint_region(hash, layout.continuation_hidden);
    fingerprint_mix(hash, layout.dflash_local_k.has_value() ? 1U : 0U);
    if (layout.dflash_local_k) { fingerprint_region(hash, *layout.dflash_local_k); }
    fingerprint_mix(hash, layout.dflash_local_v.has_value() ? 1U : 0U);
    if (layout.dflash_local_v) { fingerprint_region(hash, *layout.dflash_local_v); }
    fingerprint_mix(hash, layout.dflash_local_layer_bytes);
    fingerprint_mix(hash, layout.image_bytes);
    return hash == 0 ? 1 : hash;
}

std::uint64_t session_layout_fingerprint(const HostKVPageLayout& layout) noexcept {
    std::uint64_t hash = kFingerprintOffset;
    fingerprint_mix(hash, 0x6e696e666b767061ULL);
    fingerprint_mix(hash, layout.geometry.page_tokens);
    fingerprint_mix(hash, static_cast<std::uint32_t>(layout.geometry.device_plane_order));
    fingerprint_mix(hash, layout.geometry.planes.size());
    for (const KVPlaneGeometry& plane : layout.geometry.planes) {
        fingerprint_mix(hash, static_cast<std::uint32_t>(plane.dtype));
        fingerprint_mix(hash, std::bit_cast<std::uint32_t>(plane.leading_extent));
        fingerprint_mix(hash, std::bit_cast<std::uint32_t>(plane.head_extent));
        fingerprint_mix(hash, plane.alignment);
    }
    fingerprint_mix(hash, layout.planes.size());
    for (const HostKVPlaneLayout& plane : layout.planes) {
        fingerprint_mix(hash, plane.offset);
        fingerprint_mix(hash, plane.page_payload_bytes);
        fingerprint_mix(hash, plane.head_payload_bytes);
    }
    fingerprint_mix(hash, layout.page_stride);
    return hash == 0 ? 1 : hash;
}

void validate_continuation_session_image(const ContinuationSessionImage& image,
                                         std::uint32_t maximum_context) {
    validate_runtime(image.runtime);
    const std::size_t tokens = image.ledger.size();
    if (maximum_context == 0 || tokens == 0 || tokens > maximum_context ||
        tokens > std::numeric_limits<std::uint32_t>::max() ||
        image.ledger_frontier != tokens || image.execution_frontier == 0 ||
        image.execution_frontier > tokens || tokens - image.execution_frontier > 1U ||
        image.prefix_identity.size() != tokens || image.prefix_digests.size() != tokens ||
        image.rebuild_work.tokens != image.execution_frontier ||
        image.rebuild_tail_begin > image.execution_frontier) {
        throw std::invalid_argument("session sequence frontiers or identity are inconsistent");
    }
    for (const TokenId token : image.ledger) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= image.runtime.token_domain) {
            throw std::invalid_argument("session ledger token is outside the model domain");
        }
    }
    const auto& vision_items = image.prefix_identity.vision_items();
    if (!image.runtime.vision_enabled && !vision_items.empty()) {
        throw std::invalid_argument("session contains Vision identity while Vision is disabled");
    }
    for (const VisionItem& item : vision_items) {
        if ((item.modality != PromptModality::Image && item.modality != PromptModality::Video) ||
            item.grid.temporal <= 0 || item.grid.height <= 0 || item.grid.width <= 0 ||
            item.patch_begin > kMaximumPromptVisionRawPatches ||
            item.patch_count > kMaximumPromptVisionRawPatches - item.patch_begin ||
            item.timestamps.size() > kMaximumPromptVisionRawPatches || item.token_spans.empty()) {
            throw std::invalid_argument("session Vision identity is invalid");
        }
        std::size_t previous_end = 0;
        for (const TokenSpan span : item.token_spans) {
            if (span.count == 0 || span.begin < previous_end || span.begin > tokens ||
                span.count > tokens - span.begin) {
                throw std::invalid_argument("session Vision token spans are invalid");
            }
            previous_end = span.begin + span.count;
        }
    }
    PreparedPromptData digest_source;
    digest_source.token_ids   = image.ledger;
    digest_source.token_types = image.prefix_identity.token_types();
    digest_source.positions.reserve(3U * tokens);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto& positions = image.prefix_identity.position_axis(axis);
        digest_source.positions.insert(digest_source.positions.end(), positions.begin(),
                                       positions.end());
    }
    digest_source.vision_items = vision_items;
    digest_source.identity.rewrite_execution_frontiers =
        image.prefix_identity.rewrite_execution_frontiers();
    PrefixShortlistDigests expected_digests;
    expected_digests.assign(digest_source);
    if (expected_digests.image() != image.prefix_digests.image()) {
        throw std::invalid_argument("session prefix digest image does not match its identity");
    }
    if (!image.endpoint && !image.rewrite && image.long_anchors.empty()) {
        throw std::invalid_argument("session continuation has no checkpoint");
    }
    if (image.long_anchors.size() > 64 || image.state_count == 0 || image.state_count > 64 ||
        image.state_count > 2U + image.long_anchors.size() ||
        image.runtime.state_image_bytes > std::numeric_limits<std::size_t>::max() ||
        image.state_count > std::numeric_limits<std::size_t>::max() /
                                image.runtime.state_image_bytes ||
        image.state_payload.size() != image.state_count * image.runtime.state_image_bytes) {
        throw std::invalid_argument("session StateImage payload has an invalid shape");
    }

    std::vector<bool> state_referenced(image.state_count, false);
    std::uint32_t maximum_checkpoint_frontier = 0;
    const auto include = [&](const SessionCheckpointImage& checkpoint) {
        state_referenced[checkpoint.state_index] = true;
        maximum_checkpoint_frontier = std::max(maximum_checkpoint_frontier, checkpoint.frontier);
    };
    if (image.endpoint) {
        validate_checkpoint(*image.endpoint, runtime::CheckpointKind::SessionEndpoint,
                            image.state_count, image.execution_frontier);
        if (image.endpoint->frontier != image.execution_frontier || image.endpoint->ordinal != 0) {
            throw std::invalid_argument("session endpoint metadata is inconsistent");
        }
        include(*image.endpoint);
    }
    if (image.rewrite) {
        if (image.rewrite->kind != runtime::CheckpointKind::TurnClosure &&
            image.rewrite->kind != runtime::CheckpointKind::ResponseReplay) {
            throw std::invalid_argument("session rewrite checkpoint kind is invalid");
        }
        validate_checkpoint(*image.rewrite, image.rewrite->kind, image.state_count,
                            image.execution_frontier);
        if (image.rewrite->ordinal != 0) {
            throw std::invalid_argument("session rewrite checkpoint has an ordinal");
        }
        include(*image.rewrite);
    }
    std::unordered_set<std::uint32_t> ordinals;
    for (const auto& anchor : image.long_anchors) {
        validate_checkpoint(anchor, runtime::CheckpointKind::LongAnchor, image.state_count,
                            image.execution_frontier);
        if (anchor.ordinal == 0 || !ordinals.insert(anchor.ordinal).second) {
            throw std::invalid_argument("session long-anchor ordinals are invalid");
        }
        include(anchor);
    }
    if (std::find(state_referenced.begin(), state_referenced.end(), false) !=
        state_referenced.end()) {
        throw std::invalid_argument("session contains an unreferenced StateImage payload");
    }

    if (image.text_kv.frontier < maximum_checkpoint_frontier ||
        image.text_kv.frontier > image.execution_frontier) {
        throw std::invalid_argument("session Text KV frontier is inconsistent");
    }
    const std::uint32_t text_pages = page_count(image.text_kv.frontier, image.runtime.page_tokens);
    if (image.runtime.text_page_bytes > std::numeric_limits<std::size_t>::max() ||
        text_pages > std::numeric_limits<std::size_t>::max() / image.runtime.text_page_bytes ||
        image.text_kv.payload.size() != text_pages * image.runtime.text_page_bytes) {
        throw std::invalid_argument("session Text KV payload has an invalid shape");
    }

    const bool has_backend_layout = image.runtime.backend_page_bytes != 0;
    std::uint32_t required_backend = 0;
    if (has_backend_layout) {
        required_backend = image.runtime.speculative_backend == SpeculativeBackend::Mtp
                               ? maximum_checkpoint_frontier - 1U
                               : maximum_checkpoint_frontier;
    }
    if (image.backend_kv.frontier < required_backend ||
        image.backend_kv.frontier > image.execution_frontier) {
        throw std::invalid_argument("session Backend KV frontier is inconsistent");
    }
    const std::uint32_t backend_pages =
        page_count(image.backend_kv.frontier, image.runtime.page_tokens);
    if (image.runtime.backend_page_bytes > std::numeric_limits<std::size_t>::max() ||
        (image.runtime.backend_page_bytes != 0 &&
         backend_pages > std::numeric_limits<std::size_t>::max() /
                             image.runtime.backend_page_bytes) ||
        image.backend_kv.payload.size() != backend_pages * image.runtime.backend_page_bytes) {
        throw std::invalid_argument("session Backend KV payload has an invalid shape");
    }
    if (!has_backend_layout &&
        (image.backend_kv.frontier != 0 || !image.backend_kv.payload.empty())) {
        throw std::invalid_argument("session without a Backend KV layout contains Backend KV");
    }

    if (image.runtime.speculative_backend == SpeculativeBackend::Mtp) {
        if (image.mtp_draft_count > image.runtime.draft_window ||
            image.mtp_draft_count > image.mtp_drafts.size() ||
            image.dflash_context_frontier != 0) {
            throw std::invalid_argument("session MTP metadata is inconsistent");
        }
        for (std::uint32_t index = 0; index < image.mtp_draft_count; ++index) {
            const TokenId token = image.mtp_drafts[index];
            if (token < 0 || static_cast<std::uint32_t>(token) >= image.runtime.token_domain) {
                throw std::invalid_argument("session MTP draft is outside the model domain");
            }
        }
    } else {
        if (image.mtp_draft_count != 0) {
            throw std::invalid_argument("non-MTP session contains MTP drafts");
        }
        const bool masked = image.runtime.speculative_backend == SpeculativeBackend::DFlash ||
                            image.runtime.speculative_backend == SpeculativeBackend::DFlash2;
        if ((!masked && image.dflash_context_frontier != 0) ||
            image.dflash_context_frontier > image.execution_frontier) {
            throw std::invalid_argument("session DFlash frontier is inconsistent");
        }
    }
}

std::vector<std::uint8_t>
encode_continuation_session_image(std::string_view model_binding,
                                  const ContinuationSessionImage& image,
                                  std::uint64_t max_total_bytes) {
    validate_continuation_session_image(image, static_cast<std::uint32_t>(image.ledger.size()));
    const auto runtime_bytes  = encode_runtime(image.runtime);
    const auto sequence_bytes = encode_sequence(image);
    const auto identity_bytes = encode_identity(image);
    const auto state_bytes =
        encode_payload(image.state_count, image.runtime.state_image_bytes, image.state_payload);
    const auto text_bytes =
        encode_kv(image.text_kv, image.runtime.page_tokens, image.runtime.text_page_bytes);
    const auto backend_bytes =
        encode_kv(image.backend_kv, image.runtime.page_tokens, image.runtime.backend_page_bytes);
    const std::array sections{
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::Runtime), 0,
                                             runtime_bytes},
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::Sequence), 0,
                                             sequence_bytes},
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::Identity), 0,
                                             identity_bytes},
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::States), 0,
                                             state_bytes},
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::TextKV), 0,
                                             text_bytes},
        runtime::SessionSnapshotSectionInput{static_cast<std::uint32_t>(Section::BackendKV), 0,
                                             backend_bytes},
    };
    return runtime::encode_session_snapshot(model_binding, kQwen3_5SessionSchemaVersion, sections,
                                            max_total_bytes);
}

ContinuationSessionImage
decode_continuation_session_image(std::span<const std::uint8_t> bytes,
                                  std::string_view expected_model_binding,
                                  const SessionRuntimeBinding& expected_runtime,
                                  std::uint32_t maximum_context,
                                  std::uint64_t max_total_bytes) {
    validate_runtime(expected_runtime);
    if (maximum_context == 0) {
        throw std::invalid_argument("session maximum context is zero");
    }
    const runtime::SessionSnapshotView snapshot = runtime::decode_session_snapshot(
        bytes, expected_model_binding, max_total_bytes);
    if (snapshot.model_schema_version != kQwen3_5SessionSchemaVersion ||
        snapshot.sections.size() != 6) {
        throw std::invalid_argument("session image has an unsupported Qwen3.5 schema");
    }
    ContinuationSessionImage result;
    result.runtime = decode_runtime(require_section(snapshot, Section::Runtime).bytes);
    if (result.runtime != expected_runtime) {
        throw std::invalid_argument("session runtime binding does not match this Program");
    }
    decode_sequence(require_section(snapshot, Section::Sequence).bytes, maximum_context, result);
    decode_identity(require_section(snapshot, Section::Identity).bytes, maximum_context, result);
    result.state_payload =
        decode_payload(require_section(snapshot, Section::States).bytes, result.state_count,
                       result.runtime.state_image_bytes, "StateImage");
    // Frontiers live in the payload section header during decode so they cannot be inferred from
    // sequence metadata. Read them first through the small KV-specific envelope.
    const auto decode_kv = [&](Section type, std::uint64_t page_bytes, SessionKVImage& kv) {
        Reader reader(require_section(snapshot, type).bytes);
        if (reader.u32() != kSectionVersion) {
            throw std::invalid_argument("unsupported session KV section version");
        }
        kv.frontier = reader.u32();
        const std::uint32_t pages = reader.u32();
        if (reader.u64() != page_bytes ||
            pages != page_count(kv.frontier, result.runtime.page_tokens) ||
            (page_bytes != 0 && pages > std::numeric_limits<std::size_t>::max() / page_bytes)) {
            throw std::invalid_argument("session KV payload header is inconsistent");
        }
        const auto payload = reader.raw(static_cast<std::size_t>(pages * page_bytes));
        reader.finish();
        kv.payload.assign(payload.begin(), payload.end());
    };
    decode_kv(Section::TextKV, result.runtime.text_page_bytes, result.text_kv);
    decode_kv(Section::BackendKV, result.runtime.backend_page_bytes, result.backend_kv);
    validate_continuation_session_image(result, maximum_context);
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
