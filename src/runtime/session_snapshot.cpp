#include "runtime/session_snapshot.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace ninfer::runtime {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'N', 'I', 'N', 'F', 'S', 'N', 'P', '3'};
constexpr std::uint32_t kContainerVersion = 1;
constexpr std::uint32_t kEndianMarker     = 0x01020304U;
constexpr std::size_t kFixedHeaderBytes  = 80;
constexpr std::size_t kDirectoryBytes    = 40;
constexpr std::size_t kChecksumOffset    = 64;
constexpr std::size_t kAlignment         = 64;
constexpr std::size_t kMaxBindingBytes   = 4096;
constexpr std::size_t kMaxSections       = 64;
constexpr std::uint64_t kCrcPolynomial   = 0x42F0E1EBA9EA3693ULL;

[[noreturn]] void malformed(const char* detail) {
    throw std::invalid_argument(std::string("invalid session snapshot: ") + detail);
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* detail) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw std::overflow_error(detail); }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* detail) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        throw std::overflow_error(detail);
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value) {
    return checked_add(value, kAlignment - 1, "session snapshot alignment overflow") &
           ~(static_cast<std::uint64_t>(kAlignment) - 1);
}

void put_u32(std::span<std::uint8_t> output, std::size_t offset, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        output[offset++] = static_cast<std::uint8_t>(value >> shift);
    }
}

void put_u64(std::span<std::uint8_t> output, std::size_t offset, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        output[offset++] = static_cast<std::uint8_t>(value >> shift);
    }
}

std::uint32_t get_u32(std::span<const std::uint8_t> input, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8) {
        value |= static_cast<std::uint32_t>(input[offset++]) << shift;
    }
    return value;
}

std::uint64_t get_u64(std::span<const std::uint8_t> input, std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8) {
        value |= static_cast<std::uint64_t>(input[offset++]) << shift;
    }
    return value;
}

const std::array<std::uint64_t, 256>& crc_table() {
    static const auto table = [] {
        std::array<std::uint64_t, 256> result{};
        for (std::size_t index = 0; index < result.size(); ++index) {
            std::uint64_t crc = static_cast<std::uint64_t>(index) << 56;
            for (int bit = 0; bit < 8; ++bit) {
                crc = (crc << 1) ^ ((crc & (1ULL << 63)) != 0 ? kCrcPolynomial : 0);
            }
            result[index] = crc;
        }
        return result;
    }();
    return table;
}

std::uint64_t crc64_update(std::uint64_t crc, std::span<const std::uint8_t> bytes) {
    const auto& table = crc_table();
    for (const std::uint8_t byte : bytes) {
        crc = (crc << 8) ^ table[((crc >> 56) ^ byte) & 0xffU];
    }
    return crc;
}

std::uint64_t crc64(std::span<const std::uint8_t> bytes) { return crc64_update(0, bytes); }

std::uint64_t image_crc64(std::span<const std::uint8_t> image) {
    std::uint64_t crc = crc64_update(0, image.first(kChecksumOffset));
    const std::array<std::uint8_t, sizeof(std::uint64_t)> zero{};
    crc = crc64_update(crc, zero);
    return crc64_update(crc, image.subspan(kChecksumOffset + zero.size()));
}

void validate_common_input(std::string_view model_binding, std::uint32_t model_schema_version,
                           std::size_t section_count, std::uint64_t max_total_bytes) {
    if (model_binding.empty() || model_binding.size() > kMaxBindingBytes ||
        model_binding.find('\0') != std::string_view::npos) {
        throw std::invalid_argument("session snapshot model binding must contain 1..4096 non-NUL bytes");
    }
    if (model_schema_version == 0) {
        throw std::invalid_argument("session snapshot model schema version must be positive");
    }
    if (section_count == 0 || section_count > kMaxSections) {
        throw std::invalid_argument("session snapshot must contain 1..64 sections");
    }
    if (max_total_bytes < kFixedHeaderBytes) {
        throw std::invalid_argument("session snapshot size limit is too small");
    }
}

} // namespace

const SessionSnapshotSectionView* SessionSnapshotView::find_section(std::uint32_t type) const noexcept {
    const auto found = std::find_if(sections.begin(), sections.end(),
                                    [type](const auto& section) { return section.type == type; });
    return found == sections.end() ? nullptr : &*found;
}

std::vector<std::uint8_t>
encode_session_snapshot(std::string_view model_binding, std::uint32_t model_schema_version,
                        std::span<const SessionSnapshotSectionInput> sections,
                        std::uint64_t max_total_bytes) {
    validate_common_input(model_binding, model_schema_version, sections.size(), max_total_bytes);

    std::unordered_set<std::uint32_t> section_types;
    std::uint64_t cursor = align_up(checked_add(
        checked_add(kFixedHeaderBytes, model_binding.size(), "snapshot metadata overflow"),
        checked_mul(sections.size(), kDirectoryBytes, "snapshot directory overflow"),
        "snapshot metadata overflow"));
    const std::uint64_t payload_offset = cursor;
    for (const auto& section : sections) {
        if (section.type == 0 || !section_types.insert(section.type).second) {
            throw std::invalid_argument("session snapshot section types must be unique and nonzero");
        }
        if (section.flags != 0) {
            throw std::invalid_argument("session snapshot section flags are reserved");
        }
        cursor = checked_add(cursor, section.bytes.size(), "session snapshot payload overflow");
        cursor = align_up(cursor);
    }
    if (cursor > max_total_bytes || cursor > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("session snapshot exceeds configured size limit");
    }

    std::vector<std::uint8_t> image(static_cast<std::size_t>(cursor), 0);
    std::copy(kMagic.begin(), kMagic.end(), image.begin());
    put_u32(image, 8, kContainerVersion);
    put_u32(image, 12, kEndianMarker);
    put_u32(image, 16, kFixedHeaderBytes);
    put_u32(image, 20, 0);
    put_u64(image, 24, cursor);
    put_u64(image, 32, payload_offset);
    put_u64(image, 40, cursor - payload_offset);
    put_u32(image, 48, static_cast<std::uint32_t>(model_binding.size()));
    put_u32(image, 52, model_schema_version);
    put_u32(image, 56, static_cast<std::uint32_t>(sections.size()));
    put_u32(image, 60, 0);
    put_u64(image, 64, 0);
    put_u64(image, 72, 0);
    std::memcpy(image.data() + kFixedHeaderBytes, model_binding.data(), model_binding.size());

    cursor = payload_offset;
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto& section      = sections[index];
        const std::size_t entry  = kFixedHeaderBytes + model_binding.size() +
                                  index * kDirectoryBytes;
        put_u32(image, entry, section.type);
        put_u32(image, entry + 4, section.flags);
        put_u64(image, entry + 8, cursor);
        put_u64(image, entry + 16, section.bytes.size());
        put_u64(image, entry + 24, crc64(section.bytes));
        put_u64(image, entry + 32, 0);
        if (!section.bytes.empty()) {
            std::memcpy(image.data() + static_cast<std::size_t>(cursor), section.bytes.data(),
                        section.bytes.size());
        }
        cursor = align_up(checked_add(cursor, section.bytes.size(), "snapshot payload overflow"));
    }
    put_u64(image, kChecksumOffset, image_crc64(image));
    return image;
}

SessionSnapshotView decode_session_snapshot(std::span<const std::uint8_t> image,
                                            std::string_view expected_model_binding,
                                            std::uint64_t max_total_bytes) {
    if (image.size() < kFixedHeaderBytes) { malformed("truncated header"); }
    if (!std::equal(kMagic.begin(), kMagic.end(), image.begin())) { malformed("bad magic"); }
    if (get_u32(image, 8) != kContainerVersion) { malformed("unsupported container version"); }
    if (get_u32(image, 12) != kEndianMarker) { malformed("bad endian marker"); }
    if (get_u32(image, 16) != kFixedHeaderBytes) { malformed("bad fixed-header size"); }
    if (get_u32(image, 20) != 0 || get_u32(image, 60) != 0 || get_u64(image, 72) != 0) {
        malformed("reserved header field is nonzero");
    }

    const std::uint64_t total_bytes    = get_u64(image, 24);
    const std::uint64_t payload_offset = get_u64(image, 32);
    const std::uint64_t payload_bytes  = get_u64(image, 40);
    const std::uint32_t binding_bytes  = get_u32(image, 48);
    const std::uint32_t schema_version = get_u32(image, 52);
    const std::uint32_t section_count  = get_u32(image, 56);
    if (total_bytes != image.size()) { malformed("declared size does not match input"); }
    if (total_bytes > max_total_bytes) { malformed("configured size limit exceeded"); }
    if (binding_bytes == 0 || binding_bytes > kMaxBindingBytes || schema_version == 0 ||
        section_count == 0 || section_count > kMaxSections) {
        malformed("invalid model or section metadata");
    }

    std::uint64_t metadata_end;
    try {
        metadata_end = checked_add(
            checked_add(kFixedHeaderBytes, binding_bytes, "snapshot metadata overflow"),
            checked_mul(section_count, kDirectoryBytes, "snapshot directory overflow"),
            "snapshot metadata overflow");
    } catch (const std::overflow_error&) { malformed("metadata size overflow"); }
    if (metadata_end > total_bytes || payload_offset != align_up(metadata_end) ||
        payload_offset > total_bytes || payload_bytes != total_bytes - payload_offset) {
        malformed("noncanonical payload bounds");
    }
    if (get_u64(image, kChecksumOffset) != image_crc64(image)) {
        malformed("whole-image checksum mismatch");
    }

    const auto binding_span = image.subspan(kFixedHeaderBytes, binding_bytes);
    if (std::find(binding_span.begin(), binding_span.end(), 0) != binding_span.end()) {
        malformed("model binding contains NUL");
    }
    const std::string_view binding(reinterpret_cast<const char*>(binding_span.data()),
                                   binding_span.size());
    if (!expected_model_binding.empty() && binding != expected_model_binding) {
        malformed("model binding mismatch");
    }

    SessionSnapshotView result{
        .model_schema_version = schema_version,
        .model_binding        = binding,
    };
    result.sections.reserve(section_count);
    std::unordered_set<std::uint32_t> section_types;
    std::uint64_t cursor = payload_offset;
    for (std::size_t index = 0; index < section_count; ++index) {
        const std::size_t entry = kFixedHeaderBytes + binding_bytes + index * kDirectoryBytes;
        const std::uint32_t type = get_u32(image, entry);
        const std::uint32_t flags = get_u32(image, entry + 4);
        const std::uint64_t offset = get_u64(image, entry + 8);
        const std::uint64_t size = get_u64(image, entry + 16);
        const std::uint64_t checksum = get_u64(image, entry + 24);
        if (type == 0 || !section_types.insert(type).second) {
            malformed("duplicate or zero section type");
        }
        if (flags != 0 || get_u64(image, entry + 32) != 0) {
            malformed("reserved section field is nonzero");
        }
        if (offset != cursor || size > total_bytes - cursor) {
            malformed("noncanonical section bounds");
        }
        const auto bytes = image.subspan(static_cast<std::size_t>(offset),
                                         static_cast<std::size_t>(size));
        if (crc64(bytes) != checksum) { malformed("section checksum mismatch"); }
        result.sections.push_back(SessionSnapshotSectionView{
            .type = type,
            .flags = flags,
            .bytes = bytes,
        });
        try {
            cursor = align_up(checked_add(cursor, size, "snapshot section overflow"));
        } catch (const std::overflow_error&) { malformed("section size overflow"); }
        if (cursor > total_bytes) { malformed("section padding exceeds input"); }
    }
    if (cursor != total_bytes) { malformed("trailing bytes after final section"); }
    return result;
}

} // namespace ninfer::runtime
