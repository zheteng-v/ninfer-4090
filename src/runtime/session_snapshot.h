#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::runtime {

inline constexpr std::uint64_t kDefaultSessionSnapshotLimit = 64ULL * 1024ULL * 1024ULL * 1024ULL;

struct SessionSnapshotSectionInput {
    std::uint32_t type{};
    std::uint32_t flags{};
    std::span<const std::uint8_t> bytes;
};

struct SessionSnapshotSectionView {
    std::uint32_t type{};
    std::uint32_t flags{};
    std::span<const std::uint8_t> bytes;
};

struct SessionSnapshotView {
    std::uint32_t model_schema_version{};
    std::string_view model_binding;
    std::vector<SessionSnapshotSectionView> sections;

    [[nodiscard]] const SessionSnapshotSectionView* find_section(std::uint32_t type) const noexcept;
};

struct RetainedSessionSnapshot {
    std::vector<std::uint8_t> bytes;
    std::uint32_t tokens = 0;
    std::string session_digest;
};

// Encodes a deterministic, little-endian container. Checksums detect accidental corruption; they
// are not an authentication mechanism. The returned image owns all encoded bytes.
[[nodiscard]] std::vector<std::uint8_t>
encode_session_snapshot(std::string_view model_binding, std::uint32_t model_schema_version,
                        std::span<const SessionSnapshotSectionInput> sections,
                        std::uint64_t max_total_bytes = kDefaultSessionSnapshotLimit);

// The returned views borrow from image. Passing an expected binding makes cross-model restores
// fail before any section is exposed; an empty expected binding is useful for offline inspection.
[[nodiscard]] SessionSnapshotView
decode_session_snapshot(std::span<const std::uint8_t> image,
                        std::string_view expected_model_binding = {},
                        std::uint64_t max_total_bytes = kDefaultSessionSnapshotLimit);

} // namespace ninfer::runtime
