#pragma once

#include "runtime/session_snapshot.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::runtime {

void write_session_file_atomic(const std::filesystem::path& path,
                               std::span<const std::uint8_t> bytes);

[[nodiscard]] std::vector<std::uint8_t>
read_session_file(const std::filesystem::path& path,
                  std::uint64_t max_bytes = kDefaultSessionSnapshotLimit);

} // namespace ninfer::runtime
