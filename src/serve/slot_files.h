#pragma once

// Clients name snapshot files, but every file must remain a direct child of
// --slot-save-path. Keep the accepted language deliberately narrower than a path.

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

inline constexpr std::size_t kSlotFilenameMaxBytes = 128;

[[nodiscard]] inline std::optional<std::string> sanitize_slot_filename(std::string_view name) {
    if (name.empty() || name.size() > kSlotFilenameMaxBytes || name.front() == '.') {
        return std::nullopt;
    }
    for (const char value : name) {
        const bool accepted = (value >= 'a' && value <= 'z') ||
                              (value >= 'A' && value <= 'Z') ||
                              (value >= '0' && value <= '9') || value == '.' || value == '_' ||
                              value == '-';
        if (!accepted) { return std::nullopt; }
    }
    return std::string(name);
}

} // namespace ninfer::serve
