#include "runtime/session_file.h"

#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class Function>
bool rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    char pattern[] = "/tmp/ninfer-session-file-XXXXXX";
    char* directory = ::mkdtemp(pattern);
    if (directory == nullptr) {
        std::cerr << "mkdtemp failed\n";
        return 1;
    }
    const std::filesystem::path root(directory);
    const std::filesystem::path path = root / "slot.bin";

    const std::vector<std::uint8_t> first{1, 2, 3, 4};
    ninfer::runtime::write_session_file_atomic(path, first);
    failures += check(ninfer::runtime::read_session_file(path) == first,
                      "initial atomic write did not round-trip");

    const std::vector<std::uint8_t> replacement{9, 8, 7};
    ninfer::runtime::write_session_file_atomic(path, replacement);
    failures += check(ninfer::runtime::read_session_file(path) == replacement,
                      "replacement did not publish atomically");
    failures += check(rejects([&] {
                          ninfer::runtime::write_session_file_atomic(
                              path, std::span<const std::uint8_t>{});
                      }),
                      "empty writes must be rejected");
    failures += check(ninfer::runtime::read_session_file(path) == replacement,
                      "a rejected write damaged the previous snapshot");
    failures += check(rejects([&] { (void)ninfer::runtime::read_session_file(root / "missing"); }),
                      "missing files must be rejected");

    const std::filesystem::path empty = root / "empty.bin";
    const int empty_fd = ::open(empty.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (empty_fd >= 0) { (void)::close(empty_fd); }
    failures += check(empty_fd >= 0 &&
                          rejects([&] { (void)ninfer::runtime::read_session_file(empty); }),
                      "empty files must be rejected");

    const std::filesystem::path large = root / "large.bin";
    const int large_fd = ::open(large.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    const bool sized = large_fd >= 0 && ::ftruncate(large_fd, 5) == 0;
    if (large_fd >= 0) { (void)::close(large_fd); }
    failures += check(sized && rejects([&] { (void)ninfer::runtime::read_session_file(large, 4); }),
                      "oversized files must be rejected before allocation");
    failures += check(rejects([&] { (void)ninfer::runtime::read_session_file(root); }),
                      "directories must not be accepted as snapshots");

    std::size_t temporary_files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
            ++temporary_files;
        }
    }
    failures += check(temporary_files == 0, "atomic publication left a staging file behind");

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    failures += check(!cleanup_error, "temporary test directory cleanup failed");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
