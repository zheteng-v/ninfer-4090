#include "runtime/session_file.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace ninfer::runtime {
namespace {

std::invalid_argument file_error(const char* operation, const std::filesystem::path& path,
                                 int error = errno) {
    return std::invalid_argument(std::string(operation) + " '" + path.string() + "': " +
                                 std::strerror(error));
}

class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) noexcept : fd_(fd) {}
    ~FileDescriptor() {
        if (fd_ >= 0) { (void)::close(fd_); }
    }
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.release()) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) { (void)::close(fd_); }
            fd_ = other.release();
        }
        return *this;
    }
    FileDescriptor(const FileDescriptor&)            = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

private:
    int fd_;
};

void write_all(int fd, std::span<const std::uint8_t> bytes,
               const std::filesystem::path& path) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t remaining = bytes.size() - offset;
        const std::size_t chunk = std::min<std::size_t>(
            remaining, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
        const ssize_t written = ::write(fd, bytes.data() + offset, chunk);
        if (written > 0) {
            offset += static_cast<std::size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) { continue; }
        throw file_error("failed to write session snapshot", path);
    }
}

} // namespace

void write_session_file_atomic(const std::filesystem::path& path,
                               std::span<const std::uint8_t> bytes) {
    if (path.empty()) { throw std::invalid_argument("session snapshot path is empty"); }
    if (bytes.empty()) { throw std::invalid_argument("session snapshot payload is empty"); }

    const std::filesystem::path parent = path.has_parent_path() ? path.parent_path() : ".";
    const std::string filename         = path.filename().string();
    if (filename.empty() || filename == "." || filename == "..") {
        throw std::invalid_argument("session snapshot path has no filename");
    }

    FileDescriptor directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.get() < 0) {
        throw file_error("failed to open session snapshot directory", parent);
    }

    static std::atomic<std::uint64_t> sequence{0};
    std::string temporary;
    FileDescriptor staging;
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        temporary = "." + filename + ".tmp." + std::to_string(::getpid()) + "." +
                    std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        const int fd = ::openat(directory.get(), temporary.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) {
            staging = FileDescriptor(fd);
            break;
        }
        if (errno != EEXIST) {
            throw file_error("failed to create session snapshot staging file", path);
        }
    }
    if (staging.get() < 0) {
        throw std::invalid_argument("failed to allocate a unique session snapshot staging file");
    }

    bool published = false;
    try {
        write_all(staging.get(), bytes, path);
        if (::fsync(staging.get()) != 0) {
            throw file_error("failed to synchronize session snapshot", path);
        }
        const int raw = staging.release();
        if (::close(raw) != 0) { throw file_error("failed to close session snapshot", path); }
        if (::renameat(directory.get(), temporary.c_str(), directory.get(), filename.c_str()) != 0) {
            throw file_error("failed to publish session snapshot", path);
        }
        published = true;
        if (::fsync(directory.get()) != 0) {
            throw file_error("failed to synchronize session snapshot directory", parent);
        }
    } catch (...) {
        if (!published) { (void)::unlinkat(directory.get(), temporary.c_str(), 0); }
        throw;
    }
}

std::vector<std::uint8_t> read_session_file(const std::filesystem::path& path,
                                            std::uint64_t max_bytes) {
    if (path.empty()) { throw std::invalid_argument("session snapshot path is empty"); }
    if (max_bytes == 0) { throw std::invalid_argument("session snapshot size limit is zero"); }
    FileDescriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (file.get() < 0) { throw file_error("session snapshot file is unavailable", path); }

    struct stat status {};
    if (::fstat(file.get(), &status) != 0) {
        throw file_error("failed to inspect session snapshot file", path);
    }
    if (!S_ISREG(status.st_mode)) {
        throw std::invalid_argument("session snapshot is not a regular file");
    }
    if (status.st_size <= 0) { throw std::invalid_argument("session snapshot file is empty"); }
    const std::uint64_t size = static_cast<std::uint64_t>(status.st_size);
    if (size > max_bytes || size > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("session snapshot file exceeds the size limit");
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(file.get(), bytes.data() + offset, bytes.size() - offset);
        if (count > 0) {
            offset += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && errno == EINTR) { continue; }
        if (count == 0) {
            throw std::invalid_argument("session snapshot file was truncated while reading");
        }
        throw file_error("failed to read session snapshot file", path);
    }
    std::uint8_t extra = 0;
    ssize_t trailing   = -1;
    do {
        trailing = ::read(file.get(), &extra, 1);
    } while (trailing < 0 && errno == EINTR);
    if (trailing != 0) {
        if (trailing > 0) {
            throw std::invalid_argument("session snapshot file changed while reading");
        }
        throw file_error("failed to finish reading session snapshot file", path);
    }
    return bytes;
}

} // namespace ninfer::runtime
