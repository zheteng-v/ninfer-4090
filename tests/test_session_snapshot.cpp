#include "runtime/session_snapshot.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) { return true; }
    return false;
}

} // namespace

int main() {
    using ninfer::runtime::SessionSnapshotSectionInput;
    using ninfer::runtime::decode_session_snapshot;
    using ninfer::runtime::encode_session_snapshot;

    int failures = 0;
    const std::array<std::uint8_t, 5> ledger{1, 2, 3, 4, 5};
    const std::array<std::uint8_t, 3> identity{0xf0, 0x0d, 0x42};
    const std::vector<std::uint8_t> state(129, 0xa5);
    const std::array sections{
        SessionSnapshotSectionInput{1, 0, ledger},
        SessionSnapshotSectionInput{7, 0, identity},
        SessionSnapshotSectionInput{99, 0, state},
    };

    const auto image = encode_session_snapshot("qwen3.5-27b-v3:sm89", 3, sections);
    const auto again = encode_session_snapshot("qwen3.5-27b-v3:sm89", 3, sections);
    failures += check(image == again, "snapshot encoding is not deterministic");

    const auto decoded = decode_session_snapshot(image, "qwen3.5-27b-v3:sm89");
    failures += check(decoded.model_schema_version == 3 &&
                          decoded.model_binding == "qwen3.5-27b-v3:sm89" &&
                          decoded.sections.size() == 3,
                      "snapshot metadata did not round-trip");
    failures += check(decoded.find_section(1) != nullptr &&
                          std::equal(decoded.find_section(1)->bytes.begin(),
                                     decoded.find_section(1)->bytes.end(), ledger.begin()) &&
                          decoded.find_section(7) != nullptr &&
                          std::equal(decoded.find_section(7)->bytes.begin(),
                                     decoded.find_section(7)->bytes.end(), identity.begin()) &&
                          decoded.find_section(99) != nullptr &&
                          std::equal(decoded.find_section(99)->bytes.begin(),
                                     decoded.find_section(99)->bytes.end(), state.begin()),
                      "snapshot section payloads did not round-trip");
    failures += check(decoded.find_section(1234) == nullptr,
                      "missing snapshot section was reported as present");

    auto corrupt_payload = image;
    corrupt_payload.back() ^= 0x01;
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(corrupt_payload,
                                                        "qwen3.5-27b-v3:sm89");
                      }),
                      "payload corruption was accepted");

    auto corrupt_header = image;
    corrupt_header[72] = 1;
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(corrupt_header,
                                                        "qwen3.5-27b-v3:sm89");
                      }),
                      "reserved header corruption was accepted");
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(image, "another-model");
                      }),
                      "wrong model binding was accepted");
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(
                              std::span<const std::uint8_t>(image).first(image.size() - 1),
                              "qwen3.5-27b-v3:sm89");
                      }),
                      "truncated snapshot was accepted");
    auto trailing = image;
    trailing.push_back(0);
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(trailing, "qwen3.5-27b-v3:sm89");
                      }),
                      "snapshot with trailing data was accepted");

    const std::array duplicate_sections{
        SessionSnapshotSectionInput{4, 0, ledger},
        SessionSnapshotSectionInput{4, 0, identity},
    };
    failures += check(throws<std::invalid_argument>([&] {
                          (void)encode_session_snapshot("qwen3.5-27b-v3:sm89", 3,
                                                        duplicate_sections);
                      }),
                      "duplicate section types were accepted");
    failures += check(throws<std::invalid_argument>([&] {
                          (void)encode_session_snapshot("", 3, sections);
                      }),
                      "empty model binding was accepted");
    const std::string overlong_binding(4097, 'x');
    failures += check(throws<std::invalid_argument>([&] {
                          (void)encode_session_snapshot(overlong_binding, 3, sections);
                      }),
                      "overlong model binding was accepted");
    failures += check(throws<std::length_error>([&] {
                          (void)encode_session_snapshot("qwen3.5-27b-v3:sm89", 3, sections,
                                                        image.size() - 1);
                      }),
                      "encode size limit was not enforced");
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(image, "qwen3.5-27b-v3:sm89",
                                                        image.size() - 1);
                      }),
                      "decode size limit was not enforced");

    std::vector<std::uint8_t> legacy(80, 0);
    const std::array<std::uint8_t, 8> legacy_magic{'N', 'I', 'N', 'F', 'S', 'E', 'S', '1'};
    std::copy(legacy_magic.begin(), legacy_magic.end(), legacy.begin());
    failures += check(throws<std::invalid_argument>([&] {
                          (void)decode_session_snapshot(legacy);
                      }),
                      "legacy snapshot magic was accepted");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
