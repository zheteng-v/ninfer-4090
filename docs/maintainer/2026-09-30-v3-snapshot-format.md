# v3 session snapshot container — 2026-09-30

This note records the first state/prefix-persistence increment for the v3/sm89 integration line.
It defines and validates the model-independent disk container. It does **not** yet make the public
slot save/restore declarations operational.

## Upstream audit and prior-art review

The iteration started from `sync/2026-09-30-v3-sm89@68609b72`. A fresh remote audit found no new
source head to integrate:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architectural source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no new persistence change to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the production Ada baseline |

The old fork implementation was reviewed from `beaeb70a` (slot save/restore), `8e478945`
(digests), `8093c640` (eviction auto-save), and later catalog adaptations. Its `NINFSES1` version-3
image is host-endian and serializes deleted target-private Program state. It has model binding and
component digests, but no portable framing or whole-image checksum. Replaying it into the v3
Program would preserve the wrong ownership boundary and make corrupted or cross-architecture
images harder to reject safely.

The v3 tree also still carries public `Engine::save_slot`, `restore_slot`, `erase_slot`, and
`slot_states` declarations without corresponding implementations. Their presence is not evidence
that v3 persistence works. They will be implemented only after the model Program can export and
atomically import a complete validated state image.

## Container contract

Commit `f4b11733` introduces `runtime/session_snapshot.{h,cpp}` with these rules:

- magic `NINFSNP3`, container version 1, and an explicit little-endian marker;
- a nonempty model/artifact binding plus a positive model-state schema version;
- one to 64 unique, typed sections with 64-byte canonical alignment;
- exact total, metadata, payload, section, padding, and trailing-byte bounds;
- reserved flags and fields must remain zero, so future extensions fail closed;
- CRC64-ECMA for the complete image and for every section;
- a caller-controlled total-size limit, defaulting to 64 GiB;
- zero-copy decoded section views borrowing from the caller-owned image;
- deliberate rejection of old `NINFSES1` images.

CRC64 is an accidental-corruption detector, not an authenticity or trust mechanism. Snapshot files
remain local state and must not be accepted from an untrusted source merely because their checksum
passes.

The on-disk image is:

```text
fixed header (80 bytes)
model binding (1..4096 bytes)
section directory (40 bytes per entry)
zero padding to 64-byte boundary
section 1 + zero padding
...
section N + zero padding
```

Encoding is deterministic. Decoding accepts only this canonical layout rather than trying to
repair ambiguous images.

## Focused validation

Release configuration used CUDA 13.1, GCC 14, and `CMAKE_CUDA_ARCHITECTURES=89`. The targeted
`ninfer_session_snapshot_test` builds and passes. It covers:

- deterministic encoding and an exact three-section metadata/payload round trip;
- missing-section lookup;
- payload and reserved-header corruption;
- wrong model binding;
- truncation and trailing data;
- duplicate section types and invalid model bindings;
- encode/decode size limits;
- rejection of `NINFSES1` legacy magic.

This is a host-side format gate. It does not require or consume a production GPU window.

## Remaining integration sequence

The next persistence increments are deliberately separate:

1. [complete] define stable Qwen3.5 v3 section IDs and a complete owned staging image for the
   Program ledger, resident identity/digests, KV/state bytes, checkpoints, and continuation data;
2. [complete] validate every section and required relationship into a temporary import object
   before changing a live Program;
3. capture and restore the physical Program stores through model-independent contracts, including
   rollback/eviction tests;
4. add atomic file publication (`temporary file -> flush/fsync -> rename -> directory fsync`) and
   bounded reads;
5. implement the public Engine methods and Serve slot routes, then run save/restart/restore and
   eviction round trips on the official v3 artifact;
6. only after those gates pass, decide whether a separate, explicit offline v2 migration tool has
   enough value. Runtime auto-detection or silent conversion remains out of scope.

Production remains on the validated v2 service throughout these increments.
