# v3 durable Engine session slots — 2026-09-30

This note records the fifth session-persistence increment for the v3/sm89 integration line. It
connects the validated Qwen3.5 continuation image to the public Engine save, restore, erase, list,
and eviction-auto-save contracts through a crash-durable local file transaction. Serve routes are
still deferred; this is the Engine boundary they will call.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@b4b13fb2`. A fresh fetch found no source-head
change:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no durable-session implementation to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the Ada production baseline |

Neroued PR #335 remains open and **watch / benchmark first**. Its hybrid prefix-cache policy does
not supply exact continuation serialization, atomic disk publication, or the public Engine slot
transaction, so it does not replace this work.

## Ownership and durability

Program now exposes read-only continuation depth, digest, checkpoint, and summary views in addition
to physical export/import. EngineCore owns catalog-slot addressing, digest preconditions, file
bindings, published slot state, replacement rollback, and eviction capture. Engine owns file I/O
and the background writer, keeping all disk work outside the Program execution mutex.

The file publisher uses a unique same-directory `O_EXCL` staging file, complete EINTR-safe writes,
file `fsync`, `renameat`, and directory `fsync`. It removes an unpublished staging file on failure.
The reader admits only a non-empty bounded regular file, allocates only after `fstat`, performs a
complete read, and rejects concurrent truncation or growth. Malformed container errors are public
`std::invalid_argument` failures, matching the Engine contract.

Every image binds to the exact artifact ID plus the existing runtime-layout binding. Restoring over
an occupied slot first snapshots the resident continuation for rollback. If decoding, validation,
physical import, or catalog adoption fails, the old continuation and its file binding are restored
before the original error is returned. Explicit deletion never spills. An involuntary eviction may
spill only a session previously bound by save or restore; the existing per-path depth high-water
guard prevents a stale shallower copy from rolling a newer file back. A publication mutex orders
explicit file operations with asynchronous spills.

## Focused validation

The gate used Release, CUDA 13.1, GCC 14, `CMAKE_CUDA_ARCHITECTURES=89`, and the local 48 GiB RTX
4090. The real-artifact test used
`/data/llm/ninfer/models/qwen3_8_27b_v3.ninfer`, INT8 KV, MTP3, one lane, and a 256-token capacity.

- `ninfer_engine` compiled and linked successfully;
- `ninfer_session_snapshot_test`, `ninfer_session_file_test`,
  `ninfer_slot_spill_guard_test`, `ninfer_qwen3_5_session_image_test`, and
  `ninfer_qwen3_5_context_store_test` passed;
- `ninfer_qwen3_5_session_real_test` passed in 13.47 seconds, covering save, digest mismatch,
  delete, restore, corrupt replacement with resident-session rollback, asynchronous eviction
  auto-save, and restore into a freshly constructed Engine.

This increment changes no inference kernel or scheduling policy and makes no throughput, TTFT,
TPOT, acceptance-rate, or VRAM-performance claim. Production remains on the validated v2 `main`
line and was not started or modified by this gate.

## Next increment

Restore the OpenAI/Anthropic Serve slot routes and schema tests on top of these Engine methods,
including request-error mapping and slot metrics. Then run a small process-level restart and mixed
request/eviction soak before considering the v3 line release-ready.
