# v3 slot replacement under checkpoint pressure — 2026-09-30

This is the seventh session-persistence increment on the v3/sm89 integration line. It closes a
real-model failure in the Engine restore transaction when every private catalog entry is occupied
but the checkpoint image needed by import has no vacant Device destination.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@62404e2b`. A fresh fetch found no source-head
change:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no equivalent durable-slot fix to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the RTX 4090 serving reference |

## Reproduced defect

The HTTP session smoke used the Qwen3 8B/27B v3 artifact on one RTX 4090 48G with one active lane,
one extra Device state image, two Host state images, two private continuations, 512 tokens of INT8
KV, and MTP3. Restoring a snapshot over an occupied target worked until both private catalog cells
were populated. In that state the target could be Host-resident while the other idle continuation
owned the only cached Device state image. Releasing only the target therefore left
`Program::import_continuation` without a Device destination and the HTTP operation returned 500
with `std::bad_alloc`, despite roughly 18 GiB of otherwise free VRAM.

This was checkpoint-placement pressure, not snapshot corruption or general GPU exhaustion: erasing
the other catalog entry before the same restore made it succeed.

## Repair

- Restore now checks that physically releasing an occupied target actually consumed its Program
  capability. A failed release re-adopts the original handle and preserves its file binding.
- On import allocation pressure, Engine selects the shallowest other idle private continuation,
  spills it through the existing auto-save contract when bound, physically releases it, and retries
  the import. Active entries and the requested target are excluded, and the retry is bounded by the
  finite catalog.
- A successful replacement queues the displaced target's rollback image to its previous bound file.
  A failed replacement still imports that image as the atomic rollback path.
- Runtime slot statistics are republished after both pressure eviction and terminal failed restore.

The implementation remains inside Engine publication policy. Program continues to own physical
StateImage import/export and the service continues to own asynchronous file publication.

## Validation

Release, CUDA 13.1, GCC 14, and `CMAKE_CUDA_ARCHITECTURES=89` compiled the real session test and
`ninfer-serve`.

- The extended real Engine test passed in 24.00 seconds. It filled both private catalog entries,
  replaced the Host-resident target under Device-image pressure, observed the displaced target's
  auto-save, and restored that displaced session in a fresh Engine with the original digest.
- Six focused host tests passed: session snapshot framing, session files, spill ordering, Serve slot
  API, Serve options, and Serve metrics.
- The process-level HTTP gate restored 25-token and 18-token snapshots, then restored the latter
  over the former with the catalog full. The formerly failing request returned HTTP 200 in
  3229.614 ms, `/slots` exposed the expected 18-token digest, and metrics recorded three successful
  restores with zero operation failures.
- The pressure victim and occupied target both emitted successful auto-save events. After a clean
  server restart, the original file restored 25 tokens with its unchanged
  `3036fda0fdcebe30` digest in 1104.148 ms.

These timings qualify correctness on the intended hardware and constrained cache geometry; they
are not throughput claims.

## Next increment

Review release readiness of the accumulated v3/sm89 line against the production MTP3 profile, then
run one short mixed generation/restore soak that alternates cache hits and explicit slot operations.
Keep protocol-visible session identity out of generation responses until Engine can publish it at
response completion without relying on HTTP request ordering.
