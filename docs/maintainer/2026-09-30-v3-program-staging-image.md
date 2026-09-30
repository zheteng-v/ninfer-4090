# v3 Qwen3.5 Program staging image — 2026-09-30

This note records the second session-persistence increment for the v3/sm89 integration line. It
defines a complete, model-owned Qwen3.5 continuation image and validates it without changing live
Program stores. It does **not** yet make disk save/restore or the public slot API operational.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@19f8df16`. A fresh audit found no source-head
change to integrate:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no new persistence work to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the Ada production baseline |

The review also checked the current DFlash/DFlash2 storage split. DFlash2's official draft has only
local cyclic context in `StateImage`; it has no paged Backend KV. The staging schema therefore
binds to the actual physical Backend KV layout instead of assuming that every speculative backend
owns a second paged cache. MTP requires that cache, `None` forbids it, and masked-draft variants may
or may not have it according to their frozen model layout.

## Owned staging contract

The new host-only `ninfer_qwen3_5_session_image` library owns six typed sections inside the generic
`NINFSNP3` container:

| Section | Stable ID | Contents |
|---|---:|---|
| Runtime | `0x3501` | KV/speculation settings, model dimensions, physical layout fingerprints |
| Sequence | `0x3502` | execution/ledger frontiers, RoPE delta, speculative and rebuild state |
| Identity | `0x3503` | token ledger, exact multimodal prefix identity, shortlist digests |
| States | `0x3504` | deduplicated host `StateImage` payloads |
| Text KV | `0x3505` | committed main-cache frontier and canonical host pages |
| Backend KV | `0x3506` | optional MTP or full-attention draft-cache frontier and host pages |

Checkpoint records store an index into the deduplicated StateImage table. Endpoint, rewrite, and
long-anchor records can therefore preserve their original alias graph instead of duplicating a
shared state payload. The image also retains each checkpoint's kind, frontier, ordinal, and
rebuild-work accounting.

The runtime binding includes storage/backend/proposal choices, draft width, token domain, page
size, Vision enablement, exact State/KV payload sizes, and fingerprints of all per-image physical
layout fields. StateImage slot count is intentionally excluded: it describes server concurrency,
not the bytes in one saved image, so a valid session can move between single- and dual-lane
profiles when every other layout field matches.

## Transactional validation boundary

Decode produces a fully owned `ContinuationSessionImage`. Before a future importer reserves or
publishes a live resource, the decoder verifies:

- exact artifact binding, schema, runtime binding, section framing, checksums, and context bound;
- legal token IDs, sequence frontiers, checkpoint kinds, ordinals, and rebuild accounting;
- complete Text/Backend KV shapes derived from frontiers and physical page geometry;
- MTP and masked-draft metadata against the frozen backend and draft width;
- StateImage count, byte geometry, valid checkpoint indices, and no unreferenced payload;
- exact Text/Vision identity shape and a fresh recomputation of every prefix shortlist digest.

A failed check returns no import object and cannot partially mutate Program state. The next
increment will keep that property while mapping the staging payload into reserved State/KV stores:
all destinations must be available before any handle is published, and any failed transfer must
release its reservations.

## Focused validation

The host-only target builds without compiling the CUDA execution graph. These focused tests pass:

```text
ninfer_session_snapshot_test
ninfer_qwen3_5_session_image_test
```

The Release `sm_89` `ninfer_model_runtime` target was then rebuilt and linked successfully, which
also verifies the new static-library boundary used by the complete CUDA runtime.

Coverage includes deterministic encode/decode, exact multimodal identity, checkpoint alias
round-trip, State/KV payload round-trip, layout/artifact/context mismatch rejection, unreferenced
or out-of-range state rejection, recomputed-digest mismatch, insufficient Backend KV, layout
fingerprint sensitivity, concurrency-independent StateImage fingerprints, and DFlash2 without a
paged Backend KV layout. This is a schema and validation gate, not a GPU execution or performance
claim.

## Remaining integration sequence

1. capture live Program StateImage and host/device KV replicas into this staging image without
   weakening residency or ownership invariants;
2. reserve destinations, restore bytes, rebuild alias-aware checkpoint handles, and publish the
   complete sequence atomically, with rollback tests for every failure point;
3. add bounded atomic disk publication and implement the public Engine slot methods;
4. expose Serve routes and run save/restart/restore, eviction, single/dual-lane, MTP3, DFlash2,
   Text, and Vision canaries on the official v3 artifact.

Production remains on the validated v2 service.
