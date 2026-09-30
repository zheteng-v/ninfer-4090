# v3 Qwen3.5 Program physical import — 2026-09-30

This note records the fourth session-persistence increment for the v3/sm89 integration line. It
restores a validated Qwen3.5 staging image into an unpublished continuation with transactional
StateImage and KV ownership. It does **not** publish a disk file or expose the public Engine slot
methods.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@dfb9f38c`. The three source heads were
unchanged:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no merged session-import change to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the Ada production baseline |

Neroued PR #335 (`4c6af031`, hybrid prefix cache) remains **watch / benchmark first**. Its prefix
cache ownership may eventually replace parts of the current checkpoint catalog, but it does not
provide an exact continuation import transaction or supersede the checksummed `NINFSNP3` image.
This increment stays behind the Program-private storage boundary so the publication step can be
adapted if that PR is merged and survives the sm89 gates.

## Import transaction

`Program::import_continuation` first decodes the entire byte image and checks its artifact binding,
schema, checksums, runtime options, physical layout fingerprints, token/identity digests,
checkpoint aliases, and State/KV payload shapes. None of those failures can reserve or mutate a
Program resource.

After validation, import proceeds in five phases:

1. allocate all fallible Host metadata and reserve one unpublished continuation slot;
2. reserve every Device StateImage destination, Text/Backend address, logical page, and physical
   KV page before issuing a copy;
3. reconstruct the endpoint/rewrite/long-anchor alias graph and its exact StateImage checkpoint
   reference counts while all handles remain private;
4. enqueue all State and KV Host-to-Device copies on the Program stream and synchronize once;
5. publish StateImages, inactive KV address spaces, and finally the catalogued continuation using
   a non-throwing tail, then advance the resource revision exactly once.

KV import destinations carry zero address references and a destination pin until publication.
Their logical membership is not installed in the address space early. The final partial page keeps
its exact committed-column count, while checkpoint protection is rebuilt from the maximum
endpoint/rewrite/anchor frontier. MTP Backend KV uses the model's one-token-lagged protection
frontier; DFlash uses the checkpoint frontier; DFlash2 and no-speculation images have no paged
Backend KV. A zero-frontier Backend address is supported without consuming a physical page.

Any exception before publication drains possible asynchronous work, aborts both KV reservations,
releases every reserved StateImage in reverse order, and returns the continuation slot to `Free`.
The catalog and resource revision remain unchanged. Once publication starts, all remaining
operations are statically or structurally non-throwing; the continuation handle is created only
after the slot becomes `Catalogued`.

## Focused validation

The deliberately small gate for this ownership-only increment is:

- Release `sm_89` build and link of `ninfer_model_runtime`;
- `ninfer_qwen3_5_session_image_test` for framing, runtime binding, alias, and corruption checks;
- `ninfer_qwen3_5_context_store_test` on the RTX 4090, including StateImage H2D canonical-byte
  round-trip, unpublished KV reservation rollback, two-page KV H2D round-trip, partial-tail
  protection, and zero-frontier Backend KV publication/release.

This change adds no kernel or scheduling optimization, so it makes no throughput, TTFT, TPOT,
acceptance-rate, or VRAM-performance claim. Production remains on the validated v2 line.

## Next increment

Add durable file publication with a same-directory temporary file, complete writes, file `fsync`,
atomic rename, and directory `fsync`; then connect Engine save/load/list/delete methods to the
Program export/import transaction. The following gate must cover restart round-trip, replacement,
corrupt/truncated files, capacity failure, stale-handle rejection, deletion, and eviction without
leaking State/KV resources. Only after that should Serve routes and session metrics be restored.
