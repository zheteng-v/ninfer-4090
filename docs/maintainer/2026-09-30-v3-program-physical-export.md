# v3 Qwen3.5 Program physical export — 2026-09-30

This note records the third session-persistence increment for the v3/sm89 integration line. It
connects the validated Qwen3.5 staging image to an already catalogued live continuation through a
read-only physical export. It does **not** restore an image, publish a disk file, or make the public
Engine slot methods operational.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@ec723834`. The three source heads were
unchanged:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no new merged persistence work to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; retain as the Ada production baseline |

Neroued PR #335 (`4c6af031`, “replace the checkpoint catalog with a hybrid prefix cache”) is new
and relevant, but is classified **watch / benchmark first**. It is a 121-file architectural
replacement of the current catalog and `prefix_identity`, not a bounded persistence patch. Its
opt-in Host prefix-cache file addresses clean-restart prefix reuse, while this downstream increment
addresses an exact per-session continuation including sequence metadata, alias-aware StateImage,
and Text/speculative KV. The PR's current native-struct file records have no whole-image or
per-section checksum and no durable rename sequence with an `fsync` boundary, so they do not
supersede the `NINFSNP3` framing. The downstream export remains behind the Program-private boundary
so it can be adapted if the hybrid cache is merged and validated on sm89.

## Export contract

`Program::export_continuation` accepts only a valid `Catalogued` continuation and rejects export
while any context transaction, pending execution batch, pressure-planning session, or unsettled
State fork exists. It does not reserve a State/KV object, create a Host replica, change residency,
publish a handle, or advance the Program resource revision.

The capture performs these operations in order:

1. bind the image to the exact model/runtime and State/Text/Backend physical layouts;
2. copy the token ledger, multimodal prefix identity, digest image, speculative state, and rebuild
   accounting;
3. deduplicate endpoint, rewrite, and long-anchor StateImage handles while retaining every
   checkpoint's table index;
4. stage each immutable StateImage from its existing Host replica or directly from Device;
5. walk Text and optional Backend KV in logical page order, using a current Host replica where one
   exists and Device otherwise;
6. synchronize the compute stream, run the complete staging-image semantic validator, and only
   then encode the checksummed, model-bound `NINFSNP3` result.

Device-to-Host copies use the Program compute stream, which orders the read after the kernels that
produced the immutable continuation. Host and Device payload padding is zeroed before defined
regions are copied, and columns beyond a partial tail frontier are zeroed after synchronization.
This makes the serialized bytes independent of stale arena padding or later shared-page columns and
avoids leaking unrelated allocation contents.

The exporter supports Text-only, MTP, DFlash, and DFlash2 physical layouts. MTP and full-attention
DFlash export their paged Backend KV. DFlash2 retains its draft-local cyclic state inside the
StateImage and correctly emits no Backend KV section payload.

## Focused validation

The validation gate for this increment is intentionally smaller than a release gate:

- Release `sm_89` build and link of `ninfer_model_runtime`;
- `ninfer_qwen3_5_session_image_test` for the framing and semantic validator;
- `ninfer_qwen3_5_context_store_test` on the RTX 4090, including equal canonical export from
  Device-only, Both, and Host-only immutable StateImage placement without residency mutation.

This is a correctness and ownership-boundary result. It changes no kernels or execution schedule,
so it makes no throughput, TTFT, TPOT, acceptance-rate, or VRAM-performance claim.

## Next increment

The next bounded change is physical import. Decode and validate the complete image first, reserve
every logical StateImage, State replica, KV page, address space, and continuation slot before the
first copy, enqueue all Host-to-Device work, and publish the reconstructed alias graph only after
successful synchronization. Every failure edge must unwind all reservations while leaving the
catalog and resource revision unchanged. Atomic disk publication and the public Engine slot API
remain separate work after that transaction passes rollback tests.
