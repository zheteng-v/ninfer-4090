# v3 Serve session persistence — 2026-09-30

This is the sixth session-persistence increment on the v3/sm89 integration line. It exposes the
durable Engine slot transaction through the HTTP service without moving file ownership or session
validation out of Engine.

## Upstream audit

The iteration started from `sync/2026-09-30-v3-sm89@293ecec2`. A fresh fetch found no source-head
change:

| Source | Audited head | Decision |
|---|---:|---|
| `Neroued/ninfer` master | `d44ab584` | unchanged; retain as the v3 architecture source |
| `Neroued/ninfer` dev | `75a89050` | unchanged; no Serve slot transaction to port |
| `sergiuszm/rtx4090-port` | `aeeba414` | unchanged; used only as the mature HTTP-contract reference |

## Delivered contract

- `GET /slots` publishes every private-continuation catalog cell, including processing/retained
  state, prompt and cached depth, session digest, retained checkpoints, context limit, and whether
  speculative decoding is active.
- `POST /slots/{id}?action=save|restore|erase` validates the catalog id, accepts only a conservative
  single-component filename, preserves the optional `if_digest` precondition, and calls the public
  Engine transaction. The route is disabled with a stable 501 response unless
  `--slot-save-path` is configured.
- Stable client-visible mappings distinguish malformed requests, invalid ids/actions/filenames,
  busy slots, digest conflicts, and invalid snapshot operations. Unexpected failures remain server
  errors instead of being relabeled as client input.
- `--auto-save-evicted` is accepted only with `--slot-save-path`, is forwarded to Engine, and sends
  asynchronous spill outcomes through the service operational log. Startup creates and validates
  the configured directory before loading model weights.
- `/metrics` publishes current slot count/processing/retained gauges and cumulative successful
  save/restore/erase plus failed-operation counters. The server-start JSONL records the persistence
  directory and auto-save state.

The HTTP layer never serializes a session itself. It chooses the configured child path and delegates
all atomic publication, artifact/runtime binding, rollback, digest checks, and eviction ordering to
Engine. This preserves the ownership boundary established by the preceding durable-slot increment.

## Validation

Release, CUDA 13.1, GCC 14, and `CMAKE_CUDA_ARCHITECTURES=89` compiled `ninfer-serve` and all focused
tests. Eight host-side tests passed: OpenAI schema, Anthropic schema, Serve options, Serve metrics,
slot filename policy, slot JSON/error contract, request logging, and HTTP error handling. A startup
negative test also rejected `/dev/null` as an unusable slot directory before model loading.

The planned process-level model smoke could not run because the local RTX 4090 was already occupied
by the user's active `sglang-dual` service (about 46.3 GiB). The temporary NInfer process exited on
the expected CUDA allocation failure and was not retried; the active service was not stopped or
modified. The previous increment's real Engine restart/save/erase/restore/auto-save gate remains
valid, but it is not represented here as an HTTP-process pass.

## Next increment

When the RTX 4090 is free, run the short HTTP save/erase/restore/restart smoke plus a bounded mixed
request/eviction soak. Then add protocol-visible session identity to generation responses only if a
clean Engine observation can guarantee that identity at response completion; do not infer it from
HTTP request ordering. After that gate, review release readiness of the v3/sm89 line against the
production v2 startup profiles.
