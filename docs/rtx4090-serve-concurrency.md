# RTX 4090 48 GiB serving-concurrency program

## Objective

Make two independent requests useful on one RTX 4090 48 GiB: neither request may
wait for another request's complete prefill, and the steady two-request aggregate
decode rate must exceed the corresponding single-request rate.

This document deliberately distinguishes aggregate throughput from per-request
throughput. Two streams share compute and bandwidth, so each stream can be slower
than a single stream while their combined decode rate is higher.

## Implemented first stage: fair chunked prefill

The engine previously retained exactly one staged `prefill_lane_`. While it was
set, the admission gate rejected every pending request. A 128K request could
therefore block a short request before it was allowed to begin prefill.

The scheduler now retains an ordered collection of staged lanes. A request that
finishes an incomplete `advance_prefill()` chunk is moved to its tail. Once an
idle lane is available, a second request can be admitted despite an existing
prefill request. The existing decode/prefill alternation remains in force.

```text
before: A: chunk -> chunk -> ... -> complete; B: waiting
after:  A: chunk -> B: chunk -> A: chunk -> B: chunk -> ...
```

This is intentionally a scheduler-only change. It does not alter model weights,
the artifact format, CUDA kernels, HTTP compatibility, MTP3, or DFlash2.

### Safety invariants

- A lane occurs at most once in the staged-prefill queue.
- An unfinished prefill unit can rotate only when it owns the queue front.
- Completion, cancellation and defensive terminal cleanup remove the lane.
- Capture-pending lanes are rotated behind runnable prefill lanes.
- Admission still uses the existing resource, context-transaction, and concurrency
  limits; removing the single-prefill gate does not create additional sequence slots.

## Deployment profiles

`inferctl switch ninfer-single` is the MTP3 single-request default:

- MTP3, K=3, INT8 KV, Vision, 262144 context, one active request.
- `inferctl switch ninfer-dual-mtp3` is the 196608-per-request two-way MTP3 profile.
- `inferctl switch ninfer-dual` remains the explicit DFlash2 K=7 comparison profile.

## Measurement protocol

Run measurements only in a GPU maintenance window: a temporary benchmark server
loads a second copy of the model and must not contend with the production process.

First run the targeted fairness regression against the staged binary:

```bash
python3 tools/bench/run_serve_prefill_fairness.py \
  --serve build-native/apps/ninfer-serve \
  --artifact models/qwen3_8_27b_v3.ninfer \
  --mode mtp3 \
  --output profiles/bench/prefill-fairness/<timestamp>-mtp3
```

Repeat it with `--mode dflash2_7`. The primary result is the short request's
`queue_wait_ms`: it must be lower than the long request's complete `prefill_ms`.
The raw structured request log also reports TTFT, TPOT/decode time, accepted and
drafted speculative tokens.

Then compare aggregate decode throughput with a small matched matrix:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build-native/apps/ninfer-serve \
  --artifact rtx4090-v3=models/qwen3_8_27b_v3.ninfer \
  --mode mtp3 --mode dflash2_7 \
  --suite decode-saturation \
  --concurrency 1 --concurrency 2 \
  --sampling greedy --decode-tokens 1024 \
  --max-context 196608 --kv-capacity per-concurrency --prefill-chunk 1024 \
  --kv-dtype int8 --port 24562 \
  --output profiles/bench/rtx4090-concurrency/<timestamp>
```

`per-concurrency` resolves KV capacity to 196608 at C=1 and 393216 at C=2. This
is necessary because a C=1 server cannot legally reserve 393216 tokens, while a
full C=2 long-context point needs that capacity.

The experiment writes immutable command provenance, per-request JSONL, TTFT,
queue delay, decode batch size, aggregate decode tok/s, and speculative acceptance.

## Initial on-device evidence (2026-10-04)

These are deliberately small validation runs on the production RTX 4090 after a
restart into the fair-prefill build. They establish scheduler behavior and route
direction; they are not a replacement for the full fixed-corpus campaign above.
All runs used V3, INT8 KV, 196608 context per route, greedy decoding, no prompt
reuse in the test payloads, and a 1024-token prefill chunk.

### Long/short fairness check

The long request had 130048 prompt tokens and was submitted first. The short
request had 30 prompt tokens and arrived 250 ms later.

| Backend | Long complete prefill | Short queue wait | Short TTFT | Result |
| --- | ---: | ---: | ---: | --- |
| MTP3 | 80.57 s | 331.5 ms | 1.36 s | pass |
| DFlash2 K=7 | 81.92 s | 338.8 ms | 1.84 s | pass |

Before the scheduler change, the equivalent short request waited about 62 s
behind a 133K prefill. Both rows therefore demonstrate that the second request
now starts while the long request remains in prefill.

### Simplified matched decode check

One request produced 384 tokens; the two-request run submitted two copies of the
same 54-token prompt concurrently, each capped at 384 tokens. `Aggregate` is
total completed tokens divided by wall-clock makespan, so it includes the small
prompt/TTFT cost and is intentionally a conservative end-to-end measure.

| Backend | C=1 aggregate | C=2 aggregate | C=2 gain | C=2 per-request decode |
| --- | ---: | ---: | ---: | --- |
| MTP3 | 96.6 tok/s | 177.8 tok/s | +84.2% | 103.9 / 92.5 tok/s |
| DFlash2 K=7 | 105.5 tok/s | 105.5 tok/s | -0.1% | 55.8 / 53.8 tok/s |

MTP3 is therefore the deployed dual-request default. DFlash2 was slightly faster
in this single-request sample, but did not produce a two-request aggregation gain.
The next corpus campaign must confirm whether that pattern holds across coding,
long-context and tool-use requests before any route policy is made permanent.

## Acceptance gates

1. The long/short test shows the short request prefill begins before the long
   request has completed all prefill work.
2. At two streams, steady aggregate decode tok/s is higher than the matched
   one-stream baseline for the same speculative mode.
3. Single-stream decode throughput regresses by no more than 5%.
4. Long-context two-stream runs have no OOM, stuck lane, cancellation leak, or
   request error.

If the 512, 1024 and 2048 chunk trials differ materially, choose the smallest
chunk that meets the TTFT gate without sacrificing the aggregate-throughput gate.

## Dual MTP3/DFlash2 routing: explicit second-stage gate

The current runtime selects `--spec` while constructing `LoadOptions` and builds
one `ProgramImpl` with one immutable `speculative_backend`. MTP3 and DFlash2 have
different loaded components, KV/state layouts, workspace plans and CUDA-graph
profiles. A command-line router or two independent processes cannot share GPU
weights and would not meet the memory objective.

Do not implement process-level dual routing before the fixed-corpus protocol proves
that both modes win distinct workload classes. The initial smoke data is sufficient
to keep MTP3 as the dual-route default, but is not sufficient to justify the large
memory and correctness cost of a dual-backend runtime. If the campaign supports it,
the next design must be a single `ProgramImpl` that:

1. loads shared base weights once plus both optional speculative components;
2. owns MTP and DFlash runtime allocations separately and accounts for their exact
   reservation before admitting a request;
3. stores backend identity in each sequence and session snapshot;
4. groups decode batches by backend, never mixing incompatible CUDA graph layouts;
5. exposes a request-level route (`auto`, `mtp3`, `dflash2`) with MTP3 as fallback;
6. rejects a route that cannot fit inside the current KV/memory budget rather than
   evicting an active request.

That is a dedicated architecture change, not a safe extension of the scheduler
patch. Its entry condition is measured, repeatable evidence of a material routing
benefit after fair concurrency is established.
