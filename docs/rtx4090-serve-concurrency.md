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
  --kv-dtype int8 --port 24562 --device 0 \
  --cuda-visible-devices GPU-2f39017c-6cf6-5c22-6c8b-aff9ef65a4bd \
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

### Reproducible fixed-wave saturation result

The benchmark runner was subsequently run on the same RTX 4090 with its GPU UUID
explicitly bound, an 8192-token context, INT8 KV, greedy decoding, prefix reuse
disabled, and the fixed `long_decode_aime26_15` prompt. The MTP3 point used a
256-token output budget; DFlash2 used 512 tokens so that both its C=1 and C=2
points contained complete one-second steady-state intervals. Consequently compare
the **within-backend speedup**, not the absolute values between the two rows.

| Backend | C=1 steady decode | C=2 steady decode | C=2 batch | Within-backend speedup |
| --- | ---: | ---: | ---: | ---: |
| MTP3 | 127.0 tok/s | 255.0 tok/s | 2.00 | **2.01×** |
| DFlash2 K=7 | 148.0 tok/s | 161.4 tok/s | 2.00 | 1.09× |

The raw artifacts are retained under
`profiles/bench/rtx4090-concurrency/20261004-fixed-c1c2-8k-uuid/` (MTP3) and
`profiles/bench/rtx4090-concurrency/20261004-dflash-c1c2-8k-uuid/` (DFlash2).
The runner must use `--cuda-visible-devices GPU-2f39017c-6cf6-5c22-6c8b-aff9ef65a4bd
--device 0` on this host: physical GPU 0 is the 5060 Ti, while the RTX 4090 is
physical GPU 1. Binding its UUID creates the same logical-device mapping used by
the production service.

### Long-context code-generation result (2026-10-04)

The short fixed-wave result above is **not** a prediction for a 128K coding
session. To measure the workload that users actually see, the dedicated runner
`tools/bench/run_serve_long_context_code.py` starts a clean server per point and
submits the maintained long-NIAH document with its terminal instruction replaced
by a C++17 fair-prefill scheduler implementation task. Each request contained
130,159 prompt tokens, had a 2,048-token completion cap, used thinking mode,
INT8 KV, a 1,024-token prefill chunk, and disabled prefix reuse. C=2 submitted
two independently seeded requests concurrently; its KV capacity was 393,216.

Raw immutable reports are under
`profiles/bench/long-context-code/20261004-mtp3-dflash2-c1c2-128k/`. The
following rates are server-recorded decode rates, so the roughly 80--164 seconds
of long-context prefill are deliberately not counted as output tok/s.

| Backend | C=1 actual decode | C=1 prefill / TTFT | C=2 request decode | C=2 simultaneous-decode aggregate | Speculative acceptance (C=1 / C=2 total) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP3 | 83.5 tok/s | 80.4 s / 80.5 s | 73.6 / 58.6 tok/s | **123.9 tok/s** | 50.7% / 37.7% |
| DFlash2 K=7 | **110.2 tok/s** | 82.0 s / 82.1 s | 59.6 / 63.1 tok/s | 102.8 tok/s | 34.5% / 26.9% |

The C=2 aggregate is the token-weighted mean of one-second server intervals in
which both requests were decode-ready and no prefill was occurring (MTP3: 1,735
tokens / 14.0 s; DFlash2: 1,644 / 16.0 s). It is the appropriate comparison for
two users working concurrently. One request in each C=2 point emitted an EOS
before the cap (MTP3 1,056 tokens, DFlash2 926), so its individual decode rate
includes a tail where the other request continues alone; do not replace the
simultaneous aggregate with the sum of those two per-request averages.

This changes the operational conclusion by workload:

- At this 130K coding context, DFlash2 is 31.9% faster for one user (110.2 vs
  83.5 tok/s).
- For two active users, MTP3 has 20.6% higher simultaneous aggregate decode
  (123.9 vs 102.8 tok/s), while each user's own observed speed remains roughly
  59--74 tok/s.
- Full end-to-end completion throughput is only 15--20 tok/s because it includes
  the unavoidable cold 130K prefill. It must never be reported as decode tok/s.

The production MTP3 dual profile remains justified for concurrent work. DFlash2
is the preferred candidate for a single long-context coding profile, subject to
at least two additional seeded repetitions before a default is changed; token
acceptance and EOS length are sampling-sensitive.

### Acceptance diagnosis matrix (2026-10-04)

The long-code runner was extended with explicit fixture depth, greedy/stochastic
sampling, MTP0, MTP3, DFlash2, and proposal-head selection. This separates a
proposal-quality problem from normal context-depth and sampling effects. All rows
below use the same terminal C++ scheduler task, a 1,024-token completion cap,
thinking disabled, INT8 KV, no prefix reuse, one lane, and the optimized proposal
head. Each point is one fixed-seed clean-server run; these are diagnosis points,
not a variance estimate.

| Prompt depth | MTP0 (no spec) | MTP3 decode / acceptance | DFlash2 K=7 decode / acceptance |
| --- | ---: | ---: | ---: |
| 7,755 tokens | 51.5 tok/s | 124.9 tok/s / 63.7% | **162.0 tok/s / 42.9%** |
| 64,587 tokens | 45.6 tok/s | 111.1 tok/s / 65.9% | **129.3 tok/s / 37.4%** |
| 130,123 tokens | 40.9 tok/s | 100.2 tok/s / 67.6% | **124.0 tok/s / 40.9%** |

The lower DFlash2 acceptance is not evidence by itself of inferior weights: it
proposes seven tokens per round whereas MTP3 proposes three. On this exact task
DFlash2 is faster at every depth despite that percentage. At 130K, MTP3 and
DFlash2 respectively produce 2.45x and 3.03x the no-spec decode baseline.

At the same 130K depth, changing only the server sampling profile produced:

| Backend | Greedy decode / acceptance | Temperature 1.0, top-p .95 decode / acceptance |
| --- | ---: | ---: |
| MTP3 | 100.2 tok/s / 67.6% | 91.1 tok/s / 58.4% |
| DFlash2 K=7 | 124.0 tok/s / 40.9% | 116.4 tok/s / 37.4% |

Thus stochastic sampling accounts for a material part of the observed acceptance
loss (9.2 percentage points for MTP3 here), but does not explain all content and
depth variation. The optimized-head control also supports retaining the deployed
`--lm-head-draft` setting:

| Backend at 130K, greedy | Optimized head | Full head |
| --- | ---: | ---: |
| MTP3 | **100.2 tok/s / 67.6%** | 92.5 tok/s / 68.8% |
| DFlash2 K=7 | **124.0 tok/s / 40.9%** | 120.6 tok/s / 41.1% |

The full head gains at most 1.2 acceptance points while losing 2.8--7.7% decode
rate. This rejects a simple local proposal-head precision/configuration fault; it
does not prove that the imported MTP or DFlash2 companion weights are globally
optimal. A future artifact audit must compare target/proposal logits against the
source checkpoint on fixed token prefixes, then repeat the matrix across multiple
seeds before changing artifact conversion or quantization.

Raw reports are retained under
`profiles/bench/acceptance-matrix/20261004-{greedy-depth,stochastic-128k,full-proposal-128k}/`.

### SGLang versus NInfer long-code comparison (2026-10-04)

The comparison client `tools/bench/run_sglang_long_context_code.py` renders the
same NIAH document and terminal C++ scheduler task as the NInfer runner, then
uses OpenAI-compatible loopback SSE to measure SGLang. Both engines prepared
exactly 130,123 prompt tokens. All comparison points use greedy decoding,
thinking disabled, a 1,024-token cap, no prefix reuse, and the RTX 4090 48 GiB.
This is an operational comparison of the installed profiles, not an
equal-weights kernel benchmark: SGLang uses Qwen3.8 W4A16-AWQ, FP8 DFlash2 draft
and FP8 KV; NInfer uses its groupwise-int v3 artifact, INT8 KV and the listed
speculative backend.

| Workload | SGLang DFlash (8 drafts) | NInfer MTP3 | NInfer DFlash2 K=7 |
| --- | ---: | ---: | ---: |
| C=1 decode | 123.2 tok/s | 100.2 tok/s | **124.0 tok/s** |
| C=1 external TTFT / internal prefill | **72.75 s** | 81.7 s | 82.1 s |
| C=2 per-request decode | 100.0 / 96.6 tok/s | 91.3 / 82.2 tok/s* | 86.9 / 50.3 tok/s* |
| C=2 simultaneous decode aggregate | **193.2 tok/s** | 143.5 tok/s* | 91.3 tok/s* |
| C=2 TTFT | **74.95 / 74.89 s** | 163.28 / 162.19 s | 163.53 / 164.61 s |
| C=2 observed GPU memory | 46,328 MiB | **32,256 MiB** deployed profile | about 32 GiB class |

`*` The matched NInfer C=2 wave used the same request input but one of its two
responses stopped naturally at 316 tokens. Its simultaneous aggregate is the
server's complete one-second intervals with both decode lanes active, not the
full-wave makespan. This is the valid decode-batch comparison, but it is a short
sample and must be repeated with a multi-seed fixed-output corpus.

SGLang batches the two cold long prefills: both requests reached their first
token near 75 seconds. NInfer's fair scheduler prevents admission starvation,
but it advances one 1,024-token prefill lane at a time; each request consumed
about 81.6 seconds of compute prefill and both first tokens arrived near 163
seconds. This prefill policy, more than decode kernel speed, is the decisive
two-user interaction difference in this test.

For the presently installed profiles, SGLang is the highest immediate
two-concurrent-user throughput/TTFT option. NInfer DFlash2 is effectively tied
with it for single-user decode while reserving much more VRAM headroom, and
NInfer MTP3 remains the lower-memory dual route. Do not declare a quality winner
yet: SGLang logged that its FP8 KV cache has no supplied scale factors and falls
back to scale 1.0. Before making SGLang the production default, run a fixed
correctness/retrieval and code-completion quality suite against both artifacts,
then repeat C=2 with enough fixed requests to avoid EOS-length bias.

Raw reports are under `profiles/bench/framework-comparison/20261004-{sglang-single-128k-greedy,sglang-dual-128k-greedy,ninfer-dual-128k-greedy}/`.

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
