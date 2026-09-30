# RTX 4090 48 GiB performance program

This document is the active authority for closing the Qwen3.8-27B performance gap on the
`zheteng-v/ninfer-4090` v3/sm89 line. It records the production baseline, upstream reference
results, hardware interpretation, ranked hypotheses, experiment order, acceptance rules, and
decision log. Update it at the start and end of every performance iteration. Detailed raw reports
belong under ignored `profiles/bench/`; stable public results belong in `docs/performance/`.

Last updated: 2026-10-01. Production is
`v3-sm89-production-2026-10-01@8d3c28c09da934dbec981b18224dbfb7687cf389`.

## Mission and finish condition

The objective is not merely to run upstream v3 on Ada. It is to make one 48 GiB RTX 4090 deliver
the best reproducible NInfer performance that its sm89 execution resources allow, while preserving
correct answers, long context, Vision, durable sessions, and the public Serve contract.

Work proceeds through three performance thresholds:

1. **No-regression:** v3 matches or exceeds the tagged v2 binary on every same-method cohort.
2. **Ada-efficient:** remaining differences from the RTX 5090 reference are explained by measured
   memory, compute, occupancy, or instruction capability rather than an avoidable software stall.
3. **Stretch:** approach or exceed the official absolute groupwise-int results where sm89 supports
   the same representation, and exploit 48 GiB capacity for stronger long-context or concurrent
   service than a stock 24/32 GiB board can sustain.

A release does not satisfy this program because one easy prompt reaches a high peak. It must improve
the fixed corpus or sustained wave, keep losing cohorts visible, and pass the quality/memory gates.

## What the v3 production release changed

The 2026-10-01 release established a correctness and product baseline. Its main changes were:

- replay the upstream v3 converter, artifact loader, logical binding, and bound-instance Engine on
  the validated sm89 line;
- admit sm89 without selecting Blackwell-only TMA, cluster, block-scale MMA, PDL, NVFP4, or K8V4
  execution paths;
- integrate the maintained generic Jinja frontend and Qwen3.8 template behavior;
- restore OpenAI Chat/Responses, Anthropic, streaming cancellation, metrics, Vision, and request
  logging on the v3 Engine;
- design and implement v3 session images, physical Program export/import, crash-durable files,
  Serve slot routes, eviction auto-save, restart restore, and catalog-pressure replacement;
- qualify exact NIAH through 260,096 prompt tokens, MTP3, DFlash2 loading, mixed-protocol soak,
  cancellation, and v2 rollback;
- deploy native `inferctl` single and dual profiles from `/data/llm/ninfer`, using the official v3
  artifact, optimized MTP3, INT8 KV, Vision, and a retained v2 rollback package.

These were necessary changes, but they were not a performance-kernel campaign. No claim was made
that v3 had retuned Q4/Q5/Q6/Q8 kernels, GDN/KDA, attention, proposal verification, launch geometry,
or batching specifically for the 128-SM RTX 4090. That missing work is the purpose of this program.

## Current local evidence

The release gate provides only a directional v2/v3 comparison, not the official 75-request corpus:

| Cohort | v2 | v3 | Interpretation |
|---|---:|---:|---|
| one Python-code request, optimized MTP3 | 149.36 tok/s | 140.43 tok/s | v3 is 5.98% slower; one sample is enough to require investigation, not enough to rank kernels |
| same request MTP acceptance | 78.7% | 72.3% | output/artifact path changed; split acceptance loss from time-per-round before optimizing |
| 64K exact NIAH prefill | 1,884.70 tok/s | 1,884.52 tok/s | no measured v3 regression |
| 64K exact NIAH decode | 149.59 tok/s | 149.44 tok/s | no measured v3 regression |

The v3 long-context MTP3 matrix produced 2,155.28 / 1,884.52 / 1,599.75 / 1,251.01 prefill
tok/s at 7,680 / 64,512 / 130,048 / 260,096 prompt tokens, with exact retrieval at every depth.
Those points used INT8 KV and are not directly comparable with upstream's FP8-KV, no-speculation
prefill campaign.

The retained v2 local tournament selected MTP3 over DFlash2 K=5 for its complete workload:
126.33 versus 122.45 tok/s greedy weighted decode, and 179.55 versus 155.43 aggregate tok/s at
three concurrent requests. DFlash2 nevertheless won the v2 code, translation, structured, and
reasoning category rates. Upstream v3 has since changed DFlash2 and its artifact/runtime contracts;
the v2 decision must be re-tested, not inherited.

## Official RTX 5090 reference

The authoritative comparison is the upstream Qwen3.8 serving campaign in
[`Neroued/ninfer`](https://github.com/Neroued/ninfer/blob/d44ab58408aa389728cd8b1ee50179527e1f3e0d/docs/performance/qwen3.8-27b.md).
It was measured at revision `7f6aafedb5f20200def820cfe51ab81c09c20eeb` on one RTX 5090,
driver 617.14 and CUDA 13.4. It used FP8 E4M3 row-256 KV, prefill chunk 1,024, CUDA Graphs,
disabled prefix reuse, the common stochastic sampling profile, and optimized proposal heads.
Startup and warmup were excluded.

| Official cohort | RTX 5090 result | Correct interpretation |
|---|---:|---|
| groupwise-int MTP3, code, C=1 | 193.5 ± 5.9 tok/s | closest MTP representation reference for our artifact |
| groupwise-int DFlash2 K=7, code, C=1 | 258.6 ± 24.6 tok/s | primary absolute single-request groupwise stretch target |
| NVFP4 MTP3, code, C=1 | 205.3 ± 9.5 tok/s | Blackwell weight-format reference, not an Ada pass/fail gate |
| NVFP4 DFlash2 K=7, code, C=1 | 285.7 ± 25.2 tok/s | official peak code-category target supplied by the user |
| groupwise-int MTP3 sustained decode, C=8 | 582.4 tok/s | relevant concurrency reference for Ada-compatible weights |
| NVFP4 MTP3 sustained decode, C=8 | 922.4 tok/s | Blackwell native-FP4 saturation reference |
| groupwise-int MTP0 prefill, 7,680 | 3,331.9 ± 7.6 tok/s | comparable weight-format prefill reference |
| NVFP4 MTP0 prefill, 7,680 | 12,819.1 ± 16.8 tok/s | native-FP4/Tensor-Core path; not explained by bandwidth alone |

These results pool fixed fixtures and seeds. A single greedy 512-token response must never be
divided by one of these numbers and reported as a hardware efficiency percentage. Phase 0 below
first reproduces the upstream method locally.

## Hardware interpretation

NVIDIA's architecture references report 128 SMs and 1,008 GB/s peak memory bandwidth for RTX
4090, versus 170 SMs and 1,792 GB/s for RTX 5090. That is 1.33x as many SMs and 1.78x the peak
bandwidth, before kernel efficiency, clocks, cache behavior, power, or instruction mix. The local
48 GiB modification increases capacity; it does not increase memory bus width or bandwidth.

Blackwell also adds native FP4 Tensor Core operations. Ada has no equivalent native NVFP4 path.
The official NVFP4 figures therefore combine newer hardware instructions, a different stored
representation, less weight traffic, and 5090 scheduling. In particular, the 12,819 tok/s NVFP4
prefill value is not a reasonable first acceptance threshold for the existing Ada groupwise-int
artifact. It remains a research ceiling. The 3,331.9 tok/s groupwise-int prefill and 193.5/258.6
tok/s groupwise MTP3/DFlash2 code results are the primary cross-GPU references.

Sources:

- [NVIDIA Ada architecture, RTX 4090 specifications](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
- [NVIDIA Blackwell architecture, RTX 5090/4090 comparison and FP4 support](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

The goal is still aggressive: after same-method measurement, any gap larger than the measured
hardware roofline must be attributed and attacked. Hardware specifications are not an excuse for
software stalls.

## Ranked hypotheses

| ID | Priority | Hypothesis | Evidence now | Required decision experiment |
|---|---:|---|---|---|
| H0 | P0 | benchmark mismatch obscures the real gap | local release sample differs in KV type, sampling, fixture count, output budget and mode | run the upstream corpus and saturation tools unchanged on v2 and v3 before kernel work |
| H1 | P0 | FP8 KV is faster than deployed INT8 KV on sm89 for official cohorts | RTX 4090 has native FP8 Tensor Cores; upstream campaign uses FP8; current production uses INT8 for capacity | INT8 versus FP8 A/B at MTP0/MTP3/DFlash2, C=1/2/4/8, with exact long-context checks |
| H2 | P0 | v3 DFlash2 K=7 can reverse the v2 MTP3 decision | official groupwise code favors DFlash2 by 34%; current v3 artifact host-binds optimized DFlash2 | K=3/5/7 screen, then fixed 75-request C=1 corpus for winner; include `4201b5d2` correctness review first |
| H3 | P1 | launch plans still encode 5090-scale assumptions | upstream dev now derives attention, norm, MoE and RoPE launches from actual SM count | adapt `a667efdd`, `417eb3d6`, `e621c7d6`; compare launch geometry and end-to-end on 128 SMs |
| H4 | P1 | Q5 small-batch Linear/LinearAdd dominates lost MTP time per round | upstream PR #292 reports +11.4% MTP3 decode from Q5 K-split work on 5090 | port behind measured sm89 dispatch boundaries; sweep T=1..32 and validate end-to-end, not just Op minima |
| H5 | P1 | FP8/int8 causal attention limits prefill and deep-context decode | upstream `7f6aafed` and `d44ab584` retune split-KV/grouped small prefill after our v3 integration point | isolate applicable non-Blackwell pieces, oracle-test, then measure 8K/64K/128K/256K |
| H6 | P1 | v3 short-code regression is acceptance-driven, time-per-round-driven, or both | one sample lost acceptance and throughput while 64K decode stayed flat | replay identical token streams where possible; record rounds, accepted positions, host exposure and GPU time per round; compare v2/v3 Nsight traces |
| H7 | P2 | GDN/KDA recurrence, convolution or fused projections underutilize Ada | they are large fixed costs in Qwen3.8 decode and were not retuned in the v3 landing | kernel timeline and roofline first; tune only the top measured consumers |
| H8 | P2 | graph coverage and host launch gaps limit small batches | cold graph generation is expensive; per-round graph coverage has not been compared v2/v3 | separate startup compilation from steady graph execution; measure graph misses, CPU gaps and eager fallbacks |
| H9 | P2 | scheduler/admission prevents the 48 GiB board from converting capacity into throughput | long prefills serialize; only C=1/2 production profiles exist | run 16K decode saturation at C=1/2/4/8 with auto KV, then profile batch occupancy and waiting reasons |
| H10 | P3 | CUDA 13.4 changes code generation materially versus local 13.1 | official campaign uses 13.4 | build one isolated toolchain A/B after the workload is frozen; retain only a reproducible end-to-end gain |
| H11 | research | an Ada-specific compact weight format can capture part of NVFP4's traffic benefit | 48 GiB permits alternative resident layouts, but sm89 lacks native FP4 | profile traffic first; prefer offline Q4/Q5/FP8 layouts with independent quality gates; reject runtime repacking/emulation without a net gain |

## Execution order

### Phase 0 — freeze a comparable baseline

Do this before changing kernels.

1. Use `tools/bench/run_serve_corpus.py` and `run_serve_concurrency.py` with the upstream fixtures,
   seeds, stochastic sampling, disabled prefix reuse, text-only residency, prefill chunk 1,024 and
   FP8 KV. Record exact command, artifact hash, commit, driver, CUDA, clocks/power and warmup.
2. Run groupwise-int MTP0, MTP3 and DFlash2 K=7 at C=1. Run MTP3 decode saturation at
   C=1/2/4/8 with 16,384 context and 8,192 output tokens.
3. Run the same executable/harness against tagged v2 where the artifact contract allows it. If v2
   cannot consume a v3 fixture or option, document the narrow incompatibility rather than changing
   the workload.
4. Use a fast screening subset while iterating, but require the full 75-request corpus before a
   winner changes production.
5. Add a local scorecard row below; do not overwrite older values.

Exit condition: the project has a same-method v2/v3/official table and enough server JSONL to split
acceptance, device time, host exposure, queueing and memory behavior.

### Phase 1 — configuration and current upstream wins

1. Benchmark INT8 versus FP8 KV across the frozen cohort and NIAH depths. Production changes KV
   type only if speed, memory, and exact retrieval all pass.
2. Re-run v3 MTP K=2/3/4 and DFlash2 K=3/5/7. K=7 is mandatory because it is the official target.
3. Review/adapt upstream `4201b5d2` before trusting DFlash2 prefill state, and use PR #342-style
   per-lane records if attribution is otherwise ambiguous.
4. Sweep prefill chunks 512/1,024/2,048 only after the KV-type decision. Keep the 1,024 official
   point as control.
5. Test CUDA 13.4 only as an isolated build/toolchain factor.

Exit condition: the best existing route and configuration are known; no source-level optimization
begins while a simpler backend/KV mismatch can explain the gap.

### Phase 2 — adopt architecture-neutral upstream work

Review in this order:

1. `a667efdd`, `417eb3d6`, `e621c7d6`: derive launch capacity from the actual device rather than
   fixed 5090 assumptions. Classification: **adapt**, then benchmark on 128 SMs.
2. `4201b5d2`: DFlash per-chunk control binding. Classification: **adopt/adapt for correctness**
   before publishing v3 DFlash2 results.
3. `d44ab584`: grouped small-prefill routes. Classification: **benchmark first** for int8/fp8
   sm89 paths.
4. `7f6aafed`: FP8 split-KV causal attention. Classification: **adapt** only for instructions and
   schedules available on sm89; reject Blackwell/TMA-only pieces.
5. PR #292: Q5 K-split small-batch work. Classification: **benchmark first and retune**; its 5090
   thresholds are evidence, not Ada constants.
6. `1cfdb4d6` and other native NVFP4 work: **not applicable** to direct sm89 execution.

Each change gets its own branch and A/B. Do not merge a bundle of upstream performance commits and
then guess which one helped.

### Phase 3 — profile the unresolved gap

For the winning MTP and DFlash2 controls, capture:

- Nsight Systems: CUDA Graph launches, CPU launch gaps, memcpy/synchronization, per-round batch,
  proposal/verification ordering, and v2/v3 timeline differences;
- Nsight Compute on the top kernels: achieved HBM bandwidth, Tensor Core utilization, occupancy,
  registers, shared memory, eligible warps and stall reasons;
- hot-operation inventory by real artifact shape/format, especially Q5/Q4 Linear, LinearAdd,
  LinearSwiGLU, GDN/KDA, causal attention and proposal verification;
- separate prefill, decode, draft, verification, sampler, state transition and host exposure time.

Select only the largest measured avoidable cost. A kernel microbenchmark is not sufficient to
claim an end-to-end improvement.

### Phase 4 — sm89 kernel and dispatch work

Likely work, in evidence order:

1. retune Q5 K-split capacities and crossover bands for 128 SMs and the real Qwen3.8 shapes;
2. pipeline Q4/Q5 code/scale/activation staging where sm89 `cp.async` and shared-memory limits win;
3. tune Q8 and fused projection routes used by MTP/DFlash2 optimized heads;
4. reduce GDN/KDA recurrent and projected-convolution cost;
5. specialize FP8/INT8 causal attention split counts and grouped small-prefill schedules;
6. reduce DFlash2 K=7 proposal/verification and terminal-settlement overhead;
7. extend CUDA Graph coverage only when a trace shows uncovered steady-state work.

Every numerical kernel requires its independent oracle, odd/tail shapes, boundary dispatch tests,
and a real-model quality probe before an end-to-end benchmark.

### Phase 5 — turn 48 GiB capacity into throughput

The memory modification is an advantage only when the scheduler can keep more useful work resident.

1. Establish C=1/2/4/8 decode saturation with short contexts and automatic KV capacity.
2. Record actual batch, waiting reasons, resident state/KV, graph profile and aggregate tok/s.
3. Separate decode saturation from long-prefill admission. Do not advertise C=8 because eight
   requests fit if their prefills are serialized and steady decode never reaches batch eight.
4. After short-context saturation works, evaluate mixed short/long traffic and the current
   non-preemptive admission policy. Long-context correctness and bounded memory remain hard gates.

The 48 GiB board may beat a stock 5090 in context/lane capacity while remaining slower per token.
Report those as distinct advantages.

### Phase 6 — representation research

Do not emulate Blackwell NVFP4 merely to accept the artifact. Investigate only after profiling shows
weight traffic is the dominant residual cost:

- offline Ada-native Q4/Q5 layouts and packing that reduce bytes without runtime repacking;
- FP8-based stored/activation routes supported by sm89 Tensor Cores;
- per-layer mixed representations selected by measured hot shapes;
- quality comparison against the official groupwise artifact on AIME/GPQA/IFBench plus exact local
  regression fixtures.

Keep a representation only if load time, resident memory, end-to-end throughput, and quality beat
the current groupwise route. Native Blackwell FP4 numbers remain a research ceiling, not a reason
to ship a slow software decoder.

## Performance acceptance rules

- Use the same artifact, prompts, seeds, sampling, output budgets, KV type, context, concurrency,
  prefill chunk, graph policy, cache policy and power/clock state for A/B.
- Five samples per fixed phase point; full corpus/sustained-wave methods for release decisions.
- Treat changes below 3% as noise unless controlled repetitions and variance establish otherwise.
- Require no unexplained cohort regression above 3%. A deliberate tradeoff needs a product reason.
- Record TTFT, prefill, decode, TPOT, acceptance, tokens/round, actual batch, queue time, Host/GPU
  exposure, resident/peak VRAM and completion outcome.
- Preserve exact NIAH at 8K/64K/128K/256K, deterministic short answer, real serving contract,
  cancellation, and clean shutdown. Representation changes also require capability evaluation.
- Profile before optimizing and profile again after; an Op win must survive end-to-end measurement.
- Production changes only after rollback binary/config/artifact and the current release tag remain
  runnable.

## Scorecard

`TBD` is intentional: the release gate did not use the official method. Fill these rows in Phase 0.

| Metric | v2 sm89 same-method | v3 sm89 baseline | Best sm89 | Official 5090 reference | Next gate |
|---|---:|---:|---:|---:|---|
| groupwise MTP3 code C=1 | TBD | TBD | TBD | 193.5 tok/s | v3 >= v2 |
| groupwise DFlash2 K=7 code C=1 | TBD | TBD | TBD | 258.6 tok/s | choose local winner after full corpus |
| groupwise MTP0 prefill 7,680 | TBD | TBD | TBD | 3,331.9 tok/s | explain gap with measured roofline |
| groupwise MTP3 steady C=8 | TBD | TBD | TBD | 582.4 tok/s | reach actual batch 8 without errors |
| NVFP4 MTP3 code C=1 | not native | not native | not native | 205.3 tok/s | research reference only |
| NVFP4 DFlash2 K=7 code C=1 | not native | not native | not native | 285.7 tok/s | research reference only |
| NVFP4 MTP3 steady C=8 | not native | not native | not native | 922.4 tok/s | research reference only |
| NVFP4 MTP0 prefill 7,680 | not native | not native | not native | 12,819.1 tok/s | research reference only |

## Iteration protocol and decision log

Every performance iteration must:

1. run `tools/maintenance/upstream-audit.sh` and record immutable heads;
2. select one hypothesis from this document and state its success/rollback criteria;
3. create a bounded `perf/<topic>` branch;
4. capture the control before editing;
5. implement and qualify the smallest coherent change;
6. run the same control/candidate benchmark and quality gate;
7. append the result here, including negative results, then update the scorecard if comparable;
8. merge only a demonstrated product-level win.

| Date | Hypothesis/change | Result | Decision |
|---|---|---|---|
| 2026-10-01 | establish v3/sm89 performance program; audit `upstream/master@d44ab584`, `upstream/dev@75a89050`, `sergiuszm/rtx4090-port@aeeba414` | release evidence shows one short-code v3 regression but 64K parity; official method not yet reproduced locally | start Phase 0; no kernel claim or production parameter change |

## Immediate next iteration

The next implementation task is **Phase 0, same-method groupwise baseline**, not speculative kernel
editing. Add or adapt one reproducible campaign wrapper that runs the upstream Qwen3.8 corpus and
decode-saturation controls against explicit v2/v3 binaries and artifacts, with FP8 and INT8 KV as
separate named points. Run a short screening cohort first; if the runner and accounting agree,
complete MTP0, MTP3 and DFlash2 K=7 C=1 plus MTP3 C=1/2/4/8. The resulting trace decides whether
the first source change is the v3 regression, FP8 KV, DFlash2 correctness/current upstream work,
or 128-SM launch planning.
