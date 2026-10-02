# RTX 4090 48 GiB performance program

This document is the active authority for closing the Qwen3.8-27B performance gap on the
`zheteng-v/ninfer-4090` v3/sm89 line. It records the production baseline, upstream reference
results, hardware interpretation, ranked hypotheses, experiment order, acceptance rules, and
decision log. Update it at the start and end of every performance iteration. Detailed raw reports
belong under ignored `profiles/bench/`; stable public results belong in `docs/performance/`.

Last updated: 2026-10-02. Production is
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

The retained V2 local tournament selected MTP3 over DFlash2 K=5 for its complete workload:
126.33 versus 122.45 tok/s greedy weighted decode, and 179.55 versus 155.43 aggregate tok/s at
three concurrent requests. DFlash2 nevertheless won the V2 code, translation, structured, and
reasoning category rates. V3 has its own DFlash2 artifact/runtime contract; use the saved V2 figures
as historical comparators and evaluate V3 directly without rerunning V2.

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
| H0 | P0 | benchmark mismatch obscures the real gap | local release sample differs in KV type, sampling, fixture count, output budget and mode | use the frozen cohort and saved V2 results; run new matched measurements on V3 only |
| H1 | P0 | INT8 KV configuration and capacity are correct for the target workload | user constraint: KV cache must never be below INT8; current focused route uses INT8 | hold KV at INT8; tune only non-precision configuration such as context capacity, prefill chunk and concurrency, with exact long-context checks |
| H2 | P0 | improve V3 DFlash2 K=7 toward the +50% V2-relative goal | current V3 K7 is 213.17 tok/s versus saved V2 153.67; host exposure is negligible and device time dominates | retain K7; screen only source-attributed V3 changes, use one matched INT8 K7 request after each viable change, and compare against saved V2 data |
| H3 | P1 | launch plans still encode 5090-scale assumptions | upstream dev now derives attention, norm, MoE and RoPE launches from actual SM count | adapt `a667efdd`, `417eb3d6`, `e621c7d6`; compare launch geometry and end-to-end on 128 SMs |
| H4 | P1 | Q5 small-batch Linear/LinearAdd dominates lost MTP time per round | upstream PR #292 reports +11.4% MTP3 decode from Q5 K-split work on 5090 | port behind measured sm89 dispatch boundaries; sweep T=1..32 and validate end-to-end, not just Op minima |
| H5 | P1 | FP8/int8 causal attention limits prefill and deep-context decode | upstream `7f6aafed` and `d44ab584` retune split-KV/grouped small prefill after our v3 integration point | isolate applicable non-Blackwell pieces, oracle-test, then measure 8K/64K/128K/256K |
| H6 | P1 | v3 short-code regression is acceptance-driven, time-per-round-driven, or both | one sample lost acceptance and throughput while 64K decode stayed flat | replay identical token streams where possible; record rounds, accepted positions, host exposure and GPU time per round; compare v2/v3 Nsight traces |
| H7 | P2 | GDN/KDA recurrence, convolution or fused projections underutilize Ada | they are large fixed costs in Qwen3.8 decode and were not retuned in the v3 landing | kernel timeline and roofline first; tune only the top measured consumers |
| H8 | P2 | graph coverage and host launch gaps limit small batches | cold graph generation is expensive; per-round graph coverage has not been compared v2/v3 | separate startup compilation from steady graph execution; measure graph misses, CPU gaps and eager fallbacks |
| H9 | P2 | scheduler/admission prevents the 48 GiB board from converting capacity into throughput | DFlash2 K7 C4/C8 screens show sublinear aggregate scaling, with device wait dominating host exposure | finish one matched C2 screen, then stop the concurrency sweep unless aggregate throughput becomes a separate product objective |
| H10 | P3 | CUDA 13.4 changes code generation materially versus local 13.1 | two isolated INT8 K7 screens showed no material change versus CUDA 13.1 | retain the isolated CUDA 13.4.92 sm89 toolchain as the preferred local build/test option for environment parity, not as a performance fix; revisit only with a compiler/codegen-specific hypothesis |
| H11 | research | an Ada-specific compact weight format can capture part of NVFP4's traffic benefit | 48 GiB permits alternative resident layouts, but sm89 lacks native FP4 | profile traffic first; prefer offline Q4/Q5/FP8 layouts with independent quality gates; reject runtime repacking/emulation without a net gain |

## Execution order

### Phase 0 — freeze a comparable baseline

Do this before changing kernels.

1. Use `tools/bench/run_serve_corpus.py` and `run_serve_concurrency.py` with the upstream fixtures,
   seeds, stochastic sampling, disabled prefix reuse, text-only residency, prefill chunk 1,024 and
   INT8 KV. KV precision is a hard floor and is not a tuning variable. Record the workload and
   relevant runtime configuration.
2. Run groupwise-int MTP0, MTP3 and DFlash2 K=7 at C=1. Run MTP3 decode saturation at
   C=1/2/4/8 with 16,384 context and 8,192 output tokens.
3. Use the already-recorded V2 performance results as the comparison baseline. Do not spend another
   iteration checking V2 availability or rerunning V2; focus new measurements and implementation work
   on V3. If a required V2 metric has no existing result, leave that comparison unavailable rather
   than starting a V2 validation campaign.
4. Use a fast screening subset while iterating, but require the full 75-request corpus before a
   winner changes production.
5. Add a local scorecard row below; do not overwrite older values.

Exit condition: the project has a same-method v2/v3/official table and enough server JSONL to split
acceptance, device time, host exposure, queueing and memory behavior.

### Phase 1 — configuration and current upstream wins

1. Keep KV fixed at INT8 across all V3 cohort, NIAH-depth and concurrency measurements; vary
   capacity and scheduling parameters, not cache precision.
2. Re-run v3 DFlash2 K=7 only when a code or kernel change needs an end-to-end check; do not spend
   more time on MTP window sweeps for the focused short-code target.
3. Review/adapt upstream `4201b5d2` before trusting DFlash2 prefill state, and use PR #342-style
   per-lane records if attribution is otherwise ambiguous.
4. Sweep prefill chunks 512/1,024/2,048 with KV held at INT8. Keep the 1,024 official point as
   control.
5. CUDA 13.4.92 was screened in isolation on the matched INT8 K7 workload; retain that toolchain locally for environment parity, but do not count it as a speedup or repeat the A/B without a new compiler/codegen hypothesis.

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

1. Establish DFlash2 K7 C=1/2/4/8 decode saturation with short contexts and automatic KV capacity, holding KV at INT8 or higher.
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
| groupwise DFlash2 K=7 code C=1 | 153.67 tok/s (saved V2 point) | 213.17 tok/s (current V3 single-request screen) | V3 +38.7%; short of +50% | 258.6 tok/s | V3-only DFlash2 optimization; do not rerun V2 |
| groupwise MTP0 prefill 7,680 | TBD | TBD | TBD | 3,331.9 tok/s | explain gap with measured roofline |
| groupwise MTP3 steady C=8 | TBD | TBD | TBD | 582.4 tok/s | reach actual batch 8 without errors |
| MTP3 decode-saturation C=4, 4×1,024 output tokens | 185.9 tok/s (single screen) | 187.8 tok/s (single screen) | V3 +1.0% (within unmeasured run variance) | — | V3-only bottleneck analysis; not a C=1 comparison |
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
| 2026-10-01 | GDN Q5 value/z projection at `N=12288,K=5120,T=8`; route split4 SIMT versus K-split MMA C8 | RTX 4090, CUDA 13.1: with 256 MiB L2 flush, 30-trial median 93.184 vs 66.560 us (candidate 28.6% faster); with 1 MiB flush, 69.632 vs 45.056 us (35.3% faster). Re-ran the improved random-RowSplit A/B/oracle harness pinned to the 4090: 93.184 vs 66.560 us; screen rel-L2 `6.661e-6`; 512-row independent RowSplit Q5 FP32 oracle rel-L2 `4.221e-5` for both routes, all norms/nonzero checks passing. The hot-control time matches historical nsys's 69.2 us for the same-named split4 kernel; this suggests, but does not prove without L2 counters, resident/partially resident weights. If all 48 layer calls in the traced DFlash2 round are replaced, the hot A/B difference projects to about 1.18 ms (4.7% of the 25.36 ms device span); this is only a ceiling estimate because overlap, cache residency, and the full schedule are not represented. The production `ninfer_gdn_input_proj_test` also passed pinned to the 4090, including `T=7/8/9` around the T8 dispatch and independent Op oracle/guard checks. These remain Op-level results, not end-to-end evidence. | retain the T=8 candidate; next run one matched V3 short cohort only after the independent 5060 workload releases shared host resources, binding every CUDA process explicitly to the 4090 UUID. Do not claim the Op win as an end-to-end gain. |
| 2026-10-01 | Q5 K-split small-T dispatch for DFlash K=6/7 (`T=7/8`); compare K-split C8 with prior SIMT routes | RTX 4090, CUDA 13.1; A/B `warmup=5,repeat=30`, 256 MiB flush. At `K=6144`, `T=7/8` Linear improved 100.352→37.888 us (~62%), LinearAdd 60.416/62.464→45.056/47.104 us (~25/25%). At `K=17408`, `T=7/8` Linear improved 235.520/241.664→101.376 us (~57/58%), LinearAdd 133.120/141.312→105.472/103.424 us (~21/27%). These are Op timings without an in-bench output oracle; the production Op tests cover the routes, and Linear T=7 was just added for both K values and rebuilt, not rerun. The earlier K=6144/T6 Linear result varied around parity across 10- and 30-trial runs, so no dispatch change is justified there. At `T=4`, candidate C4 ties or slightly loses and production keeps SIMT, matching the current boundary. | retain current `T=5..8` K-split dispatch; add T7 numerical execution to the existing full-oracle test when the 5060 workload is idle, then compare one matched V3 DFlash2 K=6 and K=7 cohort before attributing an end-to-end gain. |
| 2026-10-02 | Current V3 candidate, one matched DFlash2 K=7 code request against saved V3/V2 screens | RTX 4090 UUID confirmed in server log; CUDA 13.1, V3 artifact, INT8 KV, context/KV 8,192, prefill chunk 1,024, same fixed scenario/seed and 4,096 output budget. Current binary SHA `42eff68e…`; decode 194.25 tok/s, 21.081 s, acceptance 55.4%, 840 rounds, device wait 25.09 ms/round and host exposure 13.7 us/round. Saved same-context V3 pre-change binary: 176.30 tok/s, 23.228 s, 52.9%, 871 rounds, device wait 26.65 ms/round; the current point is +10.2% decode throughput. Saved V2 point: 153.67 tok/s, 26.648 s, 51.3%; current V3 is +26.4%. Compared with the V3 control, the higher acceptance and lower per-round device wait both contribute; this is consistent with the Q5/GDN work but does not isolate either kernel. This is a one-sample screen, not a release result; 5060 Stage 2 was concurrently active on its own GPU, while the request was pinned to and logged on the 4090. A separate old Q5-threshold sample used 262,144 context and is not treated as a matched control. | keep the current candidate for focused follow-up; K7 has cleared the >=3% screen gate and is above V2 on this case. Next run one matched K6/T7 point, then inspect acceptance and per-round device wait before selecting the next kernel; do not start the full campaign yet. |
| 2026-10-02 | Current V3 candidate, one matched DFlash2 K=6 code request against saved V3 K6 control | RTX 4090 UUID confirmed; same V3 artifact, INT8 KV, context/KV 262,144, fixture, seed, sampling and 4,096 output budget. Current binary SHA `42eff68e…`; decode 185.97 tok/s, 22.020 s, acceptance 68.16%, 805 rounds, device wait 27.34 ms/round, host 15.43 us/round. Historical V3 K6 screen: 186.76 tok/s, 21.926 s, same acceptance and rounds, device wait 27.23 ms/round, host 11.84 us/round. Difference is -0.4%, below the 3% gate; no measurable end-to-end K6 gain despite Op-level K-split wins. Historical K6 control used the pre-wrapper runner with 262,144 context, confirmed from its server-start event. | do not retune K6 on this single point; the K7 and K6 results show the benefit is workload-sensitive. Profile the K7/K6 route attribution before another implementation change; preserve K7 as the current focused win and defer full matrices. |
| 2026-10-02 | Attribute the K6 result using a current-candidate Nsight Systems capture | Replayed the same K6 request on RTX 4090, then exported the capture to SQLite. Kernel table includes 64 calls to Q5 K-split MMA `(K=17408,N=5120)`, 4.653 ms aggregate (72.71 us/call), but the Q5 GDN `K=5120,N=6144` path remains SIMT split4 (64 calls, 2.560 ms aggregate, 40.0 us/call); the T8 GDN candidate is therefore not exercised by this K6 route. The trace also shows large Q4/Q5 row-split MMA and Q4 small-T totals, so the Q5 K-split Op win alone is not enough to predict this request. These are graph-capture kernel activity aggregates, not a standalone roofline or per-request additive model. Nsight's `.nsys-rep` import emitted a non-fatal target-analysis error because the unrelated concurrently running 5060 UUID was not recognized, but raw capture export to SQLite succeeded; server-start confirms the profiled workload itself ran on the 4090. | K6's neutral result is not evidence to widen its dispatch blindly; inspect the K7 trace next to confirm which new T7/T8 routes execute and whether the measured K7 gain comes from acceptance or device time. Keep the partial trace for route attribution only. |
| 2026-10-02 | Compare same-context K7 Nsight Systems traces (old and current V3 binaries) | Both requests use the same K7 prompt/seed, V3 artifact, INT8 KV, 8,192 context and 4,096 output budget on the 4090. The common first T=8 graph contains GDN Q5 `(N=12288,K=5120)` at 48 ×67.02→45.90 us (−1.01 ms of kernel time); Q4 SwiGLU Schedule16 is 64 ×111.93→108.85 us (−0.20 ms, likely noise); Q5 K=6,144 and K=17,408 routes are unchanged. The old capture also includes a later T=2 graph cluster (52-call Q4 Schedule and small-T Q4/Q5/recurrent kernels), whereas the current capture contains only the T=8 cluster. Thus the apparent disappearance of Q4DraftSmallTSchedule is a capture-window difference, not fusion or rerouting. End-to-end K7 request screens improved, but acceptance also changed 52.89→55.43%; the trace does not isolate that effect, and kernel activity totals must not be treated as a complete additive round-time model. Nsight SQLite export was usable despite an unrelated 5060 target-analysis warning. | Retain the T=8 GDN K-split candidate. Do not widen its dispatch based on the unmatched T=2 cluster. Trace/benchmark the T=2..6 GDN route separately with an independent oracle before selecting a small-T implementation; separately account for acceptance variance in repeated K7 screens. |
| 2026-10-02 | Correct GDN A/B harness to production SplitRows route | Rebuilt `ninfer_q5_ksplit_ab_bench` after changing only its GDN candidate to call the production `launch_q5_ksplit_gdn_value_z_t8` entry (`SplitRows=6144`, separate value/z output planes); the earlier A/B used a contiguous-output ksplit instantiation. On RTX 4090 UUID, CUDA 13.1, 256 MiB flush, warmup=5/repeat=30, production route median was 81.920 us split4 versus 66.560 us K-split (18.75% faster), p95 90.112 versus 68.608 us. Screen rel-L2 `6.661e-6`; 512-row independent Q5 RowSplit FP32 decode oracle rel-L2 `4.221e-5` for each route; all nonzero/norm/pass checks succeeded. This is still an Op measurement and its printed harness flag correctly says `qualification=false`; the prior 28.6% estimate is superseded for this exact production output route. | Retain the production route; use 18.75% as the current exact-route Op result. Recheck with repeated end-to-end K7 before attributing request-level gain; inspect the remaining steady-state hotspots rather than generalizing this single Op delta. |
| 2026-10-02 | Resolve T=2 graph interpretation for K=7 | Source in `src/models/qwen3_5/program/decode.cpp` computes `target_valid_columns = min(draft_window, generated_tokens_remaining-1, capacity-frontier-1)+1`; normal K=7 verification is T=8, and T=2 occurs only when the output budget or context capacity has one token of extent remaining. The old Nsight capture's separate T=2 cluster therefore is not the next steady-state target. | Supersede the previous row's suggestion to tune GDN T=2..6 immediately; prioritize the measured T=8 SIMT projection routes and only revisit partial-width GDN for terminal-tail latency if it matters to the requested metric. |
| 2026-10-02 | Reject the first Q5 attention-output dispatch experiment as a route mismatch | A shape-only A/B for plain Q5 Linear `(N=7168,K=5120,T=8)` found generic K-split 36.6% faster than generic SIMT under a 256 MiB L2 flush, but the K7 trace kernel is `q5_rowsplit_gemm_simt_kernel<...,4,8,2,true,6144>` from the specialized attention split-output Op, not `q5_dispatch`. The temporary `q5_dispatch` T=8 branch was therefore unreachable on the measured model route and was removed, along with its wrapper/test addition. The focused K7 replay had identical draft/accepted tokens, acceptance, and 840 rounds; throughput 194.253→194.351 tok/s and device wait 25.0855→25.0695 ms/round, consistent with no meaningful effect. The actual specialized Q5 attention kernel was 49.6 us baseline versus 49.9 us current in the existing traces; the generic K-split candidate at 53.2 us would not improve it. The exploratory generic-route bench/target was removed as misleading for this K7 hypothesis. | Do not use shape-only launcher comparisons to justify production dispatch. Map split-output ownership and exact operator entry before the next A/B; retain no generic N=7168 production change. |
| 2026-10-02 | Q5 attention gate/value projection T=8: exact split-output A/B and specialized route | RTX 4090 UUID, CUDA 13.1, `N=7168,K=5120,T=8,Q5_G64_FP16`; baseline reproduces `launch_q5_simt<4>` including `SplitOutput=true,SplitRow=6144` and one `(896,2)` launch, candidate uses `launch_q5_ksplit_mma_split<7168,6144,5120,8>`. With randomized old/new order, 30 trials and 256 MiB flush, median 98.304→43.008 us (p95 99.328→54.272); with 1 MiB flush, 55.296→32.768 us (p95 56.320→33.792). Screen rel-L2 `1.413e-6`; independent FP32 RowSplit decode oracle sampled 128 gate +128 value rows across the 6144 seam, old/new rel-L2 `1.714e-3`, both pass the 1e-2 screening bound. These are Op screening results, not qualification by themselves. The production T=8 branch now calls the split candidate; T=1, T=2..6, T=7, and T=9..12 retain their existing routes. `ninfer_attn_input_proj_test --q4-q5-only` passed on the pinned 4090 (exit 0); that public suite runs T=1..128 against its independent packed-weight oracle and guard checks. One matched K7 request on the rebuilt binary (`e8373309…`) reached 206.86 tok/s, 24.704 ms device-wait/round, 58.76% acceptance and 801 rounds; a same-binary/same-seed repeat was 206.66 tok/s, 24.725 ms/round, with identical accepted/drafted counts and rounds. The prior saved point was 194.35 tok/s, 25.069 ms/round, 55.43% acceptance and 840 rounds. Thus the fixed prompt/seed's changed acceptance is reproducible, but the 6.4% throughput delta cannot be attributed to kernel time alone; per-round device wait improved about 1.4%. Current V3 is about 34.6% above the saved V2 K7 point (153.67 tok/s), still below the +50% objective. Nsight Compute was attempted for the remaining Q4 route but blocked by system `ERR_NVGPUCTRPERM`; no profiling-security setting was changed. | Keep the attention Q5 T=8 route: exact-route Op and public qualification checks pass, and two new-route request runs are stable. Treat the per-round reduction as modest and the acceptance shift as prompt-specific; do not extrapolate it to the corpus. Next, benchmark the remaining production Q4 SIMT projections with a faithful small-tile MMA candidate and independent oracle, then measure one focused request only if that candidate qualifies. |
| 2026-10-02 | Reject generic fat-tile Q4 MMA for the remaining T=8 projections | Exact production SIMT baselines for GDN QK `(N=4096,K=5120,T=8)` and attention QKV `(N=7168,K=5120,T=8,SplitRow=6144)` were compared against existing 16x8, 32x8, and 16x16 row-split MMA tiles on the RTX 4090, CUDA 13.1, 30 randomized trials, 256 MiB flush. Every full-output BF16 screen and sampled independent FP32 RowSplit oracle passed the 1e-2 screening gate, but GDN candidate/baseline median ratios were 2.20, 2.20, and 2.09; attention ratios were 1.88, 1.83, and 1.85. Small output-column tiles leave too few warps/CTA and the MMA kernel iterates over 80 separate 64-value K groups, exposing synchronization and memory latency. | Reject this MMA route for T=8; do not add a production dispatch. Test the existing 16-warp Q4 small-T kernel family instead. |
| 2026-10-02 | GDN QK Q4 T=8 small-T MMA screening, route, and matched K7 replay | Exact GDN QK A/B `(N=4096,K=5120,T=8,Q4_G64_FP16)` compared production SIMT with `Q4DraftSmallTSchedule16` small-T MMA. On RTX 4090, 30 randomized trials: 256 MiB flush median 35.840→31.744 us (11.4% faster; p95 36.864→39.936), 1 MiB flush 30.720→26.624 us (13.3%; p95 31.744→27.648). Full BF16 screen rel-L2 `1.807e-6`; independent FP32 RowSplit oracle over 256 rows rel-L2 `1.714e-3` for both routes. The T=8 production dispatch now uses this small-T kernel; the first GDN Op test exposed an output-leading-stride bug because QK writes into the parent QKV tensor (`out.nb[1]/2=10240`, not 4096), which was corrected by passing the actual view stride. After the fix, `ninfer_gdn_input_proj_test` (including T=7/8/9 independent oracle and guards) and `ninfer_linear_q4_a16_test` both passed on the 4090. Two matched K7 runs on the updated binary, same artifact/INT8/context/fixture/seed, measured 213.61 and 213.30 tok/s, both 793 rounds and identical 3302/5544 accepted/drafted tokens (59.56%); device wait was 24.164 and 24.199 ms/round. The immediately preceding Q5-only build's repeated point averaged about 206.76 tok/s, 801 rounds, 58.76% acceptance, 24.715 ms/round. Thus this revision is +3.2% on the focused request and −2.2% device wait/round, but acceptance also shifted, so request-throughput change is not attributable solely to the Q4 kernel. Against the saved V2 point (153.67 tok/s), the updated focused V3 point is about +38.9%, still short of the +50% target. | Keep the route: Op screen, public independent-oracle/guard tests, and repeated end-to-end results all support the change. Preserve the stride fix and test at the sliced output boundary. Next measure the still-unoptimized attention QKV Q4 T=8 route with a split-output small-T candidate; do not run the full benchmark matrix. |
| 2026-10-02 | Reject attention QKV Q4 T=8 small-T MMA production routing | The exact split-output Op A/B `(N=7168,K=5120,T=8,Q4_G64_FP16)` passed the full BF16 screen (rel-L2 `4.401e-5`) and independent FP32 oracle across the 6144-row seam (rel-L2 `1.639e-3`), with RTX 4090 median 57.344→39.936 us (30.4% faster; p95 61.440→51.200). The public `ninfer_attn_input_proj_test --q4-q5-only` passed on the 4090 for the temporary production route. However, two same-seed K7 requests with this route averaged 211.03 tok/s versus 213.46 tok/s on the preceding build (−1.14%); device wait/round improved 24.182→23.913 ms, while speculative acceptance fell 59.56→57.88% and rounds rose 793→811. The request result is repeatable, but it does not support a net product win for this fixed workload. The Q4 production dispatch was reverted; the exact-route A/B bench and target remain as diagnostic evidence. The existing Q5 attention T=8 route and GDN QK small-T route are unchanged. | Keep attention QKV Q4 on its original SIMT route. A faster Op is insufficient when the matched speculative request regresses; retain this as evidence to include acceptance and end-to-end time in dispatch decisions. Choose the next hypothesis from current K7 trace attribution, not another shape-only launcher swap. |
| 2026-10-02 | Close Q5 LinearAdd K=6144 RowsPerCTA/KWarps schedule-only tuning | Current K7 trace shows Q5 K-split `(N=5120,K=6144,T=8,AddResidual)` at 64 × about 32 us (about 2.05–2.10 ms/graph). The proposed RowsPerCTA=8 schedule is invalid because the m16n8k16 fragment unconditionally reads and writes rows `gid` and `gid+8`. A KWarps=16-only change violates the loader's one-warp 32-lane chunk budget (`code_chunks+high_chunks+scale=41>32`; limit is 32) and its `__launch_bounds__(512,3)` resource envelope (estimated maximum register budget 42/thread versus observed 92–105). Both mechanisms would require kernel-structure redesign rather than a schedule parameter; the trial code/CMake target was removed before any GPU run, and `ninfer_linear_q5_a16_test` rebuilt successfully. No production route changed. | Close this parameter-only avenue. Do not attempt KWarps=16 without redesigning both staging and resource usage; reserve multi-CTA K-split plus reduction for a separately justified design. No other schedule-only kernel candidate currently has a credible >=3% per-round ceiling. |
| 2026-10-02 | Same-binary DFlash2 K=6 versus K=7 matched control on the rebuilt current source | Rebuilt `build/apps/ninfer-serve` (SHA-256 `441b85a66008ae25…`) after the attention Q4 production revert, then ran one request per draft window on the RTX 4090 bound by `CUDA_VISIBLE_DEVICES=GPU-2f39017c-6cf6-5c22-6c8b-aff9ef65a4bd`, same V3 artifact, INT8 KV, `max-context`/`kv-capacity` 8192, prefill chunk 1024, `--no-prefix-reuse`, pinned stochastic sampling, fixture `scenario_code_python`, seed 7632647173703958409, and a 4096-token output budget. Both points share `prefill_signature cf336425f495069a…`; the only difference is `--draft-tokens` 6 versus 7. K=6: 186.3888 tok/s, 805 rounds, 3,290/4,827 accepted/drafted (68.16%), 5.087 tokens/round, 27,268.19 us/round device wait, 32.34 us/round host, 22.1 s. K=7: 213.1728 tok/s, 792 rounds, 3,302/5,544 (59.56%), 5.169 tokens/round, 24,203.00 us/round device wait, 28.31 us/round host, 19.3 s. K=7 is +14.4% throughput with -11.2% device wait/round and 1.6% fewer rounds; prefill and TTFT are unchanged (1086.7 versus 1085.4 tok/s; 112.95 versus 112.87 ms). K=6 accepts a higher fraction of a shorter window but still yields fewer tokens per round, because six drafts cannot offset the extra rounds. The K=6 point reproduces the earlier 262144-context K=6 screen (186.76 tok/s, 21.926 s, 68.16%, 805 rounds) within 0.2%, so context capacity does not affect this short-prompt decode, and the K=7 point matches the previously reported K=7 values (213.46/213.61/213.30) within 0.2%. These are single-request configuration screens, not end-to-end or corpus-level evidence; the focused request remains about +38.7% over the saved V2 point rather than the +50% objective. | Keep K=7 as the focused short-code draft window; K=6 has no product advantage on this workload and is closed as a configuration candidate. Do not run the full K matrix. |
| 2026-10-02 | Same-binary MTP3 versus DFlash2 K=7 single-point control at C=1 | One request on the same `build/apps/ninfer-serve` (SHA-256 `441b85a66008ae25…`) with the K=7 point's artifact, INT8 KV, 8192 context/KV, prefill chunk 1024, `--no-prefix-reuse`, pinned stochastic sampling, fixture `scenario_code_python`, seed 7632647173703958409 and a 4096-token budget; the only difference from the K=7 point is `--spec mtp --draft-tokens 3`. MTP3: 147.5550 tok/s, 1,214 rounds, 2,881/3,642 accepted/drafted (79.10%), 3.373 tokens/round, 22,843.34 us/round device wait, 21.44 us/round host, 27.76 s. DFlash2 K=7 on the same binary: 213.1728 tok/s, 792 rounds, 3,302/5,544 (59.56%), 5.169 tokens/round, 24,203.00 us/round device wait, 19.21 s. MTP3 accepts a much larger fraction of its window and has the cheapest round of the three configurations (22.84 versus 24.20 versus 27.27 ms/round for K=6), but three drafts yield only 3.37 tokens/round, so it needs 53% more rounds and is 30.8% slower end to end. TTFT (115.35 versus 112.87 ms) and prefill (1063.6 versus 1085.4 tok/s) are unchanged. This is a single-request C=1 screen; it does not test the higher-concurrency MTP3 operating point. | Keep DFlash2 K=7 as the focused C=1 configuration; close MTP3 for the focused short-code request at C=1 while leaving C>1 untested. |
| 2026-10-02 | Same-binary MTP4 single-point control completes the C=1 draft-window sweep | One request on the same `build/apps/ninfer-serve` (SHA-256 `441b85a66008ae25…`) with the MTP3 point's artifact, INT8 KV, 8192 context/KV, prefill chunk 1024, `--no-prefix-reuse`, pinned stochastic sampling, fixture `scenario_code_python`, seed 7632647173703958409 and a 4096-token budget; the only difference from the MTP3 point is `--spec mtp --draft-tokens 4`. MTP4: 149.2818 tok/s, 1,085 rounds, 3,010/4,337 accepted/drafted (69.40%), 3.774 tokens/round, 25,262.93 us/round device wait, 24.63 us/round host, 27.44 s. MTP3 was 147.5550 tok/s at 1,214 rounds, 79.10% acceptance (2,881/3,642), 3.373 tokens/round, 22,843.34 us/round device wait and 27.76 s. MTP4 raises tokens/round by 11.9% but its round costs 10.6% more, so the net gain over MTP3 is 1.2%; both remain about 30% below DFlash2 K=7 and 20% below K=6. Across the four same-binary points the round cost falls with the draft window for the MTP backend (22.84/25.26 ms/round for MTP3/MTP4) while token yield falls faster, so no MTP window competes with DFlash2 K=7 at C=1. | Close the C=1 draft-window sweep: DFlash2 K=7 is the focused configuration and further MTP drafting or K expansion at C=1 is not justified. MTP concurrency scaling remains a separate, untested question. |
| 2026-10-02 | Reject 32-logical-row CTA grouping for Q4 SwiGLU T=8 | RTX 4090 sm89, CUDA 13.1; one Op shape `(N=34816,K=5120,T=8,Q4_G64_FP16)`, randomized A/B, warmup=3/repeat=15, 256 MiB flush. A prototype made one CTA stage two m16 row tiles (32 logical rows: 16 gate + 16 up), halving CTA count and activation reads from about 178.3 to 89.1 MB/call while keeping weight traffic and each row's K reduction tree unchanged. Full output matched production bit-for-bit; independent FP64 RowSplit Q4 oracle plus SiLU/product/BF16 rounding passed on 256 rows × 8 tokens, also bit-for-bit. But cold-L2 median regressed 153.600→157.696 us (+2.7%, p95 154.624→158.720 us); the larger single-buffer staging (25,088→33,792 B) outweighed the activation-traffic reduction. The hypothesis that repeated activation L2 reads explain the Q4 SwiGLU gap is rejected. The prototype, type extraction, and bench were removed; existing production Schedule16 and output-stride edits were preserved. | Keep the production route unchanged. Do not pursue row grouping further. A double-buffered K-group pipeline is a distinct, larger redesign with a per-kernel DRAM-floor ceiling of roughly 3.9% E2E and must earn an Op A/B before any production route change. |
| 2026-10-02 | Reject intra-CTA double-buffer pipeline for Q4 SwiGLU T=8 | RTX 4090 sm89, CUDA 13.1; same `(N=34816,K=5120,T=8,Q4_G64_FP16)` with randomized 3-arm A/B (production, 2-stage serialized control, true cp.async overlap), warmup=3/repeat=15, 256 MiB flush. The candidate's final K group was explicitly drained with `cp_wait<0>`; the serialized control and pipelined candidate were both bit-exact to production across all outputs, and all three matched an independent FP64 RowSplit Q4 + SiLU/product/BF16 oracle on 256 rows × 8 tokens. ptxas: production 56 regs/25,088 B static shared/0 spill; pipeline arms 40/44 regs/0 spill, 50,176 B dynamic shared; occupancy remained 2 CTAs/SM. Production median was 153.600 us, same-footprint serial control 154.624 us, pipelined candidate 152.576 us: candidate +1.32% vs the control, but only +0.67% vs production; a second run varied between +0.67 and +1.33% vs production, so net gain is within run variation (about 0.2% projected E2E). The partial result from the first build was discarded because it used a stale binary after an interrupted build; only the corrected, fully rebuilt run is used here. The pipeline prototype/bench/CMake entry were removed, and production instantiation resources were rechecked unchanged. | Reject this intra-CTA pipeline as a product optimization: CTA-level concurrency already hides most group latency for this shape. Further Q4 SwiGLU work needs a cross-CTA mechanism and a broader cumulative ceiling, not another intra-CTA schedule tweak. |
| 2026-10-02 | Q5 K-split fixed-work diagnostic, K=6144 versus K=17408 | RTX 4090 UUID, CUDA 13.1; rebuilt `ninfer_linear_bench`, pinned to the 4090, eager, warmup=3/repeat=20, 256 MiB flush. Both points use `N=5120,T=8`, the same Q5 K-split schedule and `(320,1)` grid; diagnostic `AddResidual=false`, production LinearAdd trace `AddResidual=true` (same staging/MMA/split geometry, different epilogue). K=6144: median 40.960 us for 20.64 MB packed weights (503.9 GB/s inferred); K=17408: 95.232 us for 58.49 MB (614.2 GB/s inferred). These packed-weight-bytes/time rates are not Nsight DRAM counters. At fixed grid, 2.83x more K groups raised inferred rate by 21.9%, supporting fixed per-CTA work/prologue/staging amortization over a simple occupancy-wave-only explanation; two points do not uniquely identify a model. A two-point fit suggests about 11.4 us fixed cost and 697 GB/s marginal rate, directional only. | Reject cross-CTA K splitting as the next K=6144 optimization: reducing work per CTA likely worsens fixed-cost amortization. Do not collect a third point absent a new hypothesis. The remaining C=1 projections offer no credible ≤3-change path to the ~1.81 ms/round needed for +50%; reaching that target requires a major redesign. Measure V3 C=4 aggregate MTP3 separately as the next capacity/throughput question; it is a different metric. |
| 2026-10-02 | V3 MTP3 C=4 decode-saturation screen | RTX 4090 UUID, CUDA 13.1/driver 615.71.09; V3 artifact, INT8 KV, context 8,192, KV capacity auto (32,768 tokens), prefill chunk 1,024, no prefix reuse, `long_decode_aime26_15`, fixed stochastic profile, first four saturation seeds, four concurrent 1,024-token outputs. One fixed-wave sample completed 4/4 requests and 4,096/4,096 output tokens with no errors. Steady decode was 187.80 tok/s over 20 complete seconds, average steady batch 4.00; makespan 22.805 s; MTP accepted 2,451/4,909 drafted (49.93%). Startup memory: weights 17.90 GB, KV payload 1.177 GB, sequence 1.847 GB, workspace 160 MB, graph allowance 344 MB, 30.57 GB available. The existing V2 C4 screen is 185.95 tok/s at batch 4; V3 is +1.0% on this single sample, but run-to-run variance is unmeasured and V2's schema-20 prefill signature is unreported. Treat as near parity, not a release claim. | Use the V3 point to inspect scheduler/device/host breakdown and select one V3-side throughput hypothesis. Do not rerun or validate V2. This aggregate C4 result is separate from the single-request K7 +38.7% result and cannot be used to claim C1 +50%. |

| 2026-10-02 | V3 DFlash2 K=8 screen with optimized proposal head | RTX 4090 UUID, CUDA 13.1; same V3 binary/artifact, fixture `scenario_code_python`, seed `7632647173703958409`, stochastic sampling, INT8 KV, context/KV 8,192, output limit 4,096, no prefix reuse, prefill chunk 1,024; only draft window changed from the matched K7 point. Server argv confirmed `--spec dflash2 --draft-tokens 8 --lm-head-draft`. One sample completed 4,096 tokens: 119.58 tok/s, 34.4 s decode, 908 rounds, 3,187/7,257 accepted/drafted (43.92%), 4.510 tokens/round, 37.701 ms device-wait/round, 18.67 us host/round, TTFT 112.98 ms. Matched K7 on the same binary/seed was 213.17 tok/s, 792 rounds, 59.56% acceptance, 5.169 tokens/round, and 24.203 ms device-wait/round. K8 is 43.9% slower on this screen, with 55.8% higher round cost and lower token yield; host exposure remains negligible. The complete request finished without errors; 4090 returned to idle. | Reject K=8 for the focused short-code workload; keep DFlash2 K=7 and do not sweep higher K. This is a one-request configuration screen, not a multi-seed corpus result. |
| 2026-10-02 | Historical V3 DFlash2 K=7 FP8-KV screen (superseded by the INT8 floor) | An earlier one-off screen used `--kv-dtype fp8`: 192.2 tok/s and 51.9% acceptance versus the matched INT8 point at 213.17 tok/s and 59.56%. Retain this as historical evidence only; it does not authorize repeating a lower-than-INT8 KV test. | Do not use FP8 KV in future tests or production; keep KV at INT8 or higher. |
| 2026-10-02 | DFlash2 MLP gate/up Q8→Q4 derived-artifact screens | RTX 4090 UUID, same V3 binary, K7 fixture/seed/sampling/context/output budget, with runtime KV fixed to INT8. Five DFlash2 gate/up parents `(34816,5120)` were re-encoded from the current artifact's stored Q8 values to Q4_G64 (not from pinned BF16 source weights); artifact structure and bindings were preserved. Absmax scale fit: 212.01 tok/s, 57.64% acceptance, 5.03 tokens/round, 23.72 ms device-wait/round. Least-squares scale fit: 204.19 tok/s, 54.94% acceptance, 4.84 tokens/round, 23.69 ms device-wait/round. Both are single-request screens; neither has an independent end-task quality score or multi-seed confirmation. The slight reduction in device time did not offset lower draft acceptance/token yield, and LS was slower overall. These measurements do not qualify either lower-bit weight artifact for production. | Reject both derived Q4 weight candidates for this cohort; keep the released artifact and INT8 KV. Do not trade model/cache precision for a throughput-only result. A future weight-format proposal needs a quality-preserving acceptance/quality gate and pinned source weights before production consideration. |
| 2026-10-02 | GDN recurrent-record T=8 next-key prefetch screen | RTX 4090 UUID, Qwen 27B geometry `(value_heads=48,T=8,B=1)`, production record kernel versus a bench-only schedule variant that loads the next raw key before applying the current recurrent transition, then keeps normalization and arithmetic order unchanged. A 16-node CUDA Graph per arm with randomized A/B order, 5 warmups and 30 paired samples produced bit-identical output, key/value/gate records. Production median/p95 was 9.920/9.984 us per Op; prefetch was 9.984/10.048 us (ratio 1.0065, 0.65% slower). This is below the 3% noise gate, so the prototype was removed and no production code changed. The existing single-launch eager event measurement was about 23.55 us because its timing envelope does not isolate the graph-resident kernel; the graph mode amortizes boundaries across 16 idempotent calls and measures about 9.92 us/Op. Historical Nsight Systems kernel duration was 11.293 us; the two methods agree on scale but are not interchangeable. | Reject next-key prefetch; do not route it to production. Retain the explicit recurrent CUDA-Graph benchmark option for future graph-resident comparisons. Nsight Compute remains unavailable, and Nsight Systems export on this host fails on the concurrently visible 5060 Ti UUID, so kernel-level counter attribution remains blocked. |
| 2026-10-02 | Reject 2-way K-segment Q5 FFN-down candidate at end to end | RTX 4090 UUID, CUDA 13.1; Q5_G64_FP16 LinearAdd `(N=5120,K=17408,T=8)`. The bench-only 2-segment K reduction passed a full-output comparison against production (relative L2 `6.14e-5`) and a 64-row independent FP64 RowSplit-Q5 plus BF16-residual oracle (both production/candidate `1.69e-3`). Randomized cold-L2 Op A/B medians were 106.50→100.35 us, then 107.52→102.40 us (candidate 4.8–5.8% faster at the Op scope). A matched V3 DFlash2 K7 request on the same fixture/seed/artifact/context, with KV fixed to INT8, reversed the decision: segmented candidate 207.7 tok/s and 56.7% acceptance (3,270/5,770) versus production baseline 212.7 tok/s and 59.56% (3,302/5,544), about 2.4% slower end to end. The changed FP32 reduction association perturbs draft acceptance enough to erase the local kernel gain. The segmentation kernel, route, and dedicated bench were removed; the original production route was rebuilt and `ninfer_linear_q5_a16_test` passed. | Reject K-segment reduction for this DFlash2 K7 route. Keep the existing Q5 K-split implementation and INT8 KV; do not infer model-level value from the Op-only speedup. |
| 2026-10-02 | Reject Q4 SwiGLU T=8 8-warp schedule | RTX 4090, Q4_G64_FP16 `(N=34816,K=5120,T=8)`, cold-L2 256 MiB randomized schedule A/B, 3 warmups/30 trials. The production 16-warp schedule measured 146.43 us median (151.55 p95); the existing 8-warp schedule measured 158.72 us (162.82 p95), 8.4% slower. Full BF16 screen rel-L2 was `6.13e-5`; independent FP64 RowSplit Q4 + SiLU×up oracle on 64 rows gave `1.54e-3` for both schedules. The temporary bench and target were removed; no production route changed. | Retain the existing 16-warp T=8 schedule; do not spend an end-to-end request on the slower 8-warp candidate. |
| 2026-10-02 | Reject Q5 K-split C16 for FFN-down T=8 | RTX 4090, Q5_G64_FP16 LinearAdd `(N=5120,K=17408,T=8)`, 256 MiB flush, randomized old/new order, 3 warmups/20 trials. Full-output screen was bit-exact (0/40,960 mismatches), but production C8 median/p95 was 100.35/108.54 us versus C16 122.88/128.00 us (22.4% slower). The temporary `--capacity-t8-ab` comparison mode was removed. | Keep C8 for T=8; do not route the slower wider capacity. |
| 2026-10-02 | CUDA 13.4 toolchain screen for V3 DFlash2 K7 | Built `ninfer-serve` and `ninfer_linear_q5_a16_test` in an isolated CUDA 13.4.92 toolkit/build directory for sm89; the Q5 A16 test passed. Two matched V3 requests used the same artifact, fixture/seed, sampling, 8,192 context, 4,096-token budget, DFlash2 K7, optimized proposal head, and INT8 KV. CUDA 13.4 produced 213.52 and 212.9 tok/s, 24.169 and 24.235 ms device-wait/round, and 59.56% acceptance in both runs. This is indistinguishable from the CUDA 13.1 same-source K7 control (213.17 tok/s, 24.203 ms/round) and within observed run variation; no material toolchain gain was demonstrated. | Do not promote CUDA 13.4 as a performance fix for this workload. Keep it as a reproducible isolated toolchain option only; continue investigating current V3 device-time hotspots, with KV at INT8 or higher and no V2 rerun. |
| 2026-10-02 | V3 DFlash2 K7 C2/C4/C8 decode-saturation screen | RTX 4090 UUID, same `build/apps/ninfer-serve` binary (CUDA 13.1 compile, CUDA 13.4 runtime/driver), V3 artifact, DFlash2 K7, optimized head, stochastic `long_decode_aime26_15`, first two/four/eight fixed saturation seeds, 1,024 tokens/request, context 8,192, auto KV, prefill 1,024, INT8 KV, no prefix reuse. C2/C4/C8 aggregate rates: 125.7/218.7/348.5 tok/s at steady batch 2/4/8; C2→C4 was +74%, C4→C8 +59%, so throughput scaling is sublinear. C2: 15 complete seconds, 2/2×1,024 tokens, makespan 16.342 s, acceptance 33.84%, 608 rounds, device wait 51.69–52.43 ms/round, host 26.44–26.51 μs/round, auto KV 16,384. C4: 16 complete seconds, makespan 21.191 s, 4/4×1,024 tokens, acceptance 33.33%, 1,231 rounds, device wait 55.45–62.45 ms/round, host 27.17–30.01 μs/round, auto KV 32,768. C8: 20 complete seconds, makespan 27.576 s, 8/8×1,024 tokens, acceptance 35.68%, 2,346 rounds, device wait 76.15–81.91 ms/round, host 39.1–41.6 μs/round, auto KV 65,536. The three single samples are a screening curve only, not a stable service claim or C1 improvement. An earlier C4 sample compiled with CUDA 13.4 yielded 234.8 tok/s and is not mixed into this same-binary curve. | Close the planned C2/C4/C8 aggregate screen; do not expand concurrency testing absent a separate product objective. Keep KV at INT8 or higher and do not count aggregate throughput as progress toward C1 +50%. Return effort to a concrete V3 C1 source/kernel hypothesis. |
| 2026-10-02 | Audit upstream SM-scaled launch plans and Q5 K-split for current V3 K7 | Current RTX 4090 K7 Nsight Systems trace: H3's potentially affected norm/RoPE/attention work has no applicable dispatch change at current shapes and is below the 0.73 ms/round 3% gate; MoE is absent. Q5 K-split is already active for 192 calls totaling 9.366 ms (FFN-down 4.596, o_proj 2.099, GDN value/z 2.204, attention Q5 0.467). C16 is already 22.4% slower for FFN-down T8 and the K6144 split-K diagnostic rejects further segmentation. | No-go on H3/H4 porting or capacity changes for this profile; screen one isolated main-model Q5→Q4 LinearAdd candidate next because its estimated byte reduction is the first remaining route above the 0.73 ms gate. No artifact or production route change until Op and quality screens pass. |
| 2026-10-02 | Reject main-model Q5→Q4 LinearAdd T=8 candidate | RTX 4090 sm89, CUDA 13.4, shapes `(N=5120,K=17408/6144,T=8)`, 256 MiB flush, randomized A/B, warmup=3/repeat=15. Bench-only Q4 small-T MMA + fused BF16 residual was compared with production Q5 K-split LinearAdd. Independent FP32 RowSplit oracles passed for both formats (rel-L2 1.70e-3/1.67e-3 at K17408 and 1.69e-3/1.62e-3 at K6144). Q4 was slower: 108.544 vs 97.280 us (+11.6%) and 44.032 vs 40.960 us (+7.5%); weighted over 64 calls of each shape, this adds 0.918 ms per decode round. Full-output Q4-vs-Q5 rel-L2 was 7.5%, a separate model-quality warning, not an Op correctness criterion. | Close this route before artifact conversion or production dispatch. Keep the benchmark as a screening record; do not infer all clocks/inputs will regress identically. No V2 rerun, KV remains INT8+. |

## Immediate next iteration

Do not repeat MTP-window or DFlash2 K8 tests: MTP3/MTP4 remain far below K7 at C1 and the matched K8
screen regressed to 119.58 tok/s. Use saved V2 results only; do not rerun or validate V2. V3 DFlash2
K7 C1 remains 213.17 tok/s (+38.7% over saved V2), short of the 230.5 tok/s +50% target. The
advanced-ideas note's K8 item is closed; Q8-derived DFlash2 Q4 screens did not improve end-to-end
throughput, and fetching pinned BF16 sources remains deferred because the expected gain is small
relative to download/conversion cost. Never lower KV below INT8. The DFlash2 K7 C2/C4/C8 screen is
complete at 125.7/218.7/348.5 aggregate tok/s with sublinear scaling; this separate metric does not
alter the C1 goal. The upstream launch-plan/Q5 K-split audit and the Q5→Q4 LinearAdd screen found no
remaining low-risk ≥3% kernel candidate: the latter's independent Op oracles passed, but the Q4 route
was slower by 7.5–11.6% and its weighted change was −0.918 ms/round. Do not spend time on its artifact
conversion or production routing. The current C1 single-request ceiling remains 213.17 tok/s; reaching
230.5 tok/s (+50% over saved V2) now appears to require a broader algorithm/kernel redesign, not another
small dispatch tweak. A next experiment must start from a new, specific T=8 device-time hypothesis and
show an Op-level ceiling above 0.73 ms/round before implementation. Nsight Compute remains blocked by
`ERR_NVGPUCTRPERM`; keep KV at INT8 or higher and do not rerun V2.
