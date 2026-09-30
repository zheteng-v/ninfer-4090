# RTX 4090 downstream maintenance contract

This document is the single authority for maintaining
[`zheteng-v/ninfer-4090`](https://github.com/zheteng-v/ninfer-4090). It records repository
ownership, upstream review, integration policy, validation gates, releases, and the development
roadmap. Architecture and Op details remain owned by their existing maintainer references.

## Product position

The downstream product is a native Linux NInfer server for one `sm_89` RTX 4090 with 48 GiB of
VRAM. Its production workload is Qwen3.8-27B groupwise-int, long context, one or two active
requests, Vision, OpenAI/Anthropic-compatible APIs, and MTP3 speculative decoding.

The priorities, in order, are:

1. preserve answer correctness, API contracts, and recoverable production service;
2. stay close enough to `Neroued/ninfer` that new model and artifact generations can be adopted;
3. retain or improve the best `sm_89` kernels and the 48 GiB long-context advantage;
4. publish reproducible end-to-end evidence instead of isolated peak numbers;
5. contribute generally useful fixes upstream while keeping Ada-specific code explicit.

"Faster" is never accepted from a microbenchmark alone. A performance claim needs the same model
artifact, prompt cohort, sampling policy, context depths, concurrency, GPU power/clock policy, and
quality gates on both candidates. Regressions and losing cohorts remain in the report.

## Repository and branch ownership

| Remote | Fetch source | Role |
|---|---|---|
| `origin` | `zheteng-v/ninfer-4090` | maintained downstream; the only push destination |
| `upstream` | `Neroued/ninfer` | authoritative architecture, model, artifact, runtime, and generic optimization source |
| `sergiuszm` | `sergiuszm/ninfer-4090` | validated Ada port and second source of `sm_89` fixes |

Push URLs for `upstream` and `sergiuszm` are deliberately disabled locally. Credentials, model
artifacts, runtime configuration, raw private prompts, `/data/llm/git.md`, and `.local/` must never
be committed.

Branch roles:

- `main`: deployable, reviewed, and validated on the local 48 GiB RTX 4090. It currently preserves
  the proven v2/sm89 line.
- `sync/YYYY-MM-DD-v3-sm89`: temporary integration branch based on the last validated Ada line,
  with the upstream v3 architecture replayed in its original commit order and conflicts reviewed.
- `perf/<topic>`, `fix/<topic>`, `feat/<topic>`: one bounded decision per branch.
- `vendor/sergiuszm-rtx4090-port`: optional read-only mirror of the observed vendor head. No local
  work starts from this name without a fresh audit.

Do not merge `upstream/master` wholesale into the current v2 line. At the 2026-09-30 baseline the
two sides have both rewritten core artifact/model/runtime code. A direct sm89 build of the audited
master was attempted and rejected: current master unconditionally reaches Hopper/Blackwell TMA,
cluster barrier, block-scale MMA, and PDL instructions in multiple core paths. The active migration
therefore starts from the validated Ada line and replays the upstream v3 converter, loader, and
bound-instance Engine milestones before selectively adopting later work. `main` remains the
rollback line until the v3 candidate passes every release gate.

## Mandatory iteration start

Run this before choosing or implementing work:

```bash
tools/maintenance/upstream-audit.sh
```

The command fetches all three remotes, records immutable heads and ahead/behind counts, lists new
commits, and queries recently updated pull requests from both upstream repositories. It never
merges, rebases, or pushes. If GitHub's public API is unavailable, inspect the two pull-request URLs
printed by the script before continuing.

For every iteration:

1. require a clean worktree and record the audit date and three source SHAs;
2. inspect `upstream/master`, `upstream/dev`, and recently updated/open/merged PRs;
3. inspect `sergiuszm/rtx4090-port` and its PRs;
4. classify each relevant change as `adopt`, `adapt`, `benchmark first`, `watch`, or `not applicable`;
5. select one coherent change and write its acceptance and rollback criteria before editing;
6. preserve upstream authorship; use `cherry-pick -x` only when the patch still matches the design;
7. update the sync record at the end, including rejected options and negative results.

Open PRs are evidence and design input, not release dependencies. They must be read, tested, and
attributed; they are not copied merely because a headline reports a speedup.

## Integration rules

- Prefer upstream interfaces and ownership boundaries. Ada support should be an architecture
  profile or dispatch route, not a fork-wide parallel implementation.
- Never select a CUDA path only by GPU name. Gate it by architecture and the exact required
  instruction or storage capability.
- Do not port `sm_120a` NVFP4/W4A4 kernels to Ada. Port generic scheduling, artifact, frontend,
  serving, and numerical improvements independently from Blackwell-only kernels.
- Re-evaluate old downstream fixes against current upstream code. A previous bug may already be
  fixed under a different ownership model.
- Numerical kernel changes need an independent CPU/high-precision oracle, boundary shapes, odd
  tails, and at least one real-model output check.
- Performance work begins with a profile. Keep a change only when end-to-end benefit survives the
  quality and memory gates.
- Conversion and loader changes must reject unsupported artifacts clearly; never silently reinterpret
  v2 bytes as v3 metadata.
- Maintain native startup as the primary deployment path. Containers are for reproducible build or
  test environments, not a second source tree.

## Validation and release gates

The candidate is not deployable until all applicable rows pass on the physical RTX 4090.

| Gate | Minimum evidence |
|---|---|
| Build | clean Release build for `sm_89`; no accidental `sm_120a` requirement; reproducible compiler/CUDA record |
| Unit and component | focused changed-area tests, then all runnable CTest suites; no ignored new failure |
| Artifact | v3 schema/reader/converter tests; v2-to-v3 upgrade tested on a copy; checksums and tensor inventory preserved |
| Numerics | independent oracles for changed kernels, boundary/tail shapes, deterministic short-answer and multi-turn probes |
| Model | real Qwen3.8-27B load, text, Vision, tool-call, reasoning-content, MTP3, and clean shutdown |
| Long context | exact NIAH at 8K, 64K, 128K, and 256K; prefix save/restore and edited-history reuse |
| Serving | OpenAI and Anthropic non-stream/stream, cancellation, queue timeout, `/health`, `/metrics`, and two-lane isolation |
| Performance | warm and cold TTFT, prefill tok/s, decode tok/s, TPOT, MTP acceptance, aggregate concurrency, peak/resident VRAM |
| Soak | repeated mixed requests long enough to expose allocator, cache, cancellation, and state-lifetime failures |
| Rollback | previous binary/artifact/config starts successfully on a canary port before production replacement |

Benchmarks report distribution and workload cohorts, not only the best average. Record P50/P95
where repetitions permit it, clocks/power state, driver, CUDA version, model hash, commit, exact
command, warmup policy, and raw result location. A result below 3% should be treated as noise until
repeated with controlled clocks or power.

## Current production baseline

The 2026-09-30 local baseline is commit `aeeba414459d5d6989d57d8487c9d7a2f54bddd3`, a 48 GiB RTX
4090, CUDA 13.1, and the groupwise-int Qwen3.8-27B artifact whose local benchmark record has SHA-256
`0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`.

The retained speculative profile is MTP3. In the controlled local campaign it produced 126.33
tok/s greedy and 122.74 tok/s sampled weighted decode, versus 52.17/52.26 without speculation and
122.45/118.63 for DFlash2 K=5. The three-request cohort measured 179.55 aggregate tok/s for MTP3
versus 155.43 for DFlash2 K=5. Both passed exact 8K/64K/128K/256K NIAH; MTP3 measured 1,247.37
prefill tok/s and 208,584.79 ms server TTFT at 256K. These numbers are a regression anchor, not a
claim that every prompt has the same acceptance or speed.

The deployment profiles are:

- single lane: 262,144-token context/capacity, INT8 KV, MTP3;
- two lanes: 204,800 tokens per lane, 409,600 shared KV capacity, INT8 KV, MTP3.

Raw local results remain ignored under `.local/spec-bench/`; publish only scrubbed, reproducible
reports intended for the community.

## Upstream-v3 migration roadmap

### P0 — governance and rollback baseline

- [x] establish the downstream repository without discarding the original target-repository history;
- [x] configure explicit `origin`, `upstream`, and `sergiuszm` roles;
- [x] add the repeatable upstream/PR audit;
- [ ] tag the last validated v2 production commit after a clean rebuild and smoke run;
- [ ] export a scrubbed machine-readable baseline and exact benchmark commands.

### P1 — boot a minimal v3/sm89 candidate

Create `sync/YYYY-MM-DD-v3-sm89` from the audited `sergiuszm/rtx4090-port` and replay the upstream
v3 architecture milestones in dependency order. Port only what is necessary to compile and load
on Ada:

1. CMake/CUDA architecture admission for `sm_89` and runtime capability reporting;
2. groupwise Q4/Q5/Q6/Q8 and BF16/INT8 paths already meaningful on Ada;
3. scheduling based on runtime SM count, not 5090 constants;
4. the v3 artifact reader, logical bindings, frontend resources, and official offline v2-to-v3
   upgrader unchanged until their tests pass;
5. one text-only, no-speculation, short-context Qwen3.8-27B request.

Acceptance: clean build, artifact tests, exact short-answer probe, and no unsupported Blackwell path
selected. This milestone is correctness-only; no performance claim is allowed.

Status on 2026-09-30:

- [x] upstream v3 converter (`168fdd81`), loader (`4cde7ad0`), and bound-instance Engine
  (`04350ba9`) integrated with authorship preserved;
- [x] complete Release build for `sm_89` using CUDA 13.1 and GCC 14, including `ninfer`,
  `ninfer-serve`, and `ninfer-perplexity`;
- [x] artifact reader, materializer, writer interop, and Qwen3.5 loader component tests pass;
- [x] the official 20,437,521,664-byte `qwen3_8_27b_v3.ninfer` parses as artifact v3 and host-binds
  Text (17,093,490,688 device bytes), MTP (17,544,758,272), DFlash2 (19,320,283,648), and Vision
  (17,389,210,112);
- [x] NVFP4/K8V4 runtime KV selections fail early on sm89 instead of reaching stub kernels;
- [ ] cold device materialization and an exact short-answer request (deferred to a service window;
  the production v2 process currently owns 27+ GiB on the RTX 4090).

### P2 — restore the production feature envelope

Port or redesign, in order: INT8 KV, paged long context, MTP3, state/prefix persistence, OpenAI and
Anthropic serving, Vision, E8 only if it still buys useful capacity, then DFlash2 as a research path.
Run the full release gates after each subsystem. Upgrade the existing v2 artifact on a copy and keep
the original immutable until v3 reaches production.

The initial v3 baseline intentionally defers the fork-local disk session-slot persistence and its
serve metrics. Their old implementation depended on deleted target-private Program types; they must
be ported to the new model-independent Program contracts with new round-trip and eviction tests,
not retained as declarations backed by incompatible state.

### P3 — recover and exceed the sm89 baseline

Profile the v3 candidate with Nsight Systems/Compute and target measured bottlenecks:

- Q5/Q6/Q8 small-batch linear and fused projection routes used by MTP3;
- GDN/KDA recurrence, convolution, and state movement;
- INT8/K8V4 attention at long-context prefill and decode depths;
- CUDA Graph coverage and launch overhead at one and two active lanes;
- artifact materialization and first-request TTFT;
- 48 GiB-specific KV/state allocation that a 24 GiB reference 4090 cannot exploit.

The first comparison matrix is single/dual lane at 0, 64K, 128K, 200K, and 256K depth, with
no-speculation and MTP3. Only after matching the production baseline should DFlash2, lossy INT8
prefill, alternative KV codecs, or larger speculative windows enter the tournament.

### P4 — sustainable upstream convergence

- reduce Ada-only changes to build admission, capability dispatch, kernels, and tuning tables;
- upstream generic correctness, scheduling, frontend, benchmark, and test improvements;
- rebase the v3 integration line frequently while it is private and unshipped;
- after the v3 release, prefer small regular upstream integrations over periodic large merges;
- publish a versioned scorecard and artifact compatibility table for each release.

## Initial upstream audit — 2026-09-30

| Source | Audited head | Finding |
|---|---|---|
| downstream vendor baseline | `sergiuszm/rtx4090-port@aeeba414` | current local production source |
| `Neroued/ninfer` master | `d44ab584` | 83 commits absent from the vendor line; v3 artifact/model/runtime rewrite and new generic perf work |
| `Neroued/ninfer` dev | `75a89050` | five commits beyond master, including runtime SM-count-derived launch plans and request cleanup ordering |
| `sergiuszm/ninfer-4090` | `aeeba414` | synchronized with the local vendor baseline at audit time |

The vendor and upstream master diverge after `d4929686`: the vendor has 150 unique commits and
upstream master has 83. Important upstream milestones include `168fdd81` (v3 converter),
`4cde7ad0` (v3 loader), `04350ba9` (v3 bound model parameters), and `469f014c` (offline-upgrade
guidance). The first direct-master compile established that later master kernels are not a usable
Ada baseline; P1 instead replays these architecture milestones onto the proven sm89 line.

PRs to watch from this audit include Neroued #292 (reported Q5 small-batch/MTP3 gain), #297
(workspace overflow state), #294 (structured output with speculation), #274 (shared-prefix catalog),
#273 (RMSNorm/RoPE routing), and sergiuszm #10 (Windows sm89). All were open when audited; none is
approved for downstream use without review and local evidence.

## Append-only sync record

Add one row per completed iteration. Link the detailed benchmark/test report from the change or
release rather than creating a second roadmap.

| Date | Downstream result | Neroued head | sergiuszm head | Decision and evidence |
|---|---|---|---|---|
| 2026-09-30 | maintenance baseline | `d44ab584` (`dev` `75a89050`) | `aeeba414` | established two-track v2 production/v3 migration policy; no unvalidated code merge |
| 2026-09-30 | v3/sm89 integration baseline | `d44ab584` (`dev` `75a89050`) | `aeeba414` | adopted v3 converter/loader/Engine milestones on the proven Ada base; full Release build and four component tests pass; official v3 artifact host-binds Text/MTP/DFlash2/Vision; device execution and session-slot port remain open |
