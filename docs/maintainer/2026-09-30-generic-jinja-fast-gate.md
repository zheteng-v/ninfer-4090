# Generic-Jinja v3/sm89 fast gate — 2026-09-30

This report qualifies the first generic-Jinja v3 candidate on the local 48 GiB RTX 4090. It is a
routine integration gate, not a release benchmark. Production remained on the validated v2 line
after the canary.

## Identity and scope

- branch: `feat/v3-generic-jinja-sm89`
- tested source: `672ca0df`
- base: `sync/2026-09-30-v3-sm89@d2d12057`
- artifact: `/data/llm/ninfer/models/qwen3_8_27b_v3.ninfer`
- artifact SHA-256: `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da`
- build: Release, CUDA 13.1, GCC 14, `CMAKE_CUDA_ARCHITECTURES=89`
- runtime: INT8 KV, MTP3, greedy sampling, CUDA Graphs, `--prefill-chunk 1024`
- imported upstream sequence: `b9219f3f`, `98dada0e`, and `8eaed538`

The change replaces the temporary Qwen3.8 template-hash allowlist with upstream's generic Jinja
executor. The sm89 adaptation keeps the monolithic downstream CMake layout, the fork's
`vision_max_tokens` frontend behavior, OpenAI model metadata, and
`chat_template_kwargs.reasoning_effort` alias normalization.

## Build and component results

The complete 624-target build passed, including `ninfer`, `ninfer-serve`, the Jinja executor test,
and the Qwen3.5 frontend/loader tests. Eleven focused C++ suites passed:

- artifact reader, materialization, and writer interop;
- Qwen3.5 loading and frontend;
- C++ Jinja execution;
- OpenAI schema and Responses API;
- Anthropic schema;
- serve options and request logging.

The six Python reference-template cases also matched Jinja2 3.1.6. The configured system Python did
not contain Jinja2, so that reference test was invoked with an existing Python 3.12 environment;
this is a test-only dependency and is not a runtime requirement.

## Single-lane functional and performance probes

The candidate started on loopback port 24562 with Vision enabled, 262,144-token maximum context and
KV capacity, one lane, and MTP3.

| Probe | Prompt/output | TTFT | Prefill | Decode | MTP acceptance | Result |
|---|---:|---:|---:|---:|---:|---|
| Chinese text | 66/128 | 176 ms | 440.2 tok/s | 94.3 tok/s | 67/180, 37.2% | valid text; output limit |
| OpenAI tool call | 339/75 | 259 ms | 1.45k tok/s | 157.2 tok/s | 56/63, 88.9% | `add({"a":2,"b":3})` |
| Vision, red PPM | 128/51 | 174 ms | 781.7 tok/s | 111.3 tok/s | 33/60, 55.0% | answered `红色` |

The first 32-token Vision attempt exhausted its budget inside reasoning and therefore returned no
visible answer; the repeated 96-token-budget probe stopped normally and returned the correct color.
This is expected reasoning-budget behavior, not a media failure.

## Two-lane 200K probe

The same binary then started without Vision at 204,800 maximum tokens per request,
409,600 shared INT8 KV tokens, and two active lanes. Startup reported 16.7 GiB of weights,
14.6 GiB of runtime allocation, and 15.9 GiB free after sizing. Two 128-token requests were issued
concurrently and both completed in 1.85 seconds aggregate wall time.

| Lane | TTFT | Prefill | Decode | MTP acceptance | Server total |
|---:|---:|---:|---:|---:|---:|
| 1 | 156 ms | 493.8 tok/s | 91.9 tok/s | 79/142, 55.6% | 1.7 s |
| 2 | 390 ms | 539.3 tok/s | 87.8 tok/s | 76/151, 50.3% | 1.8 s |

This probe demonstrates correct simultaneous admission and completion; it is not a statistically
controlled throughput claim.

## Operational outcome

Both canary configurations stopped cleanly. `inferctl switch ninfer-single` restored the existing
v2 production model on port 24561, and `/health` returned `{"status":"ok"}`. ComfyUI remained
running on its separate GPU throughout. The generic-Jinja candidate is suitable for the v3
integration branch, but this fast gate does not authorize replacing `main` or the production
profile.
