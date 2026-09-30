# v3/sm89 serving fast gate — 2026-09-30

This report records a focused P2 service-envelope qualification and the restoration of live
Prometheus metrics on the official v3 artifact. It is a routine fast gate, not a release soak or a
public performance comparison.

## Identity and conditions

- branch: `feat/v3-serving-gates`
- code commit: `bde0055f`
- base: `sync/2026-09-30-v3-sm89@fe500520`
- artifact: `/data/llm/ninfer/models/qwen3_8_27b_v3.ninfer`
- build: Release, CUDA 13.1, GCC 14, `CMAKE_CUDA_ARCHITECTURES=89`
- KV: INT8
- production metrics sources adapted: `4277f14b`, `656b0df7`, and `25297d06`

The iteration began with a fresh upstream audit. Heads were unchanged at
`upstream/master@d44ab584`, `upstream/dev@75a89050`, and
`sergiuszm/rtx4090-port@aeeba414`.

## Protocol contract

The repository's `tools.smoke.serve_contract` passed on a two-lane, 262,144-token server with
Vision and MTP3 enabled. It exercised:

- model listing and Anthropic token counting;
- OpenAI Chat non-streaming and SSE streaming;
- OpenAI Responses non-streaming and SSE streaming;
- stored Response retrieval, continuation via `previous_response_id`, input-item listing, and
  deletion;
- one real Vision frontend request;
- Anthropic non-streaming output.

An additional Anthropic SSE probe completed with the expected ordered event sequence:
`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`,
`message_delta`, and `message_stop`, with no protocol error.

## Scheduler and cancellation gates

The repository's audited TTFT cases were run against the v3 artifact with INT8 KV:

| Case | Result | Key evidence |
|---|---|---|
| `cancel-before-first` | pass | cancelled stream published no model output; following probe completed at 88.2 ms TTFT |
| `cancel-after-first` | pass | transport terminated after first output; following probe completed at 95.7 ms TTFT |
| `pending-timeout` | pass | blocked non-stream request returned HTTP 503 / `request_queue_timeout` after about 115 ms |
| `short-during-decode` | pass | short request completed before the 2,861-token holder; short TTFT 96.5 ms |

The two cancellation cases also verified that no protocol terminal event is emitted after client
cancellation and that the transport closes within the five-second contract.

## Prometheus metrics restoration

The v3 replay retained the `RuntimeStats` token/time fields and metrics unit test but had lost the
Serve implementation, route, test registration, request-lifetime gauge, and EngineCore time
accumulation. The port restores all of them without reintroducing `/slots` or disk persistence.

Three focused C++ tests pass: `ninfer_serve_metrics_test`, `ninfer_request_log_test`, and
`ninfer_http_transport_test`. Runtime validation established that:

- an idle completed request produced 31 computed prompt tokens, 4 decode tokens, 0.122087 seconds
  of prefill execution, and 0.076308 seconds of decode execution;
- during an active long stream, `llamacpp:requests_processing` was 1,
  `llamacpp:requests_deferred` was 0, and both live time counters continued to advance;
- the active stream was then cancelled cleanly with no protocol error.

The route emits the existing llama.cpp-compatible token/time/gauge series plus NInfer request,
prefix-cache, speculative-draft, and accepted-draft counters. When API authentication is enabled,
the normal pre-routing authentication policy also protects `/metrics`.

## Operational outcome

Every canary was stopped cleanly. `inferctl switch ninfer-single` restored the validated v2
production service on port 24561 and `/health` returned `{"status":"ok"}`. ComfyUI remained
running on its separate GPU. This serving increment is suitable for the v3 integration branch but
does not authorize a production switch.
