#!/usr/bin/env python3
"""Measure SGLang on the same long-context C++ coding workload as NInfer.

The target server must already be running.  This client uses OpenAI-compatible
SSE streaming so its TTFT and decode interval are measured at the same loopback
boundary an interactive coding client uses.  It intentionally records no model
output or API secret in its report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.run_serve_long_context_code import code_messages  # noqa: E402


GPU_4090_UUID = "GPU-2f39017c-6cf6-5c22-6c8b-aff9ef65a4bd"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24561)
    parser.add_argument("--api-key-file", type=Path, default=Path("/data/inference-control/api.key"))
    parser.add_argument("--model")
    parser.add_argument("--fixture", choices=("long_niah_8k", "long_niah_64k", "long_niah_128k"),
                        default="long_niah_128k")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--sampling", choices=("greedy", "stochastic"), default="greedy")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def authorization(api_key: str) -> dict[str, str]:
    return {
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def discover_model(args: argparse.Namespace, api_key: str) -> str:
    if args.model:
        return args.model
    connection = http.client.HTTPConnection(args.host, args.port, timeout=20)
    try:
        connection.request("GET", "/v1/models", headers=authorization(api_key))
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    require(response.status == 200, f"GET /v1/models returned HTTP {response.status}")
    payload = json.loads(body)
    models = payload.get("data", [])
    require(models and isinstance(models[0], dict) and models[0].get("id"),
            "SGLang /v1/models returned no served model")
    return str(models[0]["id"])


def sse_request(args: argparse.Namespace, api_key: str, model: str, seed: int,
                messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": args.thinking, "thinking": args.thinking},
    }
    if args.sampling == "greedy":
        payload.update({"temperature": 0.0, "top_p": 1.0})
    else:
        payload.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = authorization(api_key)
    headers["Content-Length"] = str(len(encoded))
    connection = http.client.HTTPConnection(args.host, args.port, timeout=60 * 60)
    started = time.monotonic()
    first_content: float | None = None
    finished = started
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    text_chars = 0
    reasoning_chars = 0
    try:
        connection.request("POST", "/v1/chat/completions", body=encoded, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            detail = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang returned HTTP {response.status}: {detail[:500]}")
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = str(delta.get("content") or "")
            reasoning = str(delta.get("reasoning_content") or "")
            if content or reasoning:
                if first_content is None:
                    first_content = time.monotonic()
                text_chars += len(content)
                reasoning_chars += len(reasoning)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
        finished = time.monotonic()
    finally:
        connection.close()
    require(first_content is not None, "SGLang stream completed without a content token")
    completion = int(usage.get("completion_tokens", 0))
    prompt = int(usage.get("prompt_tokens", 0))
    require(completion > 0 and prompt > 0, "SGLang stream omitted token usage")
    decode_seconds = finished - first_content
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": 1000 * (first_content - started),
        "decode_ms": 1000 * decode_seconds,
        "decode_tok_s": completion / decode_seconds if decode_seconds else None,
        "finish_reason": finish_reason,
        "text_characters": text_chars,
        "reasoning_characters": reasoning_chars,
    }


def gpu_memory_mib() -> int | None:
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        return None
    total = 0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == GPU_4090_UUID:
            total += int(parts[1])
    return total


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.max_tokens > 0, "--max-tokens must be positive")
    require(not args.output.exists(), f"output already exists: {args.output}")
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    require(api_key, "API key file is empty")
    model = discover_model(args, api_key)
    messages = code_messages(args.fixture)
    args.output.mkdir(parents=True)
    memory_before = gpu_memory_mib()
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(sse_request, args, api_key, model, 7632647173703958409 + index,
                               messages) for index in range(args.concurrency)]
        records = [future.result() for future in futures]
    finished = time.monotonic()
    memory_after = gpu_memory_mib()
    total_completion = sum(int(record["completion_tokens"]) for record in records)
    report = {
        "artifact_type": "sglang_long_context_code_bench_point",
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "server": {"host": args.host, "port": args.port, "model": model},
        "fixture": args.fixture,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "sampling": args.sampling,
        "thinking": args.thinking,
        "measurement_boundary": (
            "loopback SSE client: TTFT is request release to first content/reasoning token; "
            "decode rate is completion usage tokens divided by first-token-to-stream-end time"
        ),
        "client_makespan_seconds": finished - started,
        "aggregate_completion_tok_s": total_completion / (finished - started),
        "gpu_memory_mib_before": memory_before,
        "gpu_memory_mib_after": memory_after,
        "requests": records,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "model": model, "concurrency": args.concurrency,
        "aggregate_completion_tok_s": report["aggregate_completion_tok_s"], "requests": records,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
