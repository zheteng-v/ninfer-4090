#!/usr/bin/env python3
"""Measure whether a short request is starved by an already-running long prefill.

The workload intentionally submits a long request first and a short request after a
small fixed delay.  It is the minimal black-box regression for the scheduler's
round-robin chunked-prefill contract.  Results come from the server's structured
JSONL log, not from terminal timing output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus  # noqa: E402


MODES = ("mtp3", "dflash2_7")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, default=REPO_ROOT / "build-native/apps/ninfer-serve")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-id", default="rtx4090-v3-prefill-fairness")
    parser.add_argument("--mode", choices=MODES, default="mtp3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=24562)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=196608)
    parser.add_argument("--kv-capacity", type=int, default=393216)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--stagger-ms", type=int, default=250)
    parser.add_argument("--long-fixture", default="long_niah_128k")
    parser.add_argument("--short-fixture", default="text_smoke_zh")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise corpus.CampaignError(message)


def server_command(args: argparse.Namespace, request_log: Path) -> list[str]:
    backend, drafts = corpus.SPECULATIVE_MODES[args.mode]
    return [
        str(args.serve), str(args.artifact), "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", args.model_id, "--max-context", str(args.max_context),
        "--kv-capacity", str(args.kv_capacity), "--max-concurrency", "2",
        "--max-pending-requests", "2", "--pending-timeout-ms", "600000",
        "--prefill-chunk", str(args.prefill_chunk), "--log-stats-interval-ms", "1000",
        "--device", str(args.device), "--request-log-jsonl", str(request_log),
        "--kv-dtype", "int8", "--no-prefix-reuse", "--greedy", "--spec", backend,
        "--draft-tokens", str(drafts), "--lm-head-draft",
    ]


def request(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=24 * 60 * 60)
    try:
        return corpus.post_json(connection, payload)
    finally:
        connection.close()


def done_record(event: dict[str, Any]) -> dict[str, Any]:
    request_data = event.get("request", {})
    result = event.get("result", {})
    timing = event.get("timings_seconds", {})
    engine = event.get("engine_timing", {})
    speculative = event.get("speculative", {})
    return {
        "request_id": request_data.get("request_id"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "queue_wait_ms": 1000 * float(engine.get("queue_wait_seconds", 0.0)),
        "ttft_ms": 1000 * float(timing.get("ttft", 0.0)),
        "prefill_ms": 1000 * float(timing.get("prefill", 0.0)),
        "decode_ms": 1000 * float(timing.get("decode", 0.0)),
        "decode_tok_s": (
            float(result.get("completion_tokens", 0)) / float(timing["decode"])
            if timing.get("decode", 0.0) else None
        ),
        "speculative_backend": speculative.get("backend"),
        "drafted_tokens": speculative.get("drafted_tokens"),
        "accepted_tokens": speculative.get("accepted_tokens"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.serve.is_file(), f"serve executable not found: {args.serve}")
    require(args.artifact.is_file(), f"artifact not found: {args.artifact}")
    require(args.port > 0 and args.port < 65536, "--port must be in [1,65535]")
    require(args.max_context > 0 and args.kv_capacity >= args.max_context,
            "--kv-capacity must be at least --max-context")
    require(args.prefill_chunk > 0 and args.prefill_chunk % 128 == 0,
            "--prefill-chunk must be a positive multiple of 128")
    require(args.stagger_ms >= 0 and args.max_tokens > 0, "stagger and max tokens must be positive")

    fixtures = corpus.load_fixtures()
    require(args.long_fixture in fixtures, f"unknown long fixture: {args.long_fixture}")
    require(args.short_fixture in fixtures, f"unknown short fixture: {args.short_fixture}")
    long_fixture = fixtures[args.long_fixture]
    short_fixture = fixtures[args.short_fixture]
    require(long_fixture.prompt_tokens + args.max_tokens <= args.max_context,
            "long fixture does not fit --max-context")
    require(short_fixture.prompt_tokens + args.max_tokens <= args.max_context,
            "short fixture does not fit --max-context")

    args.output.mkdir(parents=True, exist_ok=False)
    request_log = args.output / "requests.jsonl"
    command = server_command(args, request_log)
    plan = {
        "artifact_type": "ninfer_prefill_fairness_plan",
        "schema_version": 1,
        "mode": args.mode,
        "long_fixture": {"name": long_fixture.name, "prompt_tokens": long_fixture.prompt_tokens},
        "short_fixture": {"name": short_fixture.name, "prompt_tokens": short_fixture.prompt_tokens},
        "stagger_ms": args.stagger_ms,
        "command": command,
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    long_payload = corpus.request_payload(args.model_id, long_fixture, 7632647173703958409)
    short_payload = corpus.request_payload(args.model_id, short_fixture, 7968175640111700217)
    long_payload["max_completion_tokens"] = args.max_tokens
    short_payload["max_completion_tokens"] = args.max_tokens

    with corpus.RunningServer(command, "127.0.0.1", args.port, request_log) as server:
        start = server.wait_until_ready()
        instance_id = str(start["server_instance_id"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            long_future = pool.submit(request, args.port, long_payload)
            time.sleep(args.stagger_ms / 1000)
            short_future = pool.submit(request, args.port, short_payload)
            long_future.result()
            short_future.result()
        records = [done_record(server.wait_for_request_done(instance_id)) for _ in range(2)]

    records.sort(key=lambda value: int(value.get("prompt_tokens") or 0), reverse=True)
    summary = {
        "artifact_type": "ninfer_prefill_fairness_result",
        "schema_version": 1,
        "plan": plan,
        "requests": {"long": records[0], "short": records[1]},
        "pass": records[1]["queue_wait_ms"] < records[0]["prefill_ms"],
        "criterion": "short request begins before the long request finishes its complete prefill",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
