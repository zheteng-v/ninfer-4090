#!/usr/bin/env python3
"""Benchmark long-context C++ code generation at C=1 and C=2.

Each request carries a maintained NIAH document as fixed-size historical context,
but its terminal instruction is replaced by a real scheduler/C++ coding task. The
runner starts a clean server for every point, disables prefix reuse, and records
server-side TTFT, prefill, decode, speculative acceptance, individual decode
rates, and aggregate wall-clock output throughput. It supports an acceptance
matrix across context depths, sampling profiles, proposal heads, and backends.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import http.client
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import run_serve_corpus as corpus  # noqa: E402


GPU_4090_UUID = "GPU-2f39017c-6cf6-5c22-6c8b-aff9ef65a4bd"
MODES = ("mtp0", "mtp3", "dflash2_7")
FIXTURES = ("long_niah_8k", "long_niah_64k", "long_niah_128k")
SAMPLING = ("greedy", "stochastic")
SEEDS = (7632647173703958409, 7968175640111700217)
CODE_TASK = r"""
</document>

You are maintaining a high-performance C++17 inference server. Produce production-quality code
for a fair chunked-prefill scheduler. Implement a `FairPrefillQueue` class, its synchronization
rules, cancellation-safe removal, round-robin selection, and focused unit tests. Then show the
minimal integration code that alternates decode batches and prefill chunks without starving either
request. Use complete C++ code blocks and detailed comments. Continue with implementation details
until the output limit; do not summarize early.
""".strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, default=REPO_ROOT / "build-native/apps/ninfer-serve")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="rtx4090-v3-long-code")
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--fixture", action="append", choices=FIXTURES)
    parser.add_argument("--sampling", action="append", choices=SAMPLING)
    parser.add_argument("--concurrency", action="append", type=int, choices=(1, 2))
    parser.add_argument("--port", type=int, default=24562)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default=GPU_4090_UUID)
    parser.add_argument("--max-context", type=int, default=196608)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proposal-head", choices=("optimized", "full"), default="optimized")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise corpus.CampaignError(message)


def modes(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(args.mode) if args.mode else MODES
    require(len(selected) == len(set(selected)), "duplicate --mode")
    return selected


def concurrencies(args: argparse.Namespace) -> tuple[int, ...]:
    selected = tuple(args.concurrency) if args.concurrency else (1, 2)
    require(len(selected) == len(set(selected)), "duplicate --concurrency")
    return selected


def fixtures(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(args.fixture) if args.fixture else ("long_niah_128k",)
    require(len(selected) == len(set(selected)), "duplicate --fixture")
    return selected


def sampling_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(args.sampling) if args.sampling else ("stochastic",)
    require(len(selected) == len(set(selected)), "duplicate --sampling")
    return selected


def code_messages(fixture: str) -> list[dict[str, Any]]:
    fixture_path = REPO_ROOT / "examples/cli/messages" / f"{fixture}.json"
    source = json.loads(fixture_path.read_text(encoding="utf-8"))
    document = str(source[1]["content"])
    marker = "</document>"
    require(marker in document, "long-context fixture lost its document terminator")
    return [
        {
            "role": "system",
            "content": "You are an expert C++ inference-runtime engineer. Write correct, "
                       "maintainable production code.",
        },
        {"role": "user", "content": document.split(marker, 1)[0] + CODE_TASK},
    ]


def environment(args: argparse.Namespace) -> dict[str, str]:
    require(args.device == 0, "--device must be 0 when a single GPU UUID is exposed")
    result = os.environ.copy()
    result["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    result["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return result


def server_command(args: argparse.Namespace, mode: str, concurrency: int, log: Path,
                   sampling: str) -> list[str]:
    backend, drafts = corpus.SPECULATIVE_MODES[mode]
    command = [
        str(args.serve), str(args.artifact), "--host", "127.0.0.1", "--port", str(args.port),
        "--model-id", args.model_id, "--device", str(args.device), "--max-context",
        str(args.max_context), "--kv-capacity", str(args.max_context * concurrency),
        "--max-concurrency", str(concurrency), "--max-pending-requests", str(concurrency),
        "--pending-timeout-ms", "1800000", "--prefill-chunk", str(args.prefill_chunk),
        "--kv-dtype", "int8", "--no-prefix-reuse", "--request-log-jsonl", str(log),
        "--log-stats-interval-ms", "1000", "--preserve-thinking",
    ]
    if sampling == "greedy":
        command.append("--greedy")
    else:
        command.extend(["--temperature", "1.0", "--top-p", "0.95", "--top-k", "20",
                        "--presence-penalty", "0", "--frequency-penalty", "0"])
    if backend != "none":
        command.extend(["--spec", backend, "--draft-tokens", str(drafts)])
        if args.proposal_head == "optimized":
            command.append("--lm-head-draft")
    return command


def post(port: int, payload: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60 * 60)
    started = time.monotonic()
    try:
        response = corpus.post_json(connection, payload)
    finally:
        connection.close()
    return started, time.monotonic(), response


def request_record(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result", {})
    timing = event.get("timings_seconds", {})
    engine = event.get("engine_timing", {})
    speculative = event.get("speculative", {})
    completion = int(result.get("completion_tokens", 0))
    decode_seconds = float(timing.get("decode", 0.0))
    drafted = int(speculative.get("drafted_tokens", 0))
    accepted = int(speculative.get("accepted_tokens", 0))
    return {
        "request_id": event.get("request", {}).get("request_id"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": completion,
        "model_thinking_tokens": result.get("model_thinking_tokens"),
        "queue_wait_ms": 1000 * float(engine.get("queue_wait_seconds", 0.0)),
        "ttft_ms": 1000 * float(timing.get("ttft", 0.0)),
        "prefill_ms": 1000 * float(timing.get("prefill", 0.0)),
        "decode_ms": 1000 * decode_seconds,
        "decode_tok_s": completion / decode_seconds if decode_seconds else None,
        "backend": speculative.get("backend"),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance": accepted / drafted if drafted else None,
    }


def run_point(args: argparse.Namespace, mode: str, concurrency: int, fixture: str,
              sampling: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    key = f"{fixture}-{sampling}-{args.proposal_head}-{mode}-c{concurrency}"
    log = args.output / "server" / f"{key}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = server_command(args, mode, concurrency, log, sampling)
    print(f"start {key}", flush=True)
    payloads = [
        {
            "model": args.model_id,
            "messages": messages,
            "max_completion_tokens": args.max_tokens,
            "seed": SEEDS[index],
            "stream": False,
            "enable_thinking": args.thinking,
        }
        for index in range(concurrency)
    ]
    with corpus.RunningServer(command, "127.0.0.1", args.port, log, environment(args)) as server:
        start_event = server.wait_until_ready()
        instance_id = str(start_event["server_instance_id"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(post, args.port, payload) for payload in payloads]
            clients = [future.result() for future in futures]
        records = [request_record(server.wait_for_request_done(instance_id)) for _ in range(concurrency)]

    records.sort(key=lambda row: int(row.get("request_id") or 0))
    started = min(client[0] for client in clients)
    finished = max(client[1] for client in clients)
    total_completion = sum(int(row["completion_tokens"]) for row in records)
    total_drafted = sum(int(row["drafted_tokens"]) for row in records)
    total_accepted = sum(int(row["accepted_tokens"]) for row in records)
    makespan = finished - started
    report = {
        "artifact_type": "ninfer_long_context_code_bench_point",
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "mode": mode,
        "fixture": fixture,
        "sampling": sampling,
        "proposal_head": args.proposal_head if mode != "mtp0" else None,
        "concurrency": concurrency,
        "max_context": args.max_context,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "command": command,
        "server_start": start_event,
        "client_makespan_seconds": makespan,
        "aggregate_completion_tok_s": total_completion / makespan if makespan else None,
        "aggregate_acceptance": total_accepted / total_drafted if total_drafted else None,
        "requests": records,
    }
    point_path = args.output / "points" / f"{key}.json"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    point_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "point": key, "aggregate_completion_tok_s": report["aggregate_completion_tok_s"],
        "requests": len(records), "prompt_tokens": records[0].get("prompt_tokens"),
    }, ensure_ascii=False), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.serve.is_file(), f"serve executable not found: {args.serve}")
    require(args.artifact.is_file(), f"artifact not found: {args.artifact}")
    require(args.max_context > 0 and args.max_tokens > 0, "context and output limits must be positive")
    require(args.prefill_chunk > 0 and args.prefill_chunk % 128 == 0,
            "--prefill-chunk must be a positive multiple of 128")
    require(0 < args.port < 65536, "invalid port")
    selected_modes = modes(args)
    selected_concurrency = concurrencies(args)
    selected_fixtures = fixtures(args)
    selected_sampling = sampling_profiles(args)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "artifact_type": "ninfer_long_context_code_bench_plan",
        "schema_version": 1,
        "modes": selected_modes,
        "concurrency": selected_concurrency,
        "fixtures": selected_fixtures,
        "sampling": selected_sampling,
        "proposal_head": args.proposal_head,
        "max_context": args.max_context,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "context_fixture": "selected long_niah fixture with terminal C++ scheduler task",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    reports = [
        run_point(args, mode, concurrency, fixture, sampling, code_messages(fixture))
        for fixture in selected_fixtures
        for sampling in selected_sampling
        for mode in selected_modes
        for concurrency in selected_concurrency
    ]
    (args.output / "summary.json").write_text(
        json.dumps({"plan": plan, "points": reports}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
