from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from argparse import Namespace

import pytest

from tools.bench.run_serve_corpus import (
    CampaignError,
    Fixture,
    RunSpec,
    SPECULATIVE_MODES,
    build_result_record,
    build_summary_rows,
    load_existing_records,
    parse_artifacts,
    kv_cache_name,
    server_start_prefill_signature,
    mode_display_name,
    selected_seeds,
    server_command,
    validate_server_start,
    validate_fixture_context,
)
from tools.bench import run_serve_concurrency as concurrency
from tools.bench.run_serve_concurrency import build_points
from tools.bench.run_sm89_phase0 import Candidate, commands


def test_result_record_parses_request_host_exposure() -> None:
    fixture = Fixture(
        name="fixture",
        messages=[],
        thinking=True,
        max_new=8,
        suite="test",
    )
    spec = RunSpec(
        target="qwen3_6_27b",
        model_id="qwen3.6-27b",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="greedy",
        fixture=fixture,
        seed=7,
    )
    payload = {"model": spec.model_id}
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 21,
        "event": "request_done",
        "request": {
            "model": spec.model_id,
            "requested_output_tokens": 8,
            "enable_thinking": True,
            "sampling": {"seed": 7},
        },
        "result": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish_reason": "output_limit",
        },
        "timings_seconds": {
            "prepare": 0.1,
            "vision": 0.0,
            "prefill": 0.2,
            "decode": 0.4,
            "total": 0.7,
        },
        "speculative": {
            "backend": "mtp",
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 3,
            "fallback_steps": 0,
        },
        "engine_timing": {
            "queue_wait_seconds": 0.001,
            "host_exposed_seconds": {
                "engine_boundary": 0.001,
                "program_submit": 0.002,
                "program_post": 0.003,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.015,
            },
            "device_wait_exposed_seconds": 0.3,
            "decode": {
                "host_exposed_seconds": 0.01,
                "device_wait_exposed_seconds": 0.2,
                "rounds": 2,
            },
        },
    }

    record = build_result_record(spec, "measured-prefill-bindings", payload, response, event)
    assert record["schema_version"] == 9
    assert record["kv_dtype"] == "int8"
    assert record["max_context"] == 262144
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(15.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_device_wait_us_per_round"] == pytest.approx(
        100000.0
    )


def test_arbitrary_artifact_labels_reach_the_requested_backend(tmp_path: Path) -> None:
    artifact = tmp_path / "custom.ninfer"
    artifact.touch()
    artifacts = parse_artifacts([f"org/custom={artifact}", f"org%2Fcustom={artifact}"])
    points = build_points(
        artifacts,
        Namespace(mode=["dflash7", "dflash2_7"], suite=["decode-saturation"],
                  concurrency=[1], sampling="greedy"),
    )
    assert [(point.target, point.speculative_backend) for point in points] == [
        ("org/custom", "dflash"), ("org/custom", "dflash2"),
        ("org%2Fcustom", "dflash"), ("org%2Fcustom", "dflash2"),
    ]
    assert all(point.artifact == artifact and point.model_id == point.target for point in points)
    assert len({point.key for point in points}) == len(points)
    for point in points:
        (tmp_path / f"{point.key}.json").write_text("{}")


def test_phase_zero_short_speculative_modes_have_exact_backends_and_windows() -> None:
    assert {mode: SPECULATIVE_MODES[mode] for mode in ("mtp3", "mtp4", "dflash2_6", "dflash2_7")} == {
        "mtp3": ("mtp", 3),
        "mtp4": ("mtp", 4),
        "dflash2_6": ("dflash2", 6),
        "dflash2_7": ("dflash2", 7),
    }
    assert [mode_display_name(mode) for mode in ("mtp3", "mtp4", "dflash2_6", "dflash2_7")] == [
        "MTP3", "MTP4", "DFlash2 block=7 (k=6)", "DFlash2 block=8 (k=7)"
    ]
    points = build_points(
        [("v3", Path("/tmp/v3.ninfer"))],
        Namespace(mode=["mtp3", "mtp4", "dflash2_6", "dflash2_7"],
                  suite=["decode-saturation"], concurrency=[1], sampling="stochastic"),
    )
    assert [(point.speculative_backend, point.draft_tokens) for point in points] == [
        ("mtp", 3), ("mtp", 4), ("dflash2", 6), ("dflash2", 7)
    ]


def test_resume_identity_includes_exact_draft_window(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    fixture = Fixture("scenario_code_python", [], False, 8, "scenario", "code")
    spec = RunSpec(
        target="v3",
        model_id="v3",
        artifact=artifact.resolve(),
        speculative_mode="mtp4",
        speculative_backend="mtp",
        draft_tokens=4,
        sampling_mode="greedy",
        fixture=fixture,
        seed=1,
    )
    wrong_k_record = {
        "artifact_type": "ninfer_serve_corpus_result",
        "schema_version": 9,
        "target": "v3",
        "speculative_mode": "mtp4",
        "sampling_mode": "greedy",
        "kv_dtype": "int8",
        "fixture": fixture.name,
        "seed": 1,
        "draft_tokens": 3,
        "max_context": 262144,
        "artifact_path": str(artifact),
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(wrong_k_record) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="outside this campaign"):
        load_existing_records(path, {spec.key: spec})


def test_exact_seed_selection_preserves_default_fixed_seed_prefix() -> None:
    from tools.bench.run_serve_corpus import SEEDS

    assert selected_seeds(1) == SEEDS[:1]
    assert selected_seeds(5) == SEEDS
    assert selected_seeds(5, 912910298659544128) == (912910298659544128,)
    with pytest.raises(CampaignError, match="seed-count"):
        selected_seeds(0)
    with pytest.raises(CampaignError, match="seed must be nonnegative"):
        selected_seeds(1, -1)


def test_phase_zero_kv_cache_names_are_explicit() -> None:
    assert kv_cache_name("int8") == "int8-group64"
    assert kv_cache_name("fp8") == "fp8-e4m3-row256"


def test_dflash2_k8_mode_reuses_the_backend_and_keeps_the_draft_head(tmp_path: Path) -> None:
    assert SPECULATIVE_MODES["dflash2_8"] == ("dflash2", 8)
    assert mode_display_name("dflash2_8") == "DFlash2 block=9 (k=8)"
    points = build_points(
        [("v3", tmp_path / "v3.ninfer")],
        Namespace(mode=["dflash2_8"], suite=["decode-saturation"], concurrency=[1],
                  sampling="stochastic"),
    )
    assert [(point.speculative_backend, point.draft_tokens) for point in points] == [("dflash2", 8)]
    spec = RunSpec(
        target="v3",
        model_id="v3",
        artifact=tmp_path / "v3.ninfer",
        speculative_mode="dflash2_8",
        speculative_backend="dflash2",
        draft_tokens=8,
        sampling_mode="stochastic",
        fixture=Fixture("fixture", [], True, 8, "test"),
        seed=1,
    )
    command = server_command(Path("ninfer-serve"), spec, tmp_path / "server.jsonl", 8080, 0,
                             "int8")
    assert command[command.index("--spec") + 1] == "dflash2"
    assert command[command.index("--draft-tokens") + 1] == "8"
    assert "--lm-head-draft" in command


def test_corpus_server_command_only_disables_cuda_graph_when_requested(tmp_path: Path) -> None:
    spec = RunSpec(
        target="v3",
        model_id="v3",
        artifact=tmp_path / "model.ninfer",
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="greedy",
        fixture=Fixture("fixture", [], True, 8, "test"),
        seed=1,
    )
    command_args = (Path("ninfer-serve"), spec, tmp_path / "server.jsonl", 8080, 0, "int8")
    default_command = server_command(*command_args)
    assert "--no-cuda-graph" not in default_command
    assert "--vision" not in default_command
    diagnostic_command = server_command(*command_args, cuda_graph=False, vision=True)
    assert diagnostic_command.count("--no-cuda-graph") == 1
    assert diagnostic_command.count("--vision") == 1
    short_context = server_command(*command_args, max_context=8192)
    assert short_context[short_context.index("--max-context") + 1] == "8192"
    assert short_context[short_context.index("--kv-capacity") + 1] == "8192"

    spec.artifact.touch()
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 21,
        "event": "server_start",
        "engine": {
            "device": 0,
            "max_context": 262144,
            "kv_capacity": 262144,
            "prefill_chunk": 1024,
            "kv_cache": "int8-group64",
            "cuda_graph": False,
            "vision": True,
            "prefix_reuse": False,
            "speculative_backend": "mtp",
            "speculative_draft_window": 3,
            "proposal_head": "optimized",
        },
        "sampling_defaults": {"greedy": True},
        "artifact": {
            "path": str(spec.artifact),
            "prefill_signature": "diagnostic-prefill-signature",
        },
        "server": {"public_model_id": "v3"},
        "server_instance_id": "diagnostic-server",
    }
    assert validate_server_start(
        event, spec, device=0, kv_dtype="int8", cuda_graph=False, vision=True
    ) == ("diagnostic-server", "diagnostic-prefill-signature")
    short_spec = replace(spec, max_context=8192)
    short_event = dict(event)
    short_event["engine"] = dict(event["engine"], max_context=8192, kv_capacity=8192)
    assert validate_server_start(
        short_event, short_spec, device=0, kv_dtype="int8", cuda_graph=False, vision=True
    ) == ("diagnostic-server", "diagnostic-prefill-signature")
    with pytest.raises(CampaignError, match="Engine configuration mismatch"):
        validate_server_start(
            event, short_spec, device=0, kv_dtype="int8", cuda_graph=False, vision=False
        )


def test_v2_schema_20_prefill_identity_is_explicit_and_future_missing_is_rejected() -> None:
    assert server_start_prefill_signature({"schema_version": 20, "artifact": {}}) == (
        "unreported-v2-schema20"
    )
    assert server_start_prefill_signature(
        {"schema_version": 21, "artifact": {"prefill_signature": "signature-v3"}}
    ) == "signature-v3"
    with pytest.raises(CampaignError, match="no canonical artifact prefill_signature"):
        server_start_prefill_signature({"schema_version": 21, "artifact": {}})


def test_concurrency_start_accepts_v2_schema_20_prefill_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "v2.ninfer"
    artifact.touch()
    point = build_points(
        [("v2", artifact)],
        Namespace(
            mode=["mtp3"], suite=["decode-saturation"], concurrency=[1], sampling="stochastic"
        ),
    )[0]
    args = Namespace(
        device=0, max_context=16384, kv_capacity="auto", prefill_chunk=1024, kv_dtype="int8"
    )
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 20,
        "event": "server_start",
        "engine": {
            "device": 0,
            "max_context": 16384,
            "kv_capacity_mode": "auto",
            "max_concurrency": 1,
            "max_pending_requests": 1,
            "pending_timeout_ms": concurrency.PENDING_TIMEOUT_MS,
            "prefill_chunk": 1024,
            "log_stats_interval_ms": concurrency.STATS_INTERVAL_MS,
            "kv_cache": "int8-group64",
            "cuda_graph": True,
            "prefix_reuse": False,
            "speculative_backend": "mtp",
            "speculative_draft_window": 3,
            "proposal_head": "optimized",
        },
        "sampling_defaults": {"greedy": False},
        "artifact": {"path": str(artifact)},
        "server": {"public_model_id": "v2"},
        "server_instance_id": "v2-serve-test",
    }

    assert concurrency.validate_server_start(event, point, args) == (
        "v2-serve-test", "unreported-v2-schema20"
    )


def test_phase_zero_screen_uses_one_seed_and_all_required_controls(tmp_path: Path) -> None:
    args = Namespace(output=tmp_path, phase="screen", port=8090, device=0, screen_max_context=8192)
    mtp0, mtp3, dflash2, concurrency = commands(
        Candidate("v3", Path("/tmp/ninfer-serve"), Path("/tmp/v3.ninfer")), "fp8", args
    )
    assert mtp0[mtp0.index("--mode") + 1] == "mtp0"
    assert mtp3[mtp3.index("--fixture") + 1] == "scenario_code_python"
    assert mtp3[mtp3.index("--max-context") + 1] == "8192"
    assert dflash2[dflash2.index("--mode") + 1] == "dflash2_7"
    assert all(command[command.index("--seed-count") + 1] == "1" for command in (mtp0, mtp3, dflash2))
    assert concurrency.count("--concurrency") == 1
    assert concurrency[concurrency.index("--decode-tokens") + 1] == "2048"
    assert concurrency[concurrency.index("--kv-dtype") + 1] == "fp8"
    assert concurrency[concurrency.index("--max-context") + 1] == "16384"


def test_summary_accepts_a_single_fixture_screen() -> None:
    fixture = Fixture(
        name="scenario_code_python",
        messages=[],
        thinking=False,
        max_new=8,
        suite="scenario",
        category="code",
        prompt_tokens=122,
    )
    record = {
        "target": "v2",
        "speculative_mode": "mtp3",
        "sampling_mode": "stochastic",
        "kv_dtype": "int8",
        "fixture": fixture.name,
        "seed": 7632647173703958409,
        "draft_tokens": 3,
        "max_context": 8192,
        "prefill_signature": "weights",
        "metrics": {},
    }
    records = {
        ("v2", "mtp3", "stochastic", "int8", fixture.name, record["seed"], 3, 8192): record
    }
    rows = build_summary_rows(
        records, ("v2",), ("mtp3",), "stochastic", (record["seed"],)
    )
    assert [(row["section"], row["fixture"]) for row in rows] == [
        ("scenario_fixture", "scenario_code_python"),
        ("scenario_category", ""),
    ]
    assert all(row["max_context"] == 8192 for row in rows)

    other_context = dict(record, max_context=262144)
    mixed_records = dict(records)
    mixed_records[("v2", "mtp3", "stochastic", "int8", fixture.name, record["seed"], 3, 262144)] = other_context
    with pytest.raises(CampaignError, match="one max_context"):
        build_summary_rows(mixed_records, ("v2",), ("mtp3",), "stochastic", (record["seed"],))


def test_fixture_context_requires_prompt_plus_completion_capacity() -> None:
    fixture = Fixture("scenario_code_python", [], False, 4096, "scenario", "code", 122)
    spec = RunSpec(
        target="v3", model_id="v3", artifact=Path("/tmp/v3.ninfer"),
        speculative_mode="dflash2_7", speculative_backend="dflash2", draft_tokens=7,
        sampling_mode="stochastic", fixture=fixture, seed=7632647173703958409,
        max_context=8192,
    )
    validate_fixture_context((spec,))
    too_small = replace(spec, max_context=4217)
    with pytest.raises(CampaignError, match="requires at least 4218 tokens"):
        validate_fixture_context((too_small,))


def test_resume_identity_rejects_different_max_context(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    fixture = Fixture("scenario_code_python", [], False, 8, "scenario", "code")
    spec = RunSpec(
        target="v3", model_id="v3", artifact=artifact.resolve(),
        speculative_mode="mtp3", speculative_backend="mtp", draft_tokens=3,
        sampling_mode="greedy", fixture=fixture, seed=1, max_context=8192,
    )
    record = {
        "artifact_type": "ninfer_serve_corpus_result", "schema_version": 9,
        "target": "v3", "speculative_mode": "mtp3", "sampling_mode": "greedy",
        "kv_dtype": "int8", "fixture": fixture.name, "seed": 1,
        "draft_tokens": 3, "max_context": 262144, "artifact_path": str(artifact),
    }
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(CampaignError, match="outside this campaign"):
        load_existing_records(path, {spec.key: spec})
