"""Serving configuration contracts used by the Phase 0 benchmark runners."""

from __future__ import annotations

import dataclasses
from argparse import Namespace
from pathlib import Path

import pytest

from tools.bench import run_serve_concurrency as concurrency
from tools.bench import run_serve_corpus as corpus


def _spec(artifact: Path) -> corpus.RunSpec:
    fixture = corpus.Fixture("config-test", [], False, 8, "test")
    return corpus.RunSpec(
        target="v2",
        model_id="v2",
        artifact=artifact,
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="stochastic",
        fixture=fixture,
        seed=1,
        kv_dtype="fp8",
    )


def _point(artifact: Path) -> concurrency.Point:
    return concurrency.Point(
        target="v2",
        model_id="v2",
        artifact=artifact,
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="stochastic",
        suite="decode-saturation",
        concurrency=1,
    )


def _server_start(artifact: Path, *, schema_version: int, kv_cache: str) -> dict:
    return {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": schema_version,
        "event": "server_start",
        "engine": {
            "device": 0,
            "max_context": 262144,
            "kv_capacity": 262144,
            "kv_capacity_mode": "explicit",
            "max_concurrency": 1,
            "max_pending_requests": 1,
            "pending_timeout_ms": concurrency.PENDING_TIMEOUT_MS,
            "prefill_chunk": 1024,
            "log_stats_interval_ms": concurrency.STATS_INTERVAL_MS,
            "kv_cache": kv_cache,
            "cuda_graph": True,
            "vision": False,
            "prefix_reuse": False,
            "speculative_backend": "mtp",
            "speculative_draft_window": 3,
            "proposal_head": "optimized",
        },
        "sampling_defaults": {"greedy": False},
        "artifact": {"path": str(artifact)},
        "server": {"public_model_id": "v2"},
        "server_instance_id": "v2-config-test",
    }


@pytest.mark.parametrize("kv_dtype", ["int8", "fp8"])
def test_both_server_commands_forward_requested_kv_dtype(tmp_path: Path, kv_dtype: str) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    serve = tmp_path / "ninfer-serve"
    spec = _spec(artifact)
    corpus_command = corpus.server_command(serve, spec, tmp_path / "corpus.jsonl", 24562, 0, kv_dtype)

    concurrency_args = Namespace(
        port=24562,
        max_context=16384,
        kv_capacity="auto",
        prefill_chunk=1024,
        device=0,
        kv_dtype=kv_dtype,
    )
    concurrency_command = concurrency.server_command(
        serve, _point(artifact), tmp_path / "concurrency.jsonl", concurrency_args
    )

    for command in (corpus_command, concurrency_command):
        assert command[command.index("--kv-dtype") + 1] == kv_dtype


@pytest.mark.parametrize(
    ("mode", "backend", "draft_tokens"),
    (("mtp4", "mtp", 4), ("dflash2_6", "dflash2", 6)),
)
def test_server_command_transmits_targeted_draft_window(
    tmp_path: Path, mode: str, backend: str, draft_tokens: int
) -> None:
    artifact = tmp_path / "model.ninfer"
    spec = dataclasses.replace(
        _spec(artifact),
        speculative_mode=mode,
        speculative_backend=backend,
        draft_tokens=draft_tokens,
    )

    command = corpus.server_command(
        tmp_path / "ninfer-serve", spec, tmp_path / "server.jsonl", 24562, 0, "int8"
    )

    assert command[command.index("--spec") + 1] == backend
    assert command[command.index("--draft-tokens") + 1] == str(draft_tokens)


def test_corpus_start_validation_accepts_fp8_cache_and_rejects_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    spec = _spec(artifact)
    matching = _server_start(artifact, schema_version=20, kv_cache="fp8-e4m3-row256")

    assert corpus.validate_server_start(matching, spec, device=0, kv_dtype="fp8") == (
        "v2-config-test",
        "unreported-v2-schema20",
    )
    mismatched = _server_start(artifact, schema_version=20, kv_cache="int8-group64")
    with pytest.raises(corpus.CampaignError, match="Engine configuration mismatch"):
        corpus.validate_server_start(mismatched, spec, device=0, kv_dtype="fp8")


def test_concurrency_start_validation_accepts_fp8_and_rejects_cache_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "model.ninfer"
    artifact.touch()
    point = _point(artifact)
    args = Namespace(
        device=0,
        max_context=262144,
        kv_capacity="262144",
        prefill_chunk=1024,
        kv_dtype="fp8",
    )
    matching = _server_start(artifact, schema_version=21, kv_cache="fp8-e4m3-row256")
    matching["engine"]["kv_capacity_mode"] = "explicit"
    matching["artifact"]["prefill_signature"] = "v3-test-signature"

    assert concurrency.validate_server_start(matching, point, args) == (
        "v2-config-test",
        "v3-test-signature",
    )
    mismatched = _server_start(artifact, schema_version=21, kv_cache="int8-group64")
    mismatched["artifact"]["prefill_signature"] = "v3-test-signature"
    with pytest.raises(corpus.CampaignError, match="Engine configuration mismatch"):
        concurrency.validate_server_start(mismatched, point, args)


def test_concurrency_start_validation_accepts_v2_schema20_without_prefill_signature(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "v2.ninfer"
    artifact.touch()
    point = _point(artifact)
    args = Namespace(
        device=0,
        max_context=262144,
        kv_capacity="262144",
        prefill_chunk=1024,
        kv_dtype="int8",
    )
    event = _server_start(artifact, schema_version=20, kv_cache="int8-group64")

    assert concurrency.validate_server_start(event, point, args) == (
        "v2-config-test",
        "unreported-v2-schema20",
    )
