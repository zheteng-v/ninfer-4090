"""Deterministic tests for the Phase 0 provenance manifest.

The module is intentionally dependency-injected, so these tests never call git, nvidia-smi, or any
NInfer binary. A fake subprocess runner supplies the command output.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from tools.bench.phase0_provenance import (
    Candidate,
    build_manifest,
    hash_file,
    write_manifest,
)

FIXED_NOW = datetime.datetime(2026, 10, 1, 6, 0, 0, tzinfo=datetime.timezone.utc)
FIXED_NOW_ISO = "2026-10-01T06:00:00+00:00"


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class FakeRunner:
    def __init__(self, handler: Callable[[Sequence[str]], Any]) -> None:
        self.handler = handler
        self.calls: list[list[str]] = []

    def __call__(self, command: Sequence[str], **_: Any) -> Any:
        recorded = list(command)
        self.calls.append(recorded)
        return self.handler(recorded)


def _standard_handler(
    *, head: str = "abc123def\n", status: str = "", nvidia: bool = True
) -> Callable[[Sequence[str]], Any]:
    def handler(command: Sequence[str]) -> Any:
        if command[0] == "git":
            if "rev-parse" in command:
                return _completed(head)
            if "status" in command:
                return _completed(status)
            raise AssertionError(f"unexpected git command: {command}")
        if command[0] == "nvidia-smi":
            if not nvidia:
                raise FileNotFoundError("nvidia-smi")
            if any(argument.startswith("--query-gpu") for argument in command):
                return _completed(
                    "0, NVIDIA GeForce RTX 5060 Ti, 615.71.09, 16311\n"
                    "1, NVIDIA GeForce RTX 4090, 615.71.09, 49140\n"
                )
            return _completed(
                "| NVIDIA-SMI 615.71.09  KMD Version: 615.71.09  CUDA UMD Version: 13.4 |\n"
            )
        raise AssertionError(f"unexpected command: {command}")

    return handler


def test_build_manifest_records_fields_and_sha256(tmp_path: Path) -> None:
    binary = tmp_path / "ninfer-serve"
    binary.write_bytes(b"binary-payload")
    artifact = tmp_path / "model.ninfer"
    artifact.write_bytes(b"artifact-payload")
    runner = FakeRunner(_standard_handler(status=""))

    manifest = build_manifest(
        [Candidate("v3", binary, artifact)],
        ["int8", "fp8"],
        "screen",
        tmp_path / "out",
        0,
        repo_root=tmp_path,
        runner=runner,
        now=lambda: FIXED_NOW,
    )

    assert manifest["artifact_type"] == "ninfer_phase0_provenance"
    assert manifest["schema_version"] == 1
    assert manifest["generated_at_utc"] == FIXED_NOW_ISO
    assert manifest["git"] == {"head": "abc123def", "dirty": False, "error": None}
    assert manifest["request"] == {
        "kv_dtypes": ["int8", "fp8"],
        "phase": "screen",
        "device": 0,
        "output_dir": str(tmp_path / "out"),
    }

    record = manifest["candidates"][0]
    assert record["label"] == "v3"
    assert record["binary"] == str(binary)
    assert record["artifact"] == str(artifact)
    assert record["binary_sha256"] == hashlib.sha256(b"binary-payload").hexdigest()
    assert record["binary_bytes"] == len(b"binary-payload")
    assert record["artifact_sha256"] == hashlib.sha256(b"artifact-payload").hexdigest()
    assert record["artifact_bytes"] == len(b"artifact-payload")
    assert record["error"] is None

    environment = manifest["environment"]
    assert environment["nvidia_smi_available"] is True
    assert environment["cuda_version"] == "13.4"
    assert environment["driver_version"] == "615.71.09"
    assert environment["error"] is None
    assert [gpu["index"] for gpu in environment["gpus"]] == [0, 1]
    assert environment["gpus"][1]["name"] == "NVIDIA GeForce RTX 4090"
    assert environment["gpus"][1]["memory_total_mib"] == 49140

    # Paths are passed to git explicitly, never via the process working directory or shell.
    assert all(call[0] == "git" and "-C" in call for call in runner.calls if call[0] == "git")


def test_git_dirty_and_clean_flags(tmp_path: Path) -> None:
    clean = build_manifest(
        [], ["int8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler(status="")),
        now=lambda: FIXED_NOW,
    )
    dirty = build_manifest(
        [], ["int8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler(status=" M tools/bench/x.py\n")),
        now=lambda: FIXED_NOW,
    )
    assert clean["git"]["dirty"] is False
    assert dirty["git"]["dirty"] is True


def test_git_failure_records_nulls(tmp_path: Path) -> None:
    def failing(command: Sequence[str]) -> Any:
        raise FileNotFoundError(command[0])

    manifest = build_manifest(
        [], ["int8"], "full", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(failing), now=lambda: FIXED_NOW,
    )
    assert manifest["git"]["head"] is None
    assert manifest["git"]["dirty"] is None
    assert isinstance(manifest["git"]["error"], str) and manifest["git"]["error"]


def test_nvidia_smi_missing_is_graceful(tmp_path: Path) -> None:
    manifest = build_manifest(
        [], ["fp8"], "screen", tmp_path, 1,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler(nvidia=False)),
        now=lambda: FIXED_NOW,
    )
    environment = manifest["environment"]
    assert environment["nvidia_smi_available"] is False
    assert environment["cuda_version"] is None
    assert environment["driver_version"] is None
    assert environment["gpus"] == []
    assert isinstance(environment["error"], str) and environment["error"]


def test_streaming_hash_matches_reference(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 40
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    digest, size = hash_file(path, chunk_size=7)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_missing_candidate_file_records_null_digest(tmp_path: Path) -> None:
    manifest = build_manifest(
        [Candidate("v2", tmp_path / "absent-serve", tmp_path / "absent.ninfer")],
        ["int8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler()),
        now=lambda: FIXED_NOW,
    )
    record = manifest["candidates"][0]
    assert record["binary_sha256"] is None
    assert record["artifact_sha256"] is None
    assert isinstance(record["error"], str) and record["error"]


def test_manifest_omits_argv_and_api_key(tmp_path: Path) -> None:
    manifest = build_manifest(
        [], ["int8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler()),
        now=lambda: FIXED_NOW,
    )
    serialized = json.dumps(manifest)
    assert "api_key" not in serialized
    assert "argv" not in serialized

    without_environment = build_manifest(
        [], ["int8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler()),
        now=lambda: FIXED_NOW, collect_environment_info=False,
    )
    assert without_environment["environment"] is None


def test_write_manifest_roundtrips(tmp_path: Path) -> None:
    manifest = build_manifest(
        [Candidate("v3", tmp_path / "b", tmp_path / "a")],
        ["fp8"], "screen", tmp_path, 0,
        repo_root=tmp_path, runner=FakeRunner(_standard_handler()),
        now=lambda: FIXED_NOW,
    )
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(path, manifest)

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == manifest
