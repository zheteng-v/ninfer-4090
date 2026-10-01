#!/usr/bin/env python3
"""Reproducible provenance manifest for the RTX 4090 Phase 0 serving campaign.

This module is deliberately standalone so it never touches the product runners while they are
mid-flight. It records only measurement-identity facts that the campaign scorecard needs: the UTC
generation time, the repository revision and dirty state, the exact candidate binary/artifact pairs
with streamed SHA256 digests, the requested KV/phase/device/output selection, and best-effort
CUDA/driver/GPU environment strings.

Privacy boundary: the manifest never records process argv, environment variables, API keys, or
request payloads. Candidate paths are supplied explicitly by the caller.

Everything that shells out is injectable (`runner`, `repo_root`, `now`, `hasher`) so tests can be
deterministic and so a caller can build a manifest without touching the machine.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "Candidate",
    "DEFAULT_REPO_ROOT",
    "MANIFEST_ARTIFACT_TYPE",
    "MANIFEST_SCHEMA_VERSION",
    "build_manifest",
    "collect_environment",
    "git_metadata",
    "hash_file",
    "write_manifest",
]

MANIFEST_ARTIFACT_TYPE = "ninfer_phase0_provenance"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HASH_CHUNK_BYTES = 1 << 20

Runner = Callable[..., "subprocess.CompletedProcess[str] | Any"]
HashFn = Callable[[Path], "tuple[str, int]"]

# "CUDA Version:" is the classic field; newer drivers print "CUDA UMD Version:".
_CUDA_VERSION_RE = re.compile(r"CUDA(?: UMD)? Version:\s*([0-9]+(?:\.[0-9]+)*)")


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One explicit binary/artifact pair to fingerprint."""

    label: str
    binary: Path
    artifact: Path


def hash_file(path: Path, chunk_size: int = DEFAULT_HASH_CHUNK_BYTES) -> tuple[str, int]:
    """Stream a file through SHA256 so large (multi-GiB) artifacts never load into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _invoke(runner: Runner, command: Sequence[str]) -> Any | None:
    """Run one external command, returning None when it is unavailable instead of raising."""

    try:
        return runner(list(command), capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _stdout(completed: Any | None) -> str | None:
    if completed is None:
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    output = getattr(completed, "stdout", None)
    return output if isinstance(output, str) else None


def git_metadata(repo_root: Path, runner: Runner) -> dict[str, Any]:
    """Return HEAD and dirty state; every field degrades to None when git is unavailable."""

    metadata: dict[str, Any] = {"head": None, "dirty": None, "error": None}
    head = _stdout(_invoke(runner, ["git", "-C", str(repo_root), "rev-parse", "HEAD"]))
    if head is None:
        metadata["error"] = "git rev-parse HEAD unavailable"
        return metadata
    metadata["head"] = head.strip()

    status = _stdout(_invoke(runner, ["git", "-C", str(repo_root), "status", "--porcelain"]))
    if status is None:
        metadata["error"] = "git status unavailable"
        return metadata
    metadata["dirty"] = bool(status.strip())
    return metadata


def collect_environment(runner: Runner) -> dict[str, Any]:
    """Best-effort CUDA/driver/GPU report; nulls when nvidia-smi is absent or failing."""

    environment: dict[str, Any] = {
        "nvidia_smi_available": False,
        "cuda_version": None,
        "driver_version": None,
        "gpus": [],
        "error": None,
    }
    query = _invoke(
        runner,
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )
    if query is None or getattr(query, "returncode", 1) != 0:
        environment["error"] = "nvidia-smi unavailable"
        return environment

    environment["nvidia_smi_available"] = True
    for line in str(getattr(query, "stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        try:
            index = int(fields[0])
            memory_total_mib = int(fields[3])
        except ValueError:
            continue
        environment["gpus"].append(
            {
                "index": index,
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": memory_total_mib,
            }
        )
    if environment["gpus"]:
        environment["driver_version"] = environment["gpus"][0]["driver_version"]

    banner = _stdout(_invoke(runner, ["nvidia-smi"]))
    if banner is not None:
        match = _CUDA_VERSION_RE.search(banner)
        if match:
            environment["cuda_version"] = match.group(1)
    return environment


def _utc_isoformat(moment: datetime.datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=datetime.timezone.utc)
    else:
        moment = moment.astimezone(datetime.timezone.utc)
    return moment.isoformat()


def _candidate_record(candidate: Candidate, hasher: HashFn) -> dict[str, Any]:
    record: dict[str, Any] = {
        "label": candidate.label,
        "binary": str(candidate.binary),
        "binary_sha256": None,
        "binary_bytes": None,
        "artifact": str(candidate.artifact),
        "artifact_sha256": None,
        "artifact_bytes": None,
        "error": None,
    }
    try:
        record["binary_sha256"], record["binary_bytes"] = hasher(candidate.binary)
        record["artifact_sha256"], record["artifact_bytes"] = hasher(candidate.artifact)
    except (OSError, ValueError) as exc:
        record["error"] = str(exc)
    return record


def build_manifest(
    candidates: Iterable[Candidate],
    kv_dtypes: Sequence[str],
    phase: str,
    output_dir: Path,
    device: int,
    *,
    repo_root: Path | None = None,
    runner: Runner | None = None,
    now: Callable[[], datetime.datetime] | None = None,
    hasher: HashFn | None = None,
    collect_environment_info: bool = True,
) -> dict[str, Any]:
    """Assemble a JSON-serializable provenance manifest for one Phase 0 invocation."""

    resolved_root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
    resolved_runner: Runner = runner if runner is not None else subprocess.run
    resolved_now = now if now is not None else (lambda: datetime.datetime.now(datetime.timezone.utc))
    resolved_hasher: HashFn = hasher if hasher is not None else hash_file

    manifest: dict[str, Any] = {
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_isoformat(resolved_now()),
        "git": git_metadata(resolved_root, resolved_runner),
        "request": {
            "kv_dtypes": [str(value) for value in kv_dtypes],
            "phase": str(phase),
            "device": int(device),
            "output_dir": str(output_dir),
        },
        "candidates": [
            _candidate_record(candidate, resolved_hasher) for candidate in candidates
        ],
        "environment": (
            collect_environment(resolved_runner) if collect_environment_info else None
        ),
    }
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest as deterministic UTF-8 JSON with a trailing newline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
