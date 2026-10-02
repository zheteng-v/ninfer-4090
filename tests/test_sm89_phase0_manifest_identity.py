"""Resume identity checks for the Phase 0 provenance manifest."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.bench import run_sm89_phase0 as phase0


def _manifest() -> dict:
    return {
        "git": {"head": "abc123", "dirty": True, "error": None},
        "source_fingerprints": [
            {"path": "tools/bench/run_sm89_phase0.py", "sha256": "source-a", "bytes": 1, "error": None}
        ],
        "request": {"kv_dtypes": ["int8"], "phase": "screen", "device": 0, "output_dir": "out"},
        "candidates": [{"label": "v3", "binary_sha256": "binary-a", "artifact_sha256": "artifact-a"}],
        "environment": {"cuda_version": "13.4", "driver_version": "615.71.09"},
    }


def _args(output: Path) -> Namespace:
    return Namespace(output=output, phase="screen", device=0)


def test_manifest_creation_fingerprints_phase_zero_sources(tmp_path: Path, monkeypatch) -> None:
    expected = _manifest()
    observed: dict = {}

    def fake_build(*args, **kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(phase0, "build_manifest", fake_build)
    candidate = phase0.Candidate("v3", Path("/serve"), Path("/artifact"))

    path = phase0.ensure_manifest(candidate, "int8", _args(tmp_path))

    assert observed["source_files"] == phase0.PROVENANCE_SOURCE_FILES
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_legacy_manifest_without_source_fingerprints_cannot_resume(
    tmp_path: Path, monkeypatch
) -> None:
    expected = _manifest()
    monkeypatch.setattr(phase0, "build_manifest", lambda *args, **kwargs: expected)
    candidate = phase0.Candidate("v3", Path("/serve"), Path("/artifact"))
    path = tmp_path / "v3" / "int8" / "manifest.json"
    path.parent.mkdir(parents=True)
    legacy = {key: value for key, value in expected.items() if key != "source_fingerprints"}
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(phase0.Phase0Error, match="does not match this run"):
        phase0.ensure_manifest(candidate, "int8", _args(tmp_path))
