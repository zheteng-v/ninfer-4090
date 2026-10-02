"""Behavioral tests for the Phase 0 campaign wrapper (no server/GPU required)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from tools.bench import run_sm89_phase0 as phase0


def _args(tmp_path: Path, *, phase: str = "screen", port: int = 24562, device: int = 0):
    return phase0.parse_args(
        ["--candidate", "v3=/bin/serve=/models/model.ninfer", "--output", str(tmp_path),
         "--phase", phase, "--port", str(port), "--device", str(device)]
    )


def _values(command: list[str], option: str) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command[:-1]) if value == option]


def test_parse_candidate_resolves_explicit_binary_and_artifact(tmp_path: Path) -> None:
    candidate = phase0.parse_candidate("v2=~/bin/ninfer-serve=~/models/qwen.ninfer")

    assert candidate.label == "v2"
    assert candidate.serve == Path("~/bin/ninfer-serve").expanduser().resolve()
    assert candidate.artifact == Path("~/models/qwen.ninfer").expanduser().resolve()


@pytest.mark.parametrize("value", ["", "v3", "=serve=model", "v3==model", "v3=serve="])
def test_parse_candidate_rejects_incomplete_pairs(value: str) -> None:
    with pytest.raises(phase0.Phase0Error, match="LABEL=NINFER_SERVE=MODEL_NINFER"):
        phase0.parse_candidate(value)


def test_validate_rejects_duplicate_labels_and_invalid_port_or_device(tmp_path: Path) -> None:
    candidate = phase0.Candidate("same", tmp_path / "serve", tmp_path / "model")
    args = _args(tmp_path, port=24562)
    args.dry_run = True

    with pytest.raises(phase0.Phase0Error, match="labels must be unique"):
        phase0.validate([candidate, candidate], args)
    for invalid_port in (0, 65536):
        with pytest.raises(phase0.Phase0Error, match=r"port must be in \[1, 65535\]"):
            phase0.validate([candidate], _args(tmp_path, port=invalid_port))
    invalid_device = _args(tmp_path, device=-1)
    with pytest.raises(phase0.Phase0Error, match="device must be nonnegative"):
        phase0.validate([candidate], invalid_device)


def test_dry_run_allows_nonexistent_candidate_files_and_does_not_launch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(phase0.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched runner"))
    output = io.StringIO()

    with contextlib.redirect_stdout(output):
        result = phase0.main([
            "--candidate", "v3=/missing/ninfer-serve=/missing/model.ninfer",
            "--output", str(tmp_path), "--kv-dtype", "int8", "--dry-run",
        ])

    assert result == 0
    assert "run_serve_corpus.py" in output.getvalue()
    assert "run_serve_concurrency.py" in output.getvalue()
    assert not list(tmp_path.rglob("manifest.json"))


@pytest.mark.parametrize("kv_dtype", ["int8", "fp8"])
def test_full_commands_cover_corpus_modes_seeds_and_concurrency_matrix(tmp_path: Path, kv_dtype: str) -> None:
    candidate = phase0.Candidate("v3", Path("/bin/serve"), Path("/models/model.ninfer"))
    args = _args(tmp_path, phase="full")

    corpus, concurrency = phase0.commands(candidate, kv_dtype, args)

    assert _values(corpus, "--mode") == ["mtp0", "mtp3", "dflash2_7"]
    assert _values(corpus, "--seed-count") == ["5"]
    assert _values(corpus, "--kv-dtype") == [kv_dtype]
    assert _values(concurrency, "--concurrency") == ["1", "2", "4", "8"]
    assert _values(concurrency, "--max-context") == ["16384"]
    assert _values(concurrency, "--kv-dtype") == [kv_dtype]


@pytest.mark.parametrize("kv_dtype", ["int8", "fp8"])
def test_screen_commands_are_one_seed_one_concurrency_and_2048_decode(
    tmp_path: Path, kv_dtype: str
) -> None:
    candidate = phase0.Candidate("v3", Path("/bin/serve"), Path("/models/model.ninfer"))
    args = _args(tmp_path, phase="screen")

    commands = phase0.commands(candidate, kv_dtype, args)
    corpora, concurrency = commands[:3], commands[3]

    assert [_values(command, "--mode") for command in corpora] == [["mtp0"], ["mtp3"], ["dflash2_7"]]
    assert all(_values(command, "--seed-count") == ["1"] for command in corpora)
    assert all(_values(command, "--kv-dtype") == [kv_dtype] for command in commands)
    assert all("--fixture" in command and "--no-summary" in command for command in corpora)
    assert _values(concurrency, "--concurrency") == ["1"]
    assert _values(concurrency, "--max-context") == ["16384"]
    assert _values(concurrency, "--decode-tokens") == ["2048"]


def test_selected_screen_modes_use_one_scenario_fixture_and_keep_fixed_seed(tmp_path: Path) -> None:
    args = phase0.parse_args([
        "--candidate", "v3=/bin/serve=/models/model.ninfer", "--output", str(tmp_path),
        "--kv-dtype", "fp8", "--screen-mode", "mtp3", "--screen-mode", "mtp4",
        "--screen-mode", "dflash2_6", "--screen-mode", "dflash2_7",
        "--screen-fixture", "scenario_code_python", "--screen-seed", "912910298659544128",
        "--screen-max-context", "8192",
    ])
    candidate = phase0.Candidate("v3", Path("/bin/serve"), Path("/models/model.ninfer"))

    commands = phase0.commands(candidate, "fp8", args)
    corpus_commands = commands

    assert [_values(command, "--mode") for command in corpus_commands] == [
        ["mtp3"], ["mtp4"], ["dflash2_6"], ["dflash2_7"]
    ]
    assert all(_values(command, "--fixture") == ["scenario_code_python"] for command in corpus_commands)
    assert all(_values(command, "--seed") == ["912910298659544128"] for command in corpus_commands)
    assert all(_values(command, "--max-context") == ["8192"] for command in corpus_commands)
    assert all("--seed-count" not in command for command in corpus_commands)
    assert all(_values(command, "--kv-dtype") == ["fp8"] for command in commands)
    assert all("run_serve_concurrency.py" not in command[1] for command in commands)


def test_screen_fixture_requires_explicit_mode_selection(tmp_path: Path) -> None:
    with pytest.raises(phase0.Phase0Error, match="requires at least one --screen-mode"):
        phase0.main([
            "--candidate", "v3=/missing/serve=/missing/model.ninfer", "--output", str(tmp_path),
            "--screen-fixture", "scenario_code_python", "--dry-run",
        ])


def test_full_phase_rejects_screen_only_options(tmp_path: Path) -> None:
    with pytest.raises(phase0.Phase0Error, match="require --phase screen"):
        phase0.main([
            "--candidate", "v3=/missing/serve=/missing/model.ninfer", "--output", str(tmp_path),
            "--phase", "full", "--screen-mode", "mtp4", "--dry-run",
        ])


def test_full_phase_rejects_custom_screen_max_context(tmp_path: Path) -> None:
    with pytest.raises(phase0.Phase0Error, match="require --phase screen"):
        phase0.main([
            "--candidate", "v3=/missing/serve=/missing/model.ninfer", "--output", str(tmp_path),
            "--phase", "full", "--screen-max-context", "8192", "--dry-run",
        ])
