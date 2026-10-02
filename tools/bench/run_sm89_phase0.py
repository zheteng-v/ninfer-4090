#!/usr/bin/env python3
"""Run the reproducible RTX 4090 Phase 0 serving baseline matrix.

The wrapper deliberately starts one server per candidate/KV/mode point through the
existing corpus and concurrency runners.  Candidate outputs are isolated, so v2
and v3 binaries with incompatible artifact contracts can be compared without
mixing resume records or server logs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_RUNNER = REPO_ROOT / "tools/bench/run_serve_corpus.py"
CONCURRENCY_RUNNER = REPO_ROOT / "tools/bench/run_serve_concurrency.py"
PROVENANCE_MODULE = REPO_ROOT / "tools/bench/phase0_provenance.py"
PROVENANCE_SOURCE_FILES = (
    Path(__file__).resolve(),
    CORPUS_RUNNER,
    CONCURRENCY_RUNNER,
    PROVENANCE_MODULE,
)
KV_DTYPES = ("int8", "fp8")
SCREEN_MODES = ("mtp3", "mtp4", "dflash2_6", "dflash2_7")
SCREEN_FIXTURES = ("scenario_code_python",)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.phase0_provenance import (  # noqa: E402
    Candidate as ProvenanceCandidate,
    build_manifest,
    write_manifest,
)
from tools.bench.run_serve_corpus import SEEDS as CORPUS_SEEDS  # noqa: E402


class Phase0Error(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Candidate:
    label: str
    serve: Path
    artifact: Path


def parse_candidate(value: str) -> Candidate:
    label, first, remainder = value.partition("=")
    serve, second, artifact = remainder.partition("=")
    if not first or not second or not label or not serve or not artifact:
        raise Phase0Error("--candidate must be LABEL=NINFER_SERVE=MODEL_NINFER")
    return Candidate(label, Path(serve).expanduser().resolve(), Path(artifact).expanduser().resolve())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True,
        metavar="LABEL=NINFER_SERVE=MODEL_NINFER",
        help="explicit binary/artifact pair; repeat for v2 and v3",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kv-dtype", action="append", choices=KV_DTYPES)
    parser.add_argument("--phase", choices=("screen", "full"), default="screen")
    parser.add_argument(
        "--screen-mode", action="append", choices=SCREEN_MODES,
        help="screen only these speculative modes; repeat as needed (default: baseline screen)",
    )
    parser.add_argument(
        "--screen-fixture", choices=SCREEN_FIXTURES,
        help="use this one fixture for explicitly selected --screen-mode values",
    )
    parser.add_argument(
        "--screen-seed", type=int, choices=CORPUS_SEEDS,
        help="use this exact fixed corpus seed for each screen request (default: first seed)",
    )
    parser.add_argument(
        "--screen-max-context", type=int, default=262144,
        help="Engine context for short screen corpus requests (default: 262144)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate(candidates: Sequence[Candidate], args: argparse.Namespace) -> None:
    if len({candidate.label for candidate in candidates}) != len(candidates):
        raise Phase0Error("candidate labels must be unique")
    if args.port < 1 or args.port > 65535:
        raise Phase0Error("--port must be in [1, 65535]")
    if args.device < 0:
        raise Phase0Error("--device must be nonnegative")
    if getattr(args, "screen_mode", None) and len(args.screen_mode) != len(set(args.screen_mode)):
        raise Phase0Error("duplicate --screen-mode value")
    if getattr(args, "screen_fixture", None) and not getattr(args, "screen_mode", None):
        raise Phase0Error("--screen-fixture requires at least one --screen-mode")
    if getattr(args, "screen_seed", None) is not None and args.screen_seed < 0:
        raise Phase0Error("--screen-seed must be nonnegative")
    screen_max_context = getattr(args, "screen_max_context", 262144)
    if screen_max_context < 1:
        raise Phase0Error("--screen-max-context must be positive")
    if args.phase == "full" and (
        getattr(args, "screen_mode", None)
        or getattr(args, "screen_fixture", None)
        or getattr(args, "screen_seed", None) is not None
        or screen_max_context != 262144
    ):
        raise Phase0Error(
            "--screen-mode/--screen-fixture/--screen-seed/--screen-max-context "
            "require --phase screen (and default screen max context)"
        )
    if args.dry_run:
        return
    for candidate in candidates:
        if not candidate.serve.is_file():
            raise Phase0Error(f"ninfer-serve not found: {candidate.serve}")
        if not candidate.artifact.is_file():
            raise Phase0Error(f"artifact not found: {candidate.artifact}")


def commands(candidate: Candidate, kv_dtype: str, args: argparse.Namespace) -> list[list[str]]:
    root = args.output.expanduser().resolve() / candidate.label / kv_dtype
    seed_count = "1" if args.phase == "screen" else "5"
    full_corpus = [
        sys.executable, str(CORPUS_RUNNER), "--serve", str(candidate.serve),
        "--artifact", f"{candidate.label}={candidate.artifact}",
        "--mode", "mtp0", "--mode", "mtp3", "--mode", "dflash2_7",
        "--sampling", "stochastic", "--kv-dtype", kv_dtype,
        "--seed-count", seed_count, "--port", str(args.port), "--device", str(args.device),
        "--output", str(root / "corpus"),
    ]
    full_concurrency = [
        sys.executable, str(CONCURRENCY_RUNNER), "--serve", str(candidate.serve),
        "--artifact", f"{candidate.label}={candidate.artifact}",
        "--mode", "mtp3", "--sampling", "stochastic", "--suite", "decode-saturation",
        "--concurrency", "1", "--concurrency", "2", "--concurrency", "4", "--concurrency", "8",
        "--kv-dtype", kv_dtype, "--max-context", "16384", "--kv-capacity", "auto",
        "--prefill-chunk", "1024",
        "--port", str(args.port), "--device", str(args.device), "--output", str(root / "concurrency"),
    ]
    if args.phase == "full":
        return [full_corpus, full_concurrency]

    # Preserve the established baseline screen unless the caller explicitly
    # selects speculative modes. Custom screens use one fixed scenario fixture.
    screen_corpora: list[list[str]] = []
    screen_modes = getattr(args, "screen_mode", None)
    if screen_modes:
        selected = [
            (mode, getattr(args, "screen_fixture", None) or "scenario_code_python")
            for mode in screen_modes
        ]
    else:
        selected = [
            ("mtp0", "long_niah_8k"),
            ("mtp3", "scenario_code_python"),
            ("dflash2_7", "scenario_code_python"),
        ]
    for mode, fixture in selected:
        command = list(full_corpus)
        mode_index = command.index("--mode")
        del command[mode_index : mode_index + 6]
        command[command.index("--output") + 1] = str(root / "screen" / mode)
        command.extend(("--mode", mode, "--fixture", fixture, "--no-summary"))
        command.extend(("--max-context", str(getattr(args, "screen_max_context", 262144))))
        screen_seed = getattr(args, "screen_seed", None)
        if screen_seed is not None:
            seed_count_index = command.index("--seed-count")
            del command[seed_count_index : seed_count_index + 2]
            command.extend(("--seed", str(screen_seed)))
        screen_corpora.append(command)
    if screen_modes:
        return screen_corpora

    screen_concurrency = list(full_concurrency)
    concurrency_index = screen_concurrency.index("--concurrency")
    del screen_concurrency[concurrency_index + 2 : concurrency_index + 8]
    screen_concurrency.extend(("--decode-tokens", "2048"))
    screen_concurrency[screen_concurrency.index("--output") + 1] = str(root / "screen" / "concurrency")
    return [*screen_corpora, screen_concurrency]


def ensure_manifest(candidate: Candidate, kv_dtype: str, args: argparse.Namespace) -> Path:
    output_dir = args.output.expanduser().resolve() / candidate.label / kv_dtype
    manifest_path = output_dir / "manifest.json"
    current = build_manifest(
        [
            ProvenanceCandidate(
                label=candidate.label,
                binary=candidate.serve,
                artifact=candidate.artifact,
            )
        ],
        [kv_dtype],
        args.phase,
        output_dir,
        args.device,
        source_files=PROVENANCE_SOURCE_FILES,
    )
    current["request"]["screen_max_context"] = (
        getattr(args, "screen_max_context", 262144) if args.phase == "screen" else 262144
    )
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase0Error(f"existing provenance manifest is unreadable: {manifest_path}: {exc}") from exc
        identity_fields = (
            "git",
            "source_fingerprints",
            "request",
            "candidates",
            "environment",
        )
        if any(existing.get(field) != current.get(field) for field in identity_fields):
            raise Phase0Error(
                f"existing provenance manifest does not match this run: {manifest_path}; "
                "choose a fresh --output directory"
            )
        return manifest_path

    write_manifest(manifest_path, current)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = [parse_candidate(value) for value in args.candidate]
    validate(candidates, args)
    kv_dtypes = args.kv_dtype or list(KV_DTYPES)
    if len(kv_dtypes) != len(set(kv_dtypes)):
        raise Phase0Error("duplicate --kv-dtype value")
    for candidate in candidates:
        for kv_dtype in kv_dtypes:
            if not args.dry_run:
                manifest_path = ensure_manifest(candidate, kv_dtype, args)
                print(f"manifest: {manifest_path}", flush=True)
            for command in commands(candidate, kv_dtype, args):
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Phase0Error, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
