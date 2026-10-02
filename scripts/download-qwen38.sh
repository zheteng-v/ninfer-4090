#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/qwen3_8_27b.ninfer"

mkdir -p -- "$model_dir"
# Pinned to the artifact release commit of the current v3 container (verified with an HTTP HEAD:
# x-linked-etag matches expected_sha256 and content-length is 20,437,521,664 bytes). The engine
# loads container v3; the earlier container-v2 pin (revision 3526913004b1) is kept in git history.
revision='1cbd84e7221e51186bd7f093a149912d2489625b'
expected_sha256='81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da'

printf '%s\n' "Downloading Qwen3.8-27B NInfer model (revision $revision)..."
if ! curl -L -C - --fail --output "$model" \
  "https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/$revision/qwen3_8_27b.ninfer"; then
  printf '%s\n' 'Download failed. Run this script again to resume.' >&2
  exit 1
fi
if [[ -z "${NINFER_SKIP_SHA256:-}" ]] && command -v sha256sum >/dev/null 2>&1; then
  printf '%s\n' 'Verifying SHA-256...'
  actual_sha256="$(sha256sum -- "$model" | cut -d' ' -f1)"
  if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'SHA-256 mismatch: expected %s, got %s\n' "$expected_sha256" "$actual_sha256" >&2
    exit 1
  fi
fi
printf 'Model ready: %s\n' "$model"
