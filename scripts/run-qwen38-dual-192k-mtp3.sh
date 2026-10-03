#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$scripts_dir/.." && pwd)"
server="${NINFER_SERVER:-$repo_root/build-native/apps/ninfer-serve}"
model="${1:-${NINFER_MODEL_DIR:-$repo_root/models}/qwen3_8_27b_v3.ninfer}"

if [[ ! -x "$server" ]]; then
  printf 'Missing executable: %s\nSet NINFER_SERVER to the ninfer-serve path.\n' "$server" >&2
  exit 1
fi
if [[ ! -f "$model" ]]; then
  printf 'Missing model: %s\nPass the artifact path or set NINFER_MODEL_DIR.\n' "$model" >&2
  exit 1
fi

printf '%s\n' 'Qwen3.8-27B V3: MTP3, two requests, up to 192K context each, INT8 KV'
exec "$server" "$model" \
  --host 127.0.0.1 --port 8080 \
  --model-id Qwen3.8-27B-NInfer-V3-MTP3-Dual-192K \
  --max-context 196608 --kv-capacity 393216 \
  --max-concurrency 2 --max-pending-requests 4 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision
