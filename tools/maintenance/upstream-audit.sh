#!/usr/bin/env bash
set -Eeuo pipefail

limit=12
fetch=1
allow_dirty=0

usage() {
  cat <<'EOF'
Usage: tools/maintenance/upstream-audit.sh [--no-fetch] [--allow-dirty] [--limit N]

Fetch and compare the maintained downstream against Neroued/ninfer and
sergiuszm/ninfer-4090. The report is Markdown and includes recent public PRs
when the GitHub API is available. The script never merges, rebases, or pushes.
EOF
}

while (($#)); do
  case "$1" in
    --no-fetch)
      fetch=0
      shift
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
      ;;
    --limit)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || {
        echo "error: --limit requires a positive integer" >&2
        exit 2
      }
      limit="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "error: run this script inside the ninfer repository" >&2
  exit 2
}
cd "$repo_root"

if ((allow_dirty == 0)) && [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "error: upstream audit requires a clean worktree (or pass --allow-dirty)" >&2
  exit 2
fi

declare -A expected=(
  [origin]="https://github.com/zheteng-v/ninfer-4090.git"
  [upstream]="https://github.com/Neroued/ninfer.git"
  [sergiuszm]="https://github.com/sergiuszm/ninfer-4090.git"
)

for remote in origin upstream sergiuszm; do
  actual="$(git remote get-url "$remote" 2>/dev/null || true)"
  if [[ "$actual" != "${expected[$remote]}" ]]; then
    echo "error: remote '$remote' must fetch ${expected[$remote]} (found '${actual:-missing}')" >&2
    exit 2
  fi
done

if ((fetch)); then
  for remote in origin upstream sergiuszm; do
    git fetch --quiet --prune --tags "$remote"
  done
fi

refs=(origin/main upstream/master upstream/dev sergiuszm/rtx4090-port)
for ref in "${refs[@]}"; do
  git rev-parse --verify --quiet "$ref^{commit}" >/dev/null || {
    echo "error: required ref '$ref' is unavailable" >&2
    exit 2
  }
done

escape_markdown() {
  sed 's/|/\\|/g'
}

head_sha="$(git rev-parse HEAD)"
head_short="$(git rev-parse --short=12 HEAD)"
branch="$(git symbolic-ref --quiet --short HEAD || printf '(detached)')"

printf '# NInfer upstream audit\n\n'
printf -- '- Generated: `%s`\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf -- '- Branch: `%s`\n' "$branch"
printf -- '- Local HEAD: `%s`\n\n' "$head_sha"

printf '| Source | Head | Local-only | Source-only |\n'
printf '|---|---:|---:|---:|\n'
for ref in "${refs[@]}"; do
  ref_sha="$(git rev-parse --short=12 "$ref")"
  read -r local_only source_only < <(git rev-list --left-right --count "HEAD...$ref")
  printf '| `%s` | `%s` | %s | %s |\n' "$ref" "$ref_sha" "$local_only" "$source_only"
done

for ref in upstream/master upstream/dev sergiuszm/rtx4090-port; do
  printf '\n## Recent commits only in `%s`\n\n' "$ref"
  rows="$(git log --no-merges --date=short --format='%h%x09%ad%x09%s' "HEAD..$ref" -n "$limit")"
  if [[ -z "$rows" ]]; then
    printf '_None._\n'
    continue
  fi
  while IFS=$'\t' read -r sha commit_date subject; do
    subject="$(printf '%s' "$subject" | escape_markdown)"
    printf -- '- `%s` %s — %s\n' "$sha" "$commit_date" "$subject"
  done <<<"$rows"
done

print_prs() {
  local repository="$1"
  local api_url="https://api.github.com/repos/${repository}/pulls?state=all&sort=updated&direction=desc&per_page=${limit}"
  local browse_url="https://github.com/${repository}/pulls?q=is%3Apr+sort%3Aupdated-desc"
  local payload

  printf '\n## Recently updated PRs in `%s`\n\n' "$repository"
  if ! command -v curl >/dev/null || ! command -v jq >/dev/null; then
    printf 'Public API tooling unavailable; inspect %s\n' "$browse_url"
    return
  fi

  if ! payload="$(curl --fail --silent --show-error --location \
      --header 'Accept: application/vnd.github+json' \
      --header 'X-GitHub-Api-Version: 2022-11-28' \
      "$api_url" 2>/dev/null)"; then
    printf 'Public API request failed or was rate-limited; inspect %s\n' "$browse_url"
    return
  fi

  if ! jq -e 'type == "array"' >/dev/null <<<"$payload"; then
    printf 'Unexpected API response; inspect %s\n' "$browse_url"
    return
  fi

  if [[ "$(jq 'length' <<<"$payload")" == 0 ]]; then
    printf '_No pull requests returned._\n'
    return
  fi

  jq -r '.[] | [.number, .state, (.merged_at // "-"), .updated_at, .title, .html_url] | @tsv' \
    <<<"$payload" |
    while IFS=$'\t' read -r number state merged updated title url; do
      title="$(printf '%s' "$title" | escape_markdown)"
      if [[ "$merged" != "-" ]]; then
        state="merged"
      fi
      printf -- '- [#%s](%s) `%s`, updated %s — %s\n' \
        "$number" "$url" "$state" "${updated%%T*}" "$title"
    done
}

print_prs Neroued/ninfer
print_prs sergiuszm/ninfer-4090

printf '\n## Decision reminder\n\n'
printf 'Classify relevant changes as `adopt`, `adapt`, `benchmark first`, `watch`, or `not applicable`. '
printf 'Record the audited SHAs and the decision in `docs/maintainer/downstream-maintenance.md`.\n'

printf '\n_Audit completed from `%s`; no merge, rebase, or push was performed._\n' "$head_short"
