#!/usr/bin/env bash
set -euo pipefail

# This script lives in docs/_scripts and is invoked from the repo root.
# Archive paths are relative to docs/ so the restore step can extract them there.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/docs"
if [[ ! -f .jekyll-metadata ]]; then
  echo "No Jekyll metadata to checkpoint" >&2
  exit 1
fi
if [[ -d _site ]]; then
  file_count="$(find _site -type f | wc -l)"
  if [[ "$file_count" -eq 0 ]]; then
    echo "Checkpoint _site has no rendered files" >&2
    exit 1
  fi
fi

paths=(.jekyll-metadata)
for path in \
  _site \
  backup-models.json \
  backup-benchmarking.json \
  backup-references.json
do
  if [[ -e "$path" ]]; then
    paths+=("$path")
  else
    echo "Checkpoint skipping missing ${path}" >&2
  fi
done

rm -f "$ROOT/jekyll-content.zip"
7z a -tzip "$ROOT/jekyll-content.zip" "${paths[@]}"
if [[ ! -s "$ROOT/jekyll-content.zip" ]]; then
  echo "Checkpoint zip was not written: ${ROOT}/jekyll-content.zip" >&2
  exit 1
fi
echo "Checkpoint zip written: ${ROOT}/jekyll-content.zip"
