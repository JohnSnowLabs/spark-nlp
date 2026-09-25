#!/usr/bin/env bash
set -euo pipefail

# This script lives in docs/_scripts and is invoked from the repo root.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ ! -f docs/.jekyll-metadata ]]; then
  echo "No Jekyll metadata to checkpoint" >&2
  exit 1
fi

paths=(./docs/.jekyll-metadata)
for path in \
  ./docs/_site \
  ./docs/backup-models.json \
  ./docs/backup-benchmarking.json \
  ./docs/backup-references.json
do
  if [[ -e "$path" ]]; then
    paths+=("$path")
  else
    echo "Checkpoint skipping missing ${path}" >&2
  fi
done

rm -f jekyll-content.zip
7z a -tzip jekyll-content.zip "${paths[@]}"
if [[ ! -s jekyll-content.zip ]]; then
  echo "Checkpoint zip was not written: ${ROOT}/jekyll-content.zip" >&2
  exit 1
fi
echo "Checkpoint zip written: ${ROOT}/jekyll-content.zip"
