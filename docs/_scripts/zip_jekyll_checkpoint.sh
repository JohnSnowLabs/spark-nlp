#!/usr/bin/env bash
set -euo pipefail

# This script lives in docs/_scripts and is invoked from the repo root.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ ! -f docs/.jekyll-metadata ]]; then
  echo "No Jekyll metadata to checkpoint" >&2
  exit 1
fi

rm -f jekyll-content.zip
7z a -tzip jekyll-content.zip \
  ./docs/_site \
  ./docs/.jekyll-metadata \
  ./docs/backup-models.json \
  ./docs/backup-benchmarking.json \
  ./docs/backup-references.json
if [[ ! -s jekyll-content.zip ]]; then
  echo "Checkpoint zip was not written: ${ROOT}/jekyll-content.zip" >&2
  exit 1
fi
echo "Checkpoint zip written: ${ROOT}/jekyll-content.zip"
