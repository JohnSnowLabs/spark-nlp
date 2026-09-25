#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -f docs/.jekyll-metadata ]]; then
  echo "No Jekyll metadata to checkpoint"
  exit 0
fi

rm -f jekyll-content.zip
7z a -tzip jekyll-content.zip \
  ./docs/_site \
  ./docs/.jekyll-metadata \
  ./docs/backup-models.json \
  ./docs/backup-benchmarking.json \
  ./docs/backup-references.json
echo "Checkpoint zip written: ${ROOT}/jekyll-content.zip"
