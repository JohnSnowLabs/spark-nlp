#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f jekyll-content.zip ]]; then
  printf 'No reusable Jekyll artifact; a full build is required.\n'
  printf 'restored=false\n' >> "$GITHUB_OUTPUT"
  exit 0
fi

7z x -o_site/ jekyll-content.zip
mv _site/.jekyll-metadata ./
mv _site/backup-models.json ./
mv _site/backup-benchmarking.json ./
mv _site/backup-references.json ./
rm jekyll-content.zip
printf 'restored=true\n' >> "$GITHUB_OUTPUT"
