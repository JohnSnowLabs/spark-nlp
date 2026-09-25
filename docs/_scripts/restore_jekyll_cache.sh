#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f jekyll-content.zip ]]; then
  printf 'No reusable Jekyll artifact; a full build is required.\n'
  printf 'restored=false\n' >> "$GITHUB_OUTPUT"
  exit 0
fi

7z x -o_site/ jekyll-content.zip
if [[ -f _site/.jekyll-metadata ]]; then
  mv _site/.jekyll-metadata ./
fi
for name in backup-models.json backup-benchmarking.json backup-references.json; do
  if [[ -f "_site/${name}" ]]; then
    mv "_site/${name}" ./
  fi
done
rm jekyll-content.zip
if [[ ! -f .jekyll-metadata ]]; then
  printf 'Jekyll artifact did not contain incremental metadata.\n' >&2
  exit 1
fi
printf 'restored=true\n' >> "$GITHUB_OUTPUT"
