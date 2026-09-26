#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f jekyll-content.zip ]]; then
  printf 'No reusable Jekyll artifact; a full build is required.\n'
  printf 'restored=false\n' >> "$GITHUB_OUTPUT"
  exit 0
fi

staging="$(mktemp -d)"
7z x -o"$staging" jekyll-content.zip
if [[ -d "$staging/_site" ]]; then
  rm -rf _site
  mv "$staging/_site" ./
fi
if [[ -f "$staging/.jekyll-metadata" ]]; then
  mv "$staging/.jekyll-metadata" ./
fi
for name in backup-models.json backup-benchmarking.json backup-references.json; do
  if [[ -f "$staging/$name" ]]; then
    mv "$staging/$name" ./
  fi
done
rm -rf "$staging" jekyll-content.zip
if [[ ! -f .jekyll-metadata ]]; then
  printf 'Jekyll artifact did not contain incremental metadata.\n' >&2
  exit 1
fi
printf 'restored=true\n' >> "$GITHUB_OUTPUT"
