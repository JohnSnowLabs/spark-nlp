#!/bin/bash
# this script is compatible with macOS (the sed command)

echo "Move _static"
grep -RiIl '_static' _build | xargs sed -i '' -e 's/_static/static/g'
cp -r _build/html/_static _build/html/static

echo "Move _modules"
grep -RiIl '_modules' _build | xargs sed -i '' -e 's/_modules/modules/g'
cp -r _build/html/_modules _build/html/modules

echo "Move references/_autosummary"
grep -RiIl '_autosummary' _build | xargs sed -i '' -e 's/_autosummary/autosummary/g'
cp -r _build/html/reference/_autosummary _build/html/reference/autosummary

rm -rf _build/html/_sources