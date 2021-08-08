#!/bin/bash

echo "Move _static"
grep -RiIl '_static' _build | xargs sed -i 's/_static/static/g'
mv _build/html/_static _build/html/static

echo "Move _modules"
grep -RiIl '_modules' _build | xargs sed -i 's/_modules/modules/g'
mv _build/html/_modules _build/html/modules

echo "Move references/_autosummary"
grep -RiIl '_autosummary' _build | xargs sed -i 's/_autosummary/autosummary/g'
mv _build/html/reference/_autosummary _build/html/reference/autosummary

rm -rf _build/html/_sources