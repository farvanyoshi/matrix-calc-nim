#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools/docs
for source in src/matrix_calc.nim src/matrix_calc/*.nim; do
  nim doc --hints:off --threads:on --path:src --nimcache:.tools/nimcache/doc --outdir:.tools/docs "$source"
done
