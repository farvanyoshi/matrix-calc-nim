#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-debug}
flags=(--path:src --threads:on --mm:orc)
case "$mode" in
  debug) ;;
  release) flags+=(-d:release) ;;
  arc) flags+=(-d:release --mm:arc) ;;
  guards) flags+=(-d:release --overflowChecks:off --boundChecks:off --assertions:on) ;;
  *) echo "usage: $0 [debug|release|arc|guards]" >&2; exit 2 ;;
esac
mkdir -p .tools/bin
for test_file in tests/test_*.nim; do
  test_name=$(basename "$test_file" .nim)
  nim c -r "${flags[@]}" --nimcache:".tools/nimcache/$mode/$test_name" --out:".tools/bin/${test_name}_${mode}" "$test_file"
done
