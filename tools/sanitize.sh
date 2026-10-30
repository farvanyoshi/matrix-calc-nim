#!/usr/bin/env bash
# Shape requirements: validated by the accelerated-kernel suite.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools/bin
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
for mode in goto setjmp; do
  flags=(--exceptions:"$mode")
  if [[ "$mode" == goto ]]; then
    flags+=(--opt:none)
  else
    flags+=(-d:release)
  fi
  nim c -r --threads:on --mm:orc --path:src --hints:off "${flags[@]}" \
    --passC:'-fsanitize=address,undefined -fno-omit-frame-pointer' \
    --passL:'-fsanitize=address,undefined' \
    --nimcache:".tools/nimcache/sanitize-$mode" \
    --out:".tools/bin/sanitize-$mode" tests/test_accelerated.nim
done
