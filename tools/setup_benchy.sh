#!/usr/bin/env bash
# Benchmark-only dependency, pinned and checked without modifying Nimble's global package state.
set -euo pipefail
cd "$(dirname "$0")/.."
revision=6425417b578dc54008285aad020625b4b4f61f7b
if [[ ! -d .tools/benchy/.git ]]; then
  mkdir -p .tools/benchy
  git -C .tools/benchy init -q
  git -C .tools/benchy remote add origin https://github.com/treeform/benchy.git
  git -C .tools/benchy fetch --depth 1 origin "$revision"
  git -C .tools/benchy checkout --detach FETCH_HEAD
fi
[[ $(git -C .tools/benchy rev-parse HEAD) == "$revision" ]] || { echo 'Unexpected benchy revision; preserve local changes and use the documented pin.' >&2;exit 2; }
