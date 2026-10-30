#!/usr/bin/env bash
# Shape requirements: compatible positive square benchmark inputs; sample actual main-thread stacks.
set -euo pipefail
cd "$(dirname "$0")/.."
kernel=${1:-naive}; size=${2:-512}; iterations=${3:-12}; output=${4:-.tools/profiles/sampled}
mkdir -p "$output"
nim c -d:release --threads:on --debugger:native --passC:-fno-omit-frame-pointer --path:src --path:"${BENCHY_PATH:-.tools/benchy/src}" --nimcache:.tools/profile-cache --out:.tools/profile-bench benchmarks/roadmap.nim > "$output/build.log" 2>&1
ROADMAP_MODE=profile ROADMAP_KERNEL="$kernel" ROADMAP_SIZES="$size" ROADMAP_ITERATIONS="$iterations" \
 python3 tools/sample_stacks.py --gdb "${GDB:-gdb}" --output "$output" -- .tools/profile-bench
"${FLAMEGRAPH:-flamegraph.pl}" --title='Nim main-thread stack samples' --subtitle='GDB/MI interrupt samples, not hardware counters' --countname=samples "$output/stacks.folded" > "$output/flamegraph.svg"
