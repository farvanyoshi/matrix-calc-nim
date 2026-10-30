#!/usr/bin/env bash
# Shape requirements: positive square size and repeat count; choose a matmul kernel.
# Usage: tools/profile.sh [naive|blocked|parallel|simd] [size=128] [iterations=2] [output]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
kernel=${1:-naive};size=${2:-128};iterations=${3:-2};output=${4:-.tools/profiles/$kernel}
mkdir -p "$output"
benchy_path=${BENCHY_PATH:-.tools/benchy/src}
nim c -d:release --threads:on --debugger:native --passC:-fno-omit-frame-pointer --path:src --path:"$benchy_path" --nimcache:.tools/profile-cache --out:.tools/profile-bench benchmarks/roadmap.nim > "$output/build.log" 2>&1
profiler=${VALGRIND:-$(command -v valgrind || true)}
extra=()
# In this multi-repository workspace tooling may be reused without changing either checkout.
local_tools=${PROFILING_TOOLS:-$root/../matrix-calc-rust/.tools/profiling/root}
if [[ -z "$profiler" && -x "$local_tools/usr/bin/valgrind" ]];then
  profiler="$local_tools/usr/bin/valgrind"
  export VALGRIND_LIB="$local_tools/usr/libexec/valgrind"
  extra+=("--extra-debuginfo-path=$local_tools/usr/lib/debug")
fi
[[ -n "$profiler" ]] || { echo 'Install valgrind (and matching libc debug symbols), or set VALGRIND.' >&2;exit 2; }
ROADMAP_MODE=profile ROADMAP_KERNEL="$kernel" ROADMAP_SIZES="$size" ROADMAP_ITERATIONS="$iterations" \
 "$profiler" "${extra[@]}" --tool=callgrind --cache-sim=yes --callgrind-out-file="$output/callgrind.out" .tools/profile-bench > "$output/program.csv" 2> "$output/callgrind.log"
python3 tools/callgrind_report.py "$output/callgrind.out" "$output" | tee "$output/report.json"
