#!/usr/bin/env bash
# Shape requirements: the consumer exercises 2×2 matrices, length-3 vectors, and rank-2 tensors.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools/package/source .tools/package/consumer
cp -R src matrix_calc.nimble LICENSE-MIT LICENSE-APACHE README.md .tools/package/source/
tar -czf .tools/matrix_calc-0.3.0.tar.gz -C .tools/package/source .
mkdir -p .tools/package/extracted
tar -xzf .tools/matrix_calc-0.3.0.tar.gz -C .tools/package/extracted
cat > .tools/package/consumer/main.nim <<'NIM'
import matrix_calc
let a = newMatrix(2,2,[1.0,2,3,4])
doAssert (a*identity[float64](2)).toSeq == a.toSeq
doAssert newVector([1,0,0]).cross(newVector([0,1,0])).toSeq == @[0,0,1]
doAssert (a.toTensor + newTensor([10.0,20],[2])).toSeq == @[11.0,22,13,24]
import matrix_calc/accelerated
doAssert matmulSimd(a,identity[float64](2)) == a
NIM
cd .tools/package/consumer
nim c -r --threads:on --path:../extracted/src --nimcache:cache --out:consumer main.nim
