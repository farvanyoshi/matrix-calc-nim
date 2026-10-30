version       = "0.3.0"
author        = "farvanyoshi"
description   = "Dense matrices, vectors, strided tensors, numerical linear algebra and an interactive REPL."
license       = "MIT OR Apache-2.0"
srcDir        = "src"
installExt    = @["nim", "c"]
bin           = @["matrix_calc_cli"]
requires "nim >= 2.0.8"

task test, "Run the debug library and REPL suites":
  exec "bash tools/test.sh debug"
task testRelease, "Run release numerical and integration suites":
  exec "bash tools/test.sh release"
task testArc, "Run release suites with ARC":
  exec "bash tools/test.sh arc"
task testGuards, "Verify explicit guards without compiler bounds/overflow checks":
  exec "bash tools/test.sh guards"
task docs, "Build public library documentation":
  exec "bash tools/docs.sh"

task benchmark, "Run the pinned benchy roadmap measurements":
  exec "bash tools/setup_benchy.sh"
  exec "nim c -r -d:release --threads:on --path:src --path:.tools/benchy/src --nimcache:.tools/nimcache/bench --out:.tools/roadmap-bench benchmarks/roadmap.nim"
