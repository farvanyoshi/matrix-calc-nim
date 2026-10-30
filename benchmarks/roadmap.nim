## Benchy baselines for square dense float64 matrix multiplication.
## Shape requirements: all kernels receive compatible n×n matrices.
import std/[os, strutils, strformat, monotimes, times, math]
import benchy
import matrix_calc/[core, accelerated]

var benchmarkOutput: Matrix[float64]
var benchmarkScalar: float64

proc inputs(n: int): (Matrix[float64], Matrix[float64]) =
  ## Shape requirements: n > 0; returns two n×n row-major matrices.
  (fromFn[float64](n, n, proc(r, c: int): float64 = (((r*17+c*13+1) mod 31).float64-15.0)/16.0),
   fromFn[float64](n, n, proc(r, c: int): float64 = (((r*7+c*11+3) mod 29).float64-14.0)/16.0))

proc naive(a, b: Matrix[float64]): Matrix[float64] {.noinline.} =
  ## Shape requirements: A is M×N, B is N×P. Deliberately naive i-j-k baseline.
  if cols(a) != rows(b): raise newException(ValueError, "incompatible benchmark matrices")
  result = zeros[float64](rows(a), cols(b))
  for i in 0..<rows(a):
    for j in 0..<cols(b):
      var value = 0.0
      for k in 0..<cols(a): value += a[i,k]*b[k,j]
      result[i,j] = value

proc checksum(a: Matrix[float64]): float64 =
  ## Shape requirements: any matrix; weighted logical row-major checksum.
  var i = 0
  for r in 0..<rows(a):
    for c in 0..<cols(a):
      result += a[r,c] * ((i mod 97)+1).float64
      inc i

proc run(name: string, a, b: Matrix[float64]): Matrix[float64] =
  ## Shape requirements: compatible n×n benchmark inputs; includes allocation.
  case name
  of "naive": naive(a,b)
  of "blocked": matmulBlocked(a,b,32)
  of "parallel": matmulParallel(a,b,32,parseInt(getEnv("ROADMAP_THREADS","4")))
  of "simd": matmulSimd(a,b)
  else: raise newException(ValueError, "unknown ROADMAP_KERNEL: " & name)

proc maxError(a,b: Matrix[float64]): float64 =
  ## Shape requirements: equal-shaped outputs; largest absolute element difference.
  if shape(a) != shape(b): raise newException(ValueError,"different output shapes")
  for r in 0..<rows(a):
    for c in 0..<cols(a): result = max(result, abs(a[r,c]-b[r,c]))

proc elapsed(start: MonoTime): float64 =
  ## Shape requirements: none; seconds elapsed from a monotonic timestamp.
  (getMonoTime()-start).inNanoseconds.float64 / 1_000_000_000.0

proc main() =
  ## Shape requirements: requested sizes are positive; modes full/quick/verify/profile.
  let mode = getEnv("ROADMAP_MODE", "full")
  if mode notin ["full","quick","verify","profile"]: raise newException(ValueError,"invalid ROADMAP_MODE")
  let defaultSizes = if mode in ["quick","profile"]: "128" else: "128,512,1024"
  var sizes: seq[int]
  for text in getEnv("ROADMAP_SIZES",defaultSizes).split(','):
    let n = parseInt(text)
    if n<=0: raise newException(ValueError,"benchmark size must be positive")
    sizes.add(n)
  stderr.writeLine("Nim backend: " & simdBackend() & "; workers: " & getEnv("ROADMAP_THREADS","4"))
  echo "language,kernel,size,seconds,checksum,max_error"
  for n in sizes:
    let (a,b) = inputs(n)
    if mode == "profile":
      let kernel = getEnv("ROADMAP_KERNEL","naive")
      let count = parseInt(getEnv("ROADMAP_ITERATIONS","20"))
      if count<=0: raise newException(ValueError,"iterations must be positive")
      let start = getMonoTime()
      var sum = 0.0
      for _ in 0..<count: sum += checksum(run(kernel,a,b))
      echo &"nim,{kernel},{n},{elapsed(start)/count.float64:.9f},{sum:.12f},0"
      continue
    let start = getMonoTime()
    let expected = naive(a,b)
    echo &"nim,naive,{n},{elapsed(start):.9f},{checksum(expected):.12f},0"
    for name in ["blocked","parallel","simd"]:
      let start = getMonoTime()
      let actual = run(name,a,b)
      let seconds = elapsed(start)
      let error = maxError(expected,actual)
      if error>1e-10: raise newException(ValueError,&"{name}/{n}: error {error}")
      echo &"nim,{name},{n},{seconds:.9f},{checksum(actual):.12f},{error:.12f}"
    if mode=="verify": continue
    for name in ["naive","blocked","parallel","simd"]:
      var kernelName = name
      if mode=="quick":
        timeIt(name & "/" & $n, 3):
          benchmarkOutput = run(kernelName,a,b)
      else:
        timeIt(name & "/" & $n, 10):
          benchmarkOutput = run(kernelName,a,b)

  if mode != "profile":
    let length = 1_048_576
    var x = newSeq[float64](length)
    var y = newSeq[float64](length)
    for i in 0..<length:
      x[i] = ((i mod 127).float64-63.0)/64.0
      y[i] = ((i mod 113).float64-56.0)/64.0
    let expected = dotScalar(x,y)
    for name in ["dot_scalar","dot_simd"]:
      let start = getMonoTime()
      let actual = if name=="dot_scalar": dotScalar(x,y) else: dotSimd(x,y)
      let seconds = elapsed(start)
      let error = abs(actual-expected)
      if error>1e-10: raise newException(ValueError,"dot mismatch")
      echo &"nim,{name},{length},{seconds:.9f},{actual:.12f},{error:.12f}"
      if mode!="verify":
        var kernelName = name
        timeIt(kernelName,10):
          benchmarkScalar = if kernelName=="dot_scalar": dotScalar(x,y) else: dotSimd(x,y)
    if mode!="verify": stderr.writeLine("consumed benchmark results: " & $checksum(benchmarkOutput) & ", " & $benchmarkScalar)

when isMainModule: main()
