import std/[unittest,math]
import ../src/matrix_calc/accelerated
proc close(a,b:Matrix[float64]):bool =
  if a.shape!=b.shape:return false
  for r in 0..<a.rows:
    for c in 0..<a.cols:
      if abs(a[r,c]-b[r,c])>1e-10*(1.0+abs(b[r,c])):return false
  true
suite "blocked parallel and SIMD kernels":
  test "tile tails and empty products":
    for dims in [(0,0,0),(3,0,5),(0,4,7),(6,4,0),(1,1,1),(5,7,9),(33,31,35)]:
      let (m,k,n)=dims
      let a=fromFn(m,k,proc(r,c:int):float64=sin(float64(r*13+c*7)))
      let b=fromFn(k,n,proc(r,c:int):float64=cos(float64(r*3+c*11)))
      let expected=a*b
      for blockSize in [1,3,16,32,128,high(int)]:
        check close(matmulBlocked(a,b,blockSize),expected)
        check close(matmulParallel(a,b,blockSize),expected)
      check close(matmulSimd(a,b),expected)
  test "explicit input guards":
    let a=ones[float64](2,3);let b=ones[float64](3,2);let wrong=ones[float64](4,2)
    expect DimensionError:discard matmulBlocked(a,wrong)
    expect DimensionError:discard matmulParallel(a,wrong)
    expect DimensionError:discard matmulSimd(a,wrong)
    expect MatrixArithmeticError:discard matmulBlocked(a,b,0)
    expect MatrixArithmeticError:discard matmulParallel(a,b,0)
  test "SIMD dot tails and backend":
    check simdBackend() in ["avx","neon","scalar"]
    for n in 0..129:
      var a=newSeq[float64](n);var b=newSeq[float64](n)
      for i in 0..<n:a[i]=sin(float64(i));b[i]=cos(float64(i))
      let expected=dotScalar(a,b)
      check abs(dotSimd(a,b)-expected)<1e-12*(1.0+abs(expected))
    expect DimensionError:discard dotSimd([1.0],newSeq[float64]())
    check dotSimd(newSeq[float64](),newSeq[float64]())==0.0

when defined(matrixCalcScalar):
  suite "forced portable dispatch":
    test "the fallback is selected on any host":
      check simdBackend() == "scalar"

suite "task worker configuration":
  test "isolated tasks match across worker partitions":
    let a=fromFn[float64](5,7,proc(r,c:int):float64=float64(r-c))
    let b=fromFn[float64](7,9,proc(r,c:int):float64=float64(r+c))
    for workers in [1,2,4,8]:
      check matmulParallel(a,b,3,workers)==a*b
    expect MatrixArithmeticError:discard matmulParallel(a,b,3,-1)
    expect MatrixArithmeticError:discard matmulParallel(a,b,3,65)
