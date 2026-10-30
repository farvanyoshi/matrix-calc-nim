import std/unittest
import ../src/matrix_calc/linalg
suite "real linear algebra":
  test "pivoted LU solve and inverse":
    let a = newMatrix(3, 3, [0.0, 2, 3, 4, 5, 6, 7, 8, 10])
    let l = a.lu
    check (l.permutationMatrix*a).approxEq(l.lower*l.upper, 1e-10, 1e-10)
    let b = newMatrix(3, 2, [1.0, 2, 3, 4, 5, 6])
    let x = l.solve(b)
    check (a*x).approxEq(b, 1e-10, 1e-10)
    check (a*l.inverse).approxEq(identity[float64](3), 1e-10, 1e-10)
    check abs(a.determinant+5.0) < 1e-10
    expect NumericalError: discard newMatrix(2, 2, [1.0, 2, 2, 4]).solve(ones[float64](2, 1))
    check zeros[float64](0, 0).determinant == 1.0
  test "Cholesky":
    let a = newMatrix(2, 2, [4.0'f32, 2, 2, 3])
    let c = a.cholesky
    check (c.lower*c.lower.transpose).approxEq(a, 1e-5'f32, 1e-5'f32)
    let b = newMatrix(2, 1, [1.0'f32, 3])
    check (a*c.solve(b)).approxEq(b, 1e-5'f32, 1e-5'f32)
    expect NumericalError: discard newMatrix(2, 2, [1.0, 2, 2, 1]).cholesky
  test "Householder QR and rectangular least squares":
    let a = newMatrix(3, 2, [1.0, 2, 3, 4, 5, 7])
    for q in [a.qr, a.pivotedQr]:
      check (q.q*q.r).approxEq(a*q.permutationMatrix, 1e-10, 1e-10)
      check (q.q.transpose*q.q).approxEq(identity[float64](2), 1e-10, 1e-10)
      let b = newMatrix(3, 1, [1.0, 2, 2])
      check q.applyQ(q.applyQTranspose(b)).approxEq(b, 1e-10, 1e-10)
    let x = newMatrix(3, 2, [1.0, 0, 1, 1, 1, 2]).pivotedQr.solve(newMatrix(3, 1, [1.0, 2, 2]))
    check x.approxEq(newMatrix(2, 1, [7.0/6, 0.5]), 1e-10, 1e-10)
  test "SVD reconstruction and Moore-Penrose identities":
    for a in [newMatrix(3, 2, [1.0, 2, 3, 4, 5, 6]), newMatrix(2, 3, [1.0, 2, 3, 4, 5, 6]),
        newMatrix(2, 2, [1.0, 2, 2, 4]), zeros[float64](3, 2), zeros[float64](0, 3)]:
      let s = a.svd
      check (s.u*fromDiagonal(s.singularValues)*s.vT).approxEq(a, 1e-10, 1e-10)
      let count = min(a.rows, a.cols)
      check (s.u.transpose*s.u).approxEq(identity[float64](count), 1e-10, 1e-10)
      check (s.vT*s.vT.transpose).approxEq(identity[float64](count), 1e-10, 1e-10)
      let p = s.pseudoinverse
      check (a*p*a).approxEq(a, 1e-9, 1e-9)
      check (p*a*p).approxEq(p, 1e-9, 1e-9)
      check (a*p).approxEq((a*p).transpose, 1e-9, 1e-9)
      check (p*a).approxEq((p*a).transpose, 1e-9, 1e-9)
  test "minimum norm, scales and eigenpairs":
    let a = newMatrix(1, 2, [1.0, 1])
    let fit = a.leastSquares(newMatrix(1, 1, [2.0]))
    check fit.solution.approxEq(newMatrix(2, 1, [1.0, 1]), 1e-10, 1e-10)
    for scale in [1e-200, 1e200]:
      let a = newMatrix(2, 2, [1.0, 2, 2, 4]).scalarMul(scale)
      check a.leastSquares(newMatrix(2, 1, [1.0, 2]).scalarMul(scale)).solution.approxEq(newMatrix(
          2, 1, [0.2, 0.4]), 1e-9, 1e-9)
    for a in [newMatrix(2, 2, [2.0, 1, 1, 2]), identity[float64](3), zeros[float64](0, 0)]:
      let e = a.symmetricEigen
      check (a*e.eigenvectors).approxEq(e.eigenvectors*fromDiagonal(e.eigenvalues), 1e-10, 1e-10)
      check (e.eigenvectors.transpose*e.eigenvectors).approxEq(identity[float64](a.rows), 1e-10, 1e-10)
    expect ConvergenceError: discard newMatrix(3, 3, [1.0, 2, 3, 4, 5, 6, 7, 8, 10]).svd(maxSweeps = 1)

import std/random
proc verifyGenerated[T: SomeFloat]() =
  ## Shape: deterministic finite matrices through 6×6, including deficient columns.
  var rng = initRand(73421)
  for sample in 0..<64:
    let m = rng.rand(1..6); let n = rng.rand(1..6)
    var values = newSeq[T](m*n)
    for value in values.mitems: value = T(rng.rand(-10..10))
    var a = newMatrix(m, n, values)
    if sample mod 3 == 0 and n > 1:
      for row in 0..<m: a[row, n-1] = a[row, 0]*T(2)
    let tol = T(128)*machineEpsilon[T]()*T(max(m, n))
    let s = a.svd
    let reconstructed = s.u*fromDiagonal(s.singularValues)*s.vT
    check (reconstructed-a).normFrobenius <= tol*(T(1)+a.normFrobenius)
    check (s.u.transpose*s.u).approxEq(identity[T](min(m, n)), tol, tol)
    let q = a.pivotedQr
    check (q.q*q.r-a*q.permutationMatrix).normFrobenius <= tol*(T(1)+a.normFrobenius)
    if m == n:
      let symmetric = a+a.transpose
      let e = symmetric.symmetricEigen
      let residual = symmetric*e.eigenvectors-e.eigenvectors*fromDiagonal(e.eigenvalues)
      check residual.normFrobenius <= tol*(T(1)+symmetric.normFrobenius)

suite "seeded matrix factorization verification":
  test "64 float64 rectangular and deficient cases": verifyGenerated[float64]()
  test "64 float32 rectangular and deficient cases": verifyGenerated[float32]()

suite "option guards and giant empty factorizations":
  test "invalid tolerance cannot make false comparisons pass":
    expect NumericalError: discard identity[float64](1).approxEq(identity[float64](1),Tolerance[float64](absolute:NaN,relative:0))
    expect NumericalError: discard defaultTolerance[float64](-1)
  test "zero-width tall SVD and comparison terminate":
    let a=zeros[float64](high(int),0)
    check a.approxEq(a)
    check a.svd.u.shape==(high(int),0)
