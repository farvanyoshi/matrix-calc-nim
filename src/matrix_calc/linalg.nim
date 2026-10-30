## Stable real-valued numerical algorithms. Shapes are checked explicitly.
import std/[math, algorithm]
import ./[core, errors]
export core, errors

type
  Tolerance*[T] = object
    ## Finite nonnegative absolute and relative numerical thresholds.
    absolute*, relative*: T
  Lu*[T] = object
    packed: Matrix[T]
    permutationData: seq[int]
    scale, cutoff: T
    odd: bool
  Cholesky*[T] = object
    factor: Matrix[T]
    scale: T
  Qr*[T] = object
    packed: Matrix[T]
    reflectors: seq[seq[T]]
    permutationData: seq[int]
    scale, cutoff: T
  Svd*[T] = object
    left, rightT: Matrix[T]
    values, normalizedValues: seq[T]
    scale, cutoff: T
    sweepCount: int
  SymmetricEigen*[T] = object
    values: seq[T]
    vectors: Matrix[T]
    sweepCount: int
  LeastSquaresMethod* = enum
    lsAuto, lsQr, lsSvd
  LeastSquares*[T] = object
    solution*: Matrix[T]
    residualNorm*: T
    rank*: int
    algorithm*: LeastSquaresMethod

proc machineEpsilon*[T: SomeFloat](): T =
  ## Shape: scalar precision constant used by all matrix algorithms.
  when T is float32: T(1.1920928955078125e-7)
  else: T(2.220446049250313e-16)
proc finiteScalar[T: SomeFloat](x: T): bool = classify(x) notin {fcNan, fcInf, fcNegInf}
proc tolerance*[T: SomeFloat](absolute, relative: T): Tolerance[T] =
  ## Shape: scalar tolerances must be finite and nonnegative.
  if not finiteScalar(absolute) or not finiteScalar(relative) or absolute < 0 or relative < 0:
    raise newException(NumericalError, "invalid numerical tolerance")
  Tolerance[T](absolute: absolute, relative: relative)
proc defaultTolerance*[T: SomeFloat](dimension: int): Tolerance[T] =
  ## Shape: nonnegative matrix dimension; relative tolerance is 8*n*epsilon.
  if dimension < 0: raise newException(NumericalError, "dimension must be nonnegative")
  tolerance(T(0), T(8)*T(max(1, dimension))*machineEpsilon[T]())
proc allFinite*[T: SomeFloat](a: Matrix[T]): bool =
  ## Shape: checks all M*N entries; empty matrices are finite.
  for x in a:
    if not finiteScalar(x): return false
  true
proc requireFinite[T: SomeFloat](a: Matrix[T]) =
  if not a.allFinite: raise newException(NumericalError, "nonfinite matrix input or result")
proc maxAbs*[T: SomeFloat](a: Matrix[T]): T =
  ## Shape: maximum absolute entry of M×N, or zero when empty.
  result = T(0)
  for x in a: result = max(result, abs(x))
proc normFrobenius*[T: SomeFloat](a: Matrix[T]): T =
  ## Shape: stable Euclidean norm of all M*N entries.
  result = T(0)
  for x in a: result = T(hypot(result, x))
proc approxEq*[T: SomeFloat](a, b: Matrix[T]; tol: Tolerance[T]): bool =
  ## Shape: requires equal M×N; nonfinite entries never compare approximately equal.
  discard tolerance(tol.absolute, tol.relative)
  if a.shape != b.shape: return false
  if a.isEmpty: return true
  for r in 0..<a.rows:
    for c in 0..<a.cols:
      let x = a[r, c]; let y = b[r, c]
      if not finiteScalar(x) or not finiteScalar(y): return false
      let scale = max(abs(x), abs(y))
      if scale != 0 and abs(x/scale-y/scale) > tol.absolute/scale+tol.relative: return false
  true
proc approxEq*[T: SomeFloat](a, b: Matrix[T]; atol: T = T(0); rtol: T = T(-1)): bool =
  ## Shape: equal M×N; defaults adapt to scalar precision and matrix dimensions.
  let rel = if rtol < 0: defaultTolerance[T](max(a.rows, a.cols)).relative else: rtol
  a.approxEq(b, tolerance(atol, rel))
proc normalized[T: SomeFloat](a: Matrix[T]): tuple[m: Matrix[T]; scale: T] =
  a.requireFinite
  result.scale = if a.maxAbs == 0: T(1) else: a.maxAbs
  let scale = result.scale
  result.m = a.map(proc(x: T): T = x/scale)
proc square[T](a: Matrix[T]): int =
  if not a.isSquare: raise newException(DimensionError, "factorization requires square matrix")
  a.rows
proc requireRhs[T: SomeFloat](rows: int; b: Matrix[T]) =
  b.requireFinite
  if b.rows != rows: raise newException(DimensionError, "RHS row count mismatch")
proc makeLu*[T: SomeFloat](a: Matrix[T]; tol: Tolerance[T]): Lu[T] =
  ## Shape: A must be finite N×N; factors satisfy P*A=L*U with row pivoting.
  discard tolerance(tol.absolute, tol.relative)
  let n = a.square
  let norm = normalized(a)
  result.packed = norm.m; result.scale = norm.scale
  result.cutoff = tol.absolute/norm.scale+tol.relative
  result.permutationData = newSeq[int](n)
  for i in 0..<n: result.permutationData[i] = i
  for k in 0..<n:
    var pivot = k
    for i in k+1..<n:
      if abs(result.packed[i, k]) > abs(result.packed[pivot, k]): pivot = i
    if pivot != k:
      result.packed.swapRows(k, pivot)
      swap(result.permutationData[k], result.permutationData[pivot])
      result.odd = not result.odd
    let d = result.packed[k, k]
    if d == 0: continue
    for i in k+1..<n:
      result.packed[i, k] = result.packed[i, k]/d
      for j in k+1..<n: result.packed[i, j] = result.packed[i, j]-result.packed[i, k]*result.packed[k, j]
  result.packed.requireFinite
  for r in 0..<n:
    for c in r..<n:
      if not finiteScalar(result.packed[r, c]*result.scale): raise newException(NumericalError, "unrepresentable LU factor")
proc lu*[T: SomeFloat](a: Matrix[T]): Lu[T] =
  ## Shape: finite N×N; uses dimension-aware default singularity tolerance.
  makeLu(a, defaultTolerance[T](a.rows))
proc permutation*[T](lu: Lu[T]): seq[int] =
  ## Shape: N row indices; permutation[i] identifies the original factor row.
  result = lu.permutationData
proc permutationMatrix*[T: SomeFloat](lu: Lu[T]): Matrix[T] =
  ## Shape: returns N×N P where P*A=L*U.
  fromFn(lu.packed.rows, lu.packed.rows, proc(r, c: int): T = (if lu.permutationData[r] == c: T(
      1) else: T(0)))
proc lower*[T: SomeFloat](lu: Lu[T]): Matrix[T] =
  ## Shape: returns N×N unit lower triangular L.
  fromFn(lu.packed.rows, lu.packed.rows, proc(r, c: int): T = (if r == c: T(1) elif r >
      c: lu.packed[r, c] else: T(0)))
proc upper*[T: SomeFloat](lu: Lu[T]): Matrix[T] =
  ## Shape: returns N×N upper U in original input units.
  fromFn(lu.packed.rows, lu.packed.rows, proc(r, c: int): T = (if r <= c: lu.packed[r,
      c]*lu.scale else: T(0)))
proc isSingular*[T: SomeFloat](lu: Lu[T]): bool =
  ## Shape: tests the N diagonal pivots against the selected numerical threshold.
  for i in 0..<lu.packed.rows:
    if abs(lu.packed[i, i]) <= lu.cutoff: return true
  false
proc solve*[T: SomeFloat](lu: Lu[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: retained A=N×N, B=N×K; returns N×K, rejecting numerical singularity.
  let n = lu.packed.rows
  requireRhs(n, b)
  if lu.isSingular: raise newException(NumericalError, "singular matrix")
  result = fromFn(n, b.cols, proc(r, c: int): T = b[lu.permutationData[r], c]/lu.scale)
  for i in 0..<n:
    for c in 0..<b.cols:
      var value = result[i, c]
      for k in 0..<i: value-=lu.packed[i, k]*result[k, c]
      result[i, c] = value
  for i in countdown(n-1, 0):
    for c in 0..<b.cols:
      var value = result[i, c]
      for k in i+1..<n: value-=lu.packed[i, k]*result[k, c]
      result[i, c] = value/lu.packed[i, i]
  result.requireFinite
proc solve*[T: SomeFloat](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A=N×N and B=N×K; solve with partial-row-pivoted LU.
  a.lu.solve(b)
proc inverse*[T: SomeFloat](lu: Lu[T]): Matrix[T] =
  ## Shape: invert retained nonsingular N×N A via N RHS columns.
  lu.solve(identity[T](lu.packed.rows))
proc inverse*[T: SomeFloat](a: Matrix[T]): Matrix[T] =
  ## Shape: finite nonsingular N×N; solves against identity instead of Gauss-Jordan.
  a.lu.inverse
proc signLogDeterminant*[T: SomeFloat](lu: Lu[T]): tuple[sign, logAbs: T] =
  ## Shape: determinant information for retained N×N; exact zero gives (0,-Inf).
  result.sign = if lu.odd: T(-1) else: T(1)
  result.logAbs = T(lu.packed.rows)*ln(lu.scale)
  for i in 0..<lu.packed.rows:
    let x = lu.packed[i, i]
    if x == 0: return(T(0), T(NegInf))
    if x < 0: result.sign = -result.sign
    result.logAbs+=ln(abs(x))
proc determinant*[T: SomeFloat](lu: Lu[T]): T =
  ## Shape: determinant of N×N; magnitude may overflow/underflow, use signLogDeterminant.
  let value = lu.signLogDeterminant
  if value.sign == 0: T(0) else: value.sign*exp(value.logAbs)
proc determinant*[T: SomeFloat](a: Matrix[T]): T =
  ## Shape: finite N×N; 0×0 determinant is one.
  a.lu.determinant
proc isSymmetric*[T: SomeFloat](a: Matrix[T]; tol: Tolerance[T]): bool =
  ## Shape: checks finite N×N against its transpose at an entrywise tolerance.
  a.isSquare and a.approxEq(a.transpose, tol)
proc cholesky*[T: SomeFloat](a: Matrix[T]): Cholesky[T] =
  ## Shape: finite symmetric positive-definite N×N; returns A=L*L^T.
  let n = a.square
  let tol = defaultTolerance[T](n)
  if not a.isSymmetric(tol): raise newException(NumericalError, "Cholesky requires symmetric input")
  let norm = normalized(a)
  result.factor = norm.m; result.scale = norm.scale
  let cutoff = tol.absolute/norm.scale+tol.relative
  for i in 0..<n:
    for j in 0..i:
      var value = result.factor[i, j]
      for k in 0..<j: value-=result.factor[i, k]*result.factor[j, k]
      if i == j:
        if value <= cutoff: raise newException(NumericalError, "matrix is not positive definite")
        result.factor[i, j] = sqrt(value)
      else: result.factor[i, j] = value/result.factor[j, j]
    for j in i+1..<n: result.factor[i, j] = T(0)
proc lower*[T: SomeFloat](c: Cholesky[T]): Matrix[T] =
  ## Shape: N×N lower factor in original input units.
  c.factor.scalarMul(sqrt(c.scale))
proc solve*[T: SomeFloat](c: Cholesky[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: A=N×N, B=N×K; solve using retained lower factor twice.
  let n = c.factor.rows
  requireRhs(n, b)
  result = b.scalarDiv(c.scale)
  for i in 0..<n:
    for col in 0..<b.cols:
      var value = result[i, col]
      for k in 0..<i: value-=c.factor[i, k]*result[k, col]
      result[i, col] = value/c.factor[i, i]
  for i in countdown(n-1, 0):
    for col in 0..<b.cols:
      var value = result[i, col]
      for k in i+1..<n: value-=c.factor[k, i]*result[k, col]
      result[i, col] = value/c.factor[i, i]
  result.requireFinite
proc logDeterminant*[T: SomeFloat](c: Cholesky[T]): T =
  ## Shape: logarithmic determinant of retained positive-definite N×N.
  result = T(c.factor.rows)*ln(c.scale)
  for i in 0..<c.factor.rows: result+=T(2)*ln(c.factor[i, i])
proc scaledRatio[T: SomeFloat](x, numerator, denominator, other: T): T =
  if x == 0 or numerator == 0: return T(0)
  let xx = frexp(x); let nn = frexp(numerator); let dd = frexp(denominator); let oo = frexp(other)
  result = (xx.frac*nn.frac)/(dd.frac*oo.frac)
  var exponent = xx.exp+nn.exp-dd.exp-oo.exp
  while exponent > 0:
    let step = min(exponent, 32)
    result*=T(pow(2.0, float64(step)))
    exponent-=step
  while exponent < 0:
    let step = min(-exponent, 32)
    result*=T(pow(2.0, float64(-step)))
    exponent+=step
proc makeQr*[T: SomeFloat](a: Matrix[T]; pivot: bool; tol: Tolerance[T]): Qr[T] =
  ## Shape: finite A=M×N; compact reflectors give A*P=Q*R with optional column pivoting.
  discard tolerance(tol.absolute, tol.relative)
  let norm = normalized(a)
  result.packed = norm.m; result.scale = norm.scale
  result.cutoff = tol.absolute/norm.scale+tol.relative
  result.permutationData = newSeq[int](a.cols)
  for i in 0..<a.cols: result.permutationData[i] = i
  for k in 0..<min(a.rows, a.cols):
    if pivot:
      var best = k
      var maxNorm = T(0)
      for c in k..<a.cols:
        var cn = T(0)
        for r in k..<a.rows: cn = T(hypot(cn, result.packed[r, c]))
        if cn > maxNorm: maxNorm = cn; best = c
      result.packed.swapCols(k, best)
      swap(result.permutationData[k], result.permutationData[best])
    var v = newSeq[T](a.rows-k)
    var vn = T(0)
    for r in k..<a.rows: v[r-k] = result.packed[r, k]; vn = T(hypot(vn, v[r-k]))
    if vn == 0:
      result.reflectors.add(newSeq[T]())
      continue
    let alpha = if v[0] >= 0: -vn else: vn
    v[0]-=alpha
    vn = T(0)
    for x in v: vn = T(hypot(vn, x))
    for x in v.mitems: x/=vn
    for c in k..<a.cols:
      var dot = T(0)
      for r in k..<a.rows: dot+=v[r-k]*result.packed[r, c]
      dot*=T(2)
      for r in k..<a.rows: result.packed[r, c] = result.packed[r, c]-dot*v[r-k]
    result.packed[k, k] = alpha
    for r in k+1..<a.rows: result.packed[r, k] = T(0)
    result.reflectors.add(v)
  result.packed.requireFinite
  for x in result.packed:
    if not finiteScalar(x*result.scale): raise newException(NumericalError, "unrepresentable QR factor")
proc qr*[T: SomeFloat](a: Matrix[T]): Qr[T] =
  ## Shape: finite M×N input; returns unpivoted Householder QR.
  makeQr(a, false, defaultTolerance[T](max(a.rows, a.cols)))
proc pivotedQr*[T: SomeFloat](a: Matrix[T]): Qr[T] =
  ## Shape: finite M×N input; returns column-pivoted Householder QR.
  makeQr(a, true, defaultTolerance[T](max(a.rows, a.cols)))
proc permutation*[T](qr: Qr[T]): seq[int] =
  ## Shape: N original column indices in factor order.
  qr.permutationData
proc permutationMatrix*[T: SomeFloat](qr: Qr[T]): Matrix[T] =
  ## Shape: returns N×N P with A*P=Q*R.
  fromFn(qr.packed.cols, qr.packed.cols, proc(r, c: int): T = (if qr.permutationData[c] == r: T(
      1) else: T(0)))
proc rank*[T: SomeFloat](qr: Qr[T]): int =
  ## Shape: counts min(M,N) significant R pivots; use pivoted QR for a useful estimate.
  for i in 0..<min(qr.packed.rows, qr.packed.cols):
    if abs(qr.packed[i, i]) > qr.cutoff: inc result
proc applyQImpl[T: SomeFloat](qr: Qr[T]; b: Matrix[T]; transposed: bool): Matrix[T] =
  requireRhs(qr.packed.rows, b)
  let norm = normalized(b)
  result = norm.m
  for step in 0..<qr.reflectors.len:
    let k = if transposed: step else: qr.reflectors.len-1-step
    let v = qr.reflectors[k]
    for c in 0..<b.cols:
      var dot = T(0)
      for i, x in v: dot+=x*result[k+i, c]
      dot*=T(2)
      for i, x in v: result[k+i, c] = result[k+i, c]-dot*x
  result = result.scalarMul(norm.scale)
  result.requireFinite
proc applyQ*[T: SomeFloat](qr: Qr[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: Q is M×M, B=M×K; returns Q*B without forming Q.
  applyQImpl(qr, b, false)
proc applyQTranspose*[T: SomeFloat](qr: Qr[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: Q is M×M, B=M×K; returns Q^T*B without forming Q.
  applyQImpl(qr, b, true)
proc q*[T: SomeFloat](qr: Qr[T]): Matrix[T] =
  ## Shape: returns economy M×min(M,N) Q.
  qr.applyQ(fromFn(qr.packed.rows, min(qr.packed.rows, qr.packed.cols), proc(r, c: int): T = (
      if r == c: T(1) else: T(0))))
proc qFull*[T: SomeFloat](qr: Qr[T]): Matrix[T] =
  ## Shape: returns full M×M Q.
  qr.applyQ(identity[T](qr.packed.rows))
proc r*[T: SomeFloat](qr: Qr[T]): Matrix[T] =
  ## Shape: returns economy min(M,N)×N upper factor R in original units.
  fromFn(min(qr.packed.rows, qr.packed.cols), qr.packed.cols, proc(r, c: int): T = qr.packed[r, c]*qr.scale)
proc rFull*[T: SomeFloat](qr: Qr[T]): Matrix[T] =
  ## Shape: returns M×N zero-padded R.
  qr.packed.scalarMul(qr.scale)
proc solve*[T: SomeFloat](qr: Qr[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: A=M×N with M>=N and full column rank, B=M×K; returns N×K least squares.
  requireRhs(qr.packed.rows, b)
  let n = qr.packed.cols
  if qr.packed.rows < n or qr.rank < n: raise newException(NumericalError, "QR least squares requires full column rank and M>=N")
  let norm = normalized(b)
  let y = qr.applyQTranspose(norm.m)
  var xp = y.slice(0, n, 0, b.cols)
  for i in countdown(n-1, 0):
    for c in 0..<b.cols:
      var value = xp[i, c]
      for j in i+1..<n: value-=qr.packed[i, j]*xp[j, c]
      xp[i, c] = value/qr.packed[i, i]
  result = zeros[T](n, b.cols)
  for i in 0..<n:
    for c in 0..<b.cols: result[qr.permutationData[i], c] = scaledRatio(xp[i, c], norm.scale,
        qr.scale, T(1))
  result.requireFinite
proc compensated[T: SomeFloat](terms: openArray[T]): T =
  var correction = T(0)
  for x in terms:
    let next = result+x
    if abs(result) >= abs(x): correction+=(result-next)+x
    else: correction+=(x-next)+result
    result = next
  result+=correction
proc svd*[T: SomeFloat](a: Matrix[T]; maxSweeps: int = 100; relativeTolerance: T = T(-1);
    absoluteTolerance: T = T(0)): Svd[T] =
  ## Shape: finite M×N; economy A=U*diag(s)*V^T with min(M,N) descending values.
  ## Uses scaled one-sided Jacobi directly on A, never a Gram matrix.
  if maxSweeps <= 0: raise newException(NumericalError, "SVD sweep limit must be positive")
  let rt = if relativeTolerance < 0: defaultTolerance[T](max(a.rows,
      a.cols)).relative else: relativeTolerance
  let tol = tolerance(absoluteTolerance, rt)
  if a.rows < a.cols:
    let transposed = svd(a.transpose, maxSweeps, relativeTolerance, absoluteTolerance)
    result = transposed
    result.left = transposed.rightT.transpose
    result.rightT = transposed.left.transpose
    return
  let norm = normalized(a)
  var b = norm.m
  let m = a.rows; let n = a.cols
  var v = identity[T](n)
  let convergence = T(4)*T(max(m, 1))*machineEpsilon[T]()
  var converged = n < 2
  for sweep in 0..<maxSweeps:
    if converged: break
    var changed = false
    for p in 0..<n:
      for q in p+1..<n:
        var alpha, beta: T
        for i in 0..<m: alpha = T(hypot(alpha, b[i, p])); beta = T(hypot(beta, b[i, q]))
        if alpha == 0 or beta == 0: continue
        var terms = newSeq[T](m)
        for i in 0..<m: terms[i] = (b[i, p]/alpha)*(b[i, q]/beta)
        let correlation = compensated(terms)
        if abs(correlation) <= convergence: continue
        let scale = max(alpha, beta); let aa = alpha/scale; let bb = beta/scale
        let delta = (bb-aa)*(bb+aa)
        let cross = T(2)*correlation*aa*bb
        let denominator = delta+(if delta >= 0: T(hypot(delta, cross)) else: -T(hypot(delta, cross)))
        if denominator == 0: continue
        let t = cross/denominator
        if t == 0: continue
        let cosine = T(1)/T(hypot(T(1), t)); let sine = cosine*t
        for i in 0..<m:
          let x = b[i, p]; let y = b[i, q]
          b[i, p] = cosine*x-sine*y; b[i, q] = sine*x+cosine*y
        for i in 0..<n:
          let x = v[i, p]; let y = v[i, q]
          v[i, p] = cosine*x-sine*y; v[i, q] = sine*x+cosine*y
        changed = true
    result.sweepCount = sweep+1
    converged = not changed
  if not converged: raise newException(ConvergenceError, "SVD did not converge within sweep limit")
  var norms = newSeq[T](n)
  for c in 0..<n:
    for r in 0..<m: norms[c] = T(hypot(norms[c], b[r, c]))
  var order = newSeq[int](n)
  for i in 0..<n: order[i] = i
  order.sort(proc(i, j: int): int = cmp(norms[j], norms[i]))
  result.scale = norm.scale
  result.values = newSeq[T](n); result.normalizedValues = newSeq[T](n)
  result.left = zeros[T](m, n)
  for c in 0..<n:
    let cn = norms[order[c]]
    result.normalizedValues[c] = cn; result.values[c] = cn*norm.scale
    if not finiteScalar(result.values[c]): raise newException(NumericalError, "unrepresentable singular value")
    if cn != 0:
      for r in 0..<m: result.left[r, c] = b[r, order[c]]/cn
    else:
      var best = newSeq[T](m)
      var bestNorm = T(0)
      for basis in 0..<m:
        var candidate = newSeq[T](m); candidate[basis] = T(1)
        for pass in 0..<2:
          for j in 0..<c:
            var dot = T(0)
            for r in 0..<m: dot+=candidate[r]*result.left[r, j]
            for r in 0..<m: candidate[r]-=dot*result.left[r, j]
        var candidateNorm = T(0)
        for x in candidate: candidateNorm = T(hypot(candidateNorm, x))
        if candidateNorm > bestNorm: bestNorm = candidateNorm; best = candidate
      if bestNorm == 0: raise newException(ConvergenceError, "SVD null basis completion failed")
      for r in 0..<m: result.left[r, c] = best[r]/bestNorm
  result.rightT = fromFn(n, n, proc(r, c: int): T = v[c, order[r]])
  let largest = if n == 0: T(0) else: result.normalizedValues[0]
  result.cutoff = tol.absolute/norm.scale+tol.relative*largest
proc u*[T](s: Svd[T]): Matrix[T] =
  ## Shape: economy M×min(M,N) left singular vectors as columns.
  s.left.clone
proc vT*[T](s: Svd[T]): Matrix[T] =
  ## Shape: economy min(M,N)×N transposed right singular vectors as rows.
  s.rightT.clone
proc singularValues*[T](s: Svd[T]): seq[T] =
  ## Shape: min(M,N) descending nonnegative values.
  s.values
proc sweeps*[T](s: Svd[T]): int =
  ## Shape: number of complete Jacobi pair sweeps used.
  s.sweepCount
proc rank*[T: SomeFloat](s: Svd[T]): int =
  ## Shape: numerical rank in [0,min(M,N)] at the requested threshold.
  for x in s.normalizedValues:
    if x > s.cutoff: inc result
proc spectralNorm*[T: SomeFloat](s: Svd[T]): T =
  ## Shape: largest singular value, or zero for empty M×N.
  if s.values.len == 0: T(0) else: s.values[0]
proc conditionNumber*[T: SomeFloat](s: Svd[T]): T =
  ## Shape: 2-norm condition number; empty spectrum is one, deficiency is infinity.
  if s.values.len == 0: T(1)
  elif s.rank < s.values.len: T(Inf)
  else: s.normalizedValues[0]/s.normalizedValues[^1]
proc pseudoinverse*[T: SomeFloat](s: Svd[T]): Matrix[T] =
  ## Shape: A=M×N gives N×M tolerance-truncated Moore-Penrose inverse.
  result = zeros[T](s.rightT.cols, s.left.rows)
  for r in 0..<result.rows:
    for c in 0..<result.cols:
      var value = T(0)
      for k in 0..<s.values.len:
        if s.normalizedValues[k] > s.cutoff: value+=scaledRatio(s.rightT[k, r]*s.left[c, k], T(1),
            s.scale, s.normalizedValues[k])
      result[r, c] = value
  result.requireFinite
proc solve*[T: SomeFloat](s: Svd[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: A=M×N, B=M×K; returns minimum-norm N×K least-squares solution.
  requireRhs(s.left.rows, b)
  let norm = normalized(b)
  let count = s.values.len
  var projected = zeros[T](count, b.cols)
  for k in 0..<count:
    if s.normalizedValues[k] <= s.cutoff: continue
    for c in 0..<b.cols:
      var terms = newSeq[T](b.rows)
      for r in 0..<b.rows: terms[r] = s.left[r, k]*norm.m[r, c]
      projected[k, c] = scaledRatio(compensated(terms), norm.scale, s.scale, s.normalizedValues[k])
  result = s.rightT.transpose*projected
  result.requireFinite
proc pseudoinverse*[T: SomeFloat](a: Matrix[T]): Matrix[T] =
  ## Shape: any finite M×N; returns N×M pseudoinverse via SVD.
  a.svd.pseudoinverse
proc rank*[T: SomeFloat](a: Matrix[T]): int =
  ## Shape: any finite M×N; reports SVD numerical rank.
  a.svd.rank
proc norm2*[T: SomeFloat](a: Matrix[T]): T =
  ## Shape: any finite M×N; matrix spectral norm, distinct from Frobenius norm.
  a.svd.spectralNorm
proc conditionNumber*[T: SomeFloat](a: Matrix[T]): T =
  ## Shape: any finite M×N; SVD 2-norm condition number.
  a.svd.conditionNumber
proc symmetricEigen*[T: SomeFloat](input: Matrix[T]; maxSweeps: int = 100; relativeTolerance: T = T(
    -1); absoluteTolerance: T = T(0)): SymmetricEigen[T] =
  ## Shape: finite symmetric N×N; ascending values satisfy A*V=V*diag(values).
  let n = input.square
  if maxSweeps <= 0: raise newException(NumericalError, "eigen sweep limit must be positive")
  let rt = if relativeTolerance < 0: defaultTolerance[T](n).relative else: relativeTolerance
  let tol = tolerance(absoluteTolerance, rt)
  if not input.isSymmetric(tol): raise newException(NumericalError, "eigen requires symmetric input")
  let norm = normalized(input)
  var a = norm.m
  var vectors = identity[T](n)
  for r in 0..<n:
    for c in 0..<r:
      let x = a[r, c]/T(2)+a[c, r]/T(2)
      a[r, c] = x; a[c, r] = x
  let convergence = max(tol.absolute/norm.scale+tol.relative, machineEpsilon[T]())
  var converged = n < 2
  for sweep in 0..<maxSweeps:
    if converged: break
    var changed = false
    for p in 0..<n:
      for q in p+1..<n:
        let x = a[p, q]
        if abs(x) <= convergence: continue
        let delta = (a[q, q]-a[p, p])/T(2)
        let denominator = delta+(if delta >= 0: T(hypot(delta, x)) else: -T(hypot(delta, x)))
        let t = if denominator == 0: T(1) else: x/denominator
        let cosine = T(1)/T(hypot(T(1), t)); let sine = cosine*t
        let app = a[p, p]; let aqq = a[q, q]
        for k in 0..<n:
          if k == p or k == q: continue
          let kp = a[k, p]; let kq = a[k, q]
          a[k, p] = cosine*kp-sine*kq; a[p, k] = a[k, p]
          a[k, q] = sine*kp+cosine*kq; a[q, k] = a[k, q]
        a[p, p] = app-t*x; a[q, q] = aqq+t*x; a[p, q] = T(0); a[q, p] = T(0)
        for k in 0..<n:
          let kp = vectors[k, p]; let kq = vectors[k, q]
          vectors[k, p] = cosine*kp-sine*kq; vectors[k, q] = sine*kp+cosine*kq
        changed = true
    result.sweepCount = sweep+1
    converged = not changed
  if not converged: raise newException(ConvergenceError, "eigen iteration exhausted sweeps")
  var order = newSeq[int](n)
  for i in 0..<n: order[i] = i
  order.sort(proc(i, j: int): int = cmp(a[i, i], a[j, j]))
  result.values = newSeq[T](n)
  for i in 0..<n:
    result.values[i] = a[order[i], order[i]]*norm.scale
    if not finiteScalar(result.values[i]): raise newException(NumericalError, "unrepresentable eigenvalue")
  result.vectors = fromFn(n, n, proc(r, c: int): T = vectors[r, order[c]])
proc eigenvalues*[T](e: SymmetricEigen[T]): seq[T] =
  ## Shape: N ascending real eigenvalues.
  e.values
proc eigenvectors*[T](e: SymmetricEigen[T]): Matrix[T] =
  ## Shape: N×N orthonormal eigenvectors stored as columns.
  e.vectors.clone
proc sweeps*[T](e: SymmetricEigen[T]): int =
  ## Shape: number of complete Jacobi sweeps used.
  e.sweepCount
proc leastSquares*[T: SomeFloat](a, b: Matrix[T]; algorithm: LeastSquaresMethod = lsAuto): LeastSquares[T] =
  ## Shape: A=M×N, B=M×K; QR for full-rank tall input, otherwise minimum-norm SVD.
  requireRhs(a.rows, b)
  if algorithm == lsSvd or a.rows < a.cols:
    if algorithm == lsQr: raise newException(NumericalError, "QR cannot solve wide input")
    let s = a.svd
    result.solution = s.solve(b); result.rank = s.rank; result.algorithm = lsSvd
  else:
    let q = a.pivotedQr
    if q.rank == a.cols:
      result.solution = q.solve(b); result.rank = q.rank; result.algorithm = lsQr
    else:
      if algorithm == lsQr: raise newException(NumericalError, "QR input is rank deficient")
      let s = a.svd
      result.solution = s.solve(b); result.rank = s.rank; result.algorithm = lsSvd
  result.residualNorm = (a*result.solution-b).normFrobenius

proc solveTranspose*[T: SomeFloat](lu: Lu[T]; b: Matrix[T]): Matrix[T] =
  ## Shape: retained A=N×N, B=N×K; solves A^T*X=B without refactorization.
  let n = lu.packed.rows
  requireRhs(n, b)
  if lu.isSingular: raise newException(NumericalError, "singular transposed solve")
  var z = b.scalarDiv(lu.scale)
  for i in 0..<n:
    for c in 0..<b.cols:
      var value = z[i, c]
      for k in 0..<i: value-=lu.packed[k, i]*z[k, c]
      z[i, c] = value/lu.packed[i, i]
  for i in countdown(n-1, 0):
    for c in 0..<b.cols:
      var value = z[i, c]
      for k in i+1..<n: value-=lu.packed[k, i]*z[k, c]
      z[i, c] = value
  result = zeros[T](n, b.cols)
  for r in 0..<n:
    for c in 0..<b.cols: result[lu.permutationData[r], c] = z[r, c]
  result.requireFinite
proc solveTriangular*[T: SomeFloat](a, b: Matrix[T]; lower: bool;
    unitDiagonal: bool = false): Matrix[T] =
  ## Shape: A=N×N, B=N×K; only the selected triangle is used, optional unit diagonal.
  let n = a.square
  a.requireFinite; requireRhs(n, b)
  let cutoff = defaultTolerance[T](n).relative*a.maxAbs
  if not unitDiagonal:
    for i in 0..<n:
      if abs(a[i, i]) <= cutoff: raise newException(NumericalError, "singular triangular pivot")
  result = b.clone
  for step in 0..<n:
    let i = if lower: step else: n-1-step
    for c in 0..<b.cols:
      var value = result[i, c]
      if lower:
        for k in 0..<i: value-=a[i, k]*result[k, c]
      else:
        for k in i+1..<n: value-=a[i, k]*result[k, c]
      result[i, c] = if unitDiagonal: value else: value/a[i, i]
  result.requireFinite
proc solveLower*[T: SomeFloat](a, b: Matrix[T]; unitDiagonal: bool = false): Matrix[T] =
  ## Shape: lower triangular A=N×N, B=N×K; upper entries are ignored.
  solveTriangular(a, b, true, unitDiagonal)
proc solveUpper*[T: SomeFloat](a, b: Matrix[T]; unitDiagonal: bool = false): Matrix[T] =
  ## Shape: upper triangular A=N×N, B=N×K; lower entries are ignored.
  solveTriangular(a, b, false, unitDiagonal)
proc powi*[T: SomeFloat](a: Matrix[T]; exponent: int): Matrix[T] =
  ## Shape: square N×N; negative powers use the checked inverse path.
  if exponent == low(int): raise newException(MatrixArithmeticError, "power exponent magnitude overflows int")
  if exponent < 0: a.inverse.pow(-exponent) else: a.pow(exponent)
