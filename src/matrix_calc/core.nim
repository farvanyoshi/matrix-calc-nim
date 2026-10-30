## Generic owned dense row-major matrices. All shape guards remain active in release.
import std/[options]
import ./errors
export errors

type Matrix*[T] = object
  ## Owned M×N matrix. No public mutable sequence header can invalidate its shape.
  rowCount, colCount: int
  elements: seq[T]

proc checkedCapacity[T](rows, cols: int): int =
  ## Shape: nonnegative dimensions with representable element and byte counts.
  result = checkedSize(rows, cols)
  if result > high(int) div max(sizeof(T), 1):
    raise newException(ReshapeError, "matrix byte capacity overflows int")

proc newMatrix*[T](rows, cols: int; input: openArray[T]): Matrix[T] =
  ## Shape: input.len must equal rows*cols; copies elements into owned storage.
  let count = checkedCapacity[T](rows, cols)
  if input.len != count: raise newException(ReshapeError, "matrix element count differs from shape")
  result = Matrix[T](rowCount: rows, colCount: cols, elements: newSeq[T](count))
  for i, value in input: result.elements[i] = value

proc newMatrix*[T](input: openArray[T]; rows, cols: int): Matrix[T] =
  ## Shape: input.len = rows*cols; alternate constructor order for Rust interoperation.
  newMatrix(rows, cols, input)

proc fromFn*[T](rows, cols: int; f: proc(row, col: int): T {.closure.}): Matrix[T] =
  ## Shape: returns rows×cols, calling f once per coordinate in row order.
  let count = checkedCapacity[T](rows, cols)
  result = Matrix[T](rowCount: rows, colCount: cols, elements: newSeq[T](count))
  if count == 0: return
  for r in 0..<rows:
    for c in 0..<cols: result.elements[r*cols+c] = f(r, c)

proc filled*[T](rows, cols: int; value: T): Matrix[T] =
  ## Shape: returns rows×cols copies of value.
  fromFn(rows, cols, proc(r, c: int): T = value)
proc zeros*[T](rows, cols: int): Matrix[T] =
  ## Shape: returns rows×cols additive zeros for numeric T.
  filled(rows, cols, T(0))
proc ones*[T](rows, cols: int): Matrix[T] =
  ## Shape: returns rows×cols multiplicative ones for numeric T.
  filled(rows, cols, T(1))
proc identity*[T](side: int): Matrix[T] =
  ## Shape: returns side×side identity, including 0×0.
  fromFn(side, side, proc(r, c: int): T = (if r == c: T(1) else: T(0)))
proc fromRows*[T](input: openArray[seq[T]]): Matrix[T] =
  ## Shape: all nested rows must have the same length; empty input is 0×0.
  let cols = if input.len == 0: 0 else: input[0].len
  for row in input:
    if row.len != cols: raise newException(ReshapeError, "ragged matrix input")
  result = Matrix[T](rowCount: input.len, colCount: cols, elements: newSeq[T](checkedCapacity[T](input.len, cols)))
  for r in 0..<input.len:
    for c in 0..<cols: result.elements[r*cols+c] = input[r][c]
proc rows*[T](m: Matrix[T]): int {.inline.} =
  ## Shape: number M of logical rows, including empty rows.
  m.rowCount
proc cols*[T](m: Matrix[T]): int {.inline.} =
  ## Shape: number N of logical columns, including empty columns.
  m.colCount
proc shape*[T](m: Matrix[T]): tuple[rows, cols: int] {.inline.} =
  ## Shape: returns the logical (M,N) dimensions.
  (m.rowCount, m.colCount)
proc len*[T](m: Matrix[T]): int {.inline.} =
  ## Shape: returns M*N stored elements.
  m.elements.len
proc isEmpty*[T](m: Matrix[T]): bool =
  ## Shape: true if M*N is zero, independently of either axis.
  m.len == 0
proc isSquare*[T](m: Matrix[T]): bool =
  ## Shape: true when M=N.
  m.rows == m.cols
proc toSeq*[T](m: Matrix[T]): seq[T] =
  ## Shape: returns a copy of M*N row-major elements.
  result = newSeq[T](m.len)
  for i, x in m.elements: result[i] = x
proc data*[T](m: Matrix[T]): seq[T] =
  ## Shape: compatibility accessor returning an independent M*N sequence.
  m.toSeq
proc clone*[T](m: Matrix[T]): Matrix[T] =
  ## Shape: returns an independent M×N buffer; reference elements retain T semantics.
  newMatrix(m.rows, m.cols, m.elements)
proc checkCoordinate[T](m: Matrix[T]; row, col: int) {.inline.} =
  if row < 0 or col < 0 or row >= m.rows or col >= m.cols:
    raise newException(MatrixIndexError, "matrix coordinate outside logical shape")
proc `[]`*[T](m: Matrix[T]; row, col: int): T {.inline.} =
  ## Shape: 0<=row<M and 0<=col<N, checked independently before flattening.
  m.checkCoordinate(row, col)
  m.elements[row*m.cols+col]
proc `[]=`*[T](m: var Matrix[T]; row, col: int; value: T) {.inline.} =
  ## Shape: requires valid logical coordinates; writes exactly one element.
  m.checkCoordinate(row, col)
  m.elements[row*m.cols+col] = value
proc get*[T](m: Matrix[T]; row, col: int): T =
  ## Shape: checked value accessor; requires 0<=row<M and 0<=col<N.
  m[row, col]
proc tryGet*[T](m: Matrix[T]; row, col: int): Option[T] =
  ## Shape: returns none for coordinates outside M×N.
  if row < 0 or col < 0 or row >= m.rows or col >= m.cols: none(T) else: some(m[row, col])
proc getMut*[T](m: var Matrix[T]; row, col: int): T =
  ## Shape: compatibility value accessor; use []= or set for actual mutation.
  m[row, col]
proc set*[T](m: var Matrix[T]; row, col: int; value: T) =
  ## Shape: checked element replacement in M×N.
  m[row, col] = value
iterator items*[T](m: Matrix[T]): T =
  ## Shape: traverses M*N values in row-major order.
  for x in m.elements: yield x
proc row*[T](m: Matrix[T]; index: int): seq[T] =
  ## Shape: index must be in [0,M); returns N copied elements.
  if index < 0 or index >= m.rows: raise newException(MatrixIndexError, "row outside matrix")
  result = newSeq[T](m.cols)
  for c in 0..<m.cols: result[c] = m[index, c]
proc column*[T](m: Matrix[T]; index: int): seq[T] =
  ## Shape: index must be in [0,N); returns M copied elements.
  if index < 0 or index >= m.cols: raise newException(MatrixIndexError, "column outside matrix")
  result = newSeq[T](m.rows)
  for r in 0..<m.rows: result[r] = m[r, index]
proc map*[T, U](m: Matrix[T]; f: proc(x: T): U {.closure.}): Matrix[U] =
  ## Shape: preserves M×N, including empty axes; callback exceptions propagate.
  result = Matrix[U](rowCount: m.rows, colCount: m.cols, elements: newSeq[U](m.len))
  for i, x in m.elements: result.elements[i] = f(x)
proc sameShape*[T, U](a: Matrix[T]; b: Matrix[U]) =
  ## Shape: requires identical M×N operand dimensions.
  if a.shape != b.shape: raise newException(DimensionError, "matrix shapes must match")
proc zipMap*[T, U, V](a: Matrix[T]; b: Matrix[U]; f: proc(x: T; y: U): V {.closure.}): Matrix[V] =
  ## Shape: A and B must have exactly equal dimensions, not merely equal lengths.
  sameShape(a, b)
  result = Matrix[V](rowCount: a.rows, colCount: a.cols, elements: newSeq[V](a.len))
  for i in 0..<a.len: result.elements[i] = f(a.elements[i], b.elements[i])
proc `+`*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A and B must both be M×N; returns M×N.
  zipMap(a, b, proc(x, y: T): T = x+y)
proc `-`*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A and B must both be M×N; returns M×N.
  zipMap(a, b, proc(x, y: T): T = x-y)
proc `-`*[T](a: Matrix[T]): Matrix[T] =
  ## Shape: elementwise negation preserves M×N.
  a.map(proc(x: T): T = -x)
proc hadamard*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A and B must both be M×N; returns elementwise products.
  zipMap(a, b, proc(x, y: T): T = x*y)
proc scalarMul*[T](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: scale M×N without changing dimensions.
  a.map(proc(y: T): T = y*x)
proc scalarAdd*[T](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: add x to every element of M×N.
  a.map(proc(y: T): T = y+x)
proc scalarSub*[T](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: subtract x from every element of M×N.
  a.map(proc(y: T): T = y-x)
proc scalarDiv*[T: SomeNumber](a: Matrix[T]; x: T): auto =
  ## Shape: divides M×N by nonzero x; integer operands promote through Nim /.
  if x == T(0): raise newException(MatrixArithmeticError, "division by zero")
  a.map(proc(y: T): auto = y/x)
proc elementDiv*[T: SomeNumber](a, b: Matrix[T]): auto =
  ## Shape: equal M×N matrices; all divisors must be nonzero. Integers promote.
  sameShape(a, b)
  for x in b:
    if x == T(0): raise newException(MatrixArithmeticError, "division by zero")
  zipMap(a, b, proc(x, y: T): auto = x/y)
proc scalarQuotient*[T: SomeInteger](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: M×N integer quotient truncated toward zero; zero and MIN/-1 rejected.
  if x == 0: raise newException(MatrixArithmeticError, "division by zero")
  a.map(proc(y: T): T =
    when T is SomeSignedInt:
      if y == low(T) and x == T(-1): raise newException(MatrixArithmeticError, "integer quotient overflow")
    y div x)
proc matmul*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A is M×K, B is K×N; returns M×N, including K=0.
  if a.cols != b.rows: raise newException(DimensionError, "multiplication inner dimensions differ")
  result = zeros[T](a.rows, b.cols)
  if result.len == 0 or a.cols == 0: return
  for r in 0..<a.rows:
    for k in 0..<a.cols:
      let x = a.elements[r*a.cols+k]
      for c in 0..<b.cols: result.elements[r*b.cols+c]+=x*b.elements[k*b.cols+c]
proc `*`*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: matrix multiplication requires A=M×K and B=K×N.
  matmul(a, b)
proc `*`*[T: SomeNumber](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: scalar scaling preserves M×N.
  scalarMul(a, x)
proc `*`*[T: SomeNumber](x: T; a: Matrix[T]): Matrix[T] =
  ## Shape: scalar scaling preserves M×N.
  scalarMul(a, x)
proc `/`*[T: SomeNumber](a: Matrix[T]; x: T): auto =
  ## Shape: nonzero scalar division of M×N, with Nim numeric promotion.
  scalarDiv(a, x)
proc transpose*[T](a: Matrix[T]): Matrix[T] =
  ## Shape: M×N becomes N×M in owned row-major storage.
  result = fromFn(a.cols, a.rows, proc(r, c: int): T = a[c, r])
proc reshape*[T](a: Matrix[T]; rows, cols: int): Matrix[T] =
  ## Shape: rows*cols must equal M*N; preserves flattened element order.
  newMatrix(rows, cols, a.elements)
proc slice*[T](a: Matrix[T]; rowStart, rowEnd, colStart, colEnd: int): Matrix[T] =
  ## Shape: half-open ranges must lie within M×N; returns their owned rectangle.
  if rowStart < 0 or colStart < 0 or rowStart > rowEnd or colStart > colEnd or rowEnd > a.rows or
      colEnd > a.cols:
    raise newException(MatrixIndexError, "invalid matrix slice")
  fromFn(rowEnd-rowStart, colEnd-colStart, proc(r, c: int): T = a[r+rowStart, c+colStart])
proc slice*[T](a: Matrix[T]; rr, cc: HSlice[int, int]): Matrix[T] =
  ## Shape: Nim inclusive slices rr and cc must lie within M×N.
  if rr.b == high(int) or cc.b == high(int): raise newException(MatrixIndexError, "slice endpoint overflow")
  a.slice(rr.a, rr.b+1, cc.a, cc.b+1)
proc sum*[T](a: Matrix[T]): T =
  ## Shape: reduces all M*N elements; empty sum is zero.
  result = T(0)
  for x in a: result+=x
proc product*[T](a: Matrix[T]): T =
  ## Shape: reduces all M*N elements; empty product is one.
  result = T(1)
  for x in a: result*=x
proc trace*[T](a: Matrix[T]): T =
  ## Shape: requires square M×M; 0×0 trace is zero.
  if not a.isSquare: raise newException(DimensionError, "trace requires square input")
  result = T(0)
  for i in 0..<a.rows: result+=a[i, i]
proc diagonal*[T](a: Matrix[T]): seq[T] =
  ## Shape: returns min(M,N) main-diagonal elements.
  result = newSeq[T](min(a.rows, a.cols))
  for i in 0..<result.len: result[i] = a[i, i]
proc fromDiagonal*[T](values: openArray[T]): Matrix[T] =
  ## Shape: length N produces an N×N diagonal matrix.
  result = zeros[T](values.len, values.len)
  for i, x in values: result[i, i] = x
proc swapRows*[T](a: var Matrix[T]; i, j: int) =
  ## Shape: both row indices must be valid in M×N.
  if i < 0 or j < 0 or i >= a.rows or j >= a.rows: raise newException(MatrixIndexError, "invalid row")
  for c in 0..<a.cols: swap(a.elements[i*a.cols+c], a.elements[j*a.cols+c])
proc swapCols*[T](a: var Matrix[T]; i, j: int) =
  ## Shape: both column indices must be valid in M×N.
  if i < 0 or j < 0 or i >= a.cols or j >= a.cols: raise newException(MatrixIndexError, "invalid column")
  for r in 0..<a.rows: swap(a.elements[r*a.cols+i], a.elements[r*a.cols+j])
proc hcat*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: equal row counts; concatenate columns, checking sum overflow.
  if a.rows != b.rows: raise newException(DimensionError, "hcat row mismatch")
  if b.cols > high(int)-a.cols: raise newException(ReshapeError, "column count overflow")
  fromFn(a.rows, a.cols+b.cols, proc(r, c: int): T = (if c < a.cols: a[r, c] else: b[r, c-a.cols]))
proc vcat*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: equal column counts; concatenate rows, checking sum overflow.
  if a.cols != b.cols: raise newException(DimensionError, "vcat column mismatch")
  if b.rows > high(int)-a.rows: raise newException(ReshapeError, "row count overflow")
  fromFn(a.rows+b.rows, a.cols, proc(r, c: int): T = (if r < a.rows: a[r, c] else: b[r-a.rows, c]))
proc pow*[T](a: Matrix[T]; exponent: int): Matrix[T] =
  ## Shape: square M×M and nonnegative exponent; exponent zero gives identity.
  if not a.isSquare or exponent < 0: raise newException(DimensionError, "power requires square input and nonnegative exponent")
  result = identity[T](a.rows)
  var base = a.clone
  var e = exponent
  while e > 0:
    if (e and 1) == 1: result = result*base
    e = e shr 1
    if e > 0: base = base*base
proc add*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: equal M×N; compatibility addition alias.
  a+b
proc sub*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: equal M×N; compatibility subtraction alias.
  a-b
proc mul*[T](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A=M×K, B=K×N; compatibility multiplication alias.
  a*b
proc neg*[T](a: Matrix[T]): Matrix[T] =
  ## Shape: preserves M×N; compatibility negation alias.
  -a
proc `$`*[T](a: Matrix[T]): string =
  ## Shape: pretty-print M×N with explicit shape and row separators.
  result = $a.rows & "x" & $a.cols & " ["
  for r in 0..<a.rows:
    if r > 0: result.add("; ")
    for c in 0..<a.cols:
      if c > 0: result.add(", ")
      result.add($a[r, c])
  result.add("]")

proc matmulInto*[T](a, b: Matrix[T]; output: var Matrix[T]) =
  ## Shape: A=M×K, B=K×N, output=M×N; validates before replacing the destination.
  ## Computes into a temporary so input/output aliasing cannot corrupt the result.
  if output.rows != a.rows or output.cols != b.cols:
    raise newException(DimensionError, "output shape differs from product shape")
  let product = a.matmul(b)
  output = product
proc rowSums*[T](a: Matrix[T]): Matrix[T] =
  ## Shape: M×N reduces each row to M×1; an empty row sums to zero.
  result = zeros[T](a.rows, 1)
  for r in 0..<a.rows:
    for c in 0..<a.cols: result[r, 0] = result[r, 0]+a[r, c]
proc colSums*[T](a: Matrix[T]): Matrix[T] =
  ## Shape: M×N reduces each column to 1×N; an empty column sums to zero.
  result = zeros[T](1, a.cols)
  if result.len == 0: return
  for r in 0..<a.rows:
    for c in 0..<a.cols: result[0, c] = result[0, c]+a[r, c]
proc setRow*[T](a: var Matrix[T]; row: int; values: openArray[T]) =
  ## Shape: row is in [0,M), values.len=N; validation occurs before mutation.
  if row < 0 or row >= a.rows: raise newException(MatrixIndexError, "invalid row")
  if values.len != a.cols: raise newException(DimensionError, "replacement row length mismatch")
  for c, x in values: a[row, c] = x
proc setCol*[T](a: var Matrix[T]; col: int; values: openArray[T]) =
  ## Shape: column is in [0,N), values.len=M; validation occurs before mutation.
  if col < 0 or col >= a.cols: raise newException(MatrixIndexError, "invalid column")
  if values.len != a.rows: raise newException(DimensionError, "replacement column length mismatch")
  for r, x in values: a[r, col] = x
proc checkedAddScalar[T: SomeInteger](a, b: T): T =
  when T is SomeUnsignedInt:
    if b > high(T)-a: raise newException(MatrixArithmeticError, "integer addition overflow")
  else:
    if (b > 0 and a > high(T)-b) or (b < 0 and a < low(T)-b):
      raise newException(MatrixArithmeticError, "integer addition overflow")
  a+b
proc checkedSubScalar[T: SomeInteger](a, b: T): T =
  when T is SomeUnsignedInt:
    if b > a: raise newException(MatrixArithmeticError, "unsigned subtraction underflow")
  else:
    if (b < 0 and a > high(T)+b) or (b > 0 and a < low(T)+b):
      raise newException(MatrixArithmeticError, "integer subtraction overflow")
  a-b
proc checkedMulScalar[T: SomeInteger](a, b: T): T =
  if a == 0 or b == 0: return T(0)
  when T is SomeUnsignedInt:
    if a > high(T) div b: raise newException(MatrixArithmeticError, "integer multiplication overflow")
  else:
    if (a > 0 and b > 0 and a > high(T) div b) or
       (a > 0 and b < 0 and b < low(T) div a) or
       (a < 0 and b > 0 and a < low(T) div b) or
       (a < 0 and b < 0 and a < high(T) div b):
      raise newException(MatrixArithmeticError, "integer multiplication overflow")
  a*b
proc checkedAdd*[T: SomeInteger](a, b: Matrix[T]): Matrix[T] =
  ## Shape: identical M×N integer operands; overflow is explicitly checked in every build.
  zipMap(a, b, proc(x, y: T): T = checkedAddScalar(x, y))
proc checkedSub*[T: SomeInteger](a, b: Matrix[T]): Matrix[T] =
  ## Shape: identical M×N integer operands; subtraction overflow/underflow raises.
  zipMap(a, b, proc(x, y: T): T = checkedSubScalar(x, y))
proc checkedHadamard*[T: SomeInteger](a, b: Matrix[T]): Matrix[T] =
  ## Shape: identical M×N integer operands; every element product is checked.
  zipMap(a, b, proc(x, y: T): T = checkedMulScalar(x, y))
proc checkedScalarAdd*[T: SomeInteger](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: preserve M×N with explicit scalar addition overflow checks.
  a.map(proc(y: T): T = checkedAddScalar(y, x))
proc checkedScalarSub*[T: SomeInteger](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: preserve M×N with explicit scalar subtraction overflow checks.
  a.map(proc(y: T): T = checkedSubScalar(y, x))
proc checkedScalarMul*[T: SomeInteger](a: Matrix[T]; x: T): Matrix[T] =
  ## Shape: preserve M×N with explicit scalar multiplication overflow checks.
  a.map(proc(y: T): T = checkedMulScalar(y, x))
proc checkedNegate*[T: SomeInteger](a: Matrix[T]): Matrix[T] =
  ## Shape: preserve M×N; reject signed MIN and unsigned nonzero negation.
  a.map(proc(y: T): T = checkedSubScalar(T(0), y))
proc checkedMatmul*[T: SomeInteger](a, b: Matrix[T]): Matrix[T] =
  ## Shape: A=M×K, B=K×N; checks both products and accumulated sums.
  if a.cols != b.rows: raise newException(DimensionError, "checked multiplication inner dimensions differ")
  result = zeros[T](a.rows, b.cols)
  if result.len == 0 or a.cols == 0: return
  for r in 0..<a.rows:
    for k in 0..<a.cols:
      for c in 0..<b.cols: result[r, c] = checkedAddScalar(result[r, c], checkedMulScalar(a[r, k],
          b[k, c]))
