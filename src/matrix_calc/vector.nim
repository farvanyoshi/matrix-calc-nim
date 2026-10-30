## Specialized contiguous one-dimensional storage and vector arithmetic.
## Constructors copy the input; conversions copy elements and preserve their order.
import std/[math, strutils]
import ./core

type Vector*[T] = object
  values: seq[T]

proc newVector*[T](input: openArray[T]): Vector[T] =
  ## Shape: any length n; creates a contiguous vector by copying its elements.
  result.values = newSeq[T](input.len)
  for i, x in input: result.values[i] = x

proc len*[T](v: Vector[T]): int =
  ## Shape: any vector; returns its one-dimensional element count.
  v.values.len

proc toSeq*[T](v: Vector[T]): seq[T] =
  ## Shape: n; returns an independent element buffer of length n.
  result = newSeq[T](v.len)
  for i, x in v.values: result[i] = x

proc clone*[T](v: Vector[T]): Vector[T] =
  ## Shape: n; copies the vector buffer using T's ordinary copy semantics.
  newVector(v.values)

proc `[]`*[T](v: Vector[T], index: int): T =
  ## Shape: n; index must satisfy 0 <= index < n.
  if index < 0 or index >= v.len: raise newException(ValueError, "vector index out of range")
  v.values[index]

proc `[]=`*[T](v: var Vector[T], index: int, value: T) =
  ## Shape: n; index must satisfy 0 <= index < n; replaces one element.
  if index < 0 or index >= v.len: raise newException(ValueError, "vector index out of range")
  v.values[index] = value

iterator items*[T](v: Vector[T]): T =
  ## Shape: n; visits each contiguous element once in order.
  for x in v.values: yield x

proc map*[T, U](v: Vector[T], f: proc(x: T): U): Vector[U] =
  ## Shape: n -> n; callback is called once per element in contiguous order.
  var values = newSeq[U](v.len)
  for i, x in v.values: values[i] = f(x)
  newVector(values)

proc `+`*[T, U](a: Vector[T], b: Vector[U]): auto =
  ## Shape: both vectors length n; returns elementwise sum of length n.
  if a.len != b.len: raise newException(ValueError, "vector lengths must match")
  type R = typeof(default(T) + default(U))
  var values = newSeq[R](a.len)
  for i in 0..<a.len: values[i] = a.values[i] + b.values[i]
  newVector(values)

proc `-`*[T, U](a: Vector[T], b: Vector[U]): auto =
  ## Shape: both vectors length n; returns elementwise difference of length n.
  if a.len != b.len: raise newException(ValueError, "vector lengths must match")
  type R = typeof(default(T) - default(U))
  var values = newSeq[R](a.len)
  for i in 0..<a.len: values[i] = a.values[i] - b.values[i]
  newVector(values)

proc `-`*[T](v: Vector[T]): auto =
  ## Shape: n -> n; returns elementwise negation with scalar overflow semantics.
  type R = typeof(-default(T))
  var values = newSeq[R](v.len)
  for i, x in v.values: values[i] = -x
  newVector(values)

proc scalarMul*[T, U](v: Vector[T], scalar: U): auto =
  ## Shape: n -> n; contiguous scalar multiplication without matrix indexing.
  type R = typeof(default(T) * default(U))
  var values = newSeq[R](v.len)
  for i, x in v.values: values[i] = x * scalar
  newVector(values)

proc `*`*[T, U](v: Vector[T], scalar: U): auto =
  ## Shape: n -> n; equivalent to scalarMul.
  scalarMul(v, scalar)

proc scalarDiv*[T, U](v: Vector[T], scalar: U): auto =
  ## Shape: n -> n; scalar must be nonzero; integers use Nim floating division.
  if scalar == U(0): raise newException(ValueError, "cannot divide vector by zero")
  type R = typeof(default(T) / default(U))
  var values = newSeq[R](v.len)
  for i, x in v.values: values[i] = x / scalar
  newVector(values)

proc dot*[T, U](a: Vector[T], b: Vector[U]): auto =
  ## Shape: both vectors length n; empty dot product is additive zero.
  if a.len != b.len: raise newException(ValueError, "vector lengths must match")
  type R = typeof(default(T) * default(U))
  var total = R(0)
  for i in 0..<a.len: total += a.values[i] * b.values[i]
  total

proc cross*[T, U](a: Vector[T], b: Vector[U]): auto =
  ## Shape: both vectors must have exactly 3 elements; returns their 3D cross product.
  if a.len != 3 or b.len != 3: raise newException(ValueError, "cross product requires length 3")
  newVector([a.values[1]*b.values[2]-a.values[2]*b.values[1],
             a.values[2]*b.values[0]-a.values[0]*b.values[2],
             a.values[0]*b.values[1]-a.values[1]*b.values[0]])

proc normL1*[T: SomeNumber](v: Vector[T]): float64 =
  ## Shape: n; sums absolute values in float64, returning zero for empty input.
  for x in v.values: result += abs(float64(x))

proc normL2*[T: SomeNumber](v: Vector[T]): float64 =
  ## Shape: n; scaled Euclidean norm avoids squaring overflow; empty norm is zero.
  for x in v.values: result = hypot(result, float64(x))

proc normalized*[T: SomeNumber](v: Vector[T]): Vector[float64] =
  ## Shape: n; returns a float64 unit vector; zero/nonfinite vectors are rejected.
  var scale = 0.0
  for x in v.values:
    if classify(float64(x)) in {fcNan, fcInf, fcNegInf}: raise newException(ValueError, "vector must be finite")
    scale = max(scale, abs(float64(x)))
  if scale == 0: raise newException(ValueError, "cannot normalize zero vector")
  var values = newSeq[float64](v.len)
  var norm = 0.0
  for i, x in v.values:
    values[i] = float64(x) / scale
    norm = hypot(norm, values[i])
  for x in values.mitems: x /= norm
  newVector(values)

proc toMatrix*[T](v: Vector[T], column = true): Matrix[T] =
  ## Shape: n -> n×1 when column=true, otherwise 1×n; copies elements.
  if column: newMatrix(v.toSeq, v.len, 1)
  else: newMatrix(v.toSeq, 1, v.len)

proc toVector*[T](m: Matrix[T]): Vector[T] =
  ## Shape: matrix must be 1×n or n×1, including empty vector shapes.
  if m.rows != 1 and m.cols != 1: raise newException(ValueError, "matrix must be vector-shaped")
  newVector(m.toSeq)

proc `$`*[T](v: Vector[T]): string =
  ## Shape: any n; print all elements and the one-dimensional vector length.
  var parts: seq[string]
  for x in v.values: parts.add($x)
  "Vector[" & $v.len & "] [" & parts.join(", ") & "]"
