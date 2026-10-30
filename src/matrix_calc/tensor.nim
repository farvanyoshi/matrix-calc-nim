## General n-dimensional tensors with immutable layouts and retained shared storage.
## Tensor assignment copies a handle: explicit views and handles alias element
## writes. Constructors, clone, contiguous, and Matrix/Vector conversions copy
## buffers. Layouts never resize backing storage, so retained views cannot dangle.
import std/strutils
import ./[core, vector]

type
  TensorStorage[T] = ref object
    values: seq[T]
  Tensor*[T] = object
    storage: TensorStorage[T]
    dimensions: seq[int]
    steps: seq[int]
    offset: int
    count: int

proc checkedMul(a, b: int): int =
  ## Internal signed size/stride multiplication; rejects overflow before arithmetic.
  if a == 0 or b == 0: return 0
  if a > 0:
    if b > 0:
      if a > high(int) div b: raise newException(ValueError, "tensor dimension/stride overflow")
    elif b < low(int) div a: raise newException(ValueError, "tensor dimension/stride overflow")
  else:
    if b > 0:
      if a < low(int) div b: raise newException(ValueError, "tensor dimension/stride overflow")
    elif a < high(int) div b: raise newException(ValueError, "tensor dimension/stride overflow")
  a*b

proc checkedAdd(a, b: int): int =
  ## Internal checked signed offset addition.
  if (b > 0 and a > high(int)-b) or (b < 0 and a < low(int)-b):
    raise newException(ValueError, "tensor offset overflow")
  a+b

proc copied(input: openArray[int]): seq[int] =
  ## Internal copy of shape/stride metadata.
  result = newSeq[int](input.len)
  for i,x in input: result[i] = x

proc elementCount(shape: openArray[int]): int =
  ## Internal shape product; rank-zero scalar has one element.
  result = 1
  var hasZero = false
  for n in shape:
    if n < 0: raise newException(ValueError, "negative tensor dimension")
    if n == 0: hasZero = true
  if hasZero: return 0
  for n in shape: result = checkedMul(result,n)

proc contiguousStrides(shape: openArray[int]): seq[int] =
  ## Internal canonical row-major element strides.
  result = newSeq[int](shape.len)
  for extent in shape:
    if extent == 0: return
  var step = 1
  for i in countdown(shape.high,0):
    result[i] = step
    step = checkedMul(step,shape[i])

proc newTensor*[T](input: openArray[T], shape: openArray[int]): Tensor[T] =
  ## Shape: product(shape) must equal input.len; [] denotes a one-element scalar.
  let count = elementCount(shape)
  let steps = contiguousStrides(shape)
  if count != input.len: raise newException(ValueError, "tensor shape/element count mismatch")
  if count > high(int) div max(sizeof(T),1): raise newException(ValueError, "tensor byte capacity overflow")
  result.storage = TensorStorage[T](values:newSeq[T](count))
  for i,x in input: result.storage.values[i] = x
  result.dimensions = copied(shape)
  result.steps = steps
  result.count = count

proc tensor*[T](input: openArray[T], shape: openArray[int]): Tensor[T] =
  ## Shape: same as newTensor; copies input into independent contiguous storage.
  newTensor(input,shape)

proc shape*[T](t: Tensor[T]): seq[int] =
  ## Shape: rank r; returns an independent copy of its r dimensions.
  copied(t.dimensions)

proc strides*[T](t: Tensor[T]): seq[int] =
  ## Shape: rank r; returns signed element strides, which may describe a view.
  copied(t.steps)

proc rank*[T](t: Tensor[T]): int =
  ## Shape: any tensor; scalar rank is zero.
  t.dimensions.len

proc len*[T](t: Tensor[T]): int =
  ## Shape: any tensor; product of dimensions, or zero for a default empty handle.
  t.count

proc storageIndex[T](t: Tensor[T], indices: openArray[int]): int =
  ## Internal logical coordinate validation followed by checked stride arithmetic.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  if indices.len != t.rank: raise newException(ValueError, "tensor index rank mismatch")
  result = t.offset
  for i,index in indices:
    if index < 0 or index >= t.dimensions[i]: raise newException(ValueError, "tensor index out of bounds")
    result = checkedAdd(result,checkedMul(index,t.steps[i]))
  if result < 0 or result >= t.storage.values.len: raise newException(ValueError, "invalid tensor view layout")

proc get*[T](t: Tensor[T], indices: openArray[int]): T =
  ## Shape: indices.len must equal rank and each index must be within its axis.
  let index = t.storageIndex(indices)
  t.storage.values[index]

proc set*[T](t: var Tensor[T], indices: openArray[int], value:T) =
  ## Shape: valid rank-sized coordinates; writes through all retained aliases.
  let index = t.storageIndex(indices)
  t.storage.values[index] = value

proc `[]`*[T](t: Tensor[T], indices: openArray[int]): T =
  ## Shape: valid rank-sized coordinates; checked alias of get.
  t.get(indices)

proc `[]=`*[T](t: var Tensor[T], indices: openArray[int], value:T) =
  ## Shape: valid rank-sized coordinates; checked alias of set.
  t.set(indices,value)

proc logicalIndex[T](t: Tensor[T], linear: int): int =
  ## Internal row-major flat-to-storage translation, including negative strides.
  if linear < 0 or linear >= t.count: raise newException(ValueError, "tensor flat index out of bounds")
  var rest = linear
  result = t.offset
  for axis in countdown(t.rank-1,0):
    let coordinate = rest mod t.dimensions[axis]
    rest = rest div t.dimensions[axis]
    result = checkedAdd(result,checkedMul(coordinate,t.steps[axis]))

proc toSeq*[T](t: Tensor[T]): seq[T] =
  ## Shape: n-D -> logical row-major sequence; copies values, including strided views.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  result = newSeq[T](t.len)
  for i in 0..<t.len: result[i] = t.storage.values[t.logicalIndex(i)]

iterator items*[T](t: Tensor[T]): T =
  ## Shape: any tensor; scalar yields once, zero-sized tensors yield nothing.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  for i in 0..<t.len: yield t.storage.values[t.logicalIndex(i)]

proc isContiguous*[T](t: Tensor[T]): bool =
  ## Shape: any tensor; empty tensors and unit axes do not prevent contiguity.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  if t.len == 0: return true
  var expected = 1
  for axis in countdown(t.rank-1,0):
    if t.dimensions[axis] > 1 and t.steps[axis] != expected: return false
    expected = checkedMul(expected,t.dimensions[axis])
  true

proc contiguous*[T](t: Tensor[T]): Tensor[T] =
  ## Shape: unchanged; always copies to independent canonical row-major storage.
  newTensor(t.toSeq,t.dimensions)

proc clone*[T](t: Tensor[T]): Tensor[T] =
  ## Shape: unchanged; explicit independent element-buffer copy.
  t.contiguous

proc reshape*[T](t: Tensor[T], newShape: openArray[int]): Tensor[T] =
  ## Shape: equal element count required; contiguous input shares storage, otherwise copies.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  if elementCount(newShape) != t.len: raise newException(ValueError, "reshape changes tensor element count")
  let newSteps = contiguousStrides(newShape)
  result = if t.isContiguous: t else: t.contiguous
  result.dimensions = copied(newShape)
  result.steps = newSteps

proc transpose*[T](t: Tensor[T], axes: openArray[int]): Tensor[T] =
  ## Shape: axes must be a permutation of 0..<rank; returns a zero-copy retained view.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  if axes.len != t.rank: raise newException(ValueError, "transpose permutation rank mismatch")
  var seen = newSeq[bool](t.rank)
  result = t
  result.dimensions = newSeq[int](t.rank)
  result.steps = newSeq[int](t.rank)
  for i,axis in axes:
    if axis < 0 or axis >= t.rank or seen[axis]: raise newException(ValueError, "invalid transpose permutation")
    seen[axis] = true
    result.dimensions[i] = t.dimensions[axis]
    result.steps[i] = t.steps[axis]

proc transpose*[T](t: Tensor[T]): Tensor[T] =
  ## Shape: rank r; reverses all axes, including a scalar's empty permutation.
  var axes = newSeq[int](t.rank)
  for i in 0..<t.rank: axes[i] = t.rank-1-i
  t.transpose(axes)

proc slice*[T](t: Tensor[T], axis,start,stop:int,step=1): Tensor[T] =
  ## Shape: half-open axis range; positive steps require 0<=start<=stop<=size.
  ## Negative steps require -1<=stop<=start<size; reverse uses (size-1,-1,-1).
  if t.storage.isNil or axis < 0 or axis >= t.rank: raise newException(ValueError, "invalid slice axis")
  if step == 0 or step == low(int): raise newException(ValueError, "invalid slice step")
  let size = t.dimensions[axis]
  var count = 0
  if step > 0:
    if start < 0 or start > stop or stop > size: raise newException(ValueError, "invalid positive slice range")
    let distance = stop-start
    count = distance div step + ord(distance mod step != 0)
  else:
    if stop < -1 or stop > start or start >= size or start < -1: raise newException(ValueError, "invalid negative slice range")
    let distance = start-stop
    count = distance div (-step) + ord(distance mod (-step) != 0)
  result = t
  result.dimensions = copied(t.dimensions)
  result.steps = copied(t.steps)
  result.dimensions[axis] = count
  result.count = elementCount(result.dimensions)
  if result.count == 0:
    result.offset = 0
    result.steps = newSeq[int](t.rank)
  else:
    result.offset = checkedAdd(t.offset,checkedMul(start,t.steps[axis]))
    result.steps[axis] = checkedMul(t.steps[axis],step)

proc broadcastShape*(a,b:openArray[int]):seq[int] =
  ## Shape: align trailing axes; equal extents or extent 1 are compatible, including 0 with 1.
  discard elementCount(a)
  discard elementCount(b)
  result = newSeq[int](max(a.len,b.len))
  for back in 0..<result.len:
    let x = if back < a.len: a[a.high-back] else: 1
    let y = if back < b.len: b[b.high-back] else: 1
    if x == y: result[result.high-back] = x
    elif x == 1: result[result.high-back] = y
    elif y == 1: result[result.high-back] = x
    else: raise newException(ValueError, "incompatible broadcast shapes")
  discard elementCount(result)

proc broadcastIndex[T](t: Tensor[T], outputShape:seq[int], linear:int):int =
  ## Internal aligned broadcast indexing over a logical row-major output.
  var rest = linear
  result = t.offset
  for back in 0..<outputShape.len:
    let axis = outputShape.high-back
    let coordinate = rest mod outputShape[axis]
    rest = rest div outputShape[axis]
    if back < t.rank:
      let sourceAxis = t.rank-1-back
      if t.dimensions[sourceAxis] != 1: result += coordinate*t.steps[sourceAxis]

proc map*[T,U](t:Tensor[T],f:proc(x:T):U):Tensor[U] =
  ## Shape: unchanged; applies callback in logical order and allocates independent output.
  if t.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  var values = newSeq[U](t.len)
  for i in 0..<t.len: values[i] = f(t.storage.values[t.logicalIndex(i)])
  newTensor(values,t.dimensions)

proc zipMap*[T,U,R](a:Tensor[T],b:Tensor[U],f:proc(x:T,y:U):R):Tensor[R] =
  ## Shape: operands must right-broadcast; returns independent contiguous broadcast result.
  if a.storage.isNil or b.storage.isNil: raise newException(ValueError, "uninitialized tensor")
  let dims = broadcastShape(a.dimensions,b.dimensions)
  var values = newSeq[R](elementCount(dims))
  for i in 0..<values.len: values[i] = f(a.storage.values[a.broadcastIndex(dims,i)],b.storage.values[b.broadcastIndex(dims,i)])
  newTensor(values,dims)

proc `+`*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; allocates their elementwise sum.
  a.zipMap(b,proc(x:T,y:U):auto = x+y)

proc `-`*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; allocates their elementwise difference.
  a.zipMap(b,proc(x:T,y:U):auto = x-y)

proc hadamard*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; allocates elementwise multiplication.
  a.zipMap(b,proc(x:T,y:U):auto = x*y)

proc elementDiv*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; rejects all zero divisors before computing output.
  discard broadcastShape(a.dimensions,b.dimensions)
  for x in b:
    if x == U(0): raise newException(ValueError, "tensor zero divisor")
  a.zipMap(b,proc(x:T,y:U):auto = x/y)

proc `-`*[T](t:Tensor[T]):auto =
  ## Shape: unchanged; allocates elementwise negation.
  t.map(proc(x:T):auto = -x)

proc scalarMul*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; allocates elementwise multiplication by a scalar.
  t.map(proc(x:T):auto = x*scalar)

proc scalarAdd*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; allocates elementwise addition of a scalar.
  t.map(proc(x:T):auto = x+scalar)

proc scalarDiv*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; scalar must be nonzero, including for an empty tensor.
  if scalar == U(0): raise newException(ValueError, "tensor zero scalar divisor")
  t.map(proc(x:T):auto = x/scalar)

proc `*`*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; alias of scalarMul, not tensor contraction.
  t.scalarMul(scalar)

proc fromMatrix*[T](m:Matrix[T]):Tensor[T] =
  ## Shape: m×n matrix -> rank-2 [m,n] tensor; copies data.
  newTensor(m.toSeq,[m.rows,m.cols])

proc toTensor*[T](m:Matrix[T]):Tensor[T] =
  ## Shape: m×n matrix -> rank-2 [m,n] tensor; copies data.
  fromMatrix(m)

proc toTensor*[T](v:Vector[T]):Tensor[T] =
  ## Shape: length n vector -> rank-1 [n] tensor; copies data.
  newTensor(v.toSeq,[v.len])

proc toMatrix*[T](t:Tensor[T]):Matrix[T] =
  ## Shape: rank must equal 2; copies logical row-major values from any layout.
  if t.rank != 2: raise newException(ValueError, "tensor must have rank 2 for matrix conversion")
  newMatrix(t.toSeq,t.dimensions[0],t.dimensions[1])

proc toVector*[T](t:Tensor[T]):Vector[T] =
  ## Shape: rank must equal 1; copies logical values to contiguous vector storage.
  if t.rank != 1: raise newException(ValueError, "tensor must have rank 1 for vector conversion")
  newVector(t.toSeq)

proc randomTensor*(shape:openArray[int],seed:uint64=1):Tensor[float64] =
  ## Shape: any valid shape; deterministic finite [0,1) values from a local xorshift PRNG.
  let count = elementCount(shape)
  var state = if seed == 0: 0x9E3779B97F4A7C15'u64 else: seed
  if count > high(int) div sizeof(float64): raise newException(ValueError, "tensor byte capacity overflow")
  var values = newSeq[float64](count)
  for x in values.mitems:
    state = state xor (state shl 13)
    state = state xor (state shr 7)
    state = state xor (state shl 17)
    x = float64(state shr 11) / 9007199254740992.0
  newTensor(values,shape)

proc `$`*[T](t:Tensor[T]):string =
  ## Shape: any rank; nested pretty-print uses logical layout, including strided views.
  if t.storage.isNil: return "Tensor(uninitialized)"
  proc meta(values:seq[int]):string =
    var parts:seq[string]
    for i in 0..<min(values.len,16): parts.add($values[i])
    if values.len > 16: parts.add("... (" & $(values.len-16) & " axes)")
    "@[" & parts.join(", ") & "]"
  let header = "Tensor(shape=" & meta(t.dimensions) & ", strides=" & meta(t.steps) & ") "
  if t.len == 0: return header & "[]"
  if t.rank > 8 or t.len > 128:
    var parts:seq[string]
    for i in 0..<min(t.len,64): parts.add($t.storage.values[t.logicalIndex(i)])
    if t.len > 64: parts.add("... (" & $(t.len-64) & " more)")
    return header & "preview=[" & parts.join(", ") & "]"
  proc render(axis,base:int):string =
    if axis == t.rank:
      result = $t.storage.values[base]
    else:
      var parts:seq[string]
      let shown = min(t.dimensions[axis],8)
      for i in 0..<shown: parts.add(render(axis+1,base+i*t.steps[axis]))
      if shown < t.dimensions[axis]: parts.add("... (" & $(t.dimensions[axis]-shown) & " more)")
      result = "[" & parts.join(", ") & "]"
  header & render(0,t.offset)

proc `*`*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; tensor multiplication is elementwise.
  hadamard(a,b)

proc `/`*[T,U](a:Tensor[T],b:Tensor[U]):auto =
  ## Shape: right-broadcastable operands; floating elementwise division rejects zeros.
  elementDiv(a,b)

proc `/`*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; floating scalar division rejects zero.
  scalarDiv(t,scalar)

proc `==`*[T,U](a:Tensor[T],b:Tensor[U]):bool =
  ## Shape: equal dimensions required; compares logical values, never backing addresses.
  if a.dimensions != b.dimensions or a.count != b.count: return false
  if a.storage.isNil or b.storage.isNil: return a.storage.isNil and b.storage.isNil
  for i in 0..<a.len:
    if a.storage.values[a.logicalIndex(i)] != b.storage.values[b.logicalIndex(i)]: return false
  true

proc scalarSub*[T,U](t:Tensor[T],scalar:U):auto =
  ## Shape: unchanged; allocates elementwise subtraction of a scalar.
  t.map(proc(x:T):auto = x-scalar)
