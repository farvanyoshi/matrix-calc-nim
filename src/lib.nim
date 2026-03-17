import std/sequtils
import std/strformat
import std/sugar

type
    Matrix*[T] = object
        rows*: uint
        cols*: uint
        data*: seq[T]
    
    MatrixError = object of CatchableError
    DimensionError = object of MatrixError
    ReshapeError = object of MatrixError


func len*[T](m: Matrix[T]): uint =
    m.rows * m.cols

func get*[T](m: Matrix[T]; row, col: uint): T =
    m.data[row * m.cols + col]

proc get_mut*[T](m: var Matrix[T]; row, col: uint): T =
    m.data[row * m.cols + col]

func new_matrix*[T](rows, cols: uint; input: openarray[T]): Matrix[T] =
    if rows * cols != input.len:
        raise newException(ReshapeError, &"{rows*cols}-element matrix cannot contain {input.len} input elements")
    Matrix[T](rows, cols, input)

func map*[T, U](m: Matrix[T]; f: T -> U): Matrix[U] =
    Matrix[U].new(m.rows, m.cols, m.data.mapIt(f))

func zip_map*[T, U, R](m: Matrix[T]; n: Matrix[U]; f: (T, U) -> R): Matrix[R] =
    if m.rows != n.rows or m.cols != n.cols:
        raise newException(DimensionError, &"{m.rows}x{m.cols} matrix does not have same dimensions as {n.rows}x{n.cols} matrix")
    Matrix[R].new(
        m.rows,
        m.cols,
        m.data.zip(n.data).mapIt((t) => f(t[0], t[1]))
    )

func zeros*[T](rows, cols: uint): Matrix[T] =
    Matrix[T](rows, cols, newSeq[T](rows * cols))

func ones*[T](rows, cols: uint): Matrix[T] =
    Matrix[T](rows, cols, newSeqWith[T](rows * cols, T(1)))

func identity*[T](side: uint): Matrix[T] =
    var data = newSeq[T](side * side)
    for i in 0 ..< side:
        data[i * side + i] = T(1)
    Matrix[T](side, side, data)

proc `-`*[T](self: Matrix[T]): Matrix[T] =
    self.map(x => -x)

proc `+`*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self.zip_map(other, (x, y: T) => x + y)

proc `-`*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self.zip_map(other, (x, y: T) => x - y)

proc `*`*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    if self.cols != other.rows:
        raise newException(DimensionError, &"Matrix multiplication requires the LHS rows ({rows}) is equal to the RHS columns ({cols})")
    
    let rows = self.rows
    let cols = other.cols
    let shared = self.cols

    var data = newSeq[T](rows * cols)

    for i in 0 ..< rows:
        for k in 0 ..< shared:
            let a = self.data[i * shared + k]
            for j in 0 ..< cols:
                data[i * cols + j] += a * other.data[k * cols + j]
    
    Matrix[T](rows, cols, data)

proc hadamard*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self.zip_map(other, (x, y: T) => x * y)

proc element_div*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self.zip_map(other, (x, y: T) => x / y)

proc add*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self + other

proc sub*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self - other

proc mul*[T](self: Matrix[T], other: Matrix[T]): Matrix[T] =
    self * other

proc neg*[T](self: Matrix[T]): Matrix[T] =
    -self