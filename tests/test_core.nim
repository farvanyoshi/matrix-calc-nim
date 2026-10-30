import std/[unittest, options]
import ../src/matrix_calc/core
suite "owned matrix core":
  test "construction, arithmetic, multiplication, transpose, slicing":
    let a = newMatrix(2, 3, [1, 2, 3, 4, 5, 6])
    check a.shape == (2, 3)
    check (a+a).toSeq == @[2, 4, 6, 8, 10, 12]
    check (a-a).toSeq == @[0, 0, 0, 0, 0, 0]
    check (a*2).toSeq == @[2, 4, 6, 8, 10, 12]
    check (a*a.transpose).toSeq == @[14, 32, 32, 77]
    check a.transpose.transpose == a
    check a.slice(0, 2, 1, 3).toSeq == @[2, 3, 5, 6]
    check a.reshape(3, 2).toSeq == a.toSeq
    check a.map(proc(x: int): string = $x).toSeq == @["1", "2", "3", "4", "5", "6"]
  test "copy independence and explicit mutation":
    let a = newMatrix([1, 2, 3, 4], 2, 2)
    var b = a
    b[0, 0] = 99
    check a[0, 0] == 1
    var data = a.data
    data[0] = 88
    check a[0, 0] == 1
    check b[0, 0] == 99
  test "logical bounds and shape checks":
    let a = newMatrix(2, 2, [1, 2, 3, 4])
    expect MatrixIndexError: discard a[0, 2]
    expect MatrixIndexError: discard a[-1, 0]
    check a.tryGet(0, 2).isNone
    expect DimensionError: discard a+newMatrix(1, 4, [1, 2, 3, 4])
    expect ReshapeError: discard newMatrix(-1, 0, newSeq[int]())
    expect ReshapeError: discard newMatrix(high(int), 2, newSeq[int]())
    expect MatrixArithmeticError: discard a.scalarDiv(0)
  test "empty and integer division semantics":
    check zeros[int](2, 0).shape == (2, 0)
    check (zeros[int](2, 0)*zeros[int](0, 3)) == zeros[int](2, 3)
    check zeros[int](0, 0).sum == 0
    check zeros[int](0, 0).product == 1
    check identity[int](0).trace == 0
    check newMatrix(1, 2, [3, 5]).scalarDiv(2).toSeq == @[1.5, 2.5]
    check newMatrix(1, 2, [3, -5]).scalarQuotient(2).toSeq == @[1, -2]
    check newMatrix(1, 1, [7]).pow(3).toSeq == @[343]
  test "checked integer guards work without compiler overflow checks":
    expect MatrixArithmeticError: discard newMatrix(1, 1, [high(int8)]).checkedScalarAdd(1'i8)
    expect MatrixArithmeticError: discard newMatrix(1, 1, [low(int8)]).checkedNegate
    expect MatrixArithmeticError: discard newMatrix(1, 1, [1'u8]).checkedNegate
    expect MatrixArithmeticError: discard newMatrix(1, 1, [high(uint64)]).checkedScalarMul(2'u64)
    expect MatrixArithmeticError: discard newMatrix(1, 2, [100'i8, 100]).checkedMatmul(newMatrix(2,
        1, [1'i8, 1]))
    check newMatrix(1, 2, [2'i8, 3]).checkedMatmul(newMatrix(2, 1, [4'i8, 5])).toSeq == @[23'i8]

suite "degenerate shape guards":
  test "empty giant shapes do not execute impossible loops":
    let a = zeros[int](high(int),0)
    check (a*zeros[int](0,0)).shape == (high(int),0)
    check a.checkedMatmul(zeros[int](0,0)).shape == (high(int),0)
    check a.colSums.shape == (1,0)
    expect ReshapeError: discard zeros[float64](high(int),1)
