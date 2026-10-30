import std/[unittest,strutils]
import matrix_calc/[tensor,vector,core]

suite "n-dimensional tensors":
  test "scalar ranks shapes and strict construction":
    let scalar = newTensor([7],newSeq[int](0))
    check scalar.rank == 0
    check scalar.len == 1
    check scalar.get([]) == 7
    check scalar.toSeq == @[7]
    check scalar.transpose.toSeq == @[7]
    check newTensor(newSeq[int](0),[2,0,3]).len == 0
    expect ValueError: discard newTensor([1,2],[3])
    expect ValueError: discard newTensor([1],[-1])
    expect ValueError: discard newTensor(newSeq[int](0),[high(int),2])
    check newTensor(newSeq[int](0),[0,high(int),2]).strides == @[0,0,0]
    check newTensor(newSeq[int](0),[0,high(int),2]).len == 0
    expect ValueError: discard scalar.get([0])
    expect ValueError: discard newTensor([1,2],[2]).get([-1])
  test "axis permutation slicing and retained writes":
    var a = newTensor([0,1,2,3,4,5],[2,3])
    check a.strides == @[3,1]
    var t = a.transpose
    check t.shape == @[3,2]
    check t.strides == @[1,3]
    check not t.isContiguous
    check t.toSeq == @[0,3,1,4,2,5]
    t.set([1,0],99)
    check a.get([0,1]) == 99
    check a.slice(1,0,3,2).toSeq == @[0,2,3,5]
    check a.slice(1,2,-1,-1).toSeq == @[2,99,0,5,4,3]
    check a.transpose([1,0]).transpose([1,0]).toSeq == a.toSeq
    expect ValueError: discard a.transpose([0,0])
    expect ValueError: discard a.slice(1,0,3,0)
    expect ValueError: discard a.slice(1,3,0)
    expect ValueError: discard a.get([0,3])
    a = newTensor([8],[1])
    check t.get([1,0]) == 99 # View keeps its old backing storage alive.
  test "copies reshape and empty shapes":
    let a = newTensor([1,2,3,4,5,6],[2,3])
    var b = a.contiguous
    b.set([0,0],99)
    check a.get([0,0]) == 1
    check a.reshape([3,2]).toSeq == a.toSeq
    let transposed = a.transpose
    var reshaped = transposed.reshape([6])
    check reshaped.toSeq == @[1,4,2,5,3,6]
    reshaped.set([0],77)
    check transposed.get([0,0]) == 1
    expect ValueError: discard a.reshape([5])
    let empty = newTensor(newSeq[int](0),[0,3])
    check empty.transpose.shape == @[3,0]
    check empty.transpose.toSeq.len == 0
    check empty.reshape([1,0,2]).shape == @[1,0,2]
    check empty.slice(0,0,0).shape == @[0,3]
    check a.slice(0,0,0,high(int)).strides == @[0,0]
    check newTensor(newSeq[int](0),[high(int),0]).toSeq.len == 0
    check ($newTensor(newSeq[int](0),[high(int),0])).len < 300
  test "right-aligned broadcasting includes scalar and zero axes":
    let a = newTensor([1,2,3,4,5,6],[2,3])
    let row = newTensor([10,20,30],[3])
    check (a+row).toSeq == @[11,22,33,14,25,36]
    let col = newTensor([10,20],[2,1])
    check (a+col).toSeq == @[11,12,13,24,25,26]
    let scalar = newTensor([2],newSeq[int](0))
    check hadamard(a,scalar).toSeq == @[2,4,6,8,10,12]
    check (a*scalar).toSeq == @[2,4,6,8,10,12]
    check a == a.contiguous
    check (a/scalar).toSeq == @[0.5,1,1.5,2,2.5,3]
    check (a-row).toSeq == @[-9,-18,-27,-6,-15,-24]
    check (-row).toSeq == @[-10,-20,-30]
    check (a*2).toSeq == @[2,4,6,8,10,12]
    check a.scalarAdd(1).toSeq == @[2,3,4,5,6,7]
    check a.scalarDiv(2).toSeq == @[0.5,1,1.5,2,2.5,3]
    check elementDiv(a,scalar).toSeq == @[0.5,1,1.5,2,2.5,3]
    check broadcastShape([0,3],[1,3]) == @[0,3]
    check (newTensor(newSeq[int](0),[0,3])+row).shape == @[0,3]
    expect ValueError: discard a+newTensor([1,2],[2])
    expect ValueError: discard a.scalarDiv(0)
    expect ValueError: discard elementDiv(a,newTensor([0],newSeq[int](0)))
  test "noncontiguous broadcasting conversions and diagnostics":
    let a = newTensor([1,2,3,4,5,6],[2,3]).transpose
    let row = newTensor([10,20],[2])
    check (a+row).toSeq == @[11,24,12,25,13,26]
    check a.toMatrix.shape == (3,2)
    check a.toMatrix.toTensor.toSeq == a.toSeq
    check newVector([1,2]).toTensor.toVector.toSeq == @[1,2]
    expect ValueError: discard a.toVector
    expect ValueError: discard newVector([1,2]).toTensor.toMatrix
    check "Tensor(shape=" in $a
    check "[[1, 4], [2, 5], [3, 6]]" in $a
    check randomTensor([2,3],123).toSeq == randomTensor([2,3],123).toSeq
    for x in randomTensor([100],123): check x >= 0 and x < 1
  test "generated layouts and broadcasting":
    for m in 0..6:
      for n in 0..6:
        var values:seq[int]
        for i in 0..<m*n: values.add(i)
        let a = newTensor(values,[m,n])
        check a.transpose.transpose.toSeq == values
        check a.transpose.contiguous.toSeq == a.transpose.toSeq
        let scalar = newTensor([3],newSeq[int](0))
        let sum = a+scalar
        check sum.shape == @[m,n]
        for i,x in sum.toSeq: check x == values[i]+3

  test "default handles and bounded diagnostics":
    var absent:Tensor[int]
    expect ValueError: discard absent.get([])
    expect ValueError: absent.set([],3)
    expect ValueError: discard absent.toSeq
    expect ValueError: discard absent.contiguous
    expect ValueError: discard absent.isContiguous
    expect ValueError: discard absent.scalarMul(2)
    expect ValueError: discard absent+newTensor([1],newSeq[int](0))
    check "uninitialized" in $absent
    let large=randomTensor([20,20],1)
    check ($large).len < 2000
    check "preview" in $large
    var dimensions=newSeq[int](10000)
    for x in dimensions.mitems: x=1
    let deep=newTensor([9],dimensions)
    check ($deep).len < 500
    check "preview" in $deep
