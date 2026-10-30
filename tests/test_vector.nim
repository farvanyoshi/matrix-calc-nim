import std/[unittest, math]
import matrix_calc/[vector, core]

suite "specialized vectors":
  test "arithmetic and exact 3D geometry":
    let a = newVector([1,2,3])
    let b = newVector([4,5,6])
    check (a+b).toSeq == @[5,7,9]
    check (b-a).toSeq == @[3,3,3]
    check (-a).toSeq == @[-1,-2,-3]
    check (a*3).toSeq == @[3,6,9]
    check a.dot(b) == 32
    check a.cross(b).toSeq == @[-3,6,-3]
    check b.cross(a).toSeq == @[3,-6,3]
    check abs(a.normL1-6) < 1e-12
    check abs(a.normL2-sqrt(14.0)) < 1e-12
    check abs(a.normalized.normL2-1) < 1e-12
    check a.scalarDiv(2).toSeq == @[0.5,1.0,1.5]
  test "empty vectors bounds and dimensions":
    let z = newVector(newSeq[int](0))
    check z.dot(z) == 0
    check z.normL1 == 0
    check z.normL2 == 0
    expect ValueError: discard z.normalized
    expect ValueError: discard z[0]
    expect ValueError: discard newVector([1,2]).cross(newVector([2,3]))
    expect ValueError: discard newVector([1]).dot(newVector([1,2]))
    expect ValueError: discard z.scalarDiv(0)
  test "conversions ownership and stable normalization":
    var v = newVector([1,2,3])
    var copy = v.clone
    copy[0] = 9
    check v[0] == 1
    check v.toMatrix.shape == (3,1)
    check v.toMatrix(false).shape == (1,3)
    check v.toMatrix.toVector.toSeq == v.toSeq
    expect ValueError: discard newMatrix([1,2,3,4],2,2).toVector
    let huge = newVector([1e308,1e308])
    check abs(huge.normalized.normL2-1) < 1e-12
    expect ValueError: discard newVector([Inf]).normalized
    expect ValueError: discard newVector([NaN]).normalized
    check newVector(["a","bb"]).map(proc(x:string):int=x.len).toSeq == @[1,2]
