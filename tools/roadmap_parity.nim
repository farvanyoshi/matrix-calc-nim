## Deterministic public API fixture exporter; each case records logical shape.
import std/[json,strformat]
import matrix_calc
var cases=newJArray()
proc record(name:string,shape:seq[int],data:seq[float64]) =
  ## Shape requirements: product(shape)=data.len, [] is a scalar.
  cases.add(%*{"name":name,"shape":shape,"data":data})
for m in 0..5:
  for n in 0..5:
    var values:seq[float64]
    for i in 0..<2*m*n:values.add(float64(i))
    var row:seq[float64]
    for i in 0..<n:row.add(float64(i+1))
    let a=newTensor(values,[2,m,n]);let b=newTensor(row,[1,n])
    let sum=a+b;record(&"broadcast_{m}_{n}",sum.shape,sum.toSeq)
    let t=a.transpose([2,0,1]);record(&"transpose_{m}_{n}",t.shape,t.toSeq)
    let s=a.slice(2,0,n,2);record(&"slice_{m}_{n}",s.shape,s.toSeq)
let v=newVector([3.0,4.0,0.0])
record("vector_norms",@[3],@[dot(v,v),v.normL1,v.normL2])
record("cross",@[3],cross(newVector([1.0,0.0,0.0]),newVector([0.0,1.0,0.0])).toSeq)
let a=newMatrix(3,3,[0.0,2,3,4,5,6,7,8,10])
record("inverse",@[3,3],a.inverse.toSeq)
record("determinant",@[],@[a.determinant])
record("svd",@[2],newMatrix(2,2,[1.0,2,2,4]).svd.singularValues)
record("eigen",@[2],newMatrix(2,2,[2.0,1,1,2]).symmetricEigen.eigenvalues)
echo $cases
