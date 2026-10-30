## Public facade smoke example; shape requirements are explicit in constructors.
import matrix_calc
let a=newMatrix(2,3,[1.0,2,3,4,5,6])
doAssert a.transpose.shape==(3,2)
let v=newVector([1.0,0,0]);let w=newVector([0.0,1,0])
doAssert cross(v,w).toSeq== @[0.0,0,1]
let t=newTensor([1.0,2,3,4,5,6],[2,3])
let row=newTensor([10.0,20,30],[3])
doAssert (t+row).toSeq== @[11.0,22,33,14,25,36]
echo t.transpose
echo v.normL2
