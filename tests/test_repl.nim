import std/[unittest,options,strutils]
import matrix_calc/[repl,tensor]
suite "stateful tensor calculator":
  test "assignments, matrix multiplication and broadcasting":
    var s = initSession()
    discard s.eval("A = [[1,2],[3,4]]")
    discard s.eval("B = A * transpose(A)")
    check s.getValue("B").get.toSeq == @[5.0,11.0,11.0,25.0]
    discard s.eval("C = A + [10,20]")
    check s.getValue("C").get.toSeq == @[11.0,22.0,13.0,24.0]
    discard s.eval("r = Matrix.random(3,3,7)")
    check s.getValue("r").get.shape == @[3,3]
    discard s.eval("v = cross([1,0,0],[0,1,0])")
    check s.getValue("v").get.toSeq == @[0.0,0.0,1.0]
    discard s.eval("n = norm2([3,4])")
    check s.getValue("n").get.toSeq == @[5.0]
    discard s.eval("x = reshape(tensor([1,2,3,4],[2,2]),[4])")
    check s.getValue("x").get.shape == @[4]
    discard s.eval("z = slice(A,1,0,2,2)")
    check s.getValue("z").get.toSeq == @[1.0,3.0]
  test "errors preserve variables":
    var s = initSession()
    discard s.eval("x = 2 + 3 * 4")
    for line in ["x = [[1],[2,3]]","x = missing","x = ones([-1])","x = 1e999","x = reshape([1,2],[3])","x = slice([1,2],0,0,2,0)","x = 1 / 0","x = (1+2","x = [1] trailing","x = 1; exit()"]:
      expect CatchableError: discard s.eval(line)
      check s.getValue("x").get.toSeq == @[14.0]
    check "x" in s.eval("vars").get
    discard s.eval("clear")
    check s.getValue("x").isNone
  test "resource and nesting errors recover":
    var s = initSession()
    for line in [repeat("1+",300) & "1",repeat("(",130)&"1"&repeat(")",130),"zeros([4000,1]) + zeros([1,4000])","hadamard(zeros([4000,1]),zeros([1,4000]))","zeros([0,10000001])"]:
      expect CatchableError: discard s.eval(line)
    discard s.eval("ok = 7")
    check s.getValue("ok").get.toSeq == @[7.0]

suite "array allocation guards":
  test "variables cannot be concatenated past the output limit":
    var s=initSession()
    discard s.eval("a = zeros([100000])")
    var names:seq[string]
    for i in 0..<101:names.add("a")
    expect CatchableError:discard s.eval("result = [" & names.join(",") & "]")
    check s.getValue("result").isNone
    expect CatchableError:discard s.eval("tensor([], [0,10000001])")
