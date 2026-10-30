## Stateful tensor calculator. Shape requirements are enforced before assignments
## replace existing variables; input is parsed, never executed as Nim code.
import std/[strutils, tables, options, math, algorithm]
import ./[core, tensor, vector]

type
  ReplError* = object of CatchableError
    ## Syntax, name, shape, or numerical failure in the interactive calculator.
  Session* = object
    ## Persistent float64 tensor variables, supporting scalar and n-D values.
    variables: Table[string, Tensor[float64]]
  TokenKind = enum tkNumber, tkName, tkMark
  Token = object
    kind: TokenKind
    number: float64
    name: string
    mark: char
  ExprKind = enum exNumber, exName, exArray, exCall, exNeg, exBinary
  Expr = ref object
    kind: ExprKind
    number: float64
    name: string
    children: seq[Expr]
    op: char
  Parser = object
    tokens: seq[Token]
    pos, depth: int

proc fail(message: string) {.noreturn.} = raise newException(ReplError, message)
proc finite(x: float64): bool = classify(x) notin {fcNan,fcInf,fcNegInf}
proc tokenize(line: string): seq[Token] =
  var i = 0
  while i < line.len:
    let c = line[i]
    if c in Whitespace: inc i
    elif c in Digits or (c == '.' and i+1 < line.len and line[i+1] in Digits):
      let start = i
      while i < line.len and line[i] in Digits: inc i
      if i < line.len and line[i] == '.':
        inc i
        while i < line.len and line[i] in Digits: inc i
      if i < line.len and line[i] in {'e','E'}:
        inc i
        if i < line.len and line[i] in {'+','-'}: inc i
        let exponent = i
        while i < line.len and line[i] in Digits: inc i
        if exponent == i: fail("missing exponent digits")
      let number = parseFloat(line[start..<i])
      if not finite(number): fail("number must be finite")
      result.add Token(kind: tkNumber, number: number)
    elif c in Letters or c == '_':
      let start = i
      inc i
      while i < line.len and (line[i] in Letters+Digits or line[i] in {'_', '.'}): inc i
      result.add Token(kind: tkName, name: line[start..<i])
    elif c in {'[',']','(',')','+','-','*','/',',','='}:
      result.add Token(kind: tkMark, mark: c)
      inc i
    else: fail("unexpected character at byte " & $i)
  if result.len > 4096: fail("input exceeds 4096 tokens")

proc mark(p: var Parser, c: char): bool =
  if p.pos < p.tokens.len and p.tokens[p.pos].kind == tkMark and p.tokens[p.pos].mark == c:
    inc p.pos
    result = true
proc expect(p: var Parser, c: char) =
  if not p.mark(c): fail("expected '" & $c & "'")
proc expression(p: var Parser): Expr
proc atom(p: var Parser): Expr
proc term(p: var Parser): Expr =
  result = p.atom()
  while true:
    var op: char
    if p.mark('*'): op = '*'
    elif p.mark('/'): op = '/'
    else: break
    result = Expr(kind: exBinary, op: op, children: @[result,p.atom()])
proc expression(p: var Parser): Expr =
  result = p.term()
  while true:
    var op: char
    if p.mark('+'): op = '+'
    elif p.mark('-'): op = '-'
    else: break
    result = Expr(kind: exBinary, op: op, children: @[result,p.term()])
proc atom(p: var Parser): Expr =
  inc p.depth
  if p.depth > 128: fail("expression nesting exceeds 128")
  defer: dec p.depth
  if p.mark('-'): return Expr(kind: exNeg, children: @[p.atom()])
  if p.mark('+'): return p.atom()
  if p.mark('('):
    result = p.expression()
    p.expect(')')
    return
  if p.mark('['):
    result = Expr(kind: exArray)
    if not p.mark(']'):
      while true:
        result.children.add p.expression()
        if p.mark(']'): break
        p.expect(',')
    return
  if p.pos >= p.tokens.len: fail("expected number, variable, array, or function call")
  let t = p.tokens[p.pos]
  inc p.pos
  case t.kind
  of tkNumber: result = Expr(kind: exNumber, number: t.number)
  of tkName:
    if not p.mark('('): return Expr(kind: exName, name: t.name)
    result = Expr(kind: exCall, name: t.name)
    if not p.mark(')'):
      while true:
        result.children.add p.expression()
        if p.mark(')'): break
        p.expect(',')
  else: fail("expected number, variable, array, or function call")

proc scalar(n: float64): Tensor[float64] = newTensor([n], newSeq[int]())
proc integer(t: Tensor[float64]): int =
  if t.rank != 0: fail("expected a scalar integer")
  let n = t.toSeq[0]
  if not finite(n) or n < 0 or floor(n) != n or n >= float64(high(int)):
    fail("expected a nonnegative representable integer")
  int(n)
proc dimensions(t: Tensor[float64]): seq[int] =
  if t.rank != 1: fail("shape/axes must be a vector of integers")
  for n in t.toSeq: result.add integer(scalar(n))
proc count(shape: seq[int]): int =
  result = 1
  for n in shape:
    if n > 10_000_000: fail("interactive dimension limit is 10000000")
    if n != 0 and result > high(int) div n: fail("shape overflow")
    result *= n
proc outputGuard(a,b: Tensor[float64], matmul: bool) =
  var outputShape: seq[int]
  if matmul and a.rank == 2 and b.rank == 2: outputShape = @[a.shape[0],b.shape[1]]
  else:
    let rank = max(a.rank,b.rank)
    outputShape = newSeq[int](rank)
    let ashape = a.shape
    let bshape = b.shape
    for i in 0..<rank:
      let ad = if i+a.rank >= rank: ashape[i+a.rank-rank] else: 1
      let bd = if i+b.rank >= rank: bshape[i+b.rank-rank] else: 1
      if ad != bd and ad != 1 and bd != 1: fail("incompatible broadcasting shapes")
      outputShape[i] = if ad == 1: bd else: ad
  if count(outputShape) > 10_000_000: fail("interactive allocation limit is 10000000 elements")
proc arity(args: seq[Tensor[float64]], n: int) =
  if args.len != n: fail("expected " & $n & " arguments, got " & $args.len)
proc evaluate(s: Session, e: Expr, depth = 0): Tensor[float64]
proc call(s: Session, name: string, args: seq[Tensor[float64]]): Tensor[float64] =
  let function = name.split('.')[^1].toLowerAscii
  case function
  of "matrix", "vector":
    arity(args,1)
    let rank = if function == "matrix": 2 else: 1
    if args[0].rank != rank: fail(function & " requires rank " & $rank)
    result = args[0]
  of "tensor":
    if args.len == 1: return args[0]
    arity(args,2)
    let shape = dimensions(args[1])
    if count(shape) > 10_000_000: fail("interactive allocation limit exceeded")
    result = newTensor(args[0].toSeq, shape)
  of "random", "zeros", "ones":
    if args.len == 0: fail("constructor requires dimensions")
    var shape: seq[int]
    var seed = 1'u64
    if args[0].rank == 1:
      if args.len > 2: fail("expected shape and optional seed")
      shape = dimensions(args[0])
      if args.len == 2: seed = uint64(integer(args[1]))
    else:
      if args.len notin 2..3: fail("expected rows, cols, and optional seed")
      shape = @[integer(args[0]),integer(args[1])]
      if args.len == 3: seed = uint64(integer(args[2]))
    let n = count(shape)
    if n > 10_000_000: fail("interactive allocation limit is 10000000 elements")
    if function == "random": result = randomTensor(shape,seed)
    else:
      var data = newSeq[float64](n)
      if function == "ones":
        for x in data.mitems: x = 1.0
      result = newTensor(data,shape)
  of "eye", "identity":
    arity(args,1)
    let n = integer(args[0])
    if count(@[n,n]) > 10_000_000: fail("interactive allocation limit exceeded")
    result = tensor.fromMatrix(identity[float64](n))
  of "shape":
    arity(args,1)
    var data: seq[float64]
    for n in args[0].shape: data.add float64(n)
    result = newTensor(data,[args[0].rank])
  of "transpose":
    if args.len notin 1..2: fail("transpose requires tensor and optional axes")
    if args.len == 2: result = args[0].transpose(dimensions(args[1]))
    else: result = args[0].transpose()
  of "reshape":
    arity(args,2)
    let shape = dimensions(args[1])
    if count(shape) > 10_000_000: fail("interactive allocation limit exceeded")
    result = args[0].contiguous().reshape(shape)
  of "slice":
    if args.len notin 4..5: fail("slice(A,axis,start,stop[,step])")
    result = args[0].slice(integer(args[1]),integer(args[2]),integer(args[3]),if args.len == 5: integer(args[4]) else: 1)
  of "hadamard":
    arity(args,2)
    outputGuard(args[0],args[1],false)
    result = hadamard(args[0],args[1])
  of "dot", "cross":
    arity(args,2)
    if args[0].rank != 1 or args[1].rank != 1 or args[0].shape != args[1].shape:
      fail("dot/cross require equally sized rank-1 vectors")
    let a = newVector(args[0].toSeq)
    let b = newVector(args[1].toSeq)
    if function == "dot": result = scalar(dot(a,b))
    else: result = toTensor(cross(a,b))
  of "norm1", "norm2":
    arity(args,1)
    if args[0].rank != 1: fail("vector norm requires rank 1")
    let v = newVector(args[0].toSeq)
    result = scalar(if function == "norm1": normL1(v) else: normL2(v))
  of "sum":
    arity(args,1)
    var value = 0.0
    for x in args[0].toSeq: value += x
    result = scalar(value)
  else: fail("unknown function '" & name & "'")

proc evaluate(s: Session, e: Expr, depth: int): Tensor[float64] =
  if depth > 256: fail("evaluation nesting exceeds 256")
  case e.kind
  of exNumber: result = scalar(e.number)
  of exName:
    if not s.variables.hasKey(e.name): fail("unknown variable '" & e.name & "'")
    result = s.variables[e.name]
  of exNeg: result = scalarMul(s.evaluate(e.children[0],depth+1),-1.0)
  of exArray:
    if e.children.len == 0: return newTensor(newSeq[float64](),[0])
    var values: seq[Tensor[float64]]
    for child in e.children: values.add s.evaluate(child,depth+1)
    let childShape = values[0].shape
    for value in values:
      if value.shape != childShape: fail("ragged array: sibling shapes differ")
    var shape = @[values.len]
    shape.add childShape
    if count(shape) > 10_000_000: fail("interactive allocation limit is 10000000 elements")
    var data: seq[float64]
    for value in values: data.add value.toSeq
    result = newTensor(data,shape)
  of exBinary:
    let a = s.evaluate(e.children[0],depth+1)
    let b = s.evaluate(e.children[1],depth+1)
    outputGuard(a,b,e.op == '*')
    case e.op
    of '+': result = a+b
    of '-': result = a-b
    of '*':
      if a.rank == 2 and b.rank == 2: result = tensor.fromMatrix(matmul(a.toMatrix,b.toMatrix))
      else: result = hadamard(a,b)
    of '/': result = elementDiv(a,b)
    else: fail("unknown binary operator")
  of exCall:
    var args: seq[Tensor[float64]]
    for child in e.children: args.add s.evaluate(child,depth+1)
    result = s.call(e.name,args)

proc initSession*(): Session =
  ## Create an empty calculator session. Shape requirements: none.
  Session(variables: initTable[string,Tensor[float64]]())
proc getValue*(s: Session, name: string): Option[Tensor[float64]] =
  ## Inspect a stored tensor. Shape requirements: none; missing names return none.
  if s.variables.hasKey(name): some(s.variables[name].clone()) else: none(Tensor[float64])
proc helpText*(): string =
  ## List commands and expression grammar. Shapes are enforced by the listed APIs.
  "Tensor calculator: name = expression; + - * /, parentheses, nested arrays.\n" &
  "* is matrix multiplication for rank 2, otherwise broadcasting multiplication.\n" &
  "Functions: matrix(array), vector(array), tensor(data,shape), random(rows,cols[,seed]), random(shape[,seed]), zeros(shape), ones(shape), eye(n), shape(A), transpose(A[,axes]), reshape(A,shape), slice(A,axis,start,stop[,step]), dot(a,b), cross(a,b), norm1(a), norm2(a), sum(A).\n" &
  "Matrix.random(3,3) aliases random(3,3). vars, clear, help, quit/exit."
proc eval*(s: var Session, line: string): Option[string] =
  ## Evaluate an expression, assignment, or command. Sibling array shapes must
  ## match. Errors preserve prior variable values; callbacks/code are never run.
  let line = line.strip
  if line.len == 0: return none(string)
  if line.len > 65536: fail("input exceeds 65536 bytes")
  case line
  of "help": return some(helpText())
  of "clear":
    s.variables.clear()
    return some("variables cleared")
  of "vars":
    var names: seq[string]
    for name in s.variables.keys: names.add name
    names.sort()
    var lines: seq[string]
    for name in names: lines.add name & ": " & $s.variables[name].shape
    return some(lines.join("\n"))
  else: discard
  var p = Parser(tokens: tokenize(line))
  var target = ""
  if p.tokens.len >= 2 and p.tokens[0].kind == tkName and p.tokens[1].kind == tkMark and p.tokens[1].mark == '=':
    target = p.tokens[0].name
    if '.' in target: fail("assignment requires a plain variable name")
    p.pos = 2
  let expression = p.expression()
  if p.pos != p.tokens.len: fail("unexpected trailing input")
  let value = s.evaluate(expression)
  if target.len > 0:
    s.variables[target] = value
    result = some(target & " = " & $value)
  else: result = some($value)
proc runRepl*(input: File, output: File, prompt = true) =
  ## Run a line-based session. Expression shape requirements follow eval; errors
  ## are printed without terminating later input. EOF and quit/exit terminate.
  var session = initSession()
  var line: string
  if prompt: output.writeLine("Tensor calculator. Type help or quit.")
  while true:
    if prompt:
      output.write("> ")
      output.flushFile()
    if not input.readLine(line): break
    if line.strip in ["quit","exit"]: break
    try:
      let value = session.eval(line)
      if value.isSome: output.writeLine(value.get)
    except CatchableError as e:
      output.writeLine("error: " & e.msg)
