## Blocked, parallel and runtime-SIMD float64 kernels.
## Shape requirements: multiplication accepts A(M,K), B(K,N), returning C(M,N).
## SIMD pointers never escape a synchronous call and always address validated
## nonempty seq regions. Task workers return owned row tiles; shared inputs are
## read-only and the parent alone assembles the final output.
import std/os
import ./core
export core
when compileOption("threads"):
  import std/[tasks, cpuinfo]

when defined(matrixCalcScalar):
  {.passC: "-DMC_DISABLE_SIMD".}

const simdSource = currentSourcePath.parentDir / "simd_kernels.c"
{.compile: simdSource.}
proc cBackend(): cint {.importc: "matrix_calc_simd_backend", cdecl.}
proc cDot(a,b: ptr float64; n:csize_t):float64 {.importc:"matrix_calc_dot_simd",cdecl.}
proc cAxpy(a:float64;x,y:ptr float64;n:csize_t) {.importc:"matrix_calc_axpy_simd",cdecl.}

proc simdBackend*():string =
  ## Shape requirements: none. O(1); detects usable AVX/NEON, otherwise scalar.
  case cBackend()
  of 1:"avx"
  of 2:"neon"
  else:"scalar"
proc parallelEnabled*():bool =
  ## Shape requirements: none. True when built with --threads:on.
  compileOption("threads")
proc validate(a,b:Matrix[float64];blockSize:int=1) =
  if a.cols!=b.rows:raise newException(DimensionError,"accelerated multiplication requires A(M,K), B(K,N)")
  if blockSize<=0:raise newException(MatrixArithmeticError,"blockSize must be positive")

proc dotScalar*(a,b:openArray[float64]):float64 =
  ## Shape requirements: equal lengths; empty vectors return zero. O(N) time/O(1) space.
  if a.len!=b.len:raise newException(DimensionError,"dot requires equal lengths")
  for i in 0..<a.len:result+=a[i]*b[i]
proc dotSimd*(a,b:openArray[float64]):float64 =
  ## Shape requirements: equal lengths; AVX/NEON with unaligned and scalar tails.
  ## O(N) time/O(1) space. Reduction order may change floating-point roundoff.
  if a.len!=b.len:raise newException(DimensionError,"dot requires equal lengths")
  if a.len==0:return 0.0
  cDot(unsafeAddr a[0],unsafeAddr b[0],csize_t(a.len))

proc rowTile(a,b:seq[float64];aCols,bCols,first,count,blockSize:int):seq[float64] {.gcsafe.} =
  result=newSeq[float64](count*bCols)
  var kk=0
  while kk<aCols:
    let kend=kk+min(blockSize,aCols-kk)
    var jj=0
    while jj<bCols:
      let jend=jj+min(blockSize,bCols-jj)
      for r in 0..<count:
        for k in kk..<kend:
          let value=a[(first+r)*aCols+k]
          for j in jj..<jend:result[r*bCols+j]+=value*b[k*bCols+j]
      jj=jend
    kk=kend

proc matmulBlocked*(a,b:Matrix[float64];blockSize:int=32):Matrix[float64] =
  ## Shape requirements: A(M,K), B(K,N), blockSize>0. O(MKN) work/O(MN) result.
  ## Tiled loops preserve each output cell's ascending K accumulation order.
  validate(a,b,blockSize)
  let count=checkedSize(a.rows,b.cols)
  if count==0 or a.cols==0:return zeros[float64](a.rows,b.cols)
  let ad=a.toSeq;let bd=b.toSeq
  var output=newSeq[float64](count)
  var first=0
  while first<a.rows:
    let rows=min(blockSize,a.rows-first)
    let tile=rowTile(ad,bd,a.cols,b.cols,first,rows,blockSize)
    for i,value in tile:output[first*b.cols+i]=value
    first+=rows
  newMatrix(a.rows,b.cols,output)

when compileOption("threads"):
  type TileMessage = tuple[first: int, values: seq[float64]]
  proc sendTile(a,b:seq[float64];aCols,bCols,first,count,blockSize:int;
                output:ptr Channel[TileMessage]) {.gcsafe.} =
    ## Shape requirements: isolated inputs cover the assigned valid row range.
    var tile = rowTile(a,b,aCols,bCols,first,count,blockSize)
    output[].send((first:first,values:move(tile)))
  proc executeTask(task:ptr Task) {.thread.} =
    ## Shape requirements: task points to one shared-allocated, uniquely owned Task.
    ## The worker consumes and frees its task only after invocation completes.
    try: task[].invoke()
    finally:
      `=destroy`(task[])
      deallocShared(task)

  proc startTile(thread:var Thread[ptr Task];a,b:Matrix[float64];
                 first,rows,blockSize:int;output:ptr Channel[TileMessage]) =
    ## Shape requirements: valid row partition of conforming input matrices.
    ## Ownership transfers to the worker only after thread creation succeeds.
    let job=cast[ptr Task](allocShared0(sizeof(Task)))
    if job.isNil:raise newException(OutOfMemDefect,"could not allocate isolated task")
    var transferred=false
    try:
      job[]=toTask(sendTile(a.toSeq,b.toSeq,a.cols,b.cols,first,rows,blockSize,output))
      createThread(thread,executeTask,job)
      transferred=true
    finally:
      if not transferred:
        `=destroy`(job[])
        deallocShared(job)

proc matmulParallel*(a,b:Matrix[float64];blockSize:int=32;workerCount:int=0):Matrix[float64] =
  ## Shape requirements: A(M,K), B(K,N), blockSize>0. O(MKN) work plus task/copy overhead.
  ## With --threads:on, std/tasks transfers isolated copied inputs to at most
  ## eight native worker threads by default; workerCount=1..64 selects a limit.
  ## Channel results are owned and assembled only by
  ## the parent; no mutable Nim sequence is shared between workers. Without
  ## threads this explicitly falls back to matmulBlocked.
  validate(a,b,blockSize)
  if workerCount<0 or workerCount>64:raise newException(MatrixArithmeticError,"workerCount must be 0 (automatic) or 1..64")
  let count=checkedSize(a.rows,b.cols)
  if count==0 or a.cols==0:return zeros[float64](a.rows,b.cols)
  when compileOption("threads"):
    let workers=min(a.rows,if workerCount>0:workerCount else:max(1,min(8,countProcessors())))
    var output:Channel[TileMessage]
    output.open(workers)
    var threads=newSeq[Thread[ptr Task]](workers)
    var started=0
    var received=0
    var values=newSeq[float64](count)
    try:
      for worker in 0..<workers:
        let first=worker*(a.rows div workers)+min(worker,a.rows mod workers)
        let rows=a.rows div workers+ord(worker<a.rows mod workers)
        startTile(threads[worker],a,b,first,rows,blockSize,addr output)
        inc started
      while received<started:
        let message=output.recv()
        inc received
        for i,value in message.values:values[message.first*b.cols+i]=value
      result=newMatrix(a.rows,b.cols,values)
    finally:
      # All senders must finish before the stack-allocated channel goes away.
      for i in 0..<started:joinThread(threads[i])
      while received<started:
        discard output.recv()
        inc received
      output.close()
  else:
    result=matmulBlocked(a,b,blockSize)

proc matmulSimd*(a,b:Matrix[float64]):Matrix[float64] =
  ## Shape requirements: A(M,K), B(K,N). O(MKN) work/O(MN) storage.
  ## Runtime-dispatched row AXPY; full SIMD lanes and scalar tails never cross rows.
  validate(a,b)
  let count=checkedSize(a.rows,b.cols)
  if count==0 or a.cols==0:return zeros[float64](a.rows,b.cols)
  let ad=a.toSeq;let bd=b.toSeq
  var output=newSeq[float64](count)
  for r in 0..<a.rows:
    for k in 0..<a.cols:
      cAxpy(ad[r*a.cols+k],unsafeAddr bd[k*b.cols],addr output[r*b.cols],csize_t(b.cols))
  newMatrix(a.rows,b.cols,output)
