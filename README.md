# matrix_calc

A Nim library and interactive calculator for dense matrices, specialized vectors,
strided n-dimensional tensors and real numerical linear algebra. Tested with Nim
2.2.12, the C backend, ARC and ORC. The manifest retains the 2.0.8 minimum and CI
includes that version; the older compiler is not locally verified in this run.
Production code uses the standard library plus a small included C SIMD helper.

## Library usage

```nim
import matrix_calc
let a = newMatrix(2, 3, [1.0, 2, 3, 4, 5, 6])
doAssert (a * a.transpose).toSeq == @[14.0, 32, 32, 77]
doAssert a.slice(0, 2, 1, 3).shape == (2, 2) # half-open ranges
let x = newVector([1.0, 0, 0])
let y = newVector([0.0, 1, 0])
doAssert cross(x, y).toSeq == @[0.0, 0, 1]
doAssert newVector([3.0, 4]).normL2 == 5.0
let t = newTensor([1.0, 2, 3, 4, 5, 6], [2, 3])
let row = newTensor([10.0, 20, 30], [3])
doAssert (t + row).toSeq == @[11.0, 22, 33, 14, 25, 36]
echo t.transpose
```

`Matrix[T]` owns a private row-major buffer. Constructors copy inputs, assignment
has ordinary Nim value semantics, and public data access returns a copy. Shapes
and coordinates are explicitly checked even when compiler bounds/overflow checks
are disabled. Sizes use nonnegative `int`. `[]=`/`set` provide actual mutation;
`getMut` is only a legacy value accessor. `tryGet` returns an Option. Arithmetic,
Hadamard products, scalar operations, transpose, reshape, half-open/inclusive
slicing, row/column replacement, swaps, concatenation, diagonals, reductions and
powers are available. `matmulInto` uses a temporary to support aliased inputs.
Checked integer operations reject overflow explicitly. `/` promotes integer
inputs; `scalarQuotient` provides truncating integer division.

`Vector[T]` has direct 1-D arithmetic/dot/cross kernels, L1/L2 norms and stable
normalization, with row/column Matrix and Tensor conversions. Cross requires
length three. Norms convert numeric elements to float64.

## Tensors, broadcasting and views

`newTensor(data, shape)` stores dimensions, element strides and an offset. `[]`
is a rank-zero scalar shape with exactly one element; any zero axis is empty.
Trailing broadcast axes must be equal or one must be 1. Matrix arithmetic still
requires exact compatible dimensions. Tensor `*` is elementwise scaling/product;
Matrix `*` is matrix multiplication.

Transpose/permutations and slices retain backing storage. **Nim Tensor handles
and views alias element writes**; `clone`/`contiguous` and Matrix/Vector conversions
make independent copies. Layouts do not resize the backing buffer, so views remain
valid after the original variable is rebound or destroyed. There are no raw
pointer views. This differs from Rust Tensor's copy-on-write mutation. Mutating
aliased views concurrently is not implicitly thread-safe.

Use `get`/`set` with one coordinate per axis, `shape`, `strides`, `isContiguous`,
`toSeq`, `transpose`, `slice(axis,start,stop,step)`, and `reshape`. Positive-step
ranges are half-open; negative steps additionally support reversal. Reshaping a
noncontiguous view explicitly copies logical order (Rust requires making it
contiguous first). Small values pretty-print nested arrays; large or deep tensors
use bounded previews. Invalid shapes/ranges raise catchable errors.

## Numerical linear algebra

```nim
import matrix_calc
let a = newMatrix(2, 2, [3.0, 1, 1, 2])
let b = newMatrix(2, 1, [9.0, 8])
let factor = a.lu
let solution = factor.solve(b)
doAssert solution.approxEq(newMatrix(2, 1, [2.0, 3]), 1e-12, 1e-12)
let fit = newMatrix(1, 2, [1.0, 1]).leastSquares(newMatrix(1, 1, [2.0]))
doAssert fit.solution.approxEq(newMatrix(2, 1, [1.0, 1]), 1e-12, 1e-12)
```

float32/float64 algorithms include partial-pivoted LU, transposed/triangular
solves, determinant/sign-log magnitude, inverse, Cholesky, compact Householder QR
with column pivoting and Q application, one-sided Jacobi SVD, pseudoinverse,
rank/condition/spectral norm, symmetric Jacobi eigenpairs and least squares.
Factors are reusable and accept multiple RHS columns. Auto least squares uses QR
for full-column-rank tall matrices and minimum-norm SVD otherwise.

Conventions: `P*A=L*U`, `A*P=Q*R`, `A=U*diag(s)*V^T`, `A*V=V*diag(eigenvalues)`.
Singular values descend; eigenvalues ascend. Solvers reject nonfinite data and
unrepresentable results. Default relative thresholds are `8*max(dimension,1)*eps`;
rank is tolerance-dependent. Iterations are bounded and report nonconvergence.
No normal equations are used. Global scaling cannot preserve ratios beyond the
scalar's dynamic range; ill-conditioned forward error can be large despite small
residuals. Empty shapes preserve dimensions; 0×0 determinant is one. The usual
matrix, vector and tensor shape requirements are included in `##` docstrings.

## Interactive calculator

```sh
nim c -d:release --path:src --nimcache:.tools/cli-cache --out:.tools/matrix_calc src/matrix_calc_cli.nim
.tools/matrix_calc
```

Example session:

```text
A = [[1,2],[3,4]]
B = A * transpose(A)
C = A + [10,20]
r = Matrix.random(3,3,7)
v = cross([1,0,0],[0,1,0])
shape(B)
vars
quit
```

Numbers, nested rectangular arrays, assignments, parentheses and `+ - * /` are
supported. Runtime values are float64 tensors. `*` performs matrix multiplication
for two rank-two values and otherwise broadcasts elementwise; `hadamard` is
explicit elementwise multiplication. Functions include `matrix`, `vector`,
`tensor`, seeded `random`, `zeros`, `ones`, `eye`, `shape`, `transpose`, `reshape`,
`slice`, `dot`, `cross`, `norm1`, `norm2`, and `sum`. `help` lists signatures;
`vars`, `clear`, `quit`/`exit` manage state. EOF exits cleanly. Errors preserve
existing variables and allow subsequent lines to execute. The grammar does not
execute Nim/host code. Interactive input/depth/output limits prevent accidental
giant expressions. Seeds reproduce values within each language, not across their
different PRNG algorithms.

## Blocked, parallel and SIMD execution

```nim
import matrix_calc/accelerated
let a = ones[float64](128, 128)
let b = identity[float64](128)
doAssert matmulBlocked(a, b, 32) == a
doAssert matmulParallel(a, b, 32, workerCount = 4) == a
doAssert matmulSimd(a, b) == a
```

Products require A(M,K), B(K,N), positive tile sizes. `std/tasks` isolates copied
input buffers for bounded worker threads; owned channel results are assembled by
the parent. Worker count defaults to at most eight, or choose 1..64 explicitly.
Compile with `--threads:on`; without threads, the parallel API explicitly uses a
serial blocked fallback. `parallelEnabled` exposes that distinction. Runtime
AVX/NEON intrinsics have unaligned/tail handling and scalar fallback. Compile with
`-d:matrixCalcScalar` to force and test portable dispatch. Only the included
synchronous C wrapper and private task transport use pointers, with bounded
lifetimes; tensor views do not. AVX is exercised on x86; ARM execution is not
claimed. SIMD dot reduction order may change floating-point roundoff.

## Tests, benchmarks and documentation

```sh
nimble test
nimble testRelease
nimble testArc
nimble testGuards
nimble docs
bash tools/sanitize.sh
tools/setup_benchy.sh
nimble benchmark
```

Benchy is benchmark-only and pinned by `tools/setup_benchy.sh`. Full benchmarks
cover 128², 512² and 1024² matrices. `ROADMAP_MODE=verify` checks/times one pass,
`quick` selects 128², and `ROADMAP_SIZES`/`ROADMAP_THREADS` control the workload.
See [roadmap evidence](docs/ROADMAP.md), [profiling instructions](docs/PROFILING.md)
and the actual [sampled flamegraph](docs/profiling/flamegraph.svg). The test suite
includes generated layouts and float32/64 numerical identities; 114 cases are
also compared with Rust and independently with NumPy using `tools/verify_parity.py`.
See [verification details](VALIDATION.md) for executed configurations and limits.

Install with Nimble in a normal writable environment. The public entry point is
`import matrix_calc`; `import lib` is a source compatibility re-export. Package
metadata passes `nimble check`. The cloud runner's Nimble installation attempts
to create a build-temporary directory in its read-only home, so installed-package
verification there uses a staged archive/consumer check rather than claiming a
successful Nimble install. The CLI itself is compiled and exercised separately.

## License

Copyright (c) 2026 farvanyoshi <nvnjoshi@outlook.com>.
Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
