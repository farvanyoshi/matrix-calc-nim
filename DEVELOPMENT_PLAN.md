# Nim matrix library completion plan — awaiting owner review

## Scope and working agreement

Bring the Nim library to functional parity with the completed Rust library at
`farvanyoshi/matrix-calc-rust`, commit
`d3aba2cf6a068b5c0df68aac49f15ec03bb4fee1` (version 0.2.0). This is a pinned
reference, not a promise to track future Rust changes automatically. Public
capabilities and mathematical contracts should match; language syntax, ownership,
error idioms, and defaults may differ when explicitly documented.

Only the Nim repository is changed. Work takes place on `dev`, based on Nim
commit `ca2228e`. All commits use author and committer
`farvanyoshi <nvnjoshi@outlook.com>`, with no additional attribution trailers.
Use small coherent commits; optional isolated branches use `feature/<topic>`.
Do not rewrite history. The current deliverable is planning documents only.
Implementation starts after the owner reviews and approves this plan. No remote
push or package publication is performed during this planning pass.

The Rust reference is read-only. Future reference exporters, fixtures, development
oracles, build scripts, and generated tooling belong in the Nim repository.
Apply the previously selected dual `MIT OR Apache-2.0` licensing during the
approved implementation: full texts, matching metadata and documentation, and
copyright attributed to farvanyoshi. The existing Nim manifest currently says MIT;
this planning pass does not change it or install new dependencies.

The target is a generic dense CPU library with float32/float64 numerical linear
algebra. Sparse storage, complex decompositions, general nonsymmetric eigen,
GPU/native BLAS backends, arbitrary precision, and a calculator/REPL are follow-on
work, matching the Rust release scope.

## Current state

`src/lib.nim` contains the intended matrix type, constructors, element access,
mapping, arithmetic, and multiplication. Most generic constructors/arithmetic
fail when instantiated. Problems include signed/unsigned dimension comparison,
invalid positional object construction, nonexistent `Matrix.new` calls, mapping
function objects instead of applying them, undeclared names in multiplication's
error message, and incorrect mutable-access/division return types. Public fields
also allow dimensions and storage length to diverge. Coordinate flattening does
not independently validate rows and columns. The exception types are private.

`src/matrix_calc.nim` only prints `hello`; the tests directory contains path
configuration, not a library test suite. Building that executable does not verify
these generic functions. The manifest requires Nim >=2.0.8; the prepared compiler
is 2.2.12. Both the declared minimum and current supported release must be tested
before retaining that compatibility claim.

## Design decisions proposed for approval

### Public modules and types

Provide `import matrix_calc` as the library facade, exporting focused modules:
`matrix_calc/core`, `errors`, `arithmetic`, `structure`, `views`, `numeric`,
`linalg/{lu,cholesky,qr,svd,eigen,least_squares}`, and optional `json`.
Keep `lib.nim` as a documented compatibility re-export during migration. Remove
the placeholder executable's binary declaration and turn its module into the
library facade. Existing spelling aliases must account for Nim's style-insensitive
identifiers: `new_matrix` and `newMatrix` are not two distinct APIs.

Use `Matrix[T]` with private dimensions and storage. Sizes and indices use native
`int`, matching seq/openArray lengths. Reject negative dimensions/indices and
check element-count and byte-capacity arithmetic before allocation or indexing.
Support generic storage, including strings and user-defined element types;
require numeric operations only where actually needed. Numerical decompositions
accept float32 and float64 (`float` is the platform alias, not a third precision).
Do not silently convert integer matrices into numerical fields.

### Copies, views, and mutation

First validate this design in focused compiler/runtime tests before building
arithmetic on it:

- Owned matrices have value-copy behavior for their element buffer. Assignment
  and explicit clone must not accidentally share a mutable element buffer.
  Custom copy/sink hooks may use private reference-counted backing storage;
  copying explicitly duplicates elements using T's normal copy semantics.
  Elements that are themselves ref objects retain their documented Nim reference
  semantics; matrix copying does not promise a recursive clone of object graphs.
- Views retain a private backing-storage reference and checked shape/stride/offset
  metadata. They keep elements alive independently of the owner's variable and
  deliberately alias element mutations. Mutable and immutable handles are not
  claimed to have Rust's static exclusivity guarantees.
- Structural mutation of backing storage advances a generation counter. Every
  view operation checks its generation before access and raises a typed stale-view
  error when invalidated. Check generation-counter overflow too. Rebinding an
  owner to a different matrix leaves views attached to the old retained storage.
- Do not expose a mutable seq header, resizable raw storage, escaping `var T`
  references, or raw-pointer views. Use `[]=`/setters, view setters, and explicit
  transformation procedures for mutation. Iteration yields values or retained
  views; mutable traversal uses setters/callback transformations with documented
  alias rules. The old `get_mut` is replaced with actual supported mutation APIs.
- Define copy/sink/self-assignment behavior and initialization order, including
  default zero-initialized matrices, before implementation. Keep storage handles
  valid through moves and ARC/ORC destruction.
- Output-buffer operations reject overlapping backing-storage regions, or use an
  explicit temporary when overlap is allowed; test this before any write. Safe
  multiple-view aliasing must not silently corrupt multiplication/block copies.

Acceptance of this stage requires copy-independence, view-lifetime, stale-view,
move/self-assignment, overlap, and empty-view tests under ARC and ORC. If a proposed
copy hook cannot meet that contract across supported compilers, stop that design
and document a safe alternative before extending it. Borrowing conveniences must
not be implemented by returning pointers into a movable seq. Concurrent mutation
through shared views is not implicitly thread-safe.

### Errors and naming

Use idiomatic exported `CatchableError`-based matrix exceptions for ordinary
invalid input. Include structured kinds/metadata for dimensions, indices/ranges,
negative/overflowed sizes, invalid conversions, zero division, arithmetic overflow,
nonfinite data/results, singularity, deficiency, symmetry/positive definiteness,
invalid options, nonconvergence, stale views, and forbidden overlapping output.
Do not use compiler Defects as the normal validation API or catch arbitrary
Defects to disguise errors. Do not promise recoverable out-of-memory allocation.

`[]` is checked access and `[]=` is checked mutation; `get` returns `Option[T]`.
Primary operations raise typed exceptions; do not duplicate every function solely
to imitate Rust's `Result` and `try_*` naming. Mapping accepts closures, supports
captured state, and propagates callback exceptions without swallowing them.
Validate shape and other deterministic input errors before invoking callbacks or
mutating outputs. Failed checked in-place arithmetic leaves its destination
unchanged. Document partial mutation if user element code/callbacks throw.

Use camelCase consistently, with compatibility exports only where meaningful.
Add an API mapping/migration guide rather than reproducing Rust aliases or
trait-specific operand combinations mechanically.

### Arithmetic and empty shapes

Keep row-major shape invariants and no implicit broadcasting. Matrix `*` is matrix
multiplication; Hadamard multiplication is named. Provide scalar operations,
negation, assignments, output-buffer multiplication, explicit conversions, and
callback transforms. Preserve operand numeric types unless an operation explicitly
documents promotion.

For division, use Nim's idioms deliberately: `/`, `scalarDiv`, and `elementDiv`
follow floating division (integer inputs promote to a documented float result).
Provide `scalarQuotient`/`elementQuotient` for truncating integer division and
checked integer quotient variants matching Rust's integer behavior. Reject zero
divisors consistently, including signed floating zero and a zero scalar applied
to an empty matrix. Test promotion/result inference by instantiating each overload.

Checked integer arithmetic uses explicit pre-operation range checks or supported
public primitives. It must work with compiler overflow checks disabled; signed
minimum divided by minus one, negation of minimum, unsigned subtraction, and
intermediate matrix-product overflow need tests. Cover Nim's native signed/unsigned
widths through 64 bits. Rust i128/u128 have no direct built-in Nim counterpart and
are an explicit type-coverage difference, not a hidden feature promise.

Support distinct 0×0, m×0, and 0×n shapes. Empty row/column iteration has logical
length; `(m×0)*(0×n)` gives m×n zeros. Sum/product identities are zero/one.
Min/max/mean of an existing empty reduction group are errors. Row reductions
produce m×1, column reductions 1×n. Trace/powers require square input. For 0×0,
trace is zero, determinant one, inverse 0×0. Empty SVD has rank/norm zero and
condition number one; a nonempty zero spectrum has infinite condition number.

### Numerical policy

Match the Rust algorithms already delivered, rather than the earlier superseded
bidiagonal/tridiagonal proposal: scaled partial-pivoted LU, Cholesky, Householder
QR with optional column pivoting, one-sided Jacobi SVD, and two-sided symmetric
Jacobi eigen. Preserve factor orientation and permutation conventions.

Use validated finite nonnegative absolute/relative tolerances. Initially use the
Rust dimension-aware defaults and sweep limit to simplify comparison, while
allowing documented future Nim-specific defaults. Keep SVD rank truncation
separate from convergence. Reject nonfinite solver inputs and unrepresentable
computed factors/solutions. Enforce explicit sweep limits/nonconvergence errors.
Use scaled hypot-like norms, compensated accumulation, stable reflector/rotation
construction, and exponent-aware rescaling of projected solves. Avoid overflowing
ratios whose final solution is representable. Do not form normal equations or
solve systems through an explicitly computed inverse. Ordinary scalar operations
and solver error policies remain distinct.

## Implementation milestones and acceptance gates

### 1. Establish a real core test workflow and fix the baseline

Create unittest suites that instantiate every existing generic operation for
integers and floats, plus a nonnumeric storage case. Repair construction, mapping,
coordinate validation, mutation, multiplication diagnostics, and division types.
Introduce explicit error guards and empty-shape tests. Export usable error types.
Tests must execute actual numerical results and error paths; importing generics
or building the greeting program does not count.

Gate: each original intended operation works or has a documented replacement;
known bad accesses, shapes, negative sizes, and dimension overflow are rejected
in debug and release. Assert complete values and dimensions, not merely success.

### 2. Settle storage ownership, public API, and package structure

Implement the ownership/view foundation described above. Add validated openArray,
seq/nested-row, coordinate-callback, filled, zeros, ones, identity, diagonal,
conversion and clone constructors. Default constructors copy caller storage;
transfer/zero-copy construction is added only where ownership can be demonstrated.
Add shape/size predicates, checked get/set, row/column access and traversal,
closure mapping/zipped mapping, range/exact casts, display, and consistent
arithmetic/operator/assignment forms. Require explicit immutable storage copies
or safe traversal rather than exposing the private seq for resizing.

Gate: invariants survive copies/moves/default initialization, mutable traversal,
views, callback failure and conversions under the supported compiler/memory-manager
matrix. Document source-breaking changes. Update the facade/Nimble package layout
without leaving a nonfunctional binary advertised as a REPL.

### 3. Complete matrix structure and generic operations

Add transpose (owned and view), square in-place transpose, checked reshape,
flatten/nested conversion, slicing/extraction, row/column replacement and swaps,
diagonals, concatenation and block insertion. Add vector dot/outer products,
matrix-vector multiplication, trace, sum/product, min/max/argmin/argmax, row/column
reductions, predicates, and nonnegative matrix powers by squaring. Support
mathematical operations on views through shared kernels rather than mandatory
whole-matrix copies. Match empty-group semantics and first row-major tie breaking.

Gate: transpose-twice/reshape identities, rectangular products, view versus owned
results, output alias rules, unaffected-region mutation, reductions, powers, and
invalid ranges all pass. Nonnumeric transforms do not acquire numeric constraints.

### 4. Complete integer checking and real numerical foundations

Add all checked element/scalar/negation/quotient/matrix-product operations and
atomic checked assignment equivalents from the parity table. Add exact/range
casts, finite checks, mixed absolute/relative comparison, compensated sums/dots,
stable means, vector 1/2/infinity norms, matrix Frobenius/1/infinity norms,
normalization, elementwise abs/sqrt/exp/log/power, and tolerance/options types.

Gate: exercise each integer width and float precision, zeros, signed zero,
cancellation, extreme magnitudes, invalid casts/options, nonfinite values, and
norm overflow avoidance. Checked integer results/errors are independent of
compiler debug overflow/range settings. Approximate equality never replaces
ordinary exact equality and never treats NaN as equal.

### 5. Square-system factorizations and solves

Implement reusable lower/upper triangular solves (including implicit-unit
diagonals), scaled partial-row-pivoted LU, permutation inspection, lower/upper
factors, repeated/multiple-RHS solves, transposed solves, determinant,
sign/log-absolute-determinant, inverse, and Cholesky with solves/log determinant.
Specify `P*A=L*U` and `A=L*L^T`; keep determinant-zero handling distinct from
numerical solve thresholds. Guard unrepresentable exposed factors.

Gate: analytic fixtures and reference comparisons; factor reconstruction,
permutation parity, pivot-required systems, residual checks, transpose solves,
singular/nearly singular input, symmetry/positive-definiteness failures, empty
systems, and independently scaled A/RHS all pass for float32/float64.

### 6. Rectangular solving through QR

Implement compact Householder reflectors, optional column pivoting, economy/full
Q and R, Q/Q^T application without materialization, and reusable full-column-rank
tall least squares. Use `A*P=Q*R`, with original column indices in the permutation.
Scale large RHS before reflector application. An unpivoted diagonal count is only
an estimate; numerical rank queries should use SVD, or pivoted QR where intended.
Explicit QR solving rejects wide/deficient input rather than inventing a solution.

Gate: rectangular reconstruction, orthogonality, permutation recovery, residual
optimality, large RHS, compact reflector application, deficient/wide rejection,
and multiple RHS pass. Verify both economy/full and empty factor dimensions.

### 7. Advanced numerical parity

Implement bounded scaled one-sided Jacobi SVD directly on A, with wide-input
transposition and orthonormal completion for exact null columns. Expose descending
nonnegative singular values, economy U/s/V^T, sweep counts, numerical rank,
spectral norm, condition number, and pseudoinverse. Add SVD minimum-norm solves,
rank-deficient/underdetermined least squares, automatic QR/SVD selection and
explicit algorithm/options selection. Return solution, resolved tolerance,
method, rank, and combined Frobenius residual. Add symmetric Jacobi eigenpairs,
ascending eigenvalues/column eigenvectors, and negative matrix powers via inverse.

Gate: reconstruct factors, verify orthogonality and all four Moore–Penrose
identities, minimum-norm properties and eigen residuals. Exercise repeated spectra,
null spaces, extreme scales, scale-ratio overflow, invalid/nonfinite data, and
forced nonconvergence. Compare subspaces/identities instead of requiring the same
signed eigenvectors/singular vectors from both languages.

### 8. Serialization, performance, docs, and release preparation

Add optional std/json-based serialization using Rust's `{rows,cols,data}` row-major
schema, checked dimensions/counts and type-appropriate element codecs. Reject
malformed dimensions/counts, unsupported numeric ranges and nonfinite JSON values;
do not silently truncate integers. Document std/json integer limits and provide
explicit codecs where necessary for wider values or custom T. Serialization must
not become a mandatory production dependency through the main facade.

Benchmark allocating/output-buffer and owned/view paths, copies, transpose,
rectangular multiplication, decompositions and factor reuse. Use deterministic
inputs, warmup, sufficiently long repeated samples, and sizes comparable to the
Rust measurements, including 256×256 multiplication. Reuse buffers and optimize
measured bottlenecks only; preserve numerical gates. Keep formatting, examples,
Nimble tasks, migration notes, licensing, numerical limitations, compatibility
matrix, CI and package-install smoke tests in this stage's deliverables.

Gate: documented examples execute, a fresh external consumer imports/uses the
installed package without repository-relative paths, optional JSON works, and
source/install file lists contain the modules and licenses but no local caches.
Record actual benchmark/toolchain results and any limitations. No package registry
publication is implied by verification.

## Cross-language parity and independent validation

Maintain `API_PARITY.md` as a required capability checklist. During implementation,
mark each row with its concrete Nim symbol, tests, status and justified semantic
differences. Every planned capability must be implemented or explicitly approved
as a difference; missing features cannot be hidden by differing method names.

Add versioned fixtures inside the Nim repo with input shape/data/type, operation,
explicit options, expected output/error category, reference SHA and deterministic
seed. A development-only Rust exporter in the Nim tests/tools directory can depend
on the pinned remote Rust revision and generate reference results without modifying
the Rust checkout. Its lockfile/fixture provenance are committed; its build caches
are ignored. Ordinary Nim tests consume stored fixtures without needing Rust or
network access. A dedicated CI parity job regenerates or checks them explicitly.

Supplement Rust agreement with analytic examples, invariant/property checks, and
an independent development oracle (for example nalgebra in that same test harness).
Porting the same algorithm can port the same bug. Compare integer results exactly
on common widths and matching quotient operations. Compare floating results with
justified norm-scaled residual/backward-error bounds, using explicit tolerances
when defaults differ. Compare error categories rather than strings. Test factor
orientations, repeated spectra, empty shapes and scalar-result types separately.

## Verification strategy and definition of done

Keep focused unit tests and public-facade integration tests alongside each
milestone. Include reproducible seeded generation at least covering the Rust
categories: 128 structural, 128 SVD/QR/least-squares, 96 LU/Cholesky/eigen and 64
single-precision cases. Counts alone are not evidence: retain boundary, analytic,
callback/alias failure and numerical-identity tests too.

Define tested Nimble tasks for core/full/parity tests, examples, docs, benchmarks
and package smoke checks. Run meaningful tests under debug and release with
assertions/test checks demonstrably active, plus guard tests with compiler
arithmetic/bounds checks disabled. Avoid `-d:danger` as a workaround for failures.
Use the C backend, ARC and ORC, and the declared minimum compiler (2.0.8) plus the
supported 2.2 line. Keep the minimum pin only if demonstrated. Run C-backend
AddressSanitizer/UndefinedBehaviorSanitizer checks for view/copy/lifetime tests
where supported; Nim has no direct Miri equivalent. Document platform limitations
and do not claim unrun configurations passed. Do not use fast-math flags that
invalidate IEEE or finite-value checks.

Final review re-runs agreed gates, validates public import/package installation,
checks dependency/license metadata, audits author/committer identities, and
confirms the Rust checkout is unchanged. Record commands, versions, test counts,
passed/failed/skipped outcomes, fixtures, intentional differences and benchmark
results in a Nim VALIDATION.md. Keep `dev` green after the initial repair. Push
only the reviewed/authorized branch when that phase is reached; do not equate a
push with Nimble registry publication or Rust changes.

## Status

- [x] Inspect Nim baseline and pin the Rust parity target.
- [x] Create Nim `dev` and configure the requested local commit identity.
- [x] Write the proposed design, milestones, parity matrix and validation strategy.
- [ ] Owner reviews/approves this plan.
- [ ] Implement milestones 1–8 with their acceptance gates.
- [ ] Complete final cross-language/independent validation and documentation.
- [ ] Push the approved implementation branch when authorized for that phase.
