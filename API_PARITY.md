> Historical planning record. The later tensor roadmap request supersedes its
> scope/order/approval status; current implementation evidence is in [docs/ROADMAP.md](docs/ROADMAP.md).

# Proposed Rust → Nim capability mapping

Reference: Rust version 0.2.0 at `d3aba2cf6a068b5c0df68aac49f15ec03bb4fee1`.
All rows below are **planned, not implemented/verified**. The current Nim facade
is a placeholder; the final symbol spelling is settled alongside type/ownership
proofs in milestone 2. This table tracks capability, not a literal translation of
Rust traits, aliases, or Result-returning signatures.

| Rust capability | Proposed Nim API / behavior | Gate |
| --- | --- | --- |
| Generic owned Matrix<T> | Private `Matrix[T]`, int dimensions, owned buffer-copy semantics | Copy/sink/self-assignment, default initialization, negative/overflow sizes |
| new/from_vec/from_rows/from_fn/filled | `newMatrix`, `fromRows`, `fromFn`, `filled`; validated openArray/seq input | Count/ragged errors, closure state, nonnumeric elements |
| zeros/ones/identity | `zeros`, `ones`, `identity` | Integer/float values and empty shapes |
| shape/rows/cols/len/is_empty/is_square | `shape`, `rows`, `cols`, `len`, `isEmpty`, `isSquare` | Logical versus storage dimensions |
| slices/into_vec/clone | Safe traversal, explicit `toSeq`/clone and optional proven ownership transfer | No mutable seq-header escape or hidden buffer alias |
| get/get_mut/indexing/set | `get: Option[T]`, checked `[]`/`[]=`, `set`; no escaping mutable element reference | Both coordinates, error types, actual mutation |
| element/row/column iteration | Value iterators and retained row/column views; explicit mutation operations | Empty groups, traversal order and length, alias/lifetime rules |
| map/try_map/zip_map | Closure-capable `map`/`zipMap`; callback exceptions propagate | Result-type inference, callback count/short circuit, shape-first validation |
| range/exact casts | `convertTo`/`convertExact` or equivalent checked conversions | Range, fractional loss, NaN/infinity, signed/unsigned boundaries |
| matrix/scalar arithmetic and negation | `+`, `-`, `*`, unary `-`; named scalar methods and assignments | Shape checks, input independence, output types |
| Hadamard product | `hadamard` | Equal-shape requirement and expected values |
| floating scalar/element division | `scalarDiv`, `elementDiv`, `/`; integer input promotes explicitly | Result precision documented, zero and signed zero rejection |
| Rust integer division | `scalarQuotient`, `elementQuotient` | Truncation toward zero and minimum/-1 behavior |
| checked integer operations | `checkedAdd/Sub/Hadamard/Negate`, scalar/quotient/product and assignment equivalents | All native widths through 64 bits, checks-disabled builds, atomic failure |
| matmul/matmul_into/matvec | Matrix `*`, `matmul`, `matmulInto`, `matvec` | Rectangles, zero inner dimension, overlap policy, buffer reuse |
| transpose/into/in-place | Owned/view transpose; explicit moved variant if safe; square in-place transpose | Round trip, nonnumeric elements, view sharing |
| reshape/flatten/nested conversion | Checked reshape and row-major conversions | Order, copy/mutation rules, failure leaves shape unchanged |
| slicing and views | Retained strided read/write views, checked generation, half-open bounds | Lifetime after owner destruction/rebinding, invalidation, no dangling pointers |
| row/column replacement/swaps | `setRow`, `setCol`, `swapRows`, `swapCols` | Validate full request before writes |
| diagonal and block structure | Diagonal extraction/construction, `hcat`, `vcat`, `insertBlock` | Dimension overflow, empty ranges, overlapping inputs |
| dot/outer | `dot`, `outer` on row/column vectors | Equal lengths where required, empty vectors, orientation |
| sums/products/trace | Generic whole/axis reductions and square trace | Correct identities, row m×1 / column 1×n shapes |
| extrema/arg extrema | Whole/axis min/max, first row-major argmin/argmax | Empty or unordered groups, ties |
| matrix powers | Nonnegative squaring algorithm; negative real powers via inverse | Square constraint, exponent zero, signed exponent limits |
| real predicates/comparison | Finite checks, maxAbs, symmetry, explicit approximate comparison | NaN/infinity, overflow-safe tolerance comparison |
| compensated sums/dots/means | Scaled compensated kernels and whole/axis means | Cancellation, large/small scales, empty groups |
| vector/matrix norms | Vector 1/2/inf, matrix Frobenius/1/inf; normalization | Stable scaling, zero vector, distinction from spectral norm |
| floating element transforms | abs/sqrt/exp/log/power functions | IEEE policy and documented result types |
| triangular systems | Lower/upper, implicit-unit diagonal, multiple RHS | Ignored-triangle convention, pivots and transpose usage |
| LU | Reusable partial-row-pivoted factors, permutations, P/L/U | P*A=L*U; scaled inputs and row swaps |
| square/transposed solves | Factor reuse, solve, transposed solve, multiple RHS | Norm-scaled residuals; singular and nonsquare errors |
| determinant/inverse | Determinant, sign/log magnitude, inverse through solves | Parity, 0×0, tiny determinant versus singular threshold |
| Cholesky | Reusable SPD factor, solve, determinant/log determinant | Symmetry, positive definiteness, A=L*L^T |
| QR | Compact Householder and pivoted forms; full/economy Q/R; apply Q/Q^T | A*P=Q*R, rank-estimate limitations and large RHS |
| least squares | Auto / explicit QR or SVD; diagnostics and multiple RHS | Tall, wide, deficient, minimum-norm and residual behavior |
| SVD | Scaled one-sided Jacobi, economy U/s/V^T, sorted values, bounded sweeps | Reconstruction, orthogonality, null completion, nonconvergence |
| rank/norm2/condition | SVD-based numerical rank, spectral norm and condition | Explicit thresholds, zero/empty spectra |
| pseudoinverse | Tolerance-truncated SVD pseudoinverse | All four Moore–Penrose identities |
| symmetric eigen | Two-sided Jacobi; ascending values and column eigenvectors | A*V=V*Lambda, repeated values, orthogonality |
| options/errors/diagnostics | Validated tolerances/limits, exported typed exceptions, factor and fit metadata | Shape, numeric, overlap/stale-view and convergence failures |
| serde serialization | Optional std/json module with `{rows,cols,data}` schema and element codecs | Cross-language round trip, numeric ranges, malformed shape rejection |
| formatting/docs/examples | `$`, API docs, migration/numerics guides and runnable examples | Clean public-facade import and documented differences |
| CI/benchmarks/packaging/license | Nimble tasks, debug/release + ARC/ORC, sanitizers, verified install, dual license | Recorded outcomes and package contents |

Accepted differences proposed for review:

1. CamelCase, typed exceptions and Option accessors replace a mechanical Rust
   spelling/Result/trait mapping.
2. Native int sizes replace unsigned size parameters; negative input is rejected.
3. Floating integer-input division follows Nim promotion; explicit quotient
   operations provide Rust-like integer results.
4. Retained views can alias in Nim and use generation validation. They do not
   claim Rust's compile-time exclusive borrowing. Normal matrix buffer copies
   remain independent, while reference-typed elements retain Nim semantics.
5. Primitive checked integer coverage stops at Nim's native widths through 64
   bits; there is no built-in i128/u128 promise.
6. JSON is an optional standard-library module, with documented element codec and
   integer-range rules, instead of a serde feature flag.

All other rows require equivalent useful functionality. Default differences must
be documented and overridden explicitly in comparison fixtures. Neither matching
method names nor matching Rust outputs alone establishes numerical correctness.
