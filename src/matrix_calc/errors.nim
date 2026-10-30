## Catchable, explicit matrix validation failures, independent of compiler checks.
type
  MatrixError* = object of CatchableError
    ## Base class for recoverable shape and numerical errors.
  DimensionError* = object of MatrixError
    ## Operands do not satisfy an operation's shape requirements.
  ReshapeError* = object of MatrixError
    ## Dimensions and element counts disagree.
  MatrixIndexError* = object of MatrixError
    ## A logical coordinate or range is invalid.
  MatrixArithmeticError* = object of MatrixError
    ## Invalid scalar arithmetic, including zero division and overflow.
  NumericalError* = object of MatrixError
    ## Invalid/nonfinite input, singularity, or failed factorization.
  ConvergenceError* = object of NumericalError
    ## A bounded numerical iteration exhausted its sweeps.

proc checkedSize*(rows, cols: int): int =
  ## Shape: nonnegative rows and cols; checks product before any allocation.
  if rows < 0 or cols < 0:
    raise newException(ReshapeError, "matrix dimensions must be nonnegative")
  if rows != 0 and cols > high(int) div rows:
    raise newException(ReshapeError, "matrix element count overflows int")
  rows * cols
