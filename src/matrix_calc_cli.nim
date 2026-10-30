## Executable tensor calculator. Shapes and errors are documented by repl.eval.
import std/terminal
import matrix_calc/repl
when isMainModule:
  runRepl(stdin,stdout,isatty(stdin))
