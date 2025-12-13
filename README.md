# rsdsl_proc_macro_no_raw

A PoC Rust proc-macro DSL that parses a Redstone-like spec into a **fully structured AST**
(with **no "raw string" fallbacks**). Unsupported statements trigger compile errors.

Current coverage:
- `model Name { ... }`
- declarations: `index`, `enum`, `scenario`, `pin`, `fn`, `place/state/shape/sources`
- rule statements: `require`, `def <->`, `let`, `add`, `exclude`, `force`, nested `feature { ... }`
- expressions: `!`, `&`/`and`, `|`/`or`, `+`, `-`, `*`, comparisons (`==`, `<=`, `>=`, `<`, `>`),
  calls `f(...)`, var refs `X[...]`, `OR{...}`, `sum(binder) expr`

Run:
```bash
cargo run -p rsdsl --example not_gate
```
