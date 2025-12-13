# rsdsl (Redstone / Routing Solver DSL)

This repository contains two crates:

- `rsdsl` — runtime AST + SCIP `.lp` code generator + re-exported proc-macro
- `rsdsl_macros` — `rsdsl!{ ... }` proc-macro parser

## What you get
- A compact DSL to write ILP models for placement/routing/state-propagation problems.
- Deterministic lowering to SCIP LP format.
- A practical `sources` + `add` + `OR(...)` pattern to avoid gigantic OR expressions.

## Quick start

```bash
# from repo root
cargo run -p rsdsl --example not_gate > out.lp
scip -c "read out.lp optimize display solution quit"
```

## Docs
- See `SPEC.md` for the full language specification and lowering semantics.
- See `rsdsl/examples/not_gate.rs` for a complete model example.
