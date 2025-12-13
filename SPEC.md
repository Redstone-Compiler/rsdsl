# RSDSL Full Spec (v0.1)

This document defines **RSDSL** (Redstone / Routing Solver DSL): a small DSL that compiles into a **0/1 ILP (SCIP .lp)**.
It is designed to express **placement + connectivity + state-propagation** problems with:
- **Domain-indexed binary variables**
- **Quantified rules** (`forall`)
- **Boolean definitions** (`def X <-> expr`)
- **Source-sets** (`sources`) accumulated via `add/exclude` and later materialized by `OR(SourceSet[...])`
- **Feature gating** (`feature NAME { ... }`)
- **Observed pins** (`Observe(PIN, s=...)`) as externally-provided binary variables

The runtime crate (`rsdsl`) provides:
- an AST (`ModelSpec`)
- a proc-macro (`rsdsl!{ ... }`) that parses the DSL into that AST
- a code generator (`codegen_scip_lp`) that lowers the AST into a `.lp` file

---

## 1. Core concepts

### 1.1 Domains
A **domain** is a finite set of symbols used as indices for variables:
- Built-in: `Cell` (from the instance), plus any `enum` or `scenario` you declare.
- Domains are referenced by name in variable declarations and quantifiers.

**Domain values** are just identifiers (e.g. `GROUND`, `N`) or cell-ids like `x0_z0`.

### 1.2 Variables
Every declared variable is **binary** (`0/1`).

A variable declaration specifies:
- its **kind** (used by compilation and special lowering)
- a **name**
- an **index signature** (list of domains)
- a **type tag** (string for your own bookkeeping)

Kinds:
- `place`: physical placement decisions (block/dust/torch/repeater)
- `shape`: shape/orientation decisions (dust axis, connection flags)
- `state`: per-scenario logic state (block powered, dust powered, torch output, repeater output)
- `sources`: special “multiset-like” collectors. You don’t constrain them directly; you `add` to them and later take `OR(...)`.

### 1.3 Rules and statements
Rules are blocks of statements. Every statement ultimately emits linear constraints.

Statements:
- `forall (binders...) { ... }` — expands to the cartesian product of binder domains.
- `feature NAME { ... }` — only included if `NAME` is enabled in the instance.
- `require EXPR;` — asserts a boolean or linear condition.
- `def LHS <-> RHS;` — defines a **binary variable** as equivalent to a boolean expression.
- `add TARGET += VALUE [where COND];` — appends a term to a `sources` target (optionally guarded).
- `exclude TARGET += VALUE;` — removes a previously-added term (or prevents it from being added).
- `force A == B;` — hard equality, used for pinning, truth tables, etc.

### 1.4 Expressions
Expressions are either:
- **Linear**: built from `+ - *`, numeric literals, parameters, variables, and `sum(...)`
- **Boolean**: built from `!`, `and`, `or`, `->`, `<->`, `OR(...)`, variable refs, and a small set of calls.

The compiler **linearizes** boolean expressions into MILP constraints using standard encodings.

---

## 2. Syntax

### 2.1 Top-level
```
model <Name> {
  <decl>*
  <rule>*
  <objective>?
}
```

### 2.2 Declarations
**Index (metadata only):**
```
index Cell = (x,z) in Grid;
```

**Enum domain:**
```
enum Dir { N, E, S, W }
```

**Scenario domain:**
```
scenario s in {0,1};
```

**Pin (metadata only; used by Observe):**
```
pin IN : ObsPoint;
```

**Variable:**
```
place   S[Cell]        : Solid;
state   BP[s,Cell]     : Bool;
shape   Conn[Layer,Cell,Dir] : Bool;
sources DustSrc[s,Layer,Cell] : Bool;
```

### 2.3 Rules
```
rule <RuleName> { <stmt>* }
```

### 2.4 Statements
```
forall (c in Cell, (c,d) in Cell * Dir) { ... }
feature CrossDust { ... }

require <expr>;
def <varref> <-> <expr>;

add <varref> += <expr> [where <expr>];
exclude <varref> += <expr>;

force <expr> == <expr>;
```

### 2.5 Variable reference
```
S[c]
D[GROUND, c]
TO_wall[s, c, E]
```

Indices can use function calls like `neigh(c,d)`.

### 2.6 Expressions (operators)
- Unary: `!x`, `-x`
- Multiplicative: `a * b` (only `const * linear` in linear contexts)
- Additive: `a + b`, `a - b`
- Comparison (linear): `<=`, `>=`, `==`
- Boolean: `and`, `or`, `->` (implies), `<->` (iff)

Special forms:
- `OR(SourceSet[...])` — materializes “OR of all added terms”.
- `OR{a,b,c}` — explicit OR-list (boolean).
- `sum((i,j) in DomA * DomB) <linear_expr>`

---

## 3. Semantics and ILP lowering

### 3.1 Boolean linearization
All boolean variables are binary (`0/1`). The compiler introduces auxiliary binary vars as needed.

**NOT**
`y <-> !x` becomes: `y + x = 1`

**AND**
`z <-> (x and y)` becomes:
- `z <= x`
- `z <= y`
- `z >= x + y - 1`

**OR**
`z <-> (x or y)` becomes:
- `z >= x`
- `z >= y`
- `z <= x + y`

**IMPLIES**
`x -> y` is compiled as `x <= y` (equivalently `x - y <= 0`).

**IFF**
`x <-> y` compiles as `(x -> y) and (y -> x)`.

### 3.2 `def`
`def V[...] <-> BOOL_EXPR;` compiles `V - BOOLAUX = 0` where `BOOLAUX` is the lowered boolean expression.

### 3.3 `require`
- If the expression is a comparison (`<=`, `>=`, `==`), it becomes that linear constraint.
- Otherwise it is treated as boolean and forced to `1`: `BOOLAUX = 1`.

### 3.4 `force`
`force A == B;` emits `A - B = 0`.
If `A` or `B` is a boolean expression, it will be lowered to an auxiliary binary first.

### 3.5 `sources` + `add/exclude` + `OR(...)`
A `sources` variable is not a real decision variable; it is an **accumulator**.

- `add DustSrc[s,l,c] += term where cond;` registers `(term AND cond)` into a bag keyed by `(s,l,c)`.
- `exclude ...` prevents a specific term from contributing.
- `OR(DustSrc[s,l,c])` creates a new auxiliary variable `z` with OR-constraints over all registered terms.

This is a practical pattern for expressing “OR of many possible incoming power sources” without manually writing the OR every time.

### 3.6 Calls supported by the code generator
These calls appear in indices or boolean expressions:

Index calls (used inside `[...]`):
- `neigh(c,d)` — returns adjacent cell in direction `d` (or `__NONE__` if out-of-bounds)
- `back(c,d)` — synonym for `neigh(c, opp(d))`
- `opp(d)` — opposite direction
- `supportForWallTorch(c,d)` — treated as `neigh(c, opp(d))`

Boolean calls:
- `Observe(PIN, s=<int>)` — returns a binary variable name from the instance callback.
- `OR(x)` — special: if `x` is a `sources` varref, materializes OR-of-added-terms.

The shipped generator also includes a few placeholder calls returning constants for convenience (`ClearUp`, etc.).
You can extend this set by editing `rsdsl/src/scip_codegen.rs`.

---

## 4. Instance interface

At runtime you provide:
- `cells: Vec<Cell>` — defines the `Cell` domain and cell-id strings (`x{X}_z{Z}`).
- `scenarios: Vec<i32>` — defines scenario domain values.
- `params: HashMap<String,f64>` — used for constants in objectives and linear expressions.
- `features: HashSet<String>` — enables/disables `feature NAME { ... }` blocks.
- `observe: Fn(pin, scenario)->ConcreteVar` — mapping for `Observe(...)`.

---

## 5. Output format

The compiler emits a SCIP-compatible **LP file**:

- `Minimize` / `Maximize` objective
- `Subject To` constraints
- `Binary` list of all variables (including aux and observed vars)
- `End`

---

## 6. Determinism & naming

Variable names are deterministic:
- `VarName__idx0__idx1__...`
- Aux variables: `__aux_<kind>_<n>`
- Observed vars: whatever your `observe(...)` returns (must be unique and stable)

Out-of-bounds neighbor (`neigh`) returns `__NONE__` which lowers to constant `0`.

---

## 7. What this spec is (and isn’t)

This DSL is **not** a full redstone simulator. It’s a modeling layer that lets you:
- declare indexed binary vars
- express constraints and “OR-of-many-sources” patterns
- compile into a solver-friendly MILP

For accuracy, you’ll iterate the physical semantics (dust connections, torch power rules, repeater directionality, vertical connectivity, occlusion, etc.) inside the same framework.

---

## 8. Example: NOT gate (minimal line)

See `rsdsl/examples/not_gate.rs` for a complete end-to-end example that:
- defines a model in the DSL
- pins IN/OUT truth table across two scenarios
- generates a `.lp` text to stdout
