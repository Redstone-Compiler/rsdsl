// NOTE: In your workspace, enable the `macros` feature (and set the correct `rsdsl_macros` path)
// if you want `rsdsl!{...}` to be available.
// This is gated so `cargo test` can run even when the proc-macro crate isn't present.
pub use rsdsl_macros::rsdsl;

pub mod scip_codegen;
pub use scip_codegen::codegen_scip_lp;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A concrete boolean variable in the generated ILP (0/1).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConcreteVar {
    pub name: String,
}
impl ConcreteVar {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
    pub fn lp_name(&self) -> String {
        self.name.clone()
    }
}

/// Runtime instance: provides concrete domains (cells, enums), feature flags, parameters, and Observe() mapping.
#[derive(Clone)]
pub struct Instance {
    pub cells: Vec<Cell>,
    pub scenarios: Vec<i32>,
    pub params: HashMap<String, f64>,
    pub features: HashSet<String>,
    pub observe: Arc<dyn Fn(&str, i32) -> ConcreteVar + Send + Sync>,
}
impl Instance {
    pub fn new(
        cells: Vec<Cell>,
        scenarios: Vec<i32>,
        params: HashMap<String, f64>,
        features: HashSet<String>,
        observe: Arc<dyn Fn(&str, i32) -> ConcreteVar + Send + Sync>,
    ) -> Self {
        Self {
            cells,
            scenarios,
            params,
            features,
            observe,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    pub x: i32,
    pub z: i32,
}
impl Cell {
    pub fn id(&self) -> String {
        format!("x{}_z{}", self.x, self.z)
    }
}

/// Model specification AST (produced by the proc-macro).
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: String,
    pub decls: Vec<Decl>,
    pub rules: Vec<Rule>,
    pub objective: Option<Objective>,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Index {
        name: String,
        domain: String,
    },
    Enum {
        name: String,
        variants: Vec<String>,
    },
    Scenario {
        name: String,
        values: Vec<i32>,
    },
    Pin {
        name: String,
        ty: String,
    },
    Fn {
        name: String,
        args: Vec<(String, String)>,
        ret: String,
    },
    Var {
        kind: VarKind,
        name: String,
        indices: Vec<String>,
        ty: String,
    },
    // objective weights etc are passed via Instance.params
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VarKind {
    Place,
    State,
    Shape,
    Sources,
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub name: String,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    ForAll {
        binders: Vec<Binder>,
        body: Vec<Stmt>,
    },
    Feature {
        name: String,
        body: Vec<Stmt>,
    },
    Let {
        name: String,
        expr: Expr,
    },
    Require(Expr),
    Def {
        lhs: VarRef,
        rhs: Expr,
    },
    Add {
        exclude: bool,
        target: VarRef,
        value: Expr,
        cond: Option<Expr>,
    },
    ForceEq {
        lhs: Expr,
        rhs: Expr,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Binder {
    pub vars: Vec<String>,
    pub domains: Vec<String>, // cartesian product in order
}

#[derive(Clone, Debug, PartialEq)]
pub struct VarRef {
    pub name: String,
    pub indices: Vec<Expr>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Sym(String),
    Lit(f64),
    Var(VarRef),
    NamedArg {
        name: String,
        value: Box<Expr>,
    },

    Not(Box<Expr>),
    Neg(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Implies(Box<Expr>, Box<Expr>),
    Iff(Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),

    Eq(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),

    OrList(Vec<Expr>),
    Call {
        name: String,
        args: Vec<Expr>,
    },
    Sum {
        binders: Vec<Binder>,
        body: Box<Expr>,
    },

    Paren(Box<Expr>),
}

#[derive(Clone, Debug)]
pub struct Objective {
    pub sense: ObjSense,
    pub expr: Expr, // linear expression
}

#[derive(Clone, Debug)]
pub enum ObjSense {
    Minimize,
    Maximize,
}
