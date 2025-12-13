use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub decls: Vec<Decl>,
    pub rules: Vec<Rule>,
    pub objective: Option<Objective>,
}

impl ModelSpec {
    pub fn debug_print(&self) {
        println!("== Model: {} ==", self.name);
        println!("Decls: {}", self.decls.len());
        println!("Rules: {}", self.rules.len());
        if let Some(obj) = &self.objective {
            println!("Objective: {:?}", obj);
        }
    }
    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Decl {
    Index(IndexDecl),
    Enum(EnumDecl),
    Scenario(ScenarioDecl),
    Pin(PinDecl),
    Fn(FnDecl),
    Var(VarDecl),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDecl {
    pub name: String,
    /// We keep RHS as a structured token list for now (not a string).
    pub rhs_toks: Vec<Tok>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: String,
    pub value: Option<String>, // e.g. "=0"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioDecl {
    pub name: String,
    pub values: Vec<String>, // e.g. ["0","1"]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinDecl {
    pub name: String,
    pub ty: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnDecl {
    pub name: String,
    pub args: Vec<String>,
    pub ret: String,
    pub ret_optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarDecl {
    pub kind: VarKind,
    pub name: String,
    pub indices: Vec<String>,
    pub ty: Option<String>,         // after ':'
    pub domain_leq: Option<String>, // after '<=' (domain link)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VarKind {
    Place,
    State,
    Shape,
    Sources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub name: String,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Require(Expr),
    Def {
        lhs: VarRef,
        op: DefOp,
        rhs: Expr,
    },
    Let {
        name: String,
        value: Expr,
    },
    Add {
        target: VarRef,
        value: Expr,
        cond: Option<Expr>,
        exclude: bool,
    },
    ForceEq {
        lhs: Expr,
        rhs: Expr,
    },
    /// Explicit binder block: forall(v in Domain, ...) { body }
    ForAll {
        binders: Vec<(String, String)>,
        body: Vec<Stmt>,
    },
    Feature {
        name: String,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DefOp {
    Iff,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarRef {
    pub name: String,
    pub indices: Vec<Expr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub sense: ObjSense,
    pub body: Expr,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    Sym(String),
    Lit(String),

    Var(VarRef),
    Call { name: String, args: Vec<Expr> },
    NamedArg { name: String, value: Box<Expr> }, // used inside Call args

    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),

    Eq(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),

    Implies(Box<Expr>, Box<Expr>),

    OrList(Vec<Expr>), // OR{...}
    Sum { binder: Binder, body: Box<Expr> },

    Paren(Box<Expr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binder {
    pub toks: Vec<Tok>, // binder tokens (e.g. `c in Cell` or `(c,d) in Cell * Dir`)
}

/// Serializable token representation (so we avoid "strings" for unparsed RHS fragments).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tok {
    Ident(String),
    Lit(String),
    Punct(char),
    Group { delim: TokDelim, inner: Vec<Tok> },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TokDelim {
    Paren,
    Bracket,
    Brace,
    None,
}

pub mod scip_codegen;
