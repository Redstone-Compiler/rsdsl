use crate::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("unknown domain `{0}`")]
    UnknownDomain(String),
    #[error("unknown variable `{0}`")]
    UnknownVar(String),
    #[error("variable `{0}` used with wrong arity: expected {1}, got {2}")]
    WrongArity(String, usize, usize),
    #[error("unsupported expression in linear context: {0:?}")]
    UnsupportedLinear(Expr),
    #[error("unsupported boolean context: {0:?}")]
    UnsupportedBool(Expr),
    #[error("unsupported call `{0}`")]
    UnsupportedCall(String),
    #[error("Observe(pin, s=..) requires integer scenario, got `{0}`")]
    BadScenario(String),
    #[error("missing scenario binder `s` for call `{0}`")]
    MissingScenarioBinder(String),
}

#[derive(Clone, Debug)]
struct LinearExpr {
    terms: BTreeMap<String, f64>,
    constant: f64,
}
impl LinearExpr {
    fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
            constant: 0.0,
        }
    }
    fn from_const(v: f64) -> Self {
        let mut e = Self::zero();
        e.constant = v;
        e
    }
    fn from_var(v: &str, c: f64) -> Self {
        let mut e = Self::zero();
        if c != 0.0 {
            e.terms.insert(v.to_string(), c);
        }
        e
    }
    fn add_inplace(&mut self, other: &LinearExpr) {
        self.constant += other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) += *v;
        }
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    fn sub_inplace(&mut self, other: &LinearExpr) {
        self.constant -= other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) -= *v;
        }
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    fn scale(&self, k: f64) -> Self {
        let mut e = Self::zero();
        e.constant = self.constant * k;
        for (n, c) in self.terms.iter() {
            e.terms.insert(n.clone(), c * k);
        }
        e
    }
}

#[derive(Clone, Copy, Debug)]
enum Sense {
    Le,
    Ge,
    Eq,
}

#[derive(Clone, Debug)]
struct Constraint {
    name: String,
    expr: LinearExpr, // lhs
    sense: Sense,
    rhs: f64,
}

#[derive(Clone, Debug)]
struct Ilp {
    objective: LinearExpr,
    sense: ObjSense,
    constraints: Vec<Constraint>,
    binaries: BTreeSet<String>,
}

impl Ilp {
    fn new() -> Self {
        Self {
            objective: LinearExpr::zero(),
            sense: ObjSense::Minimize,
            constraints: vec![],
            binaries: BTreeSet::new(),
        }
    }
}

#[derive(Clone)]
struct Env {
    // domain values
    domains: HashMap<String, Vec<String>>,
    // var signature: name -> domain names per index
    sigs: HashMap<String, Vec<String>>,
    // var kind for special lowering
    var_kinds: HashMap<String, VarKind>,
    // features + params + observe
    features: HashSet<String>,
    params: HashMap<String, f64>,
    observe: std::sync::Arc<dyn Fn(&str, i32) -> ConcreteVar + Send + Sync>,
    // cell set for topo bounds
    cell_set: HashSet<String>,
}

#[derive(Clone, Debug, Default)]
struct Ctx {
    bind: HashMap<String, String>,
    lets: HashMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SourceKey {
    // canonical key name for source set: DustSrc__s__layer__cell
    name: String,
}

#[derive(Clone, Debug)]
struct SourceDB {
    adds: HashMap<SourceKey, Vec<Expr>>,
    excludes: HashMap<SourceKey, Vec<Expr>>,
}

impl SourceDB {
    fn new() -> Self {
        Self {
            adds: HashMap::new(),
            excludes: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct Generator {
    env: Env,
    ilp: Ilp,
    sources: SourceDB,
    // cache: sources key -> OR var
    sources_or_cache: HashMap<SourceKey, String>,
    aux_id: usize,
}

pub fn codegen_scip_lp(spec: &ModelSpec, inst: &Instance) -> Result<String, CodegenError> {
    let env = build_env(spec, inst)?;
    let mut gen = Generator {
        env,
        ilp: Ilp::new(),
        sources: SourceDB::new(),
        sources_or_cache: HashMap::new(),
        aux_id: 0,
    };

    // constants
    gen.ilp.binaries.insert("__const0".to_string());
    gen.ilp.binaries.insert("__const1".to_string());
    gen.ilp.constraints.push(Constraint {
        name: "__fix_const0".to_string(),
        expr: LinearExpr::from_var("__const0", 1.0),
        sense: Sense::Eq,
        rhs: 0.0,
    });
    gen.ilp.constraints.push(Constraint {
        name: "__fix_const1".to_string(),
        expr: LinearExpr::from_var("__const1", 1.0),
        sense: Sense::Eq,
        rhs: 1.0,
    });

    // 1) collect sources
    gen.collect_sources(spec)?;

    // 2) emit constraints (Require/Def/ForceEq) + objective
    gen.emit_model(spec)?;

    // normalize constraints: move constants to rhs
    for c in gen.ilp.constraints.iter_mut() {
        if c.expr.constant.abs() > 1e-12 {
            c.rhs -= c.expr.constant;
            c.expr.constant = 0.0;
        }
    }
    if gen.ilp.objective.constant.abs() > 1e-12 {
        // drop constant objective
        gen.ilp.objective.constant = 0.0;
    }

    Ok(emit_lp(&gen.ilp))
}

fn build_env(spec: &ModelSpec, inst: &Instance) -> Result<Env, CodegenError> {
    let mut domains: HashMap<String, Vec<String>> = HashMap::new();

    // default Cell domain from instance
    let cell_vals: Vec<String> = inst.cells.iter().map(|c| c.id()).collect();
    domains.insert("Cell".to_string(), cell_vals.clone());

    // default Layer/Dir if declared; we'll override from enums
    domains.insert("Layer".to_string(), vec!["GROUND".into(), "TOP".into()]);
    domains.insert(
        "Dir".to_string(),
        vec!["N".into(), "E".into(), "S".into(), "W".into()],
    );

    // Scenario domain
    domains.insert(
        "s".to_string(),
        inst.scenarios.iter().map(|v| v.to_string()).collect(),
    );
    domains.insert(
        "Sc".to_string(),
        inst.scenarios.iter().map(|v| v.to_string()).collect(),
    );

    let mut sigs: HashMap<String, Vec<String>> = HashMap::new();
    let mut var_kinds: HashMap<String, VarKind> = HashMap::new();

    for d in &spec.decls {
        match d {
            Decl::Enum { name, variants } => {
                domains.insert(name.clone(), variants.clone());
            }
            Decl::Scenario { name, values } => {
                domains.insert(name.clone(), values.iter().map(|v| v.to_string()).collect());
            }
            Decl::Var {
                kind,
                name,
                indices,
                ..
            } => {
                sigs.insert(name.clone(), indices.clone());
                var_kinds.insert(name.clone(), kind.clone());
            }
            Decl::Index { name, .. } => {
                // index decl indicates a domain name exists; already covered by instance for Cell.
                if name != "Cell" && !domains.contains_key(name) {
                    domains.insert(name.clone(), vec![]);
                }
            }
            _ => {}
        }
    }

    let cell_set = cell_vals.into_iter().collect::<HashSet<_>>();

    Ok(Env {
        domains,
        sigs,
        var_kinds,
        features: inst.features.clone(),
        params: inst.params.clone(),
        observe: inst.observe.clone(),
        cell_set,
    })
}

impl Generator {
    fn fresh_aux(&mut self, prefix: &str) -> String {
        let n = self.aux_id;
        self.aux_id += 1;
        format!("__aux_{}_{}", prefix, n)
    }

    fn is_feature_on(&self, name: &str) -> bool {
        self.env.features.contains(name)
    }

    fn domain_vals(&self, name: &str) -> Result<Vec<String>, CodegenError> {
        self.env
            .domains
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::UnknownDomain(name.to_string()))
    }

    fn eval_index(&self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Sym(s) => {
                if let Some(v) = ctx.lets.get(s) {
                    return Ok(v.clone());
                }
                if let Some(v) = ctx.bind.get(s) {
                    return Ok(v.clone());
                }
                // enum literal or domain value
                Ok(s.clone())
            }
            Expr::Lit(v) => Ok(format!("{}", *v as i64)),
            Expr::Paren(x) => self.eval_index(x, ctx),
            Expr::Call { name, args } => self.eval_index_call(name, args, ctx),
            _ => Err(CodegenError::UnsupportedLinear(e.clone())),
        }
    }

    fn parse_cell_id(id: &str) -> Option<(i32, i32)> {
        // x{X}_z{Z}
        let re = regex::Regex::new(r"^x(-?\d+)_z(-?\d+)$").ok()?;
        let caps = re.captures(id)?;
        let x = caps.get(1)?.as_str().parse::<i32>().ok()?;
        let z = caps.get(2)?.as_str().parse::<i32>().ok()?;
        Some((x, z))
    }

    fn eval_index_call(
        &self,
        name: &str,
        args: &[Expr],
        ctx: &Ctx,
    ) -> Result<String, CodegenError> {
        match name {
            "opp" => {
                let d = self.eval_index(&args[0], ctx)?;
                Ok(match d.as_str() {
                    "N" => "S".into(),
                    "S" => "N".into(),
                    "E" => "W".into(),
                    "W" => "E".into(),
                    _ => d,
                })
            }
            "neigh" | "back" | "front" | "supportForWallTorch" => {
                let c = self.eval_index(&args[0], ctx)?;
                let d = if name == "back" {
                    let dd = self.eval_index(&args[1], ctx)?;
                    match dd.as_str() {
                        "N" => "S".into(),
                        "S" => "N".into(),
                        "E" => "W".into(),
                        "W" => "E".into(),
                        _ => dd,
                    }
                } else if name == "supportForWallTorch" {
                    let dd = self.eval_index(&args[1], ctx)?;
                    match dd.as_str() {
                        "N" => "S".into(),
                        "S" => "N".into(),
                        "E" => "W".into(),
                        "W" => "E".into(),
                        _ => dd,
                    }
                } else {
                    self.eval_index(&args[1], ctx)?
                };
                let Some((x, z)) = Self::parse_cell_id(&c) else {
                    return Ok("__NONE__".into());
                };
                let (dx, dz) = match d.as_str() {
                    "N" => (0, -1),
                    "S" => (0, 1),
                    "E" => (1, 0),
                    "W" => (-1, 0),
                    _ => (0, 0),
                };
                let (nx, nz) = (x + dx, z + dz);
                let nid = format!("x{}_z{}", nx, nz);
                if self.env.cell_set.contains(&nid) {
                    Ok(nid)
                } else {
                    Ok("__NONE__".into())
                }
            }
            _ => Ok(format!("{}({})", name, args.len())),
        }
    }

    fn var_sig(&self, name: &str) -> Result<Vec<String>, CodegenError> {
        self.env
            .sigs
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::UnknownVar(name.to_string()))
    }

    fn var_kind(&self, name: &str) -> Option<VarKind> {
        self.env.var_kinds.get(name).cloned()
    }

    fn var_name(&mut self, vr: &VarRef, ctx: &Ctx) -> Result<String, CodegenError> {
        let sig = self.var_sig(&vr.name)?;
        if sig.len() != vr.indices.len() {
            return Err(CodegenError::WrongArity(
                vr.name.clone(),
                sig.len(),
                vr.indices.len(),
            ));
        }
        let mut parts = vec![vr.name.clone()];
        for idx in &vr.indices {
            let s = self.eval_index(idx, ctx)?;
            if s == "__NONE__" {
                return Ok("__const0".into());
            }
            parts.push(s);
        }
        let n = parts.join("__");
        // every declared var is binary in this model (0/1)
        self.ilp.binaries.insert(n.clone());
        Ok(n)
    }

    fn as_const_param(&self, e: &Expr, ctx: &Ctx) -> Option<f64> {
        match e {
            Expr::Lit(v) => Some(*v),
            Expr::Sym(s) => self.env.params.get(s).cloned(),
            Expr::Paren(x) => self.as_const_param(x, ctx),
            _ => None,
        }
    }

    fn eval_linear(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match e {
            Expr::Lit(v) => Ok(LinearExpr::from_const(*v)),
            Expr::Sym(s) => {
                if let Some(v) = self.env.params.get(s) {
                    Ok(LinearExpr::from_const(*v))
                } else if s == "__const0" {
                    Ok(LinearExpr::from_var("__const0", 1.0))
                } else if s == "__const1" {
                    Ok(LinearExpr::from_var("__const1", 1.0))
                } else {
                    // allow linear context only for params or constants
                    Err(CodegenError::UnsupportedLinear(e.clone()))
                }
            }
            Expr::Var(vr) => {
                // treat as 0/1 variable
                let n = self.var_name(vr, ctx)?;
                Ok(LinearExpr::from_var(&n, 1.0))
            }
            Expr::Add(a, b) => {
                let mut x = self.eval_linear(a, ctx)?;
                let y = self.eval_linear(b, ctx)?;
                x.add_inplace(&y);
                Ok(x)
            }
            Expr::Sub(a, b) => {
                let mut x = self.eval_linear(a, ctx)?;
                let y = self.eval_linear(b, ctx)?;
                x.sub_inplace(&y);
                Ok(x)
            }
            Expr::Mul(a, b) => {
                // only constant * linear
                if let Some(k) = self.as_const_param(a, ctx) {
                    let x = self.eval_linear(b, ctx)?;
                    Ok(x.scale(k))
                } else if let Some(k) = self.as_const_param(b, ctx) {
                    let x = self.eval_linear(a, ctx)?;
                    Ok(x.scale(k))
                } else {
                    Err(CodegenError::UnsupportedLinear(e.clone()))
                }
            }
            Expr::Sum { binders, body } => {
                let mut acc = LinearExpr::zero();
                self.expand_binders(binders, ctx, |g, cctx| {
                    let t = g.eval_linear(body, &cctx)?;
                    acc.add_inplace(&t);
                    Ok(())
                })?;
                Ok(acc)
            }
            Expr::Paren(x) => self.eval_linear(x, ctx),
            _ => Err(CodegenError::UnsupportedLinear(e.clone())),
        }
    }

    fn eval_linear_or_boolish(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match self.eval_linear(e, ctx) {
            Ok(x) => Ok(x),
            Err(CodegenError::UnsupportedLinear(_)) => {
                let v = self.eval_bool(e, ctx)?;
                Ok(LinearExpr::from_var(&v, 1.0))
            }
            Err(e) => Err(e),
        }
    }

    fn eval_bool(&mut self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Lit(v) => Ok(if (*v - 0.0).abs() < 1e-9 {
                "__const0".into()
            } else {
                "__const1".into()
            }),
            Expr::Var(vr) => {
                // sources varref is special
                if self.var_kind(&vr.name) == Some(VarKind::Sources) {
                    let key = self.sources_key(vr, ctx)?;
                    self.ensure_sources_or(&key)
                } else {
                    self.var_name(vr, ctx)
                }
            }
            Expr::Not(x) => {
                let a = self.eval_bool(x, ctx)?;
                if a == "__const0" {
                    return Ok("__const1".into());
                }
                if a == "__const1" {
                    return Ok("__const0".into());
                }
                let y = self.fresh_aux("not");
                self.ilp.binaries.insert(y.clone());
                // y + a = 1
                let mut lhs = LinearExpr::from_var(&y, 1.0);
                lhs.add_inplace(&LinearExpr::from_var(&a, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("not_{}", y),
                    expr: lhs,
                    sense: Sense::Eq,
                    rhs: 1.0,
                });
                Ok(y)
            }
            Expr::And(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                if x == "__const0" || y == "__const0" {
                    return Ok("__const0".into());
                }
                if x == "__const1" {
                    return Ok(y);
                }
                if y == "__const1" {
                    return Ok(x);
                }
                let z = self.fresh_aux("and");
                self.ilp.binaries.insert(z.clone());
                // z <= x
                self.ilp.constraints.push(Constraint {
                    name: format!("and_le1_{}", z),
                    expr: LinearExpr::from_var(&z, 1.0).sub(LinearExpr::from_var(&x, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                // z <= y
                self.ilp.constraints.push(Constraint {
                    name: format!("and_le2_{}", z),
                    expr: LinearExpr::from_var(&z, 1.0).sub(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                // z >= x + y - 1  <=>  z - x - y >= -1
                let mut expr = LinearExpr::from_var(&z, 1.0);
                expr.sub_inplace(&LinearExpr::from_var(&x, 1.0));
                expr.sub_inplace(&LinearExpr::from_var(&y, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("and_ge_{}", z),
                    expr,
                    sense: Sense::Ge,
                    rhs: -1.0,
                });
                Ok(z)
            }
            Expr::Or(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                if x == "__const1" || y == "__const1" {
                    return Ok("__const1".into());
                }
                if x == "__const0" {
                    return Ok(y);
                }
                if y == "__const0" {
                    return Ok(x);
                }
                let z = self.fresh_aux("or");
                self.ilp.binaries.insert(z.clone());
                // z >= x  => -z + x <= 0? actually z - x >=0 => -z + x <=0
                self.ilp.constraints.push(Constraint {
                    name: format!("or_ge1_{}", z),
                    expr: LinearExpr::from_var(&x, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("or_ge2_{}", z),
                    expr: LinearExpr::from_var(&y, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                // z <= x + y  => z - x - y <=0
                let mut expr = LinearExpr::from_var(&z, 1.0);
                expr.sub_inplace(&LinearExpr::from_var(&x, 1.0));
                expr.sub_inplace(&LinearExpr::from_var(&y, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("or_le_{}", z),
                    expr,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(z)
            }
            Expr::OrList(xs) => {
                if xs.is_empty() {
                    return Ok("__const0".into());
                }
                let mut terms = vec![];
                for x in xs {
                    terms.push(self.eval_bool(x, ctx)?);
                }
                if terms.iter().any(|t| t == "__const1") {
                    return Ok("__const1".into());
                }
                terms.retain(|t| t != "__const0");
                if terms.is_empty() {
                    return Ok("__const0".into());
                }
                if terms.len() == 1 {
                    return Ok(terms[0].clone());
                }
                let z = self.fresh_aux("orlist");
                self.ilp.binaries.insert(z.clone());
                // z >= each
                for (i, t) in terms.iter().enumerate() {
                    self.ilp.constraints.push(Constraint {
                        name: format!("orlist_ge{}_{}", i, z),
                        expr: LinearExpr::from_var(t, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
                        sense: Sense::Le,
                        rhs: 0.0,
                    });
                }
                // z <= sum(terms)
                let mut expr = LinearExpr::from_var(&z, 1.0);
                for t in &terms {
                    expr.sub_inplace(&LinearExpr::from_var(t, 1.0));
                }
                self.ilp.constraints.push(Constraint {
                    name: format!("orlist_le_{}", z),
                    expr,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(z)
            }
            Expr::Implies(a, b) => {
                // a -> b == (!a) OR b
                let na = Expr::Not(a.clone());
                let or = Expr::Or(Box::new(na), b.clone());
                self.eval_bool(&or, ctx)
            }
            Expr::Iff(a, b) => {
                // (a->b) and (b->a)
                let ab = Expr::Implies(a.clone(), b.clone());
                let ba = Expr::Implies(b.clone(), a.clone());
                let and = Expr::And(Box::new(ab), Box::new(ba));
                self.eval_bool(&and, ctx)
            }
            Expr::Call { name, args } => self.eval_bool_call(name, args, ctx),
            Expr::Paren(x) => self.eval_bool(x, ctx),
            Expr::NamedArg { .. } => Err(CodegenError::UnsupportedBool(e.clone())),
            // comparisons are not reified (only allowed at top-level Require/ForceEq)
            Expr::Eq(..)
            | Expr::Le(..)
            | Expr::Ge(..)
            | Expr::Lt(..)
            | Expr::Gt(..)
            | Expr::Add(..)
            | Expr::Sub(..)
            | Expr::Mul(..)
            | Expr::Sum { .. }
            | Expr::Sym(_)
            | Expr::Neg(_) => Err(CodegenError::UnsupportedBool(e.clone())),
        }
    }

    fn eval_bool_call(
        &mut self,
        name: &str,
        args: &[Expr],
        ctx: &Ctx,
    ) -> Result<String, CodegenError> {
        match name {
            "OR" => {
                // OR(X) where X is a sources varref
                if args.len() == 1 {
                    if let Expr::Var(vr) = &args[0] {
                        if self.var_kind(&vr.name) == Some(VarKind::Sources) {
                            let key = self.sources_key(vr, ctx)?;
                            return self.ensure_sources_or(&key);
                        }
                    }
                }
                // otherwise treat as OR-list of args
                let or = Expr::OrList(args.to_vec());
                self.eval_bool(&or, ctx)
            }
            "Observe" => {
                // Observe(PIN, s=0) -> ConcreteVar
                let pin = match args.get(0) {
                    Some(Expr::Sym(p)) => p.clone(),
                    Some(Expr::Var(vr)) => vr.name.clone(),
                    Some(x) => match x {
                        Expr::Paren(p) => match &**p {
                            Expr::Sym(pn) => pn.clone(),
                            _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
                        },
                        _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
                    },
                    None => return Err(CodegenError::UnsupportedCall("Observe".into())),
                };
                let mut sc: Option<String> = None;
                for a in args.iter().skip(1) {
                    if let Expr::NamedArg { name, value } = a {
                        if name == "s" {
                            sc = Some(self.eval_index(value, ctx)?);
                        }
                    }
                }
                let Some(sc) = sc else {
                    return Err(CodegenError::BadScenario("missing".into()));
                };
                let sci: i32 = sc
                    .parse::<i32>()
                    .map_err(|_| CodegenError::BadScenario(sc.clone()))?;
                let cv = (self.env.observe)(&pin, sci);
                let vname = cv.lp_name();
                self.ilp.binaries.insert(vname.clone());
                Ok(vname)
            }
            "TorchOut" => {
                // TorchOut(stand,c) or TorchOut(wall,c,d) uses scenario binder `s`
                let s = ctx
                    .bind
                    .get("s")
                    .cloned()
                    .ok_or_else(|| CodegenError::MissingScenarioBinder("TorchOut".into()))?;
                let kind = match args.get(0) {
                    Some(Expr::Sym(k)) => k.clone(),
                    _ => return Err(CodegenError::UnsupportedCall("TorchOut".into())),
                };
                if kind == "stand" {
                    let c = self.eval_index(&args[1], ctx)?;
                    let vr = VarRef {
                        name: "TO_stand".into(),
                        indices: vec![Expr::Sym(s), Expr::Sym(c)],
                    };
                    self.var_name(&vr, ctx)
                } else if kind == "wall" {
                    let c = self.eval_index(&args[1], ctx)?;
                    let d = self.eval_index(&args[2], ctx)?;
                    let vr = VarRef {
                        name: "TO_wall".into(),
                        indices: vec![Expr::Sym(s), Expr::Sym(c), Expr::Sym(d)],
                    };
                    self.var_name(&vr, ctx)
                } else {
                    Err(CodegenError::UnsupportedCall("TorchOut".into()))
                }
            }
            "RepOut" => {
                let s = ctx
                    .bind
                    .get("s")
                    .cloned()
                    .ok_or_else(|| CodegenError::MissingScenarioBinder("RepOut".into()))?;
                let c = self.eval_index(&args[0], ctx)?;
                let d = self.eval_index(&args[1], ctx)?;
                let vr = VarRef {
                    name: "RO".into(),
                    indices: vec![Expr::Sym(s), Expr::Sym(c), Expr::Sym(d)],
                };
                self.var_name(&vr, ctx)
            }
            "CandidateDustAdj" => {
                // CandidateDustAdj(l,c,d) -> D[l, neigh(c,d)]
                let l = self.eval_index(&args[0], ctx)?;
                let c = self.eval_index(
                    &Expr::Call {
                        name: "neigh".into(),
                        args: vec![args[1].clone(), args[2].clone()],
                    },
                    ctx,
                )?;
                let vr = VarRef {
                    name: "D".into(),
                    indices: vec![Expr::Sym(l), Expr::Sym(c)],
                };
                self.var_name(&vr, ctx)
            }
            "ExistsConnectionCandidate" => {
                // OR of neighbor placements
                let l = self.eval_index(&args[0], ctx)?;
                let neigh = self.eval_index(
                    &Expr::Call {
                        name: "neigh".into(),
                        args: vec![args[1].clone(), args[2].clone()],
                    },
                    ctx,
                )?;
                if neigh == "__NONE__" {
                    return Ok("__const0".into());
                }
                let mut ors: Vec<Expr> = vec![];
                // dust neighbor
                ors.push(Expr::Var(VarRef {
                    name: "D".into(),
                    indices: vec![Expr::Sym(l.clone()), Expr::Sym(neigh.clone())],
                }));
                // block neighbor
                ors.push(Expr::Var(VarRef {
                    name: "S".into(),
                    indices: vec![Expr::Sym(neigh.clone())],
                }));
                // standing torch neighbor
                ors.push(Expr::Var(VarRef {
                    name: "T_stand".into(),
                    indices: vec![Expr::Sym(neigh.clone())],
                }));
                // wall torches / repeaters neighbor (any dir)
                let dirs = self.domain_vals("Dir")?;
                for d in dirs {
                    ors.push(Expr::Var(VarRef {
                        name: "T_wall".into(),
                        indices: vec![Expr::Sym(neigh.clone()), Expr::Sym(d.clone())],
                    }));
                    ors.push(Expr::Var(VarRef {
                        name: "R".into(),
                        indices: vec![Expr::Sym(neigh.clone()), Expr::Sym(d)],
                    }));
                }
                self.eval_bool(&Expr::OrList(ors), ctx)
            }
            "ClearUp" | "ClearDown" | "AllowCrossChoice" | "Touches" => Ok("__const1".into()),
            "TorchPowersCell" => Ok("__const0".into()),
            _ => Err(CodegenError::UnsupportedCall(name.to_string())),
        }
    }

    fn sources_key(&mut self, vr: &VarRef, ctx: &Ctx) -> Result<SourceKey, CodegenError> {
        // build a canonical string from indices
        let mut parts = vec![vr.name.clone()];
        for idx in &vr.indices {
            let s = self.eval_index(idx, ctx)?;
            if s == "__NONE__" {
                parts.push("__NONE__".into());
            } else {
                parts.push(s);
            }
        }
        Ok(SourceKey {
            name: parts.join("__"),
        })
    }

    fn ensure_sources_or(&mut self, key: &SourceKey) -> Result<String, CodegenError> {
        if let Some(v) = self.sources_or_cache.get(key) {
            return Ok(v.clone());
        }
        // collect terms minus excludes
        let mut terms = self.sources.adds.get(key).cloned().unwrap_or_default();
        let ex = self.sources.excludes.get(key).cloned().unwrap_or_default();
        terms.retain(|t| !ex.iter().any(|e| e == t));

        // lower terms to bool vars
        let ctx = Ctx::default(); // should not be used for concrete exprs (already concretized); still ok.
        let mut term_vars: Vec<String> = vec![];
        for t in terms {
            let v = self.eval_bool(&t, &ctx)?;
            if v == "__const1" {
                self.sources_or_cache.insert(key.clone(), "__const1".into());
                return Ok("__const1".into());
            }
            if v != "__const0" {
                term_vars.push(v);
            }
        }
        term_vars.sort();
        term_vars.dedup();

        if term_vars.is_empty() {
            self.sources_or_cache.insert(key.clone(), "__const0".into());
            return Ok("__const0".into());
        }
        if term_vars.len() == 1 {
            self.sources_or_cache
                .insert(key.clone(), term_vars[0].clone());
            return Ok(term_vars[0].clone());
        }

        let z = self.fresh_aux("srcor");
        self.ilp.binaries.insert(z.clone());
        // z >= each: -z + t <= 0
        for (i, t) in term_vars.iter().enumerate() {
            self.ilp.constraints.push(Constraint {
                name: format!("srcor_ge{}_{}", i, z),
                expr: LinearExpr::from_var(t, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
                sense: Sense::Le,
                rhs: 0.0,
            });
        }
        // z <= sum(t): z - sum(t) <= 0
        let mut expr = LinearExpr::from_var(&z, 1.0);
        for t in &term_vars {
            expr.sub_inplace(&LinearExpr::from_var(t, 1.0));
        }
        self.ilp.constraints.push(Constraint {
            name: format!("srcor_le_{}", z),
            expr,
            sense: Sense::Le,
            rhs: 0.0,
        });

        self.sources_or_cache.insert(key.clone(), z.clone());
        Ok(z)
    }

    fn collect_sources(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
        for r in &spec.rules {
            self.collect_stmt_block(&r.body, &Ctx::default())?;
        }
        Ok(())
    }

    fn collect_stmt_block(&mut self, body: &[Stmt], ctx: &Ctx) -> Result<(), CodegenError> {
        for st in body {
            self.collect_stmt(st, ctx)?;
        }
        Ok(())
    }

    /// Find "implicit" binders that are referenced in a statement without an explicit `forall`.
    ///
    /// Example:
    ///   def DP0[s, x1_z0] <-> (D0[x1_z0] and DP0[s, x0_z0]);
    /// should expand over `s in Sc` even if the rule omitted:
    ///   forall (s in Sc) { ... }
    ///
    /// We infer binders only from *index positions* of `VarRef` occurrences.
    /// If an index is `Sym(name)` and `name` is not bound by the current `Ctx`, and it is
    /// not a literal value of the expected domain, then we treat it as a missing binder.
    fn implicit_binders_for_stmt(&self, st: &Stmt, ctx: &Ctx) -> Result<Vec<Binder>, CodegenError> {
        let mut sym_to_domain: BTreeMap<String, String> = BTreeMap::new();

        let mut add_unbound = |var_name: &str, indices: &[Expr]| -> Result<(), CodegenError> {
            let sig = self
                .env
                .sigs
                .get(var_name)
                .ok_or_else(|| CodegenError::UnknownVar(var_name.to_string()))?
                .clone();

            if sig.len() != indices.len() {
                return Err(CodegenError::WrongArity(
                    var_name.to_string(),
                    sig.len(),
                    indices.len(),
                ));
            }

            for (dn, idx_expr) in sig.iter().zip(indices.iter()) {
                // Only infer binders from a raw symbol in index position.
                let Expr::Sym(sym) = idx_expr else { continue };

                // Already bound => not implicit.
                if ctx.bind.contains_key(sym) || ctx.lets.contains_key(sym) {
                    continue;
                }

                // If `sym` is a literal value in the expected domain, keep it as a literal.
                let dom_vals = self.domain_vals(dn)?;
                if dom_vals.iter().any(|v| v == sym) {
                    continue;
                }

                // Otherwise treat it as an implicit binder var.
                sym_to_domain
                    .entry(sym.clone())
                    .or_insert_with(|| dn.clone());
            }
            Ok(())
        };

        fn visit_expr<F>(e: &Expr, f: &mut F)
        where
            F: FnMut(&VarRef),
        {
            match e {
                Expr::Var(vr) => f(vr),

                Expr::Not(x) | Expr::Paren(x) | Expr::Neg(x) => {
                    visit_expr(x, f);
                }

                Expr::And(a, b)
                | Expr::Or(a, b)
                | Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Le(a, b)
                | Expr::Ge(a, b)
                | Expr::Eq(a, b)
                | Expr::Implies(a, b)
                | Expr::Iff(a, b)
                | Expr::Lt(a, b)
                | Expr::Gt(a, b) => {
                    visit_expr(a, f);
                    visit_expr(b, f);
                }

                Expr::Sum { body, .. } => {
                    visit_expr(body, f);
                }

                Expr::Call { args, .. } => {
                    for a in args {
                        visit_expr(a, f);
                    }
                }

                Expr::OrList(exprs) => {
                    for e in exprs {
                        visit_expr(e, f);
                    }
                }

                Expr::NamedArg { value, .. } => {
                    visit_expr(value, f);
                }

                Expr::Sym(_) | Expr::Lit(_) => {}
            }
        }

        match st {
            Stmt::Require(e) => {
                visit_expr(e, &mut |vr| {
                    let _ = add_unbound(&vr.name, &vr.indices);
                });
            }
            Stmt::Def { lhs, rhs } => {
                add_unbound(&lhs.name, &lhs.indices)?;
                visit_expr(rhs, &mut |vr| {
                    let _ = add_unbound(&vr.name, &vr.indices);
                });
            }
            Stmt::ForceEq { lhs, rhs } => {
                visit_expr(lhs, &mut |vr| {
                    let _ = add_unbound(&vr.name, &vr.indices);
                });
                visit_expr(rhs, &mut |vr| {
                    let _ = add_unbound(&vr.name, &vr.indices);
                });
            }
            Stmt::Add {
                target,
                value,
                cond,
                ..
            } => {
                add_unbound(&target.name, &target.indices)?;
                visit_expr(value, &mut |vr| {
                    let _ = add_unbound(&vr.name, &vr.indices);
                });
                if let Some(c) = cond {
                    visit_expr(c, &mut |vr| {
                        let _ = add_unbound(&vr.name, &vr.indices);
                    });
                }
            }
            Stmt::ForAll { .. } | Stmt::Feature { .. } | Stmt::Let { .. } => {}
        }

        // Turn each inferred `sym -> domain` into a single-var binder.
        let mut binders: Vec<Binder> = vec![];
        for (sym, dom) in sym_to_domain {
            binders.push(Binder {
                vars: vec![sym],
                domains: vec![dom],
            });
        }
        Ok(binders)
    }

    fn collect_stmt(&mut self, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // Implicit binder expansion for statements that reference domain variables (e.g. `s`)
        // without an explicit `forall`.
        //
        // Example: `DP0[s, ...]` -> expand `s in Sc` into `s=0/1`
        if !matches!(
            st,
            Stmt::ForAll { .. } | Stmt::Feature { .. } | Stmt::Let { .. }
        ) {
            let ib = self.implicit_binders_for_stmt(st, ctx)?;
            if !ib.is_empty() {
                return self.expand_binders(&ib, ctx, |g, cctx| g.collect_stmt(st, &cctx));
            }
        }
        match st {
            Stmt::ForAll { binders, body } => {
                self.expand_binders(binders, ctx, |g, cctx| g.collect_stmt_block(body, &cctx))?;
            }
            Stmt::Feature { name, body } => {
                if self.is_feature_on(name) {
                    self.collect_stmt_block(body, ctx)?;
                }
            }
            Stmt::Let { name, expr } => {
                let mut nctx = ctx.clone();
                let v = self.eval_index(expr, ctx)?;
                nctx.lets.insert(name.clone(), v);
                // let affects subsequent statements only within same block, but for simplicity
                // we treat let as local in code expansion by requiring explicit nesting.
                // In our spec, lets are used inside forall blocks in small scope; we emulate by not supporting statement-seq let here.
                let _ = nctx;
            }
            Stmt::Add {
                exclude,
                target,
                value,
                cond,
            } => {
                if self.var_kind(&target.name) != Some(VarKind::Sources) {
                    return Ok(());
                }
                let key = self.sources_key(target, ctx)?;
                let mut val = concretize_expr(value, ctx)?;
                if let Some(c) = cond {
                    let cc = concretize_expr(c, ctx)?;
                    val = Expr::And(Box::new(val), Box::new(cc));
                }
                if *exclude {
                    self.sources.excludes.entry(key).or_default().push(val);
                } else {
                    self.sources.adds.entry(key).or_default().push(val);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn emit_model(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
        for r in &spec.rules {
            self.emit_stmt_block(&r.name, &r.body, &Ctx::default())?;
        }
        if let Some(obj) = &spec.objective {
            self.ilp.sense = obj.sense.clone();
            let lin = self.eval_linear(&obj.expr, &Ctx::default())?;
            self.ilp.objective = lin;
        }
        Ok(())
    }

    fn emit_stmt_block(
        &mut self,
        rname: &str,
        body: &[Stmt],
        ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        for st in body {
            self.emit_stmt(rname, st, ctx)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, rname: &str, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // Implicit binder expansion for statements that reference domain variables (e.g. `s`)
        // without an explicit `forall`.
        //
        // Example: `DP0[s, ...]` -> expand `s in Sc` into `s=0/1`
        if !matches!(
            st,
            Stmt::ForAll { .. } | Stmt::Feature { .. } | Stmt::Let { .. } | Stmt::Add { .. }
        ) {
            let ib = self.implicit_binders_for_stmt(st, ctx)?;
            if !ib.is_empty() {
                return self.expand_binders(&ib, ctx, |g, cctx| g.emit_stmt(rname, st, &cctx));
            }
        }
        match st {
            Stmt::ForAll { binders, body } => {
                self.expand_binders(binders, ctx, |g, cctx| {
                    g.emit_stmt_block(rname, body, &cctx)
                })?;
            }
            Stmt::Feature { name, body } => {
                if self.is_feature_on(name) {
                    self.emit_stmt_block(rname, body, ctx)?;
                }
            }
            Stmt::Let { name, expr } => {
                // sequence-scoped let: we support by mutating ctx clone and emitting following statements in the same block
                // This simple emitter cannot reorder; therefore we require lets to be inside their own ForAll blocks where used immediately.
                let _ = (name, expr);
            }
            Stmt::Require(e) => self.emit_require(rname, e, ctx)?,
            Stmt::Def { lhs, rhs } => self.emit_def(rname, lhs, rhs, ctx)?,
            Stmt::ForceEq { lhs, rhs } => self.emit_force_eq(rname, lhs, rhs, ctx)?,
            Stmt::Add { .. } => (), // already handled via SourceDB
        }

        Ok(())
    }

    fn emit_require(&mut self, rname: &str, e: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        // If top-level is comparison, emit linear constraint
        match e {
            Expr::Le(a, b) => {
                let mut lhs = self.eval_linear(a, ctx)?;
                let rhs = self.eval_linear(b, ctx)?;
                lhs.sub_inplace(&rhs);
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_le_{}", rname, self.aux_id),
                    expr: lhs,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(())
            }
            Expr::Ge(a, b) => {
                let mut lhs = self.eval_linear(a, ctx)?;
                let rhs = self.eval_linear(b, ctx)?;
                lhs.sub_inplace(&rhs);
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_ge_{}", rname, self.aux_id),
                    expr: lhs,
                    sense: Sense::Ge,
                    rhs: 0.0,
                });
                Ok(())
            }
            Expr::Eq(a, b) => {
                let mut lhs = self.eval_linear(a, ctx)?;
                let rhs = self.eval_linear(b, ctx)?;
                lhs.sub_inplace(&rhs);
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_eq_{}", rname, self.aux_id),
                    expr: lhs,
                    sense: Sense::Eq,
                    rhs: 0.0,
                });
                Ok(())
            }
            Expr::Implies(a, b) => {
                // For boolean implies: a <= b
                let av = self.eval_bool(a, ctx)?;
                let bv = self.eval_bool(b, ctx)?;
                let mut lhs = LinearExpr::from_var(&av, 1.0);
                lhs.sub_inplace(&LinearExpr::from_var(&bv, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_impl_{}", rname, self.aux_id),
                    expr: lhs,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(())
            }
            _ => {
                let v = self.eval_bool(e, ctx)?;
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_bool_{}", rname, self.aux_id),
                    expr: LinearExpr::from_var(&v, 1.0),
                    sense: Sense::Eq,
                    rhs: 1.0,
                });
                Ok(())
            }
        }
    }

    fn emit_def(
        &mut self,
        rname: &str,
        lhs: &VarRef,
        rhs: &Expr,
        ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        let lv = self.var_name(lhs, ctx)?;
        let rv = self.eval_bool(rhs, ctx)?;
        let mut lhs_expr = LinearExpr::from_var(&lv, 1.0);
        lhs_expr.sub_inplace(&LinearExpr::from_var(&rv, 1.0));
        self.ilp.constraints.push(Constraint {
            name: format!("{}_def_{}", rname, self.aux_id),
            expr: lhs_expr,
            sense: Sense::Eq,
            rhs: 0.0,
        });
        Ok(())
    }

    fn emit_force_eq(
        &mut self,
        rname: &str,
        lhs: &Expr,
        rhs: &Expr,
        ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        // force allows linear expressions
        let mut le = self.eval_linear_or_boolish(lhs, ctx)?;
        let re = self.eval_linear_or_boolish(rhs, ctx)?;
        le.sub_inplace(&re);
        self.ilp.constraints.push(Constraint {
            name: format!("{}_force_{}", rname, self.aux_id),
            expr: le,
            sense: Sense::Eq,
            rhs: 0.0,
        });
        Ok(())
    }

    fn expand_binders<F>(&mut self, binders: &[Binder], ctx: &Ctx, f: F) -> Result<(), CodegenError>
    where
        F: FnMut(&mut Generator, Ctx) -> Result<(), CodegenError>,
    {
        fn cartesian(domains: &[Vec<String>]) -> Vec<Vec<String>> {
            let mut acc: Vec<Vec<String>> = vec![vec![]];
            for dom in domains {
                let mut next = vec![];
                for a in &acc {
                    for v in dom {
                        let mut b = a.clone();
                        b.push(v.clone());
                        next.push(b);
                    }
                }
                acc = next;
            }
            acc
        }

        fn rec<F>(
            g: &mut Generator,
            binders: &[Binder],
            i: usize,
            ctx: &Ctx,
            f: &mut F,
        ) -> Result<(), CodegenError>
        where
            F: FnMut(&mut Generator, Ctx) -> Result<(), CodegenError>,
        {
            if i == binders.len() {
                return f(g, ctx.clone());
            }
            let b = &binders[i];
            let mut doms: Vec<Vec<String>> = vec![];
            for dn in &b.domains {
                doms.push(g.domain_vals(dn)?);
            }
            let tuples = cartesian(&doms);
            for t in tuples {
                if t.len() != b.vars.len() {
                    // if binder vars mismatch, skip
                    continue;
                }
                let mut nctx = ctx.clone();
                for (var, val) in b.vars.iter().zip(t.iter()) {
                    nctx.bind.insert(var.clone(), val.clone());
                }
                rec(g, binders, i + 1, &nctx, f)?;
            }
            Ok(())
        }

        let mut ff = f;
        rec(self, binders, 0, ctx, &mut ff)
    }
}

// helper: deterministic concretization for sources terms
fn concretize_expr(e: &Expr, ctx: &Ctx) -> Result<Expr, CodegenError> {
    fn rec(e: &Expr, ctx: &Ctx) -> Result<Expr, CodegenError> {
        Ok(match e {
            Expr::Sym(s) => {
                if let Some(v) = ctx.lets.get(s) {
                    Expr::Sym(v.clone())
                } else if let Some(v) = ctx.bind.get(s) {
                    Expr::Sym(v.clone())
                } else {
                    Expr::Sym(s.clone())
                }
            }
            Expr::Lit(v) => Expr::Lit(*v),
            Expr::Var(vr) => {
                let mut idx = vec![];
                for i in &vr.indices {
                    idx.push(rec(i, ctx)?);
                }
                Expr::Var(VarRef {
                    name: vr.name.clone(),
                    indices: idx,
                })
            }
            Expr::NamedArg { name, value } => Expr::NamedArg {
                name: name.clone(),
                value: Box::new(rec(value, ctx)?),
            },
            Expr::Not(x) => Expr::Not(Box::new(rec(x, ctx)?)),
            Expr::Neg(x) => Expr::Neg(Box::new(rec(x, ctx)?)),
            Expr::And(a, b) => Expr::And(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Or(a, b) => Expr::Or(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Implies(a, b) => Expr::Implies(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Iff(a, b) => Expr::Iff(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Add(a, b) => Expr::Add(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Sub(a, b) => Expr::Sub(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Mul(a, b) => Expr::Mul(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Eq(a, b) => Expr::Eq(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Le(a, b) => Expr::Le(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Ge(a, b) => Expr::Ge(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Lt(a, b) => Expr::Lt(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::Gt(a, b) => Expr::Gt(Box::new(rec(a, ctx)?), Box::new(rec(b, ctx)?)),
            Expr::OrList(xs) => Expr::OrList(
                xs.iter()
                    .map(|x| rec(x, ctx))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Expr::Call { name, args } => Expr::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| rec(a, ctx))
                    .collect::<Result<Vec<_>, _>>()?,
            },
            Expr::Sum { binders, body } => Expr::Sum {
                binders: binders.clone(),
                body: Box::new(rec(body, ctx)?),
            },
            Expr::Paren(x) => Expr::Paren(Box::new(rec(x, ctx)?)),
        })
    }
    rec(e, ctx)
}

trait LinExt {
    fn sub(self, other: LinearExpr) -> LinearExpr;
}
impl LinExt for LinearExpr {
    fn sub(mut self, other: LinearExpr) -> LinearExpr {
        self.sub_inplace(&other);
        self
    }
}

fn emit_lp(ilp: &Ilp) -> String {
    let mut out = String::new();
    match ilp.sense {
        ObjSense::Minimize => out.push_str("Minimize\n obj: "),
        ObjSense::Maximize => out.push_str("Maximize\n obj: "),
    }
    out.push_str(&fmt_lin(&ilp.objective));
    out.push('\n');
    out.push_str("Subject To\n");
    for c in &ilp.constraints {
        out.push_str(&format!(
            " {}: {} {} {}\n",
            c.name,
            fmt_lin(&c.expr),
            fmt_sense(c.sense),
            fmt_num(c.rhs)
        ));
    }
    out.push_str("Binary\n");
    for b in &ilp.binaries {
        out.push_str(&format!(" {}\n", b));
    }
    out.push_str("End\n");
    out
}

fn fmt_sense(s: Sense) -> &'static str {
    match s {
        Sense::Le => "<=",
        Sense::Ge => ">=",
        Sense::Eq => "=",
    }
}

fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.6}", v)
    }
}

fn fmt_lin(e: &LinearExpr) -> String {
    let mut parts: Vec<String> = vec![];
    for (n, c) in e.terms.iter() {
        if (c - 1.0).abs() < 1e-12 {
            parts.push(format!("+1 {}", n));
        } else if (c + 1.0).abs() < 1e-12 {
            parts.push(format!("-1 {}", n));
        } else {
            parts.push(format!("{:+.6} {}", c, n));
        }
    }
    if parts.is_empty() {
        parts.push(format!("+0"));
    }
    if e.constant.abs() > 1e-12 {
        parts.push(format!("{:+.6}", e.constant));
    }
    parts.join(" ")
}
