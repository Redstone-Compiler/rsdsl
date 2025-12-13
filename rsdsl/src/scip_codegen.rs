use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    Binder, Decl, EnumDecl, Expr, ModelSpec, ObjSense, Objective, Rule, ScenarioDecl, Stmt, Tok,
    TokDelim, VarDecl, VarKind, VarRef,
};

/// Concrete cell coordinate (x,z) for 2D grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cell {
    pub x: i32,
    pub z: i32,
}

/// Concrete direction for typical 2D neighborhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dir {
    N,
    E,
    S,
    W,
}

impl Dir {
    pub fn all() -> [Dir; 4] {
        [Dir::N, Dir::E, Dir::S, Dir::W]
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Dir::N => "N",
            Dir::E => "E",
            Dir::S => "S",
            Dir::W => "W",
        }
    }
    pub fn opp(&self) -> Dir {
        match self {
            Dir::N => Dir::S,
            Dir::S => Dir::N,
            Dir::E => Dir::W,
            Dir::W => Dir::E,
        }
    }
    pub fn delta(&self) -> (i32, i32) {
        match self {
            Dir::N => (0, -1),
            Dir::S => (0, 1),
            Dir::E => (1, 0),
            Dir::W => (-1, 0),
        }
    }
}

/// Concrete layer (GROUND/TOP) for 2-layer modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Layer {
    GROUND,
    TOP,
}
impl Layer {
    pub fn all() -> [Layer; 2] {
        [Layer::GROUND, Layer::TOP]
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Layer::GROUND => "GROUND",
            Layer::TOP => "TOP",
        }
    }
}

/// Runtime instantiation data needed to expand quantifiers into a flat ILP.
pub struct Instance {
    /// All cells used for `Cell` domain.
    pub cells: Vec<Cell>,
    /// Scenario values for scenario decls (usually [0,1]).
    pub scenarios: Vec<i32>,
    /// Optional parameters (e.g. wT, wS).
    pub params: HashMap<String, f64>,
    /// Pin observe mapping: Observe(PIN, s=<scenario>) -> a concrete variable reference.
    pub observe: std::sync::Arc<dyn Fn(&str, i32) -> ConcreteVar + Send + Sync>,
}

/// A concrete variable instance: name + fully evaluated indices (already formatted).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConcreteVar {
    pub name: String,
    pub indices: Vec<String>,
}
impl ConcreteVar {
    pub fn lp_name(&self) -> String {
        if self.indices.is_empty() {
            return sanitize(&self.name);
        }
        let mut s = sanitize(&self.name);
        for idx in &self.indices {
            s.push_str("__");
            s.push_str(&sanitize(idx));
        }
        s
    }
}

#[derive(Debug, Clone)]
pub struct LinearExpr {
    pub constant: f64,
    pub terms: BTreeMap<String, f64>, // var -> coeff
}
impl LinearExpr {
    pub fn zero() -> Self {
        Self {
            constant: 0.0,
            terms: BTreeMap::new(),
        }
    }
    pub fn from_var(v: &str, coeff: f64) -> Self {
        let mut t = BTreeMap::new();
        t.insert(v.to_string(), coeff);
        Self {
            constant: 0.0,
            terms: t,
        }
    }
    pub fn from_const(c: f64) -> Self {
        Self {
            constant: c,
            terms: BTreeMap::new(),
        }
    }
    pub fn add(mut self, other: LinearExpr) -> Self {
        self.constant += other.constant;
        for (k, v) in other.terms {
            *self.terms.entry(k).or_insert(0.0) += v;
        }
        self
    }
    pub fn sub(mut self, other: LinearExpr) -> Self {
        self.constant -= other.constant;
        for (k, v) in other.terms {
            *self.terms.entry(k).or_insert(0.0) -= v;
        }
        self
    }
    pub fn scale(mut self, s: f64) -> Self {
        self.constant *= s;
        for v in self.terms.values_mut() {
            *v *= s;
        }
        self
    }
    pub fn to_lp(&self) -> String {
        let mut out = String::new();
        let mut first = true;
        // terms
        for (var, coeff) in &self.terms {
            if *coeff == 0.0 {
                continue;
            }
            if first {
                first = false;
                out.push_str(&format!("{:+} {}", coeff, var));
            } else {
                out.push_str(&format!(" {:+} {}", coeff, var));
            }
        }
        if self.constant != 0.0 || first {
            if first {
                out.push_str(&format!("{:+}", self.constant));
            } else {
                out.push_str(&format!(" {:+}", self.constant));
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub enum Sense {
    Eq,
    Le,
    Ge,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: String,
    pub expr: LinearExpr,
    pub sense: Sense,
    pub rhs: f64,
}

pub struct Ilp {
    pub objective_min: bool,
    pub objective: LinearExpr,
    pub constraints: Vec<Constraint>,
    pub binaries: BTreeSet<String>,
}
impl Ilp {
    pub fn to_scip_lp(&self) -> String {
        let mut out = String::new();

        out.push_str(if self.objective_min {
            "Minimize\n"
        } else {
            "Maximize\n"
        });

        // objective: 상수항은 목적함수에 영향 없으니 버리는 게 안전
        let mut obj = self.objective.clone();
        obj.constant = 0.0;

        out.push_str(" obj: ");
        out.push_str(&obj.to_lp());
        out.push_str("\nSubject To\n");

        for c in &self.constraints {
            // 🔥 핵심: LHS constant를 RHS로 넘김
            let mut lhs = c.expr.clone();
            let rhs = c.rhs - lhs.constant;
            lhs.constant = 0.0;

            out.push_str(&format!(" {}: {} ", sanitize(&c.name), lhs.to_lp()));
            match c.sense {
                Sense::Eq => out.push_str("= "),
                Sense::Le => out.push_str("<= "),
                Sense::Ge => out.push_str(">= "),
            }
            out.push_str(&format!("{:+}\n", rhs));
        }

        out.push_str("Binary\n");
        for v in &self.binaries {
            out.push_str(" ");
            out.push_str(v);
            out.push_str("\n");
        }
        out.push_str("End\n");
        out
    }
}

#[derive(Debug)]
pub enum CodegenError {
    MissingScenario,
    Unsupported(String),
    UnknownSymbol(String),
    ConflictingBinder { var: String, a: String, b: String },
    BadBinder(String),
    IndexEval(String),
}

/// Primary entry: expand `spec` into a flat SCIP .lp ILP string.
pub fn codegen_scip_lp(spec: &ModelSpec, inst: &Instance) -> Result<String, CodegenError> {
    let env = Env::from_spec(spec, inst)?;
    let mut gen = Generator::new(env);
    gen.lower_model(spec)?;
    Ok(gen.ilp.to_scip_lp())
}

/// Build-time environment extracted from the spec + instance.
struct Env {
    // declared var signatures: name -> indices domains
    vars: HashMap<String, Vec<String>>,
    // enum domains: name -> variants (strings)
    enums: HashMap<String, Vec<String>>,
    // scenario domains: name -> values (strings)
    scenarios: HashMap<String, Vec<String>>,
    // runtime instance
    inst_cells: Vec<Cell>,
    inst_scenarios: Vec<i32>,
    params: HashMap<String, f64>,
    observe: std::sync::Arc<dyn Fn(&str, i32) -> ConcreteVar + Send + Sync>,
}

impl Env {
    fn from_spec(spec: &ModelSpec, inst: &Instance) -> Result<Self, CodegenError> {
        let mut vars = HashMap::new();
        let mut enums = HashMap::new();
        let mut scenarios = HashMap::new();

        for d in &spec.decls {
            match d {
                Decl::Enum(EnumDecl { name, variants }) => {
                    enums.insert(
                        name.clone(),
                        variants.iter().map(|v| v.name.clone()).collect(),
                    );
                }
                Decl::Scenario(ScenarioDecl { name, values }) => {
                    scenarios.insert(name.clone(), values.clone());
                }
                Decl::Var(VarDecl { name, indices, .. }) => {
                    vars.insert(name.clone(), indices.clone());
                }
                _ => {}
            }
        }

        Ok(Self {
            vars,
            enums,
            scenarios,
            inst_cells: inst.cells.clone(),
            inst_scenarios: inst.scenarios.clone(),
            params: inst.params.clone(),
            observe: inst.observe.clone(),
        })
    }

    fn domain_values(&self, dom: &str) -> Result<Vec<String>, CodegenError> {
        if dom == "Cell" {
            return Ok(self
                .inst_cells
                .iter()
                .map(|c| format!("x{}_z{}", c.x, c.z))
                .collect());
        }
        if let Some(v) = self.enums.get(dom) {
            return Ok(v.clone());
        }
        if let Some(v) = self.scenarios.get(dom) {
            return Ok(v.clone());
        }
        // Common aliases: Sc for scenario
        if dom == "Sc" {
            return Ok(self.inst_scenarios.iter().map(|v| v.to_string()).collect());
        }
        Err(CodegenError::Unsupported(format!("unknown domain {dom}")))
    }

    fn param(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }
}

struct Generator {
    env: Env,
    ilp: Ilp,
    aux_counter: usize,
}

impl Generator {
    fn new(env: Env) -> Self {
        Self {
            env,
            ilp: Ilp {
                objective_min: true,
                objective: LinearExpr::zero(),
                constraints: vec![],
                binaries: BTreeSet::new(),
            },
            aux_counter: 0,
        }
    }

    fn lower_model(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
        // Objective
        if let Some(obj) = &spec.objective {
            self.ilp.objective_min = obj.sense == ObjSense::Minimize;
            let expr = self.eval_linear_expr(&obj.body, &Ctx::default())?;
            self.ilp.objective = expr;
        }

        // Fix boolean constants as binaries:
        self.ilp.binaries.insert("__const0".into());
        self.ilp.binaries.insert("__const1".into());
        self.ilp.constraints.push(Constraint {
            name: "const0".into(),
            expr: LinearExpr::from_var("__const0", 1.0),
            sense: Sense::Eq,
            rhs: 0.0,
        });
        self.ilp.constraints.push(Constraint {
            name: "const1".into(),
            expr: LinearExpr::from_var("__const1", 1.0),
            sense: Sense::Eq,
            rhs: 1.0,
        });

        // Rules
        for rule in &spec.rules {
            self.lower_rule(rule)?;
        }

        Ok(())
    }

    fn lower_rule(&mut self, rule: &Rule) -> Result<(), CodegenError> {
        // We process statements sequentially, with `let` bindings captured as AST in ctx.
        self.lower_block(&rule.name, &rule.body, &Ctx::default())
    }

    fn lower_block(
        &mut self,
        rule_name: &str,
        body: &[Stmt],
        base_ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        // We want to expand each statement with implicit forall. We do it per-stmt.
        let mut ctx = base_ctx.clone();
        for (i, st) in body.iter().enumerate() {
            match st {
                Stmt::Let { name, value } => {
                    ctx.lets.insert(name.clone(), value.clone());
                }
                Stmt::Feature { name: _, body } => {
                    // For now, we always include features. You can add an enable/disable map later.
                    self.lower_block(rule_name, body, &ctx)?;
                }
                Stmt::ForAll { binders, body } => {
                    // Explicit forall(...) { ... } block expansion.
                    self.expand_forall_block(
                        rule_name,
                        i,
                        binders,
                        body,
                        &ctx,
                        0,
                        &mut HashMap::new(),
                    )?;
                }
                _ => {
                    // Expand this statement for all inferred binder vars.
                    let let_names: BTreeSet<String> = ctx.lets.keys().cloned().collect();
                    let binders = infer_binders(st, &self.env.vars, &let_names)?;
                    let binder_list: Vec<(String, String)> = binders.into_iter().collect(); // (var, domain)
                    self.expand_forall(
                        rule_name,
                        i,
                        st,
                        &ctx,
                        &binder_list,
                        0,
                        &mut HashMap::new(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn expand_forall(
        &mut self,
        rule_name: &str,
        stmt_idx: usize,
        st: &Stmt,
        ctx: &Ctx,
        binder_list: &[(String, String)],
        depth: usize,
        assign: &mut HashMap<String, String>,
    ) -> Result<(), CodegenError> {
        if depth == binder_list.len() {
            // Evaluate statement under this assignment.
            let mut local = ctx.clone();
            local.binders = assign.clone();
            self.lower_stmt(rule_name, stmt_idx, st, &local)?;
            return Ok(());
        }
        let (v, dom) = &binder_list[depth];
        let vals = self.env.domain_values(dom)?;
        for val in vals {
            assign.insert(v.clone(), val);
            self.expand_forall(rule_name, stmt_idx, st, ctx, binder_list, depth + 1, assign)?;
        }
        Ok(())
    }

    fn expand_forall_block(
        &mut self,
        rule_name: &str,
        stmt_idx: usize,
        binders: &[(String, String)],
        body: &[Stmt],
        ctx: &Ctx,
        depth: usize,
        assign: &mut HashMap<String, String>,
    ) -> Result<(), CodegenError> {
        if depth == binders.len() {
            let mut local = ctx.clone();
            local.binders = assign.clone();
            // In explicit forall, DO NOT run implicit binder inference; evaluate body directly.
            self.lower_block_explicit(rule_name, stmt_idx, body, &local)?;
            return Ok(());
        }
        let (v, dom) = &binders[depth];
        let vals = self.env.domain_values(dom)?;
        for val in vals {
            assign.insert(v.clone(), val);
            self.expand_forall_block(rule_name, stmt_idx, binders, body, ctx, depth + 1, assign)?;
        }
        Ok(())
    }

    fn lower_block_explicit(
        &mut self,
        rule_name: &str,
        base_idx: usize,
        body: &[Stmt],
        base_ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        let mut ctx = base_ctx.clone();
        for (j, st) in body.iter().enumerate() {
            match st {
                Stmt::Let { name, value } => {
                    ctx.lets.insert(name.clone(), value.clone());
                }
                Stmt::Feature { name: _, body } => {
                    self.lower_block_explicit(rule_name, base_idx, body, &ctx)?;
                }
                Stmt::ForAll { binders, body } => {
                    self.expand_forall_block(
                        rule_name,
                        base_idx * 1000 + j,
                        binders,
                        body,
                        &ctx,
                        0,
                        &mut HashMap::new(),
                    )?;
                }
                _ => {
                    let idx = base_idx * 1000 + j;
                    self.lower_stmt(rule_name, idx, st, &ctx)?;
                }
            }
        }
        Ok(())
    }

    fn lower_stmt(
        &mut self,
        rule_name: &str,
        stmt_idx: usize,
        st: &Stmt,
        ctx: &Ctx,
    ) -> Result<(), CodegenError> {
        match st {
            Stmt::ForAll { binders, body } => {
                self.expand_forall_block(
                    rule_name,
                    stmt_idx,
                    binders,
                    body,
                    ctx,
                    0,
                    &mut HashMap::new(),
                )?;
            }
            Stmt::Require(e) => {
                let v = self.eval_bool_expr(e, ctx)?;
                // require v == 1
                self.add_eq1(&format!("{rule_name}_req{stmt_idx}_{}", ctx.suffix()), &v);
            }
            Stmt::Def { lhs, op: _, rhs } => {
                let lhs_v = self.eval_varref(lhs, ctx)?;
                self.ilp.binaries.insert(lhs_v.clone());
                let rhs_v = self.eval_bool_expr(rhs, ctx)?;
                // lhs == rhs
                self.add_eq_var(
                    &format!("{rule_name}_def{stmt_idx}_{}", ctx.suffix()),
                    &lhs_v,
                    &rhs_v,
                );
            }
            Stmt::ForceEq { lhs, rhs } => {
                // Special-case: Observe(pin, s=..) is treated via instance mapping.
                let le = self.eval_linear_expr(lhs, ctx)?;
                let re = self.eval_linear_expr(rhs, ctx)?;
                // move to LHS: le - re == 0
                let expr = le.sub(re);
                self.ilp.constraints.push(Constraint {
                    name: format!("{rule_name}_force{stmt_idx}_{}", ctx.suffix()),
                    expr,
                    sense: Sense::Eq,
                    rhs: 0.0,
                });
            }
            Stmt::Add { .. } => {
                // sources wiring not lowered in this minimal codegen
            }
            Stmt::Feature { .. } => {}
            Stmt::Let { .. } => {}
        }
        Ok(())
    }

    fn add_eq1(&mut self, base: &str, v: &str) {
        self.ilp.constraints.push(Constraint {
            name: format!("{base}_{}", self.aux_counter),
            expr: LinearExpr::from_var(v, 1.0),
            sense: Sense::Eq,
            rhs: 1.0,
        });
        self.aux_counter += 1;
    }

    fn add_eq_var(&mut self, base: &str, a: &str, b: &str) {
        // a - b == 0
        let expr = LinearExpr::from_var(a, 1.0).sub(LinearExpr::from_var(b, 1.0));
        self.ilp.constraints.push(Constraint {
            name: format!("{base}_{}", self.aux_counter),
            expr,
            sense: Sense::Eq,
            rhs: 0.0,
        });
        self.aux_counter += 1;
    }

    fn new_aux(&mut self, prefix: &str) -> String {
        let v = format!("__aux_{}_{}", prefix, self.aux_counter);
        self.aux_counter += 1;
        let v = sanitize(&v);
        self.ilp.binaries.insert(v.clone());
        v
    }

    fn eval_varref(&mut self, vr: &VarRef, ctx: &Ctx) -> Result<String, CodegenError> {
        let mut idxs = vec![];
        for e in &vr.indices {
            idxs.push(self.eval_index(e, ctx)?);
        }
        let cv = ConcreteVar {
            name: vr.name.clone(),
            indices: idxs,
        };
        let vname = cv.lp_name();
        self.ilp.binaries.insert(vname.clone());
        Ok(vname)
    }

    fn eval_index(&mut self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Sym(s) => {
                if let Some(v) = ctx.binders.get(s) {
                    return Ok(v.clone());
                }
                if let Some(v) = ctx.lets.get(s) {
                    return self.eval_index(v, ctx);
                }
                // Allow Layer/Dir literals as symbols
                Ok(s.clone())
            }
            Expr::Lit(l) => Ok(l.clone()),
            Expr::Call { name, args } => {
                // handle neigh(c,d) / supportForWallTorch(c,d) as pure string evaluation for now:
                // return "NONE" if cannot evaluate -> caller should avoid using it for out-of-grid; we can't in this minimal engine.
                let mut ev = vec![];
                for a in args {
                    match a {
                        Expr::NamedArg { name: _, value } => ev.push(self.eval_index(value, ctx)?),
                        _ => ev.push(self.eval_index(a, ctx)?),
                    }
                }
                Ok(format!("{name}({})", ev.join(",")))
            }
            Expr::Paren(x) => self.eval_index(x, ctx),
            _ => Err(CodegenError::IndexEval(format!(
                "unsupported index expr: {e:?}"
            ))),
        }
    }

    fn eval_linear_expr(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match e {
            Expr::Lit(l) => {
                let v: f64 = l
                    .parse()
                    .map_err(|_| CodegenError::Unsupported(format!("bad literal {l}")))?;
                Ok(LinearExpr::from_const(v))
            }
            Expr::Var(vr) => {
                let v = self.eval_varref(vr, ctx)?;
                Ok(LinearExpr::from_var(&v, 1.0))
            }
            Expr::Sym(s) => {
                if let Some(v) = ctx.binders.get(s) {
                    return Ok(LinearExpr::from_const(v.parse::<f64>().unwrap_or(0.0)));
                }
                if let Some(v) = self.env.param(s) {
                    return Ok(LinearExpr::from_const(v));
                }
                Err(CodegenError::UnknownSymbol(s.clone()))
            }
            Expr::Add(a, b) => Ok(self
                .eval_linear_expr(a, ctx)?
                .add(self.eval_linear_expr(b, ctx)?)),
            Expr::Sub(a, b) => Ok(self
                .eval_linear_expr(a, ctx)?
                .sub(self.eval_linear_expr(b, ctx)?)),
            Expr::Mul(a, b) => {
                // allow const * linear
                let la = self.eval_linear_expr(a, ctx)?;
                let lb = self.eval_linear_expr(b, ctx)?;
                if la.terms.is_empty() {
                    Ok(lb.scale(la.constant))
                } else if lb.terms.is_empty() {
                    Ok(la.scale(lb.constant))
                } else {
                    Err(CodegenError::Unsupported("nonlinear mul".into()))
                }
            }
            Expr::Sum { binder, body } => {
                let (bind_vars, domains) = parse_binder(binder)?;
                if bind_vars.len() != domains.len() {
                    return Err(CodegenError::BadBinder(
                        "binder vars/domains mismatch".into(),
                    ));
                }
                let mut acc = LinearExpr::zero();
                // Build cartesian product.
                let dom_vals: Vec<Vec<String>> = domains
                    .iter()
                    .map(|d| self.env.domain_values(d))
                    .collect::<Result<_, _>>()?;
                for combo in cartesian(&dom_vals) {
                    let mut c2 = ctx.clone();
                    for (i, v) in bind_vars.iter().enumerate() {
                        c2.binders.insert(v.clone(), combo[i].clone());
                    }
                    acc = acc.add(self.eval_linear_expr(body, &c2)?);
                }
                Ok(acc)
            }
            Expr::Paren(x) => self.eval_linear_expr(x, ctx),
            Expr::Call { name, args } => {
                if name == "Observe" {
                    // Observe(IN, s=0)
                    let mut pin: Option<String> = None;
                    let mut sc: Option<i32> = None;
                    for a in args {
                        match a {
                            Expr::Sym(s) => {
                                if pin.is_none() {
                                    pin = Some(s.clone());
                                }
                            }
                            Expr::NamedArg { name, value } if name == "s" => {
                                let sv = self.eval_index(value, ctx)?;
                                sc = Some(sv.parse().map_err(|_| {
                                    CodegenError::Unsupported("Observe s must be int".into())
                                })?);
                            }
                            _ => {}
                        }
                    }
                    let pin =
                        pin.ok_or_else(|| CodegenError::Unsupported("Observe missing pin".into()))?;
                    let sc =
                        sc.ok_or_else(|| CodegenError::Unsupported("Observe missing s=".into()))?;
                    let cv = (self.env.observe)(&pin, sc);
                    let vname = cv.lp_name();
                    self.ilp.binaries.insert(vname.clone());
                    return Ok(LinearExpr::from_var(&vname, 1.0));
                }
                Err(CodegenError::Unsupported(format!(
                    "call {name} in linear expr"
                )))
            }
            Expr::NamedArg { .. } => Err(CodegenError::Unsupported("named arg as expr".into())),
            _ => Err(CodegenError::Unsupported(format!(
                "unsupported linear expr: {e:?}"
            ))),
        }
    }

    fn eval_bool_expr(&mut self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Var(vr) => self.eval_varref(vr, ctx),
            Expr::Lit(l) => {
                if l == "0" {
                    return Ok("__const0".into());
                }
                if l == "1" {
                    return Ok("__const1".into());
                }
                Err(CodegenError::Unsupported(format!(
                    "bool literal must be 0/1, got {l}"
                )))
            }
            Expr::Not(x) => {
                let a = self.eval_bool_expr(x, ctx)?;
                if a == "__const0" {
                    return Ok("__const1".into());
                }
                if a == "__const1" {
                    return Ok("__const0".into());
                }
                let v = self.new_aux("not");
                // v + a = 1
                let expr = LinearExpr::from_var(&v, 1.0).add(LinearExpr::from_var(&a, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("not_{}", v),
                    expr,
                    sense: Sense::Eq,
                    rhs: 1.0,
                });
                Ok(v)
            }
            Expr::And(a, b) => {
                let x = self.eval_bool_expr(a, ctx)?;
                let y = self.eval_bool_expr(b, ctx)?;
                if x == "__const0" || y == "__const0" {
                    return Ok("__const0".into());
                }
                if x == "__const1" {
                    return Ok(y);
                }
                if y == "__const1" {
                    return Ok(x);
                }
                let v = self.new_aux("and");
                // v <= x, v <= y, v >= x+y-1
                self.ilp.constraints.push(Constraint {
                    name: format!("and_le1_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0).sub(LinearExpr::from_var(&x, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("and_le2_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0).sub(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                let expr = LinearExpr::from_var(&v, 1.0)
                    .sub(LinearExpr::from_var(&x, 1.0))
                    .sub(LinearExpr::from_var(&y, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("and_ge_{}", v),
                    expr,
                    sense: Sense::Ge,
                    rhs: -1.0,
                });
                Ok(v)
            }
            Expr::Or(a, b) => {
                let x = self.eval_bool_expr(a, ctx)?;
                let y = self.eval_bool_expr(b, ctx)?;
                if x == "__const1" || y == "__const1" {
                    return Ok("__const1".into());
                }
                if x == "__const0" {
                    return Ok(y);
                }
                if y == "__const0" {
                    return Ok(x);
                }
                let v = self.new_aux("or");
                // v >= x, v >= y, v <= x+y
                self.ilp.constraints.push(Constraint {
                    name: format!("or_ge1_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0).sub(LinearExpr::from_var(&x, 1.0)),
                    sense: Sense::Ge,
                    rhs: 0.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("or_ge2_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0).sub(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Ge,
                    rhs: 0.0,
                });
                let expr = LinearExpr::from_var(&v, 1.0)
                    .sub(LinearExpr::from_var(&x, 1.0))
                    .sub(LinearExpr::from_var(&y, 1.0));
                self.ilp.constraints.push(Constraint {
                    name: format!("or_le_{}", v),
                    expr,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(v)
            }
            Expr::Implies(a, b) => {
                // (1-a) OR b
                let na = Expr::Not(a.clone());
                let or = Expr::Or(Box::new(na), b.clone());
                self.eval_bool_expr(&or, ctx)
            }
            Expr::Eq(a, b) => {
                // boolean equality: a == b
                let x = self.eval_bool_expr(a, ctx)?;
                let y = self.eval_bool_expr(b, ctx)?;
                if x == y {
                    return Ok("__const1".into());
                }
                let v = self.new_aux("eq");
                // v = 1 - (x xor y). linearize with 4 constraints:
                // v <= 1 - x + y; v <= 1 + x - y; v >= 1 - x - y; v >= -1 + x + y
                self.ilp.constraints.push(Constraint {
                    name: format!("eq_c1_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0)
                        .add(LinearExpr::from_var(&x, 1.0))
                        .sub(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Le,
                    rhs: 1.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("eq_c2_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0)
                        .sub(LinearExpr::from_var(&x, 1.0))
                        .add(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Le,
                    rhs: 1.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("eq_c3_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0)
                        .add(LinearExpr::from_var(&x, 1.0))
                        .add(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Ge,
                    rhs: 1.0,
                });
                self.ilp.constraints.push(Constraint {
                    name: format!("eq_c4_{}", v),
                    expr: LinearExpr::from_var(&v, 1.0)
                        .sub(LinearExpr::from_var(&x, 1.0))
                        .sub(LinearExpr::from_var(&y, 1.0)),
                    sense: Sense::Ge,
                    rhs: -1.0,
                });
                Ok(v)
            }
            Expr::OrList(xs) => {
                // fold OR over list
                let mut it = xs.iter();
                let Some(first) = it.next() else {
                    return Ok("__const0".into());
                };
                let mut acc = self.eval_bool_expr(first, ctx)?;
                for x in it {
                    let rhs = self.eval_bool_expr(x, ctx)?;
                    let tmp = self.new_aux("orlist");
                    // tmp = acc OR rhs
                    // tmp >= acc; tmp >= rhs; tmp <= acc+rhs
                    self.ilp.constraints.push(Constraint {
                        name: format!("orlist_ge1_{}", tmp),
                        expr: LinearExpr::from_var(&tmp, 1.0).sub(LinearExpr::from_var(&acc, 1.0)),
                        sense: Sense::Ge,
                        rhs: 0.0,
                    });
                    self.ilp.constraints.push(Constraint {
                        name: format!("orlist_ge2_{}", tmp),
                        expr: LinearExpr::from_var(&tmp, 1.0).sub(LinearExpr::from_var(&rhs, 1.0)),
                        sense: Sense::Ge,
                        rhs: 0.0,
                    });
                    self.ilp.constraints.push(Constraint {
                        name: format!("orlist_le_{}", tmp),
                        expr: LinearExpr::from_var(&tmp, 1.0)
                            .sub(LinearExpr::from_var(&acc, 1.0))
                            .sub(LinearExpr::from_var(&rhs, 1.0)),
                        sense: Sense::Le,
                        rhs: 0.0,
                    });
                    acc = tmp;
                }
                Ok(acc)
            }
            Expr::Paren(x) => self.eval_bool_expr(x, ctx),
            Expr::Call { name, .. } => {
                // allow Observe in bool context
                let le = self.eval_linear_expr(e, ctx)?;
                if le.terms.len() == 1 && le.constant == 0.0 {
                    return Ok(le.terms.keys().next().unwrap().clone());
                }
                Err(CodegenError::Unsupported(format!(
                    "call {name} in bool expr"
                )))
            }
            _ => Err(CodegenError::Unsupported(format!(
                "unsupported bool expr: {e:?}"
            ))),
        }
    }
}

/// Statement-local context for evaluation.
#[derive(Clone, Default)]
struct Ctx {
    binders: HashMap<String, String>,
    lets: HashMap<String, Expr>,
}
impl Ctx {
    fn suffix(&self) -> String {
        // stable suffix for constraint naming: join binder assigns
        let mut kv: Vec<_> = self.binders.iter().collect();
        kv.sort_by_key(|(k, _)| *k);
        kv.into_iter()
            .map(|(k, v)| format!("{k}{v}"))
            .collect::<Vec<_>>()
            .join("_")
    }
}

/// Infer binder variables and their domain names from a statement by looking at varrefs with known signatures.
fn infer_binders(
    st: &Stmt,
    sigs: &HashMap<String, Vec<String>>,
    let_names: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>, CodegenError> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut visit_vr = |vr: &VarRef| -> Result<(), CodegenError> {
        let Some(domains) = sigs.get(&vr.name) else {
            return Ok(());
        };
        for (i, idx) in vr.indices.iter().enumerate() {
            if let Expr::Sym(v) = idx {
                if let Some(dom) = domains.get(i) {
                    if let_names.contains(v) {
                        continue;
                    }
                    if let Some(prev) = map.get(v) {
                        if prev != dom {
                            return Err(CodegenError::ConflictingBinder {
                                var: v.clone(),
                                a: prev.clone(),
                                b: dom.clone(),
                            });
                        }
                    } else {
                        map.insert(v.clone(), dom.clone());
                    }
                }
            }
        }
        Ok(())
    };

    fn walk_expr(
        e: &Expr,
        sigs: &HashMap<String, Vec<String>>,
        visit_vr: &mut dyn FnMut(&VarRef) -> Result<(), CodegenError>,
    ) -> Result<(), CodegenError> {
        match e {
            Expr::Var(vr) => visit_vr(vr)?,
            Expr::Not(x) => walk_expr(x, sigs, visit_vr)?,
            Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Eq(a, b)
            | Expr::Le(a, b)
            | Expr::Ge(a, b)
            | Expr::Lt(a, b)
            | Expr::Gt(a, b)
            | Expr::Implies(a, b) => {
                walk_expr(a, sigs, visit_vr)?;
                walk_expr(b, sigs, visit_vr)?;
            }
            Expr::OrList(xs) => {
                for x in xs {
                    walk_expr(x, sigs, visit_vr)?;
                }
            }
            Expr::Sum { binder: _, body } => walk_expr(body, sigs, visit_vr)?,
            Expr::Call { args, .. } => {
                for a in args {
                    match a {
                        Expr::NamedArg { value, .. } => walk_expr(value, sigs, visit_vr)?,
                        _ => walk_expr(a, sigs, visit_vr)?,
                    }
                }
            }
            Expr::Paren(x) => walk_expr(x, sigs, visit_vr)?,
            Expr::NamedArg { value, .. } => walk_expr(value, sigs, visit_vr)?,
            Expr::Sym(_) | Expr::Lit(_) => {}
        }
        Ok(())
    }

    match st {
        Stmt::ForAll { binders, body } => {
            // 1) explicit binders: validate duplicates + build "shadow" set
            let mut shadow: BTreeSet<String> = let_names.iter().cloned().collect();

            // (선택) ForAll에 있는 binder들끼리 충돌 검사
            for (v, dom) in binders {
                if let Some(prev) = map.get(v) {
                    if prev != dom {
                        return Err(CodegenError::ConflictingBinder {
                            var: v.clone(),
                            a: prev.clone(),
                            b: dom.clone(),
                        });
                    }
                } else {
                    // 여기서 map에 넣어둘지 말지 선택:
                    // - 넣으면: "ForAll은 explicit binder를 가지고 있다"는 정보를 리턴함
                    // - 안 넣으면: infer_binders 결과가 ForAll 바깥 확장에 영향 주지 않음
                    //
                    // 추천: ForAll은 codegen이 별도 처리하므로 'map에는 넣지 않는 것'이 안전함.
                    // map.insert(v.clone(), dom.clone());
                }
                shadow.insert(v.clone());
            }

            // 2) body scan: free symbols에 대한 충돌 검사(및 선택적으로 추론 결과 수집)
            // shadow(= explicit binder + let)를 넘겨서, 그 이름들은 binder로 추론되지 않게 함.
            for s in body {
                let inner = infer_binders(s, sigs, &shadow)?;

                // ✅ 옵션 A (보수적): 바디에서 새로 추론된 binder는 "그냥 충돌검사용"으로만 쓰고 버림
                // let _ = inner;

                // ✅ 옵션 B (편의): 바디에서 명시되지 않은 자유 심볼을 추가 binder로 허용하고 map에 합침
                for (v, dom) in inner {
                    if let Some(prev) = map.get(&v) {
                        if prev != &dom {
                            return Err(CodegenError::ConflictingBinder {
                                var: v.clone(),
                                a: prev.clone(),
                                b: dom.clone(),
                            });
                        }
                    } else {
                        map.insert(v, dom);
                    }
                }
            }

            // 3) ForAll 자체는 implicit-expansion 대상으로 삼지 않게 빈 맵 리턴하고 싶으면:
            // return Ok(BTreeMap::new());

            // 4) 옵션 B를 썼다면 "바디에서 새로 추론된 것만" 리턴:
        }
        Stmt::Require(e) => walk_expr(e, sigs, &mut visit_vr)?,
        Stmt::Def { lhs, rhs, .. } => {
            visit_vr(lhs)?;
            walk_expr(rhs, sigs, &mut visit_vr)?;
        }
        Stmt::ForceEq { lhs, rhs } => {
            walk_expr(lhs, sigs, &mut visit_vr)?;
            walk_expr(rhs, sigs, &mut visit_vr)?;
        }
        Stmt::Add {
            target,
            value,
            cond,
            ..
        } => {
            visit_vr(target)?;
            walk_expr(value, sigs, &mut visit_vr)?;
            if let Some(c) = cond {
                walk_expr(c, sigs, &mut visit_vr)?;
            }
        }
        Stmt::Let { .. } => {}
        Stmt::Feature { body, .. } => {
            for s in body {
                let _ = infer_binders(s, sigs, let_names)?;
            }
        }
    }
    Ok(map)
}

/// Parse binder tokens into (vars, domains) with minimal patterns:
/// - `c in Cell`
/// - `(c,d) in Cell * Dir`
fn parse_binder(b: &Binder) -> Result<(Vec<String>, Vec<String>), CodegenError> {
    // Convert token stream into a flat string-ish pattern.
    let toks = &b.toks;

    // Helper: extract identifiers from a token list.
    fn toks_to_idents(ts: &[Tok]) -> Vec<String> {
        let mut out = vec![];
        for t in ts {
            match t {
                Tok::Ident(s) => out.push(s.clone()),
                Tok::Group { delim: _, inner } => out.extend(toks_to_idents(inner)),
                _ => {}
            }
        }
        out
    }

    // Pattern 1: ident in Ident
    if toks.len() == 3 {
        if let (Tok::Ident(v), Tok::Ident(in_kw), Tok::Ident(dom)) = (&toks[0], &toks[1], &toks[2])
        {
            if in_kw == "in" {
                return Ok((vec![v.clone()], vec![dom.clone()]));
            }
        }
    }

    // Pattern 2: group(parens) in Ident * Ident
    if toks.len() == 5 {
        if let (
            Tok::Group {
                delim: TokDelim::Paren,
                inner,
            },
            Tok::Ident(in_kw),
            Tok::Ident(dom1),
            Tok::Punct('*'),
            Tok::Ident(dom2),
        ) = (&toks[0], &toks[1], &toks[2], &toks[3], &toks[4])
        {
            if in_kw == "in" {
                let vars = toks_to_idents(inner);
                if vars.len() != 2 {
                    return Err(CodegenError::BadBinder(
                        "tuple binder must have 2 vars".into(),
                    ));
                }
                return Ok((vars, vec![dom1.clone(), dom2.clone()]));
            }
        }
    }

    Err(CodegenError::BadBinder(format!(
        "unsupported binder tokens: {toks:?}"
    )))
}

/// Cartesian product helper.
fn cartesian(dom_vals: &[Vec<String>]) -> Vec<Vec<String>> {
    if dom_vals.is_empty() {
        return vec![vec![]];
    }
    let mut acc: Vec<Vec<String>> = vec![vec![]];
    for vals in dom_vals {
        let mut next = vec![];
        for prefix in &acc {
            for v in vals {
                let mut p = prefix.clone();
                p.push(v.clone());
                next.push(p);
            }
        }
        acc = next;
    }
    acc
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
