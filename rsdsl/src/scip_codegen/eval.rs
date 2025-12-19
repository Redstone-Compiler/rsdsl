use crate::{Expr, VarKind, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};

impl Generator {
    pub(crate) fn domain_vals(&self, name: &str) -> Result<Vec<String>, CodegenError> {
        self.env
            .domains
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::UnknownDomain(name.to_string()))
    }

    pub(crate) fn eval_index(&self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
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
        // expected: x{X}_z{Z}
        let (xpart, zpart) = id.split_once("_z")?;
        let xstr = xpart.strip_prefix('x')?;
        let x = xstr.parse::<i32>().ok()?;
        let z = zpart.parse::<i32>().ok()?;
        Some((x, z))
    }

    fn eval_index_call(&self, name: &str, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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
                let d = if name == "back" || name == "supportForWallTorch" {
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

    pub(crate) fn var_name(&mut self, vr: &VarRef, ctx: &Ctx) -> Result<String, CodegenError> {
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

    pub(crate) fn eval_linear(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
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

    pub(crate) fn eval_linear_or_boolish(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match self.eval_linear(e, ctx) {
            Ok(x) => Ok(x),
            Err(CodegenError::UnsupportedLinear(_)) => {
                let v = self.eval_bool(e, ctx)?;
                Ok(LinearExpr::from_var(&v, 1.0))
            }
            Err(e) => Err(e),
        }
    }

    fn bool_const(v: f64) -> String {
        if (v - 0.0).abs() < 1e-9 { "__const0".into() } else { "__const1".into() }
    }

    fn lower_not(&mut self, a: String) -> Result<String, CodegenError> {
        if a == "__const0" { return Ok("__const1".into()); }
        if a == "__const1" { return Ok("__const0".into()); }

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

    fn lower_and(&mut self, x: String, y: String) -> Result<String, CodegenError> {
        if x == "__const0" || y == "__const0" { return Ok("__const0".into()); }
        if x == "__const1" { return Ok(y); }
        if y == "__const1" { return Ok(x); }

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

    fn lower_or(&mut self, x: String, y: String) -> Result<String, CodegenError> {
        if x == "__const1" || y == "__const1" { return Ok("__const1".into()); }
        if x == "__const0" { return Ok(y); }
        if y == "__const0" { return Ok(x); }

        let z = self.fresh_aux("or");
        self.ilp.binaries.insert(z.clone());

        // z >= x  <=>  -z + x <= 0
        self.ilp.constraints.push(Constraint {
            name: format!("or_ge1_{}", z),
            expr: LinearExpr::from_var(&x, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
            sense: Sense::Le,
            rhs: 0.0,
        });
        // z >= y
        self.ilp.constraints.push(Constraint {
            name: format!("or_ge2_{}", z),
            expr: LinearExpr::from_var(&y, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
            sense: Sense::Le,
            rhs: 0.0,
        });
        // z <= x + y  <=> z - x - y <= 0
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

    fn lower_or_list(&mut self, mut terms: Vec<String>) -> Result<String, CodegenError> {
        if terms.is_empty() {
            return Ok("__const0".into());
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
        terms.sort();
        terms.dedup();

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

    pub(crate) fn eval_bool(&mut self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Lit(v) => Ok(Self::bool_const(*v)),

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
                self.lower_not(a)
            }

            Expr::And(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                self.lower_and(x, y)
            }

            Expr::Or(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                self.lower_or(x, y)
            }

            Expr::OrList(xs) => {
                let mut terms = Vec::with_capacity(xs.len());
                for x in xs {
                    terms.push(self.eval_bool(x, ctx)?);
                }
                self.lower_or_list(terms)
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

            Expr::Sym(_) | Expr::Neg(_) | Expr::NamedArg { .. } | Expr::Sum { .. } | Expr::Eq(..)
            | Expr::Le(..) | Expr::Ge(..) | Expr::Lt(..) | Expr::Gt(..) | Expr::Add(..) | Expr::Sub(..)
            | Expr::Mul(..) => Err(CodegenError::UnsupportedBool(e.clone())),
        }
    }

    fn eval_bool_call(&mut self, name: &str, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        match name {
            "OR" => self.call_or(args, ctx),
            "Observe" => self.call_observe(args, ctx),
            "TorchOut" => self.call_torch_out(args, ctx),
            "RepOut" => self.call_rep_out(args, ctx),
            "CandidateDustAdj" => self.call_candidate_dust_adj(args, ctx),
            "ExistsConnectionCandidate" => self.call_exists_connection_candidate(args, ctx),
            "ClearUp" | "ClearDown" | "AllowCrossChoice" | "Touches" => Ok("__const1".into()),
            "TorchPowersCell" => Ok("__const0".into()),
            _ => Err(CodegenError::UnsupportedCall(name.to_string())),
        }
    }

    fn call_or(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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

    fn call_observe(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        // Observe(PIN, s=0) -> ConcreteVar
        let pin = match args.get(0) {
            Some(Expr::Sym(p)) => p.clone(),
            Some(Expr::Var(vr)) => vr.name.clone(),
            Some(Expr::Paren(p)) => match &**p {
                Expr::Sym(pn) => pn.clone(),
                _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
            },
            _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
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

    fn call_torch_out(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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

    fn call_rep_out(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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

    fn call_candidate_dust_adj(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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

    fn call_exists_connection_candidate(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
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
}
