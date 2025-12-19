use crate::{Expr, ModelSpec, Stmt, VarKind, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::{Generator, SourceKey};
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};
use std::collections::BTreeSet;

impl Generator {
    pub(crate) fn sources_key(&self, vr: &VarRef, ctx: &Ctx) -> Result<SourceKey, CodegenError> {
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

    pub(crate) fn ensure_sources_or(&mut self, key: &SourceKey) -> Result<String, CodegenError> {
        if let Some(v) = self.sources_or_cache.get(key) {
            return Ok(v.clone());
        }
        // collect terms minus excludes
        let mut terms = self.sources.adds.get(key).cloned().unwrap_or_default();
        let ex = self.sources.excludes.get(key).cloned().unwrap_or_default();
        terms.retain(|t| !ex.iter().any(|e| e == t));

        // lower terms to bool vars
        let ctx = Ctx::default(); // expressions were concretized already
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

        // normalize
        let mut set: BTreeSet<String> = BTreeSet::new();
        for v in term_vars { set.insert(v); }
        let term_vars: Vec<String> = set.into_iter().collect();

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

    pub(crate) fn collect_sources(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
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

    fn collect_stmt(&mut self, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // Implicit binder expansion for statements that reference domain variables (e.g. `s`)
        // without an explicit `forall`.
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
            Stmt::Let { .. } => {
                // source collection does not need lets; lets are supported by macro expansion in rules.
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
}

/// helper: deterministic concretization for sources terms
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
