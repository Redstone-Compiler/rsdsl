use crate::{Binder, Expr, Stmt, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use std::collections::BTreeMap;

impl Generator {
    /// Expand `forall (vars in domains)` by enumerating cartesian products of concrete domain values.
    pub(crate) fn expand_binders<F>(
        &mut self,
        binders: &[Binder],
        ctx: &Ctx,
        mut f: F,
    ) -> Result<(), CodegenError>
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

        rec(self, binders, 0, ctx, &mut f)
    }

    fn collect_varrefs_expr(e: &Expr, out: &mut Vec<VarRef>) {
        match e {
            Expr::Var(vr) => out.push(vr.clone()),

            Expr::Not(x) | Expr::Paren(x) | Expr::Neg(x) => Self::collect_varrefs_expr(x, out),

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
                Self::collect_varrefs_expr(a, out);
                Self::collect_varrefs_expr(b, out);
            }

            Expr::Sum { body, .. } => Self::collect_varrefs_expr(body, out),

            Expr::Call { args, .. } => {
                for a in args {
                    Self::collect_varrefs_expr(a, out);
                }
            }

            Expr::OrList(exprs) => {
                for e in exprs {
                    Self::collect_varrefs_expr(e, out);
                }
            }

            Expr::NamedArg { value, .. } => Self::collect_varrefs_expr(value, out),

            Expr::Sym(_) | Expr::Lit(_) => {}
        }
    }

    /// Find "implicit" binders that are referenced in a statement without an explicit `forall`.
    ///
    /// Example:
    ///   def DP0[s, x1_z0] <-> (D0[x1_z0] and DP0[s, x0_z0]);
    /// should expand over `s in Sc` even if the rule omitted:
    ///   forall (s in Sc) { ... }
    ///
    /// We infer binders only from *index positions* of `VarRef` occurrences.
    pub(crate) fn implicit_binders_for_stmt(
        &self,
        st: &Stmt,
        ctx: &Ctx,
    ) -> Result<Vec<Binder>, CodegenError> {
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

                sym_to_domain
                    .entry(sym.clone())
                    .or_insert_with(|| dn.clone());
            }
            Ok(())
        };

        let mut vrs: Vec<VarRef> = vec![];

        match st {
            Stmt::Require(e) => {
                Self::collect_varrefs_expr(e, &mut vrs);
            }
            Stmt::Def { lhs, rhs } => {
                vrs.push(lhs.clone());
                Self::collect_varrefs_expr(rhs, &mut vrs);
            }
            Stmt::ForceEq { lhs, rhs } => {
                Self::collect_varrefs_expr(lhs, &mut vrs);
                Self::collect_varrefs_expr(rhs, &mut vrs);
            }
            Stmt::Add {
                target,
                value,
                cond,
                ..
            } => {
                vrs.push(target.clone());
                Self::collect_varrefs_expr(value, &mut vrs);
                if let Some(c) = cond {
                    Self::collect_varrefs_expr(c, &mut vrs);
                }
            }
            Stmt::ForAll { .. } | Stmt::Feature { .. } | Stmt::Let { .. } => {}
        }

        for vr in &vrs {
            add_unbound(&vr.name, &vr.indices)?;
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
}
