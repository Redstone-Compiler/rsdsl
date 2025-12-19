use crate::{Expr, ModelSpec, ObjSense, Rule, Stmt, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};

impl Generator {
    pub(crate) fn emit_model(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
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

    fn emit_stmt_block(&mut self, rname: &str, body: &[Stmt], ctx: &Ctx) -> Result<(), CodegenError> {
        for st in body {
            self.emit_stmt(rname, st, ctx)?;
        }
        Ok(())
    }

    fn emit_stmt(&mut self, rname: &str, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // Implicit binder expansion for statements that reference domain variables (e.g. `s`)
        // without an explicit `forall`.
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
                self.expand_binders(binders, ctx, |g, cctx| g.emit_stmt_block(rname, body, &cctx))?;
            }
            Stmt::Feature { name, body } => {
                if self.is_feature_on(name) {
                    self.emit_stmt_block(rname, body, ctx)?;
                }
            }
            Stmt::Let { .. } => {
                // sequence-scoped let: not supported in the simple emitter
            }
            Stmt::Require(e) => self.emit_require(rname, e, ctx)?,
            Stmt::Def { lhs, rhs } => self.emit_def(rname, lhs, rhs, ctx)?,
            Stmt::ForceEq { lhs, rhs } => self.emit_force_eq(rname, lhs, rhs, ctx)?,
            Stmt::Add { .. } => (), // already handled by SourceDB in collect_sources()
        }

        Ok(())
    }

    fn emit_require(&mut self, rname: &str, e: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();

        // If top-level is comparison, emit linear constraint
        match e {
            Expr::Le(a, b) => {
                let mut lhs = self.eval_linear(a, ctx)?;
                let rhs = self.eval_linear(b, ctx)?;
                lhs.sub_inplace(&rhs);
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_le_{}", rname, cid),
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
                    name: format!("{}_req_ge_{}", rname, cid),
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
                    name: format!("{}_req_eq_{}", rname, cid),
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
                    name: format!("{}_req_impl_{}", rname, cid),
                    expr: lhs,
                    sense: Sense::Le,
                    rhs: 0.0,
                });
                Ok(())
            }
            _ => {
                let v = self.eval_bool(e, ctx)?;
                self.ilp.constraints.push(Constraint {
                    name: format!("{}_req_bool_{}", rname, cid),
                    expr: LinearExpr::from_var(&v, 1.0),
                    sense: Sense::Eq,
                    rhs: 1.0,
                });
                Ok(())
            }
        }
    }

    fn emit_def(&mut self, rname: &str, lhs: &VarRef, rhs: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();
        let lv = self.var_name(lhs, ctx)?;
        let rv = self.eval_bool(rhs, ctx)?;
        let mut lhs_expr = LinearExpr::from_var(&lv, 1.0);
        lhs_expr.sub_inplace(&LinearExpr::from_var(&rv, 1.0));
        self.ilp.constraints.push(Constraint {
            name: format!("{}_def_{}", rname, cid),
            expr: lhs_expr,
            sense: Sense::Eq,
            rhs: 0.0,
        });
        Ok(())
    }

    fn emit_force_eq(&mut self, rname: &str, lhs: &Expr, rhs: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();
        // force allows linear expressions
        let mut le = self.eval_linear_or_boolish(lhs, ctx)?;
        let re = self.eval_linear_or_boolish(rhs, ctx)?;
        le.sub_inplace(&re);
        self.ilp.constraints.push(Constraint {
            name: format!("{}_force_{}", rname, cid),
            expr: le,
            sense: Sense::Eq,
            rhs: 0.0,
        });
        Ok(())
    }
}
