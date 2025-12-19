//! 제약 조건 및 목적 함수 생성
//! 
//! 이 모듈은 ModelSpec의 규칙들을 ILP 제약 조건으로 변환하고,
//! 목적 함수를 생성합니다.

use crate::{Expr, ModelSpec, Stmt, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};

impl Generator {
    /// 모델의 모든 규칙과 목적 함수를 ILP 제약 조건으로 생성합니다.
    /// 
    /// 각 규칙의 문장들을 처리하여 제약 조건을 생성하고,
    /// 목적 함수가 있으면 선형 표현식으로 변환합니다.
    pub(crate) fn emit_model(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
        // 모든 규칙의 문장 블록 처리
        for r in &spec.rules {
            self.emit_stmt_block(&r.name, &r.body, &Ctx::default())?;
        }
        // 목적 함수가 있으면 설정
        if let Some(obj) = &spec.objective {
            self.ilp.sense = obj.sense.clone();
            let lin = self.eval_linear(&obj.expr, &Ctx::default())?;
            self.ilp.objective = lin;
        }
        Ok(())
    }

    /// 문장 블록의 모든 문장을 순차적으로 처리합니다.
    fn emit_stmt_block(&mut self, rname: &str, body: &[Stmt], ctx: &Ctx) -> Result<(), CodegenError> {
        for st in body {
            self.emit_stmt(rname, st, ctx)?;
        }
        Ok(())
    }

    /// 단일 문장을 처리하여 제약 조건을 생성합니다.
    /// 
    /// 명시적 `forall`이 없는 문장에서 암시적 바인더를 추론하여 확장합니다.
    fn emit_stmt(&mut self, rname: &str, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // 명시적 `forall`이 없는 문장에서 도메인 변수(예: `s`)를 참조하는 경우
        // 암시적 바인더 확장 수행
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
            // ForAll: 바인더를 확장하여 내부 문장 블록 처리
            Stmt::ForAll { binders, body } => {
                self.expand_binders(binders, ctx, |g, cctx| g.emit_stmt_block(rname, body, &cctx))?;
            }
            // Feature: 피처가 활성화된 경우에만 내부 문장 처리
            Stmt::Feature { name, body } => {
                if self.is_feature_on(name) {
                    self.emit_stmt_block(rname, body, ctx)?;
                }
            }
            // Let: 시퀀스 스코프 let은 단순 emitter에서 지원하지 않음
            Stmt::Let { .. } => {
                // sequence-scoped let: not supported in the simple emitter
            }
            // Require: 제약 조건 생성
            Stmt::Require(e) => self.emit_require(rname, e, ctx)?,
            // Def: 정의 제약 조건 생성 (lhs <-> rhs)
            Stmt::Def { lhs, rhs } => self.emit_def(rname, lhs, rhs, ctx)?,
            // ForceEq: 강제 등식 제약 조건 생성
            Stmt::ForceEq { lhs, rhs } => self.emit_force_eq(rname, lhs, rhs, ctx)?,
            // Add: Sources 수집 단계에서 이미 처리됨
            Stmt::Add { .. } => (), // already handled by SourceDB in collect_sources()
        }

        Ok(())
    }

    /// `Require` 문장을 제약 조건으로 생성합니다.
    /// 
    /// 표현식의 최상위 연산자에 따라 적절한 제약 조건을 생성합니다:
    /// - 비교 연산자 (Le, Ge, Eq): 선형 제약 조건
    /// - Implies: 불리언 함의 제약 조건 (a -> b는 a <= b로 변환)
    /// - 기타: 불리언 표현식을 평가하여 1과 같다는 제약 조건
    fn emit_require(&mut self, rname: &str, e: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();

        // 최상위가 비교 연산자면 선형 제약 조건 생성
        match e {
            // a <= b: a - b <= 0
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
            // a >= b: a - b >= 0
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
            // a == b: a - b == 0
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
            // a -> b: 불리언 함의는 a <= b로 변환 (a가 1이면 b도 1이어야 함)
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
            // 기타: 불리언 표현식을 평가하여 1과 같다는 제약 조건 생성
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

    /// `Def` 문장을 제약 조건으로 생성합니다.
    /// 
    /// `lhs <-> rhs` 형태의 정의를 `lhs - rhs == 0` 제약 조건으로 변환합니다.
    /// 양쪽 모두 불리언 표현식으로 평가됩니다.
    fn emit_def(&mut self, rname: &str, lhs: &VarRef, rhs: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();
        let lv = self.var_name(lhs, ctx)?;
        let rv = self.eval_bool(rhs, ctx)?;
        // lhs - rhs == 0
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

    /// `ForceEq` 문장을 제약 조건으로 생성합니다.
    /// 
    /// `lhs == rhs` 형태의 강제 등식을 생성합니다.
    /// 선형 표현식 또는 불리언 표현식을 허용합니다.
    fn emit_force_eq(&mut self, rname: &str, lhs: &Expr, rhs: &Expr, ctx: &Ctx) -> Result<(), CodegenError> {
        let cid = self.next_cst_id();
        // force는 선형 표현식 또는 불리언 표현식을 허용
        let mut le = self.eval_linear_or_boolish(lhs, ctx)?;
        let re = self.eval_linear_or_boolish(rhs, ctx)?;
        // lhs - rhs == 0
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
