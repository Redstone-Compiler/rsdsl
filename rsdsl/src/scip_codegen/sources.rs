//! Sources 변수 처리
//!
//! 이 모듈은 Sources 타입 변수에 대한 Add/Exclude 문을 수집하고,
//! OR 변수로 변환하는 기능을 제공합니다.

use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::{Generator, SourceKey};
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};
use crate::{Expr, ModelSpec, Stmt, VarKind, VarRef};
use std::collections::BTreeSet;

impl Generator {
    /// Sources 변수 참조를 SourceKey로 변환합니다.
    ///
    /// 변수 이름과 인덱스들을 `__`로 연결하여 정규화된 키를 생성합니다.
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

    /// Sources 키에 대한 OR 변수를 생성하거나 캐시에서 반환합니다.
    ///
    /// Sources 변수는 여러 Add 문으로 구성되며, 이들의 OR로 변환됩니다.
    /// Exclude 문으로 제외된 항목은 제거됩니다.
    pub(crate) fn ensure_sources_or(&mut self, key: &SourceKey) -> Result<String, CodegenError> {
        // 캐시 확인
        if let Some(v) = self.sources_or_cache.get(key) {
            return Ok(v.clone());
        }
        // Add 항목 수집 (Exclude 항목 제외)
        let mut terms = self.sources.adds.get(key).cloned().unwrap_or_default();
        let ex = self.sources.excludes.get(key).cloned().unwrap_or_default();
        terms.retain(|t| !ex.iter().any(|e| e == t));

        // 항목들을 불리언 변수로 변환
        let ctx = Ctx::default(); // expressions were concretized already
        let mut term_vars: Vec<String> = vec![];
        for t in terms {
            let v = self.eval_bool(&t, &ctx)?;
            // 1이 포함되면 결과는 1
            if v == "__const1" {
                self.sources_or_cache.insert(key.clone(), "__const1".into());
                return Ok("__const1".into());
            }
            // 0이 아닌 항만 추가
            if v != "__const0" {
                term_vars.push(v);
            }
        }

        // 정규화: 중복 제거 및 정렬
        let mut set: BTreeSet<String> = BTreeSet::new();
        for v in term_vars {
            set.insert(v);
        }
        let term_vars: Vec<String> = set.into_iter().collect();

        // 최적화: 빈 리스트는 0, 단일 항은 항 자체
        if term_vars.is_empty() {
            self.sources_or_cache.insert(key.clone(), "__const0".into());
            return Ok("__const0".into());
        }
        if term_vars.len() == 1 {
            self.sources_or_cache
                .insert(key.clone(), term_vars[0].clone());
            return Ok(term_vars[0].clone());
        }

        // OR 변수 생성 및 제약 조건 추가
        let z = self.fresh_aux("srcor");
        self.ilp.binaries.insert(z.clone());
        // z >= each term: -z + t <= 0
        for (i, t) in term_vars.iter().enumerate() {
            self.ilp.constraints.push(Constraint {
                name: format!("srcor_ge{}_{}", i, z),
                expr: LinearExpr::from_var(t, 1.0).sub(LinearExpr::from_var(&z, 1.0)),
                sense: Sense::Le,
                rhs: 0.0,
            });
        }
        // z <= sum(terms): z - sum(terms) <= 0
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

        // 캐시에 저장
        self.sources_or_cache.insert(key.clone(), z.clone());
        Ok(z)
    }

    /// 모델의 모든 규칙에서 Sources 변수에 대한 Add/Exclude 문을 수집합니다.
    pub(crate) fn collect_sources(&mut self, spec: &ModelSpec) -> Result<(), CodegenError> {
        for r in &spec.rules {
            self.collect_stmt_block(&r.body, &Ctx::default())?;
        }
        Ok(())
    }

    /// 문장 블록의 모든 문장을 순차적으로 처리하여 Sources를 수집합니다.
    fn collect_stmt_block(&mut self, body: &[Stmt], ctx: &Ctx) -> Result<(), CodegenError> {
        for st in body {
            self.collect_stmt(st, ctx)?;
        }
        Ok(())
    }

    /// 단일 문장을 처리하여 Sources Add/Exclude 문을 수집합니다.
    ///
    /// 명시적 `forall`이 없는 문장에서 암시적 바인더를 추론하여 확장합니다.
    fn collect_stmt(&mut self, st: &Stmt, ctx: &Ctx) -> Result<(), CodegenError> {
        // 명시적 `forall`이 없는 문장에서 도메인 변수(예: `s`)를 참조하는 경우
        // 암시적 바인더 확장 수행
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
            // ForAll: 바인더를 확장하여 내부 문장 블록 처리
            Stmt::ForAll { binders, body } => {
                self.expand_binders(binders, ctx, |g, cctx| g.collect_stmt_block(body, &cctx))?;
            }
            // Feature: 피처가 활성화된 경우에만 내부 문장 처리
            Stmt::Feature { name, body } => {
                if self.is_feature_on(name) {
                    self.collect_stmt_block(body, ctx)?;
                }
            }
            // Let: Sources 수집에는 필요 없음 (매크로 확장에서 처리됨)
            Stmt::Let { .. } => {
                // source collection does not need lets; lets are supported by macro expansion in rules.
            }
            // Add: Sources 타입 변수에 대한 Add/Exclude 문 수집
            Stmt::Add {
                exclude,
                target,
                value,
                cond,
            } => {
                // Sources 타입이 아니면 무시
                if self.var_kind(&target.name) != Some(VarKind::Sources) {
                    return Ok(());
                }
                let key = self.sources_key(target, ctx)?;
                // 표현식을 구체화 (바인딩된 변수 값으로 치환)
                let mut val = concretize_expr(value, ctx)?;
                // 조건이 있으면 AND로 결합
                if let Some(c) = cond {
                    let cc = concretize_expr(c, ctx)?;
                    val = Expr::And(Box::new(val), Box::new(cc));
                }
                // Exclude 또는 Add에 추가
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

/// Sources 항목을 위한 결정론적 구체화 헬퍼 함수
///
/// 표현식의 바인딩된 변수들을 구체적인 값으로 치환합니다.
/// 이는 Sources 항목을 비교 가능한 형태로 만들기 위해 필요합니다.
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
