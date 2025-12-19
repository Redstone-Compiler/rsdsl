//! 바인더 확장 및 암시적 바인더 추론
//!
//! 이 모듈은 `forall` 바인더를 구체적인 도메인 값들의 카르테시안 곱으로 확장하고,
//! 명시적 바인더가 없는 문장에서 암시적 바인더를 추론합니다.

use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use crate::{Binder, Expr, Stmt, VarRef};
use std::collections::BTreeMap;

impl Generator {
    /// `forall (vars in domains)` 바인더를 구체적인 도메인 값들의 카르테시안 곱으로 확장합니다.
    ///
    /// 각 바인더의 도메인 값들을 조합하여 모든 가능한 바인딩을 생성하고,
    /// 각 바인딩에 대해 주어진 클로저 `f`를 실행합니다.
    ///
    /// # 예시
    /// `forall (s in Sc, c in Cell)`는 Sc와 Cell의 모든 조합에 대해 확장됩니다.
    pub(crate) fn expand_binders<F>(
        &mut self,
        binders: &[Binder],
        ctx: &Ctx,
        mut f: F,
    ) -> Result<(), CodegenError>
    where
        F: FnMut(&mut Generator, Ctx) -> Result<(), CodegenError>,
    {
        /// 여러 도메인의 카르테시안 곱을 계산합니다.
        ///
        /// 예: [[1,2], [a,b]] -> [[1,a], [1,b], [2,a], [2,b]]
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

        /// 재귀적으로 바인더를 확장하는 헬퍼 함수
        ///
        /// 각 바인더에 대해 도메인 값들의 조합을 생성하고,
        /// 변수에 값을 바인딩한 새로운 컨텍스트로 다음 바인더를 처리합니다.
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
            // 모든 바인더를 처리했으면 클로저 실행
            if i == binders.len() {
                return f(g, ctx.clone());
            }
            let b = &binders[i];
            // 현재 바인더의 각 도메인에 대한 값들을 수집
            let mut doms: Vec<Vec<String>> = vec![];
            for dn in &b.domains {
                doms.push(g.domain_vals(dn)?);
            }
            // 도메인들의 카르테시안 곱 계산
            let tuples = cartesian(&doms);
            for t in tuples {
                // 튜플 길이와 변수 개수가 일치하지 않으면 스킵
                if t.len() != b.vars.len() {
                    continue;
                }
                // 새로운 컨텍스트에 변수-값 바인딩 추가
                let mut nctx = ctx.clone();
                for (var, val) in b.vars.iter().zip(t.iter()) {
                    nctx.bind.insert(var.clone(), val.clone());
                }
                // 다음 바인더로 재귀
                rec(g, binders, i + 1, &nctx, f)?;
            }
            Ok(())
        }

        rec(self, binders, 0, ctx, &mut f)
    }

    /// 표현식에서 모든 변수 참조를 재귀적으로 수집합니다.
    ///
    /// 표현식 트리를 순회하며 모든 `VarRef`를 `out` 벡터에 추가합니다.
    /// 이는 암시적 바인더 추론에 사용됩니다.
    fn collect_varrefs_expr(e: &Expr, out: &mut Vec<VarRef>) {
        match e {
            Expr::Var(vr) => out.push(vr.clone()),

            // 단항 연산자: 하위 표현식 재귀 처리
            Expr::Not(x) | Expr::Paren(x) | Expr::Neg(x) => Self::collect_varrefs_expr(x, out),

            // 이항 연산자: 양쪽 하위 표현식 모두 처리
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

            // Sum 표현식: 본문만 처리 (바인더는 이미 처리됨)
            Expr::Sum { body, .. } => Self::collect_varrefs_expr(body, out),

            // 함수 호출: 모든 인자 처리
            Expr::Call { args, .. } => {
                for a in args {
                    Self::collect_varrefs_expr(a, out);
                }
            }

            // OrList: 모든 항목 처리
            Expr::OrList(exprs) => {
                for e in exprs {
                    Self::collect_varrefs_expr(e, out);
                }
            }

            // 명명된 인자: 값만 처리
            Expr::NamedArg { value, .. } => Self::collect_varrefs_expr(value, out),

            // 리터럴/심볼: 변수 참조 없음
            Expr::Sym(_) | Expr::Lit(_) => {}
        }
    }

    /// 명시적 `forall` 없이 문장에서 참조되는 "암시적" 바인더를 찾습니다.
    ///
    /// # 예시
    /// ```
    /// def DP0[s, x1_z0] <-> (D0[x1_z0] and DP0[s, x0_z0]);
    /// ```
    /// 위 문장은 명시적인 `forall (s in Sc)`가 없어도 `s`에 대한 암시적 바인더를 추론하여
    /// `forall (s in Sc) { ... }`로 확장됩니다.
    ///
    /// # 동작 방식
    /// `VarRef`의 인덱스 위치에서 심볼을 찾아, 이미 바인딩되지 않았고
    /// 도메인 리터럴이 아닌 경우 암시적 바인더로 추론합니다.
    pub(crate) fn implicit_binders_for_stmt(
        &self,
        st: &Stmt,
        ctx: &Ctx,
    ) -> Result<Vec<Binder>, CodegenError> {
        // 심볼 -> 도메인 이름 매핑
        let mut sym_to_domain: BTreeMap<String, String> = BTreeMap::new();

        // 바인딩되지 않은 인덱스 심볼을 찾아 암시적 바인더로 추가하는 클로저
        let mut add_unbound = |var_name: &str, indices: &[Expr]| -> Result<(), CodegenError> {
            // 변수의 시그니처 가져오기 (각 인덱스의 도메인)
            let sig = self
                .env
                .sigs
                .get(var_name)
                .ok_or_else(|| CodegenError::UnknownVar(var_name.to_string()))?
                .clone();

            // 인덱스 개수 검증
            if sig.len() != indices.len() {
                return Err(CodegenError::WrongArity(
                    var_name.to_string(),
                    sig.len(),
                    indices.len(),
                ));
            }

            // 각 인덱스 위치를 검사
            for (dn, idx_expr) in sig.iter().zip(indices.iter()) {
                // 심볼이 아니면 스킵
                let Expr::Sym(sym) = idx_expr else { continue };

                // 이미 바인딩되어 있으면 암시적 바인더가 아님
                if ctx.bind.contains_key(sym) || ctx.lets.contains_key(sym) {
                    continue;
                }

                // 심볼이 도메인의 리터럴 값이면 리터럴로 유지
                let dom_vals = self.domain_vals(dn)?;
                if dom_vals.iter().any(|v| v == sym) {
                    continue;
                }

                // 암시적 바인더로 추가
                sym_to_domain
                    .entry(sym.clone())
                    .or_insert_with(|| dn.clone());
            }
            Ok(())
        };

        // 문장에서 모든 변수 참조 수집
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
            // ForAll, Feature, Let은 이미 바인더를 가지고 있거나 처리할 필요 없음
            Stmt::ForAll { .. } | Stmt::Feature { .. } | Stmt::Let { .. } => {}
        }

        // 각 변수 참조에서 바인딩되지 않은 인덱스 심볼 찾기
        for vr in &vrs {
            add_unbound(&vr.name, &vr.indices)?;
        }

        // 추론된 `sym -> domain` 매핑을 단일 변수 바인더로 변환
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
