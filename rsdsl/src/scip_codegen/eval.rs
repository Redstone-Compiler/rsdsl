//! 표현식 평가 및 변환
//! 
//! 이 모듈은 표현식을 평가하여 선형 표현식이나 불리언 변수로 변환합니다.
//! 인덱스 평가, 변수 이름 생성, 불리언 연산자 변환 등을 처리합니다.

use crate::{Expr, VarKind, VarRef};
use crate::scip_codegen::env::Ctx;
use crate::scip_codegen::error::CodegenError;
use crate::scip_codegen::generator::Generator;
use crate::scip_codegen::linear::{Constraint, LinearExpr, Sense};

impl Generator {
    /// 도메인 이름에 해당하는 모든 값을 반환합니다.
    pub(crate) fn domain_vals(&self, name: &str) -> Result<Vec<String>, CodegenError> {
        self.env
            .domains
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::UnknownDomain(name.to_string()))
    }

    /// 인덱스 표현식을 평가하여 문자열 값으로 변환합니다.
    /// 
    /// 변수 인덱스나 함수 호출 결과를 구체적인 값으로 변환합니다.
    pub(crate) fn eval_index(&self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            Expr::Sym(s) => {
                // Let 바인딩 우선 확인
                if let Some(v) = ctx.lets.get(s) {
                    return Ok(v.clone());
                }
                // 바인더 바인딩 확인
                if let Some(v) = ctx.bind.get(s) {
                    return Ok(v.clone());
                }
                // 바인딩되지 않으면 enum 리터럴이나 도메인 값으로 간주
                Ok(s.clone())
            }
            // 리터럴 값은 정수로 변환
            Expr::Lit(v) => Ok(format!("{}", *v as i64)),
            // 괄호는 내부 표현식 평가
            Expr::Paren(x) => self.eval_index(x, ctx),
            // 함수 호출은 특별 처리
            Expr::Call { name, args } => self.eval_index_call(name, args, ctx),
            _ => Err(CodegenError::UnsupportedLinear(e.clone())),
        }
    }

    /// 셀 ID 문자열을 파싱하여 (x, z) 좌표를 반환합니다.
    /// 
    /// 형식: `x{X}_z{Z}` (예: "x0_z1" -> (0, 1))
    fn parse_cell_id(id: &str) -> Option<(i32, i32)> {
        // expected: x{X}_z{Z}
        let (xpart, zpart) = id.split_once("_z")?;
        let xstr = xpart.strip_prefix('x')?;
        let x = xstr.parse::<i32>().ok()?;
        let z = zpart.parse::<i32>().ok()?;
        Some((x, z))
    }

    /// 인덱스 컨텍스트에서 함수 호출을 평가합니다.
    /// 
    /// 지원하는 함수:
    /// - `opp`: 방향의 반대 방향 반환
    /// - `neigh`, `back`, `front`, `supportForWallTorch`: 인접 셀 계산
    fn eval_index_call(&self, name: &str, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        match name {
            // opp: 방향의 반대 방향 반환 (N<->S, E<->W)
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
            // 인접 셀 계산 함수들
            "neigh" | "back" | "front" | "supportForWallTorch" => {
                let c = self.eval_index(&args[0], ctx)?;
                // back과 supportForWallTorch는 방향을 반대로 사용
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

                // 셀 ID 파싱 실패 시 __NONE__ 반환
                let Some((x, z)) = Self::parse_cell_id(&c) else {
                    return Ok("__NONE__".into());
                };

                // 방향에 따른 좌표 오프셋 계산
                let (dx, dz) = match d.as_str() {
                    "N" => (0, -1),
                    "S" => (0, 1),
                    "E" => (1, 0),
                    "W" => (-1, 0),
                    _ => (0, 0),
                };

                // 인접 셀 좌표 계산
                let (nx, nz) = (x + dx, z + dz);
                let nid = format!("x{}_z{}", nx, nz);
                // 셀 집합에 존재하면 반환, 없으면 __NONE__
                if self.env.cell_set.contains(&nid) {
                    Ok(nid)
                } else {
                    Ok("__NONE__".into())
                }
            }
            // 알 수 없는 함수는 이름과 인자 개수만 반환
            _ => Ok(format!("{}({})", name, args.len())),
        }
    }

    /// 변수의 시그니처(각 인덱스의 도메인)를 반환합니다.
    fn var_sig(&self, name: &str) -> Result<Vec<String>, CodegenError> {
        self.env
            .sigs
            .get(name)
            .cloned()
            .ok_or_else(|| CodegenError::UnknownVar(name.to_string()))
    }

    /// 변수 참조를 LP 변수 이름으로 변환합니다.
    /// 
    /// 형식: `변수명__인덱스1__인덱스2__...`
    /// 예: `D__GROUND__x0_z1`
    /// 
    /// 인덱스가 `__NONE__`이면 상수 0 변수로 변환합니다.
    /// 모든 선언된 변수는 이 모델에서 이진 변수(0/1)입니다.
    pub(crate) fn var_name(&mut self, vr: &VarRef, ctx: &Ctx) -> Result<String, CodegenError> {
        let sig = self.var_sig(&vr.name)?;
        // 인덱스 개수 검증
        if sig.len() != vr.indices.len() {
            return Err(CodegenError::WrongArity(
                vr.name.clone(),
                sig.len(),
                vr.indices.len(),
            ));
        }
        let mut parts = vec![vr.name.clone()];
        // 각 인덱스를 평가하여 이름에 추가
        for idx in &vr.indices {
            let s = self.eval_index(idx, ctx)?;
            // __NONE__ 인덱스는 상수 0으로 처리
            if s == "__NONE__" {
                return Ok("__const0".into());
            }
            parts.push(s);
        }
        let n = parts.join("__");
        // 모든 선언된 변수는 이진 변수(0/1)로 등록
        self.ilp.binaries.insert(n.clone());
        Ok(n)
    }

    /// 표현식이 상수나 파라미터인지 확인하고 값을 반환합니다.
    /// 
    /// 선형 표현식에서 곱셈의 상수 인자를 찾는 데 사용됩니다.
    fn as_const_param(&self, e: &Expr, ctx: &Ctx) -> Option<f64> {
        match e {
            Expr::Lit(v) => Some(*v),
            Expr::Sym(s) => self.env.params.get(s).cloned(),
            Expr::Paren(x) => self.as_const_param(x, ctx),
            _ => None,
        }
    }

    /// 표현식을 선형 표현식으로 평가합니다.
    /// 
    /// 지원하는 연산:
    /// - 리터럴, 파라미터, 변수
    /// - 덧셈, 뺄셈
    /// - 상수 * 선형 표현식 (곱셈)
    /// - Sum: 바인더를 확장하여 합계 계산
    pub(crate) fn eval_linear(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match e {
            // 리터럴: 상수로 변환
            Expr::Lit(v) => Ok(LinearExpr::from_const(*v)),
            Expr::Sym(s) => {
                // 파라미터 확인
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
            // 변수: 0/1 변수로 처리
            Expr::Var(vr) => {
                // treat as 0/1 variable
                let n = self.var_name(vr, ctx)?;
                Ok(LinearExpr::from_var(&n, 1.0))
            }
            // 덧셈: 두 선형 표현식의 합
            Expr::Add(a, b) => {
                let mut x = self.eval_linear(a, ctx)?;
                let y = self.eval_linear(b, ctx)?;
                x.add_inplace(&y);
                Ok(x)
            }
            // 뺄셈: 두 선형 표현식의 차
            Expr::Sub(a, b) => {
                let mut x = self.eval_linear(a, ctx)?;
                let y = self.eval_linear(b, ctx)?;
                x.sub_inplace(&y);
                Ok(x)
            }
            // 곱셈: 상수 * 선형 표현식만 지원
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
            // Sum: 바인더를 확장하여 모든 조합에 대해 합계 계산
            Expr::Sum { binders, body } => {
                let mut acc = LinearExpr::zero();
                self.expand_binders(binders, ctx, |g, cctx| {
                    let t = g.eval_linear(body, &cctx)?;
                    acc.add_inplace(&t);
                    Ok(())
                })?;
                Ok(acc)
            }
            // 괄호: 내부 표현식 평가
            Expr::Paren(x) => self.eval_linear(x, ctx),
            _ => Err(CodegenError::UnsupportedLinear(e.clone())),
        }
    }

    /// 표현식을 선형 표현식으로 평가하거나, 실패하면 불리언 변수로 변환합니다.
    /// 
    /// ForceEq 등에서 선형 표현식과 불리언 표현식을 모두 허용할 때 사용됩니다.
    pub(crate) fn eval_linear_or_boolish(&mut self, e: &Expr, ctx: &Ctx) -> Result<LinearExpr, CodegenError> {
        match self.eval_linear(e, ctx) {
            Ok(x) => Ok(x),
            // 선형 평가 실패 시 불리언으로 평가
            Err(CodegenError::UnsupportedLinear(_)) => {
                let v = self.eval_bool(e, ctx)?;
                Ok(LinearExpr::from_var(&v, 1.0))
            }
            Err(e) => Err(e),
        }
    }

    /// 불리언 상수 값을 변수 이름으로 변환합니다.
    /// 
    /// 0.0 -> "__const0", 그 외 -> "__const1"
    fn bool_const(v: f64) -> String {
        if (v - 0.0).abs() < 1e-9 { "__const0".into() } else { "__const1".into() }
    }

    /// NOT 연산을 제약 조건으로 변환합니다.
    /// 
    /// `y = NOT a`는 `y + a = 1` 제약 조건으로 변환됩니다.
    /// 상수 최적화: NOT 0 = 1, NOT 1 = 0
    fn lower_not(&mut self, a: String) -> Result<String, CodegenError> {
        // 상수 최적화
        if a == "__const0" { return Ok("__const1".into()); }
        if a == "__const1" { return Ok("__const0".into()); }

        // 보조 변수 생성
        let y = self.fresh_aux("not");
        self.ilp.binaries.insert(y.clone());

        // y + a = 1 (y = NOT a)
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

    /// AND 연산을 제약 조건으로 변환합니다.
    /// 
    /// `z = x AND y`는 다음 제약 조건으로 변환됩니다:
    /// - z <= x
    /// - z <= y
    /// - z >= x + y - 1
    /// 
    /// 상수 최적화: AND 0 = 0, AND 1 = y, x AND 1 = x
    fn lower_and(&mut self, x: String, y: String) -> Result<String, CodegenError> {
        // 상수 최적화
        if x == "__const0" || y == "__const0" { return Ok("__const0".into()); }
        if x == "__const1" { return Ok(y); }
        if y == "__const1" { return Ok(x); }

        // 보조 변수 생성
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

    /// OR 연산을 제약 조건으로 변환합니다.
    /// 
    /// `z = x OR y`는 다음 제약 조건으로 변환됩니다:
    /// - z >= x
    /// - z >= y
    /// - z <= x + y
    /// 
    /// 상수 최적화: OR 1 = 1, OR 0 = y, x OR 0 = x
    fn lower_or(&mut self, x: String, y: String) -> Result<String, CodegenError> {
        // 상수 최적화
        if x == "__const1" || y == "__const1" { return Ok("__const1".into()); }
        if x == "__const0" { return Ok(y); }
        if y == "__const0" { return Ok(x); }

        // 보조 변수 생성
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

    /// OR 리스트를 제약 조건으로 변환합니다.
    /// 
    /// `z = OR(terms)`는 다음 제약 조건으로 변환됩니다:
    /// - z >= 각 term
    /// - z <= sum(terms)
    /// 
    /// 최적화: 빈 리스트 = 0, 1이 포함 = 1, 단일 항 = 항 자체
    fn lower_or_list(&mut self, mut terms: Vec<String>) -> Result<String, CodegenError> {
        // 빈 리스트는 0
        if terms.is_empty() {
            return Ok("__const0".into());
        }
        // 1이 포함되면 결과는 1
        if terms.iter().any(|t| t == "__const1") {
            return Ok("__const1".into());
        }
        // 0 항목 제거
        terms.retain(|t| t != "__const0");
        if terms.is_empty() {
            return Ok("__const0".into());
        }
        // 단일 항이면 항 자체 반환
        if terms.len() == 1 {
            return Ok(terms[0].clone());
        }
        // 중복 제거
        terms.sort();
        terms.dedup();

        // 보조 변수 생성
        let z = self.fresh_aux("orlist");
        self.ilp.binaries.insert(z.clone());
        // z >= each term
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

    /// 표현식을 불리언 변수로 평가합니다.
    /// 
    /// 불리언 연산자들은 제약 조건으로 변환되고, 보조 변수가 생성됩니다.
    pub(crate) fn eval_bool(&mut self, e: &Expr, ctx: &Ctx) -> Result<String, CodegenError> {
        match e {
            // 리터럴: 상수로 변환
            Expr::Lit(v) => Ok(Self::bool_const(*v)),

            // 변수: Sources 타입은 특별 처리, 그 외는 일반 변수 이름 생성
            Expr::Var(vr) => {
                // sources varref is special
                if self.var_kind(&vr.name) == Some(VarKind::Sources) {
                    let key = self.sources_key(vr, ctx)?;
                    self.ensure_sources_or(&key)
                } else {
                    self.var_name(vr, ctx)
                }
            }

            // NOT: 제약 조건으로 변환
            Expr::Not(x) => {
                let a = self.eval_bool(x, ctx)?;
                self.lower_not(a)
            }

            // AND: 제약 조건으로 변환
            Expr::And(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                self.lower_and(x, y)
            }

            // OR: 제약 조건으로 변환
            Expr::Or(a, b) => {
                let x = self.eval_bool(a, ctx)?;
                let y = self.eval_bool(b, ctx)?;
                self.lower_or(x, y)
            }

            // OR 리스트: 제약 조건으로 변환
            Expr::OrList(xs) => {
                let mut terms = Vec::with_capacity(xs.len());
                for x in xs {
                    terms.push(self.eval_bool(x, ctx)?);
                }
                self.lower_or_list(terms)
            }

            // Implies: a -> b == (!a) OR b로 변환
            Expr::Implies(a, b) => {
                // a -> b == (!a) OR b
                let na = Expr::Not(a.clone());
                let or = Expr::Or(Box::new(na), b.clone());
                self.eval_bool(&or, ctx)
            }

            // Iff: (a->b) and (b->a)로 변환
            Expr::Iff(a, b) => {
                // (a->b) and (b->a)
                let ab = Expr::Implies(a.clone(), b.clone());
                let ba = Expr::Implies(b.clone(), a.clone());
                let and = Expr::And(Box::new(ab), Box::new(ba));
                self.eval_bool(&and, ctx)
            }

            // 함수 호출: 특별 처리
            Expr::Call { name, args } => self.eval_bool_call(name, args, ctx),

            // 괄호: 내부 표현식 평가
            Expr::Paren(x) => self.eval_bool(x, ctx),

            // 지원하지 않는 표현식
            Expr::Sym(_) | Expr::Neg(_) | Expr::NamedArg { .. } | Expr::Sum { .. } | Expr::Eq(..)
            | Expr::Le(..) | Expr::Ge(..) | Expr::Lt(..) | Expr::Gt(..) | Expr::Add(..) | Expr::Sub(..)
            | Expr::Mul(..) => Err(CodegenError::UnsupportedBool(e.clone())),
        }
    }

    /// 불리언 컨텍스트에서 함수 호출을 평가합니다.
    fn eval_bool_call(&mut self, name: &str, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        match name {
            "OR" => self.call_or(args, ctx),
            "Observe" => self.call_observe(args, ctx),
            "TorchOut" => self.call_torch_out(args, ctx),
            "RepOut" => self.call_rep_out(args, ctx),
            "CandidateDustAdj" => self.call_candidate_dust_adj(args, ctx),
            "ExistsConnectionCandidate" => self.call_exists_connection_candidate(args, ctx),
            // 항상 true를 반환하는 함수들
            "ClearUp" | "ClearDown" | "AllowCrossChoice" | "Touches" => Ok("__const1".into()),
            // 항상 false를 반환하는 함수
            "TorchPowersCell" => Ok("__const0".into()),
            _ => Err(CodegenError::UnsupportedCall(name.to_string())),
        }
    }

    /// OR 함수 호출을 처리합니다.
    /// 
    /// `OR(X)`에서 X가 Sources 변수면 Sources OR 변환을 사용하고,
    /// 그 외에는 인자들의 OR 리스트로 처리합니다.
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

    /// Observe 함수 호출을 처리합니다.
    /// 
    /// `Observe(PIN, s=시나리오)`를 ConcreteVar로 변환합니다.
    /// 시나리오는 정수여야 합니다.
    fn call_observe(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        // Observe(PIN, s=0) -> ConcreteVar
        // PIN 인자 추출
        let pin = match args.get(0) {
            Some(Expr::Sym(p)) => p.clone(),
            Some(Expr::Var(vr)) => vr.name.clone(),
            Some(Expr::Paren(p)) => match &**p {
                Expr::Sym(pn) => pn.clone(),
                _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
            },
            _ => return Err(CodegenError::UnsupportedCall("Observe".into())),
        };

        // 시나리오 인자 추출 (명명된 인자 s)
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

        // 시나리오를 정수로 변환
        let sci: i32 = sc
            .parse::<i32>()
            .map_err(|_| CodegenError::BadScenario(sc.clone()))?;

        // Observe 함수 호출하여 ConcreteVar 생성
        let cv = (self.env.observe)(&pin, sci);
        let vname = cv.lp_name();
        self.ilp.binaries.insert(vname.clone());
        Ok(vname)
    }

    /// TorchOut 함수 호출을 처리합니다.
    /// 
    /// `TorchOut(stand, c)` 또는 `TorchOut(wall, c, d)`를 변수 참조로 변환합니다.
    /// 시나리오 바인더 `s`가 필요합니다.
    fn call_torch_out(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        // TorchOut(stand,c) or TorchOut(wall,c,d) uses scenario binder `s`
        // 시나리오 바인더 확인
        let s = ctx
            .bind
            .get("s")
            .cloned()
            .ok_or_else(|| CodegenError::MissingScenarioBinder("TorchOut".into()))?;
        // 종류 추출 (stand 또는 wall)
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

    /// RepOut 함수 호출을 처리합니다.
    /// 
    /// `RepOut(c, d)`를 변수 참조로 변환합니다.
    /// 시나리오 바인더 `s`가 필요합니다.
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

    /// CandidateDustAdj 함수 호출을 처리합니다.
    /// 
    /// `CandidateDustAdj(l, c, d)` -> `D[l, neigh(c, d)]`로 변환합니다.
    fn call_candidate_dust_adj(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        // CandidateDustAdj(l,c,d) -> D[l, neigh(c,d)]
        let l = self.eval_index(&args[0], ctx)?;
        // neigh(c, d) 계산
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

    /// ExistsConnectionCandidate 함수 호출을 처리합니다.
    /// 
    /// 인접 셀에 배치 가능한 블록/토치/리피터가 있는지 확인합니다.
    /// 인접 셀의 모든 가능한 배치에 대한 OR를 계산합니다.
    fn call_exists_connection_candidate(&mut self, args: &[Expr], ctx: &Ctx) -> Result<String, CodegenError> {
        // OR of neighbor placements
        let l = self.eval_index(&args[0], ctx)?;
        // 인접 셀 계산
        let neigh = self.eval_index(
            &Expr::Call {
                name: "neigh".into(),
                args: vec![args[1].clone(), args[2].clone()],
            },
            ctx,
        )?;
        // 인접 셀이 없으면 false
        if neigh == "__NONE__" {
            return Ok("__const0".into());
        }
        // 인접 셀의 모든 가능한 배치에 대한 OR 리스트 생성
        let mut ors: Vec<Expr> = vec![];
        // 먼지 인접
        ors.push(Expr::Var(VarRef {
            name: "D".into(),
            indices: vec![Expr::Sym(l.clone()), Expr::Sym(neigh.clone())],
        }));
        // 블록 인접
        ors.push(Expr::Var(VarRef {
            name: "S".into(),
            indices: vec![Expr::Sym(neigh.clone())],
        }));
        // 서 있는 토치 인접
        ors.push(Expr::Var(VarRef {
            name: "T_stand".into(),
            indices: vec![Expr::Sym(neigh.clone())],
        }));
        // 벽 토치 / 리피터 인접 (모든 방향)
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
