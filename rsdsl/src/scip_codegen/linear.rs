//! 선형 표현식 및 ILP 구조
//! 
//! 이 모듈은 선형 표현식, 제약 조건, ILP 구조를 정의하고,
//! LP 파일 형식으로 출력하는 기능을 제공합니다.

use crate::ObjSense;
use std::collections::{BTreeMap, BTreeSet};

/// 선형 표현식
/// 
/// 변수 항들과 상수항으로 구성된 선형 표현식입니다.
#[derive(Clone, Debug)]
pub(crate) struct LinearExpr {
    /// 변수 이름 -> 계수 매핑
    pub(crate) terms: BTreeMap<String, f64>,
    /// 상수항
    pub(crate) constant: f64,
}
impl LinearExpr {
    /// 0으로 초기화된 선형 표현식을 생성합니다.
    pub(crate) fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
            constant: 0.0,
        }
    }
    /// 상수만 있는 선형 표현식을 생성합니다.
    pub(crate) fn from_const(v: f64) -> Self {
        let mut e = Self::zero();
        e.constant = v;
        e
    }
    /// 단일 변수 항만 있는 선형 표현식을 생성합니다.
    pub(crate) fn from_var(v: &str, c: f64) -> Self {
        let mut e = Self::zero();
        if c != 0.0 {
            e.terms.insert(v.to_string(), c);
        }
        e
    }
    /// 다른 선형 표현식을 더합니다 (in-place).
    /// 
    /// 계수가 0에 가까운 항은 제거합니다.
    pub(crate) fn add_inplace(&mut self, other: &LinearExpr) {
        self.constant += other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) += *v;
        }
        // 0에 가까운 항 제거
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    /// 다른 선형 표현식을 뺍니다 (in-place).
    /// 
    /// 계수가 0에 가까운 항은 제거합니다.
    pub(crate) fn sub_inplace(&mut self, other: &LinearExpr) {
        self.constant -= other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) -= *v;
        }
        // 0에 가까운 항 제거
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    /// 선형 표현식에 상수를 곱합니다.
    pub(crate) fn scale(&self, k: f64) -> Self {
        let mut e = Self::zero();
        e.constant = self.constant * k;
        for (n, c) in self.terms.iter() {
            e.terms.insert(n.clone(), c * k);
        }
        e
    }
    /// 다른 선형 표현식을 뺍니다 (소유권 이동).
    pub(crate) fn sub(mut self, other: LinearExpr) -> LinearExpr {
        self.sub_inplace(&other);
        self
    }
}

/// 제약 조건의 부등호 방향
#[derive(Clone, Copy, Debug)]
pub(crate) enum Sense {
    /// <= (작거나 같음)
    Le,
    /// >= (크거나 같음)
    Ge,
    /// == (같음)
    Eq,
}

/// 선형 제약 조건
/// 
/// 형식: `expr sense rhs` (예: `x + y <= 5`)
#[derive(Clone, Debug)]
pub(crate) struct Constraint {
    /// 제약 조건 이름
    pub(crate) name: String,
    /// 좌변 선형 표현식
    pub(crate) expr: LinearExpr, // lhs
    /// 부등호 방향
    pub(crate) sense: Sense,
    /// 우변 값
    pub(crate) rhs: f64,
}

/// 정수 선형 계획법 (ILP) 문제
#[derive(Clone, Debug)]
pub(crate) struct Ilp {
    /// 목적 함수
    pub(crate) objective: LinearExpr,
    /// 최소화 또는 최대화
    pub(crate) sense: ObjSense,
    /// 제약 조건 목록
    pub(crate) constraints: Vec<Constraint>,
    /// 이진 변수 집합 (0 또는 1만 가능)
    pub(crate) binaries: BTreeSet<String>,
}

impl Ilp {
    /// 새로운 ILP 문제를 생성합니다.
    pub(crate) fn new() -> Self {
        Self {
            objective: LinearExpr::zero(),
            sense: ObjSense::Minimize,
            constraints: vec![],
            binaries: BTreeSet::new(),
        }
    }
}

/// ILP를 LP 파일 형식 문자열로 변환합니다.
/// 
/// SCIP 호환 형식으로 출력합니다.
pub(crate) fn emit_lp(ilp: &Ilp) -> String {
    let mut out = String::new();
    // 목적 함수 헤더
    match ilp.sense {
        ObjSense::Minimize => out.push_str("Minimize\n obj: "),
        ObjSense::Maximize => out.push_str("Maximize\n obj: "),
    }
    out.push_str(&fmt_lin(&ilp.objective));
    out.push('\n');
    // 제약 조건 섹션
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
    // 이진 변수 섹션
    out.push_str("Binary\n");
    for b in &ilp.binaries {
        out.push_str(&format!(" {}\n", b));
    }
    out.push_str("End\n");
    out
}

/// 부등호 방향을 문자열로 변환합니다.
fn fmt_sense(s: Sense) -> &'static str {
    match s {
        Sense::Le => "<=",
        Sense::Ge => ">=",
        Sense::Eq => "=",
    }
}

/// 숫자를 문자열로 변환합니다.
/// 
/// 정수면 정수로, 실수면 소수점 6자리까지 출력합니다.
fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.6}", v)
    }
}

/// 선형 표현식을 문자열로 변환합니다.
/// 
/// 계수가 1 또는 -1이면 간단히 표시하고, 그 외에는 소수점 6자리까지 출력합니다.
fn fmt_lin(e: &LinearExpr) -> String {
    let mut parts: Vec<String> = vec![];
    // 변수 항들
    for (n, c) in e.terms.iter() {
        if (c - 1.0).abs() < 1e-12 {
            parts.push(format!("+1 {}", n));
        } else if (c + 1.0).abs() < 1e-12 {
            parts.push(format!("-1 {}", n));
        } else {
            parts.push(format!("{:+.6} {}", c, n));
        }
    }
    // 항이 없으면 0 표시
    if parts.is_empty() {
        parts.push("+0".to_string());
    }
    // 상수항 추가
    if e.constant.abs() > 1e-12 {
        parts.push(format!("{:+.6}", e.constant));
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_num_int() {
        assert_eq!(fmt_num(3.0), "3");
        assert_eq!(fmt_num(-2.0), "-2");
    }

    #[test]
    fn test_linear_sub() {
        let a = LinearExpr::from_var("x", 1.0);
        let b = LinearExpr::from_var("y", 2.0);
        let c = a.sub(b);
        assert!(c.terms.contains_key("x"));
        assert!(c.terms.contains_key("y"));
        assert_eq!(c.terms["x"], 1.0);
        assert_eq!(c.terms["y"], -2.0);
    }
}
