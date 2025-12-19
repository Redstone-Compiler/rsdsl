use crate::ObjSense;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub(crate) struct LinearExpr {
    pub(crate) terms: BTreeMap<String, f64>,
    pub(crate) constant: f64,
}
impl LinearExpr {
    pub(crate) fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
            constant: 0.0,
        }
    }
    pub(crate) fn from_const(v: f64) -> Self {
        let mut e = Self::zero();
        e.constant = v;
        e
    }
    pub(crate) fn from_var(v: &str, c: f64) -> Self {
        let mut e = Self::zero();
        if c != 0.0 {
            e.terms.insert(v.to_string(), c);
        }
        e
    }
    pub(crate) fn add_inplace(&mut self, other: &LinearExpr) {
        self.constant += other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) += *v;
        }
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    pub(crate) fn sub_inplace(&mut self, other: &LinearExpr) {
        self.constant -= other.constant;
        for (k, v) in other.terms.iter() {
            *self.terms.entry(k.clone()).or_insert(0.0) -= *v;
        }
        self.terms.retain(|_, c| c.abs() > 1e-12);
    }
    pub(crate) fn scale(&self, k: f64) -> Self {
        let mut e = Self::zero();
        e.constant = self.constant * k;
        for (n, c) in self.terms.iter() {
            e.terms.insert(n.clone(), c * k);
        }
        e
    }
    pub(crate) fn sub(mut self, other: LinearExpr) -> LinearExpr {
        self.sub_inplace(&other);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Sense {
    Le,
    Ge,
    Eq,
}

#[derive(Clone, Debug)]
pub(crate) struct Constraint {
    pub(crate) name: String,
    pub(crate) expr: LinearExpr, // lhs
    pub(crate) sense: Sense,
    pub(crate) rhs: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Ilp {
    pub(crate) objective: LinearExpr,
    pub(crate) sense: ObjSense,
    pub(crate) constraints: Vec<Constraint>,
    pub(crate) binaries: BTreeSet<String>,
}

impl Ilp {
    pub(crate) fn new() -> Self {
        Self {
            objective: LinearExpr::zero(),
            sense: ObjSense::Minimize,
            constraints: vec![],
            binaries: BTreeSet::new(),
        }
    }
}

pub(crate) fn emit_lp(ilp: &Ilp) -> String {
    let mut out = String::new();
    match ilp.sense {
        ObjSense::Minimize => out.push_str("Minimize\n obj: "),
        ObjSense::Maximize => out.push_str("Maximize\n obj: "),
    }
    out.push_str(&fmt_lin(&ilp.objective));
    out.push('\n');
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
    out.push_str("Binary\n");
    for b in &ilp.binaries {
        out.push_str(&format!(" {}\n", b));
    }
    out.push_str("End\n");
    out
}

fn fmt_sense(s: Sense) -> &'static str {
    match s {
        Sense::Le => "<=",
        Sense::Ge => ">=",
        Sense::Eq => "=",
    }
}

fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.6}", v)
    }
}

fn fmt_lin(e: &LinearExpr) -> String {
    let mut parts: Vec<String> = vec![];
    for (n, c) in e.terms.iter() {
        if (c - 1.0).abs() < 1e-12 {
            parts.push(format!("+1 {}", n));
        } else if (c + 1.0).abs() < 1e-12 {
            parts.push(format!("-1 {}", n));
        } else {
            parts.push(format!("{:+.6} {}", c, n));
        }
    }
    if parts.is_empty() {
        parts.push("+0".to_string());
    }
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
