use crate::VarKind;
use crate::scip_codegen::env::{Ctx, Env};
use crate::scip_codegen::linear::{Constraint, Ilp, LinearExpr, Sense};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SourceKey {
    /// canonical key name for source set: DustSrc__s__layer__cell
    pub(crate) name: String,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceDB {
    pub(crate) adds: HashMap<SourceKey, Vec<crate::Expr>>,
    pub(crate) excludes: HashMap<SourceKey, Vec<crate::Expr>>,
}

impl SourceDB {
    pub(crate) fn new() -> Self {
        Self {
            adds: HashMap::new(),
            excludes: HashMap::new(),
        }
    }
}

pub(crate) struct Generator {
    pub(crate) env: Env,
    pub(crate) ilp: Ilp,
    pub(crate) sources: SourceDB,
    /// cache: sources key -> OR var
    pub(crate) sources_or_cache: HashMap<SourceKey, String>,
    aux_var_id: usize,
    cst_id: usize,
}

impl Generator {
    pub(crate) fn new(env: Env) -> Self {
        Self {
            env,
            ilp: Ilp::new(),
            sources: SourceDB::new(),
            sources_or_cache: HashMap::new(),
            aux_var_id: 0,
            cst_id: 0,
        }
    }

    pub(crate) fn ilp(&self) -> &Ilp {
        &self.ilp
    }

    pub(crate) fn fresh_aux(&mut self, prefix: &str) -> String {
        let n = self.aux_var_id;
        self.aux_var_id += 1;
        format!("__aux_{}_{}", prefix, n)
    }

    pub(crate) fn next_cst_id(&mut self) -> usize {
        let n = self.cst_id;
        self.cst_id += 1;
        n
    }

    pub(crate) fn init_constants(&mut self) {
        self.ilp.binaries.insert("__const0".to_string());
        self.ilp.binaries.insert("__const1".to_string());
        self.ilp.constraints.push(Constraint {
            name: "__fix_const0".to_string(),
            expr: LinearExpr::from_var("__const0", 1.0),
            sense: Sense::Eq,
            rhs: 0.0,
        });
        self.ilp.constraints.push(Constraint {
            name: "__fix_const1".to_string(),
            expr: LinearExpr::from_var("__const1", 1.0),
            sense: Sense::Eq,
            rhs: 1.0,
        });
    }

    pub(crate) fn is_feature_on(&self, name: &str) -> bool {
        self.env.features.contains(name)
    }

    pub(crate) fn var_kind(&self, name: &str) -> Option<VarKind> {
        self.env.var_kinds.get(name).cloned()
    }

    pub(crate) fn normalize(&mut self) {
        for c in self.ilp.constraints.iter_mut() {
            if c.expr.constant.abs() > 1e-12 {
                c.rhs -= c.expr.constant;
                c.expr.constant = 0.0;
            }
        }
        if self.ilp.objective.constant.abs() > 1e-12 {
            self.ilp.objective.constant = 0.0;
        }
    }
}
