use crate::{Cell, Decl, Instance, ModelSpec, VarKind};
use crate::scip_codegen::error::CodegenError;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Env {
    /// domain values: domain_name -> concrete values (as strings)
    pub(crate) domains: HashMap<String, Vec<String>>,
    /// variable signature: var_name -> domain names per index
    pub(crate) sigs: HashMap<String, Vec<String>>,
    /// kind of each variable (for special lowering)
    pub(crate) var_kinds: HashMap<String, VarKind>,
    /// feature flags + parameters + Observe() mapping
    pub(crate) features: HashSet<String>,
    pub(crate) params: HashMap<String, f64>,
    pub(crate) observe: Arc<dyn Fn(&str, i32) -> crate::ConcreteVar + Send + Sync>,
    /// for topo bounds: which cell ids exist
    pub(crate) cell_set: HashSet<String>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Ctx {
    pub(crate) bind: HashMap<String, String>,
    pub(crate) lets: HashMap<String, String>,
}

pub(crate) fn build_env(spec: &ModelSpec, inst: &Instance) -> Result<Env, CodegenError> {
    let mut domains: HashMap<String, Vec<String>> = HashMap::new();

    // default Cell domain from instance
    let cell_vals: Vec<String> = inst.cells.iter().map(Cell::id).collect();
    domains.insert("Cell".to_string(), cell_vals.clone());

    // default Layer/Dir if declared; can be overridden by enums
    domains.insert("Layer".to_string(), vec!["GROUND".into(), "TOP".into()]);
    domains.insert("Dir".to_string(), vec!["N".into(), "E".into(), "S".into(), "W".into()]);

    // Scenario domain
    domains.insert("s".to_string(), inst.scenarios.iter().map(|v| v.to_string()).collect());
    domains.insert("Sc".to_string(), inst.scenarios.iter().map(|v| v.to_string()).collect());

    let mut sigs: HashMap<String, Vec<String>> = HashMap::new();
    let mut var_kinds: HashMap<String, VarKind> = HashMap::new();

    for d in &spec.decls {
        match d {
            Decl::Enum { name, variants } => {
                domains.insert(name.clone(), variants.clone());
            }
            Decl::Scenario { name, values } => {
                domains.insert(name.clone(), values.iter().map(|v| v.to_string()).collect());
            }
            Decl::Var { kind, name, indices, .. } => {
                sigs.insert(name.clone(), indices.clone());
                var_kinds.insert(name.clone(), kind.clone());
            }
            Decl::Index { name, .. } => {
                // index decl indicates a domain name exists; already covered by instance for Cell.
                if name != "Cell" && !domains.contains_key(name) {
                    domains.insert(name.clone(), vec![]);
                }
            }
            _ => {}
        }
    }

    let cell_set = cell_vals.into_iter().collect::<HashSet<_>>();

    Ok(Env {
        domains,
        sigs,
        var_kinds,
        features: inst.features.clone(),
        params: inst.params.clone(),
        observe: inst.observe.clone(),
        cell_set,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConcreteVar, ObjSense, Objective, Rule, Stmt, Expr, VarRef, Decl};

    #[test]
    fn test_build_env_defaults() {
        let spec = ModelSpec {
            name: "M".into(),
            decls: vec![
                Decl::Var { kind: VarKind::State, name: "X".into(), indices: vec!["Sc".into(), "Cell".into()], ty: "Bool".into() },
            ],
            rules: vec![Rule { name: "r".into(), body: vec![Stmt::Require(Expr::Lit(1.0))]}],
            objective: Some(Objective { sense: ObjSense::Minimize, expr: Expr::Lit(0.0)}),
        };
        let inst = Instance::new(
            vec![Cell{x:0,z:0}],
            vec![0,1],
            HashMap::new(),
            HashSet::new(),
            Arc::new(|pin, s| ConcreteVar::new(format!("OBS_{}_{}", pin, s))),
        );
        let env = build_env(&spec, &inst).unwrap();
        assert_eq!(env.domains["Cell"], vec!["x0_z0"]);
        assert_eq!(env.domains["Sc"], vec!["0","1"]);
    }
}
