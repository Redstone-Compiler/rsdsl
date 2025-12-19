//! 코드 생성 환경 설정
//! 
//! 이 모듈은 ModelSpec과 Instance로부터 코드 생성에 필요한 환경 정보를 구축합니다.
//! 도메인 값, 변수 시그니처, 피처 플래그 등을 관리합니다.

use crate::{Cell, Decl, Instance, ModelSpec, VarKind};
use crate::scip_codegen::error::CodegenError;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// 코드 생성에 필요한 환경 정보
#[derive(Clone)]
pub(crate) struct Env {
    /// 도메인 값: 도메인 이름 -> 구체적인 값들 (문자열)
    pub(crate) domains: HashMap<String, Vec<String>>,
    /// 변수 시그니처: 변수 이름 -> 각 인덱스의 도메인 이름들
    pub(crate) sigs: HashMap<String, Vec<String>>,
    /// 각 변수의 종류 (특수 변환을 위해 사용)
    pub(crate) var_kinds: HashMap<String, VarKind>,
    /// 피처 플래그, 파라미터, Observe() 매핑
    pub(crate) features: HashSet<String>,
    /// 파라미터 값: 파라미터 이름 -> 값
    pub(crate) params: HashMap<String, f64>,
    /// Observe 함수: (pin, scenario) -> ConcreteVar
    pub(crate) observe: Arc<dyn Fn(&str, i32) -> crate::ConcreteVar + Send + Sync>,
    /// 토폴로지 경계 계산용: 존재하는 셀 ID 집합
    pub(crate) cell_set: HashSet<String>,
}

/// 변수 바인딩 컨텍스트
/// 
/// 표현식 평가 및 코드 생성 중 변수 바인딩 정보를 유지합니다.
#[derive(Clone, Debug, Default)]
pub(crate) struct Ctx {
    /// 바인더에서 바인딩된 변수: 변수 이름 -> 값
    pub(crate) bind: HashMap<String, String>,
    /// Let 문에서 바인딩된 변수: 변수 이름 -> 값
    pub(crate) lets: HashMap<String, String>,
}

/// ModelSpec과 Instance로부터 코드 생성 환경을 구축합니다.
/// 
/// # 처리 내용
/// 1. 기본 도메인 설정 (Cell, Layer, Dir, Scenario)
/// 2. 선언된 Enum 및 Scenario 도메인 추가
/// 3. 변수 시그니처 및 종류 수집
/// 4. 피처 플래그 및 파라미터 설정
pub(crate) fn build_env(spec: &ModelSpec, inst: &Instance) -> Result<Env, CodegenError> {
    let mut domains: HashMap<String, Vec<String>> = HashMap::new();

    // 인스턴스에서 Cell 도메인 값 추출
    let cell_vals: Vec<String> = inst.cells.iter().map(Cell::id).collect();
    domains.insert("Cell".to_string(), cell_vals.clone());

    // 기본 Layer/Dir 도메인 (선언되면 enum으로 덮어쓸 수 있음)
    domains.insert("Layer".to_string(), vec!["GROUND".into(), "TOP".into()]);
    domains.insert("Dir".to_string(), vec!["N".into(), "E".into(), "S".into(), "W".into()]);

    // 시나리오 도메인 (s와 Sc는 동일한 값 사용)
    domains.insert("s".to_string(), inst.scenarios.iter().map(|v| v.to_string()).collect());
    domains.insert("Sc".to_string(), inst.scenarios.iter().map(|v| v.to_string()).collect());

    let mut sigs: HashMap<String, Vec<String>> = HashMap::new();
    let mut var_kinds: HashMap<String, VarKind> = HashMap::new();

    // 선언들을 처리하여 도메인, 변수 시그니처 등을 수집
    for d in &spec.decls {
        match d {
            // Enum 선언: 도메인에 variant 값들 추가
            Decl::Enum { name, variants } => {
                domains.insert(name.clone(), variants.clone());
            }
            // Scenario 선언: 도메인에 시나리오 값들 추가
            Decl::Scenario { name, values } => {
                domains.insert(name.clone(), values.iter().map(|v| v.to_string()).collect());
            }
            // 변수 선언: 시그니처와 종류 저장
            Decl::Var { kind, name, indices, .. } => {
                sigs.insert(name.clone(), indices.clone());
                var_kinds.insert(name.clone(), kind.clone());
            }
            // Index 선언: 도메인 이름이 존재함을 나타냄 (Cell은 인스턴스에서 이미 처리됨)
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
    use crate::{ConcreteVar, ObjSense, Objective, Rule, Stmt, Expr, Decl};

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
