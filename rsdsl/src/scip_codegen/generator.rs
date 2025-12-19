//! 코드 생성기 메인 구조
//! 
//! 이 모듈은 코드 생성기의 핵심 구조와 기본 메서드를 정의합니다.
//! Sources 데이터베이스, 보조 변수 생성, 제약 조건 정규화 등을 처리합니다.

use crate::VarKind;
use crate::scip_codegen::env::Env;
use crate::scip_codegen::linear::{Constraint, Ilp, LinearExpr, Sense};
use std::collections::HashMap;

/// Sources 변수 집합을 식별하는 키
/// 
/// 형식: `변수명__인덱스1__인덱스2__...` (예: `DustSrc__s__layer__cell`)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SourceKey {
    /// Sources 집합의 정규화된 키 이름
    pub(crate) name: String,
}

/// Sources 변수에 대한 Add/Exclude 문을 저장하는 데이터베이스
#[derive(Clone, Debug)]
pub(crate) struct SourceDB {
    /// 각 Sources 키에 추가할 표현식들
    pub(crate) adds: HashMap<SourceKey, Vec<crate::Expr>>,
    /// 각 Sources 키에서 제외할 표현식들
    pub(crate) excludes: HashMap<SourceKey, Vec<crate::Expr>>,
}

impl SourceDB {
    /// 새로운 Sources 데이터베이스를 생성합니다.
    pub(crate) fn new() -> Self {
        Self {
            adds: HashMap::new(),
            excludes: HashMap::new(),
        }
    }
}

/// 코드 생성기
/// 
/// ModelSpec을 ILP로 변환하는 핵심 구조체입니다.
pub(crate) struct Generator {
    /// 코드 생성 환경 (도메인, 변수 시그니처 등)
    pub(crate) env: Env,
    /// 생성 중인 ILP (제약 조건, 목적 함수 등)
    pub(crate) ilp: Ilp,
    /// Sources 변수에 대한 Add/Exclude 문 저장소
    pub(crate) sources: SourceDB,
    /// Sources 키 -> OR 변수 이름 캐시
    pub(crate) sources_or_cache: HashMap<SourceKey, String>,
    /// 보조 변수 ID 카운터
    aux_var_id: usize,
    /// 제약 조건 ID 카운터
    cst_id: usize,
}

impl Generator {
    /// 새로운 코드 생성기를 생성합니다.
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

    /// 생성된 ILP에 대한 참조를 반환합니다.
    pub(crate) fn ilp(&self) -> &Ilp {
        &self.ilp
    }

    /// 새로운 보조 변수 이름을 생성합니다.
    /// 
    /// 형식: `__aux_{prefix}_{id}`
    pub(crate) fn fresh_aux(&mut self, prefix: &str) -> String {
        let n = self.aux_var_id;
        self.aux_var_id += 1;
        format!("__aux_{}_{}", prefix, n)
    }

    /// 다음 제약 조건 ID를 반환하고 카운터를 증가시킵니다.
    pub(crate) fn next_cst_id(&mut self) -> usize {
        let n = self.cst_id;
        self.cst_id += 1;
        n
    }

    /// 상수 변수를 초기화합니다.
    /// 
    /// `__const0 = 0`과 `__const1 = 1` 제약 조건을 추가합니다.
    pub(crate) fn init_constants(&mut self) {
        self.ilp.binaries.insert("__const0".to_string());
        self.ilp.binaries.insert("__const1".to_string());
        // __const0 = 0
        self.ilp.constraints.push(Constraint {
            name: "__fix_const0".to_string(),
            expr: LinearExpr::from_var("__const0", 1.0),
            sense: Sense::Eq,
            rhs: 0.0,
        });
        // __const1 = 1
        self.ilp.constraints.push(Constraint {
            name: "__fix_const1".to_string(),
            expr: LinearExpr::from_var("__const1", 1.0),
            sense: Sense::Eq,
            rhs: 1.0,
        });
    }

    /// 피처 플래그가 활성화되어 있는지 확인합니다.
    pub(crate) fn is_feature_on(&self, name: &str) -> bool {
        self.env.features.contains(name)
    }

    /// 변수의 종류를 반환합니다.
    pub(crate) fn var_kind(&self, name: &str) -> Option<VarKind> {
        self.env.var_kinds.get(name).cloned()
    }

    /// 제약 조건을 정규화합니다.
    /// 
    /// 좌변의 상수항을 우변으로 이동시킵니다.
    /// 목적 함수의 상수항은 제거합니다 (최적화에 영향 없음).
    pub(crate) fn normalize(&mut self) {
        for c in self.ilp.constraints.iter_mut() {
            if c.expr.constant.abs() > 1e-12 {
                c.rhs -= c.expr.constant;
                c.expr.constant = 0.0;
            }
        }
        // 목적 함수의 상수항 제거 (최적화 결과에 영향 없음)
        if self.ilp.objective.constant.abs() > 1e-12 {
            self.ilp.objective.constant = 0.0;
        }
    }
}
