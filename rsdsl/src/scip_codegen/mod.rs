//! SCIP 코드 생성 모듈
//!
//! 이 모듈은 ModelSpec과 Instance를 SCIP 호환 LP 파일 형식으로 변환합니다.

mod binders;
mod emit;
mod env;
mod error;
mod eval;
mod generator;
mod linear;
mod sources;

pub use error::CodegenError;

use crate::{Instance, ModelSpec};
use env::build_env;
use generator::Generator;
use linear::emit_lp;

/// 진입점: `ModelSpec`과 `Instance`를 SCIP 호환 LP 파일 문자열로 변환합니다.
///
/// # 처리 단계
/// 1. 환경 구축: 도메인, 변수 시그니처, 피처 플래그 등을 설정
/// 2. 상수 초기화: __const0, __const1 등 기본 상수 변수 생성
/// 3. 소스 수집: Sources 타입 변수에 대한 Add/Exclude 문 수집
/// 4. 모델 생성: Require/Def/ForceEq 제약 조건 및 목적 함수 생성
/// 5. 정규화: 제약 조건의 상수항을 우변으로 이동
/// 6. LP 형식 출력: 최종 LP 파일 문자열 생성
pub fn codegen_scip_lp(spec: &ModelSpec, inst: &Instance) -> Result<String, CodegenError> {
    // 환경 구축: 도메인 값, 변수 시그니처, 피처 플래그 등을 설정
    let env = build_env(spec, inst)?;
    let mut gen = Generator::new(env);

    // 상수 변수 초기화 (__const0 = 0, __const1 = 1)
    gen.init_constants();

    // 1단계: Sources 타입 변수에 대한 Add/Exclude 문 수집
    gen.collect_sources(spec)?;

    // 2단계: 제약 조건 (Require/Def/ForceEq) 및 목적 함수 생성
    gen.emit_model(spec)?;

    // 제약 조건 정규화: 좌변의 상수항을 우변으로 이동
    gen.normalize();

    // LP 파일 형식으로 출력
    Ok(emit_lp(gen.ilp()))
}
