//! 코드 생성 에러 타입
//! 
//! 코드 생성 과정에서 발생할 수 있는 다양한 에러를 정의합니다.

use crate::Expr;
use thiserror::Error;

/// 코드 생성 중 발생하는 에러
#[derive(Debug, Error)]
pub enum CodegenError {
    /// 알 수 없는 도메인 이름
    #[error("unknown domain `{0}`")]
    UnknownDomain(String),
    /// 알 수 없는 변수 이름
    #[error("unknown variable `{0}`")]
    UnknownVar(String),
    /// 변수의 인덱스 개수가 시그니처와 일치하지 않음
    #[error("variable `{0}` used with wrong arity: expected {1}, got {2}")]
    WrongArity(String, usize, usize),
    /// 선형 컨텍스트에서 지원하지 않는 표현식
    #[error("unsupported expression in linear context: {0:?}")]
    UnsupportedLinear(Expr),
    /// 불리언 컨텍스트에서 지원하지 않는 표현식
    #[error("unsupported boolean context: {0:?}")]
    UnsupportedBool(Expr),
    /// 지원하지 않는 함수 호출
    #[error("unsupported call `{0}`")]
    UnsupportedCall(String),
    /// Observe 함수에서 정수 시나리오가 필요한데 다른 값이 제공됨
    #[error("Observe(pin, s=..) requires integer scenario, got `{0}`")]
    BadScenario(String),
    /// 시나리오 바인더 `s`가 필요한 함수 호출에서 바인더가 없음
    #[error("missing scenario binder `s` for call `{0}`")]
    MissingScenarioBinder(String),
}
