use crate::Expr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodegenError {
    #[error("unknown domain `{0}`")]
    UnknownDomain(String),
    #[error("unknown variable `{0}`")]
    UnknownVar(String),
    #[error("variable `{0}` used with wrong arity: expected {1}, got {2}")]
    WrongArity(String, usize, usize),
    #[error("unsupported expression in linear context: {0:?}")]
    UnsupportedLinear(Expr),
    #[error("unsupported boolean context: {0:?}")]
    UnsupportedBool(Expr),
    #[error("unsupported call `{0}`")]
    UnsupportedCall(String),
    #[error("Observe(pin, s=..) requires integer scenario, got `{0}`")]
    BadScenario(String),
    #[error("missing scenario binder `s` for call `{0}`")]
    MissingScenarioBinder(String),
}
