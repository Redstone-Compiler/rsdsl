mod env;
mod error;
mod linear;
mod generator;
mod eval;
mod sources;
mod emit;
mod binders;

pub use error::CodegenError;

use crate::{Instance, ModelSpec};
use env::build_env;
use generator::Generator;
use linear::emit_lp;

/// Entry point: lower a `ModelSpec` + `Instance` into an LP file string (SCIP-compatible).
pub fn codegen_scip_lp(spec: &ModelSpec, inst: &Instance) -> Result<String, CodegenError> {
    let env = build_env(spec, inst)?;
    let mut gen = Generator::new(env);

    // constants
    gen.init_constants();

    // 1) collect sources
    gen.collect_sources(spec)?;

    // 2) emit constraints (Require/Def/ForceEq) + objective
    gen.emit_model(spec)?;

    // normalize constraints: move constants to rhs
    gen.normalize();

    Ok(emit_lp(gen.ilp()))
}
