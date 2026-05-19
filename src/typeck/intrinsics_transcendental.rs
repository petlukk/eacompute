//! Type checks for the f32-vector transcendental approximation family:
//! `exp_poly_f32` (v1.11.0), `tanh_approx_f32` / `log_approx_f32` /
//! `sin_approx_f32` / `cos_approx_f32` (v1.14.0).
//!
//! All five share the same shape — accept `f32xN` for any vector width;
//! reject scalar, f64, f16, and integer vectors. Codegen retains a
//! defense-in-depth guard against malformed args. The shared shape means
//! these definitions could collapse into one helper parameterized by
//! intrinsic name and libm fallback, but the per-intrinsic error messages
//! (pointing at the right libm scalar fallback) are stable enough that
//! the deduplication isn't worth the indirection.

use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// Type-check `exp_poly_f32(v: f32xN) -> f32xN`. Restricts to f32-element
    /// vectors only — scalar f32, f64xN, f16xN, integer vectors all rejected
    /// at typeck. Codegen retains a defense-in-depth guard.
    pub(super) fn check_exp_poly_f32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("exp_poly_f32 expects 1 argument, got {}", args.len()),
                span.clone(),
            ));
        }
        let t = self.check_expr(&args[0], locals)?;
        match &t {
            Type::Vector { elem, .. } if **elem == Type::F32 => Ok(t),
            Type::Vector { elem, .. } => Err(CompileError::type_error(
                format!("exp_poly_f32 expects f32 element type, got {elem}"),
                span.clone(),
            )),
            Type::F32 | Type::FloatLiteral => Err(CompileError::type_error(
                "exp_poly_f32 expects f32 vector, got scalar; use exp() for scalar libm-precision"
                    .to_string(),
                span.clone(),
            )),
            _ => Err(CompileError::type_error(
                format!("exp_poly_f32 expects float vector, got {t}"),
                span.clone(),
            )),
        }
    }

    /// Type-check `tanh_approx_f32(v: f32xN) -> f32xN`. Same f32-vector-only
    /// shape as `exp_poly_f32`; scalar / f64 / f16 / integer all rejected.
    pub(super) fn check_tanh_approx_f32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("tanh_approx_f32 expects 1 argument, got {}", args.len()),
                span.clone(),
            ));
        }
        let t = self.check_expr(&args[0], locals)?;
        match &t {
            Type::Vector { elem, .. } if **elem == Type::F32 => Ok(t),
            Type::Vector { elem, .. } => Err(CompileError::type_error(
                format!("tanh_approx_f32 expects f32 element type, got {elem}"),
                span.clone(),
            )),
            Type::F32 | Type::FloatLiteral => Err(CompileError::type_error(
                "tanh_approx_f32 expects f32 vector, got scalar; use tanh() for scalar libm-precision"
                    .to_string(),
                span.clone(),
            )),
            _ => Err(CompileError::type_error(
                format!("tanh_approx_f32 expects float vector, got {t}"),
                span.clone(),
            )),
        }
    }

    /// Type-check `log_approx_f32(v: f32xN) -> f32xN`. Same f32-vector-only
    /// shape as `exp_poly_f32`; scalar / f64 / f16 / integer all rejected.
    pub(super) fn check_log_approx_f32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("log_approx_f32 expects 1 argument, got {}", args.len()),
                span.clone(),
            ));
        }
        let t = self.check_expr(&args[0], locals)?;
        match &t {
            Type::Vector { elem, .. } if **elem == Type::F32 => Ok(t),
            Type::Vector { elem, .. } => Err(CompileError::type_error(
                format!("log_approx_f32 expects f32 element type, got {elem}"),
                span.clone(),
            )),
            Type::F32 | Type::FloatLiteral => Err(CompileError::type_error(
                "log_approx_f32 expects f32 vector, got scalar; use log() for scalar libm-precision"
                    .to_string(),
                span.clone(),
            )),
            _ => Err(CompileError::type_error(
                format!("log_approx_f32 expects float vector, got {t}"),
                span.clone(),
            )),
        }
    }

    /// Type-check `sin_approx_f32(v: f32xN) -> f32xN` and
    /// `cos_approx_f32(v: f32xN) -> f32xN`. Same f32-vector-only shape as
    /// `exp_poly_f32` — scalar / f64 / f16 / integer all rejected. Error
    /// text varies by intrinsic name to point at the right libm fallback.
    pub(super) fn check_sin_cos_approx_f32(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument, got {}", args.len()),
                span.clone(),
            ));
        }
        let libm_fallback = if name == "cos_approx_f32" {
            "cos"
        } else {
            "sin"
        };
        let t = self.check_expr(&args[0], locals)?;
        match &t {
            Type::Vector { elem, .. } if **elem == Type::F32 => Ok(t),
            Type::Vector { elem, .. } => Err(CompileError::type_error(
                format!("{name} expects f32 element type, got {elem}"),
                span.clone(),
            )),
            Type::F32 | Type::FloatLiteral => Err(CompileError::type_error(
                format!(
                    "{name} expects f32 vector, got scalar; use {libm_fallback}() for scalar libm-precision"
                ),
                span.clone(),
            )),
            _ => Err(CompileError::type_error(
                format!("{name} expects float vector, got {t}"),
                span.clone(),
            )),
        }
    }
}
