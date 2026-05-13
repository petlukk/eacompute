use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// bsrli_i8x16(v: i8x16, imm: 0..=15) -> i8x16
    /// Byte shift right within 128-bit lane, shifting in zeros.
    /// x86: vpsrldq (SSE2). ARM: ext(v, zero, imm).
    pub(super) fn check_bsrli_i8x16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        self.check_byteshift("bsrli_i8x16", args, locals, span, Type::I8, 16)
    }

    /// bsrli_i8x32(v: i8x32, imm: 0..=15) -> i8x32
    /// Byte shift right within each 128-bit lane, shifting in zeros.
    /// x86-only: vpsrldq (AVX2).
    pub(super) fn check_bsrli_i8x32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        self.check_byteshift("bsrli_i8x32", args, locals, span, Type::I8, 32)
    }

    /// bslli_i8x16(v: i8x16, imm: 0..=15) -> i8x16
    /// Byte shift left within 128-bit lane, shifting in zeros.
    /// x86: vpslldq (SSE2). ARM: ext(zero, v, 16-imm).
    pub(super) fn check_bslli_i8x16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        self.check_byteshift("bslli_i8x16", args, locals, span, Type::I8, 16)
    }

    /// bslli_i8x32(v: i8x32, imm: 0..=15) -> i8x32
    /// Byte shift left within each 128-bit lane, shifting in zeros.
    /// x86-only: vpslldq (AVX2).
    pub(super) fn check_bslli_i8x32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        self.check_byteshift("bslli_i8x32", args, locals, span, Type::I8, 32)
    }

    fn check_byteshift(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        expected_elem: Type,
        expected_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                format!("{name} expects 2 arguments: ({expected_elem}x{expected_width}, imm)"),
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let vec_ty = Type::Vector {
            elem: Box::new(expected_elem.clone()),
            width: expected_width,
        };
        if a != vec_ty {
            return Err(CompileError::type_error(
                format!(
                    "{name} expects {expected_elem}x{expected_width} as first argument, got {a}"
                ),
                span.clone(),
            ));
        }
        let imm = self.eval_const_expr(&args[1]).map_err(|_| {
            CompileError::type_error(
                format!("{name} requires a compile-time integer constant as second argument"),
                span.clone(),
            )
        })?;
        let imm_val = match imm {
            super::const_eval::ConstValue::Integer(v) => v,
            _ => {
                return Err(CompileError::type_error(
                    format!("{name} requires an integer immediate, not a float or bool"),
                    span.clone(),
                ));
            }
        };
        if !(0..=15).contains(&imm_val) {
            return Err(CompileError::type_error(
                format!("{name} immediate must be 0..=15, got {imm_val}"),
                span.clone(),
            ));
        }
        Ok(vec_ty)
    }
}
