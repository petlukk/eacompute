use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// cvt_f16_f32(i16x4) -> f32x4, cvt_f16_f32(i16x8) -> f32x8, cvt_f16_f32(i16x16) -> f32x16
    pub(super) fn check_cvt_f16_f32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "cvt_f16_f32 expects 1 argument: i16x4, i16x8 or i16x16",
                span.clone(),
            ));
        }
        let arg = self.check_expr(&args[0], locals)?;
        match &arg {
            Type::Vector { elem, width }
                if matches!(elem.as_ref(), Type::I16)
                    && (*width == 4 || *width == 8 || *width == 16) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::F32),
                    width: *width,
                })
            }
            _ => Err(CompileError::type_error(
                format!("cvt_f16_f32 expects i16x4, i16x8 or i16x16, got {arg}"),
                args[0].span().clone(),
            )),
        }
    }

    /// cvt_f32_f16(f32x4) -> i16x4, cvt_f32_f16(f32x8) -> i16x8
    pub(super) fn check_cvt_f32_f16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "cvt_f32_f16 expects 1 argument: f32x4 or f32x8",
                span.clone(),
            ));
        }
        let arg = self.check_expr(&args[0], locals)?;
        match &arg {
            Type::Vector { elem, width }
                if matches!(elem.as_ref(), Type::F32) && (*width == 4 || *width == 8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: *width,
                })
            }
            _ => Err(CompileError::type_error(
                format!("cvt_f32_f16 expects f32x4 or f32x8, got {arg}"),
                args[0].span().clone(),
            )),
        }
    }
}
