use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// round_f32x8_i32x8(f32x8) -> i32x8
    /// Round-to-nearest-even (banker's rounding) on both x86 and ARM.
    pub(super) fn check_round_f32x8_i32x8(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "round_f32x8_i32x8 expects 1 argument (f32x8 vector)",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 8 } if matches!(elem.as_ref(), Type::F32) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("round_f32x8_i32x8 expects f32x8, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    /// pack_sat_i32x8(i32x8, i32x8) -> i16x16
    /// Saturating narrow two i32x8 into one i16x16.
    /// x86: vpackssdw. ARM: sqxtn + sqxtn2.
    pub(super) fn check_pack_sat_i32x8(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "pack_sat_i32x8 expects 2 arguments (i32x8, i32x8)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (Type::Vector { elem: ea, width: 8 }, Type::Vector { elem: eb, width: 8 })
                if matches!(ea.as_ref(), Type::I32) && matches!(eb.as_ref(), Type::I32) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: 16,
                })
            }
            _ => Err(CompileError::type_error(
                format!("pack_sat_i32x8 expects (i32x8, i32x8), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// pack_sat_i16x16(i16x16, i16x16) -> i8x32
    /// Saturating narrow two i16x16 into one i8x32.
    /// x86: vpacksswb. ARM: sqxtn + sqxtn2.
    pub(super) fn check_pack_sat_i16x16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "pack_sat_i16x16 expects 2 arguments (i16x16, i16x16)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (
                Type::Vector {
                    elem: ea,
                    width: 16,
                },
                Type::Vector {
                    elem: eb,
                    width: 16,
                },
            ) if matches!(ea.as_ref(), Type::I16) && matches!(eb.as_ref(), Type::I16) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I8),
                    width: 32,
                })
            }
            _ => Err(CompileError::type_error(
                format!("pack_sat_i16x16 expects (i16x16, i16x16), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }
}
