use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    pub(super) fn check_widen_i8_f32(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        output_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument (i8x16 or u8x16 vector)"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 16 } if matches!(elem.as_ref(), Type::I8 | Type::U8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::F32),
                    width: output_width,
                })
            }
            _ => Err(CompileError::type_error(
                format!("{name} expects i8x16 or u8x16, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_widen_u8_i32(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        output_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument (u8x16 vector)"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 16 } if matches!(elem.as_ref(), Type::U8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: output_width,
                })
            }
            _ => Err(CompileError::type_error(
                format!("{name} expects u8x16, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_narrow_f32x4_i8(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "narrow_f32x4_i8 expects 1 argument (f32x4 vector)",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 4 } if matches!(elem.as_ref(), Type::F32) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I8),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("narrow_f32x4_i8 expects f32x4, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_sat_add(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "sat_add expects 2 arguments",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        if a != b {
            return Err(CompileError::type_error(
                format!("sat_add arguments must have the same type, got ({a}, {b})"),
                span.clone(),
            ));
        }
        match &a {
            Type::Vector { elem, width }
                if matches!(
                    (elem.as_ref(), *width),
                    (Type::I8, 16) | (Type::U8, 16) | (Type::I16, 8) | (Type::U16, 8)
                ) =>
            {
                Ok(a)
            }
            _ => Err(CompileError::type_error(
                format!("sat_add supports i8x16, u8x16, i16x8, u16x8; got {a}"),
                span.clone(),
            )),
        }
    }

    pub(super) fn check_sat_sub(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "sat_sub expects 2 arguments",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        if a != b {
            return Err(CompileError::type_error(
                format!("sat_sub arguments must have the same type, got ({a}, {b})"),
                span.clone(),
            ));
        }
        match &a {
            Type::Vector { elem, width }
                if matches!(
                    (elem.as_ref(), *width),
                    (Type::I8, 16) | (Type::U8, 16) | (Type::I16, 8) | (Type::U16, 8)
                ) =>
            {
                Ok(a)
            }
            _ => Err(CompileError::type_error(
                format!("sat_sub supports i8x16, u8x16, i16x8, u16x8; got {a}"),
                span.clone(),
            )),
        }
    }

    pub(super) fn check_widen_u8_u16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "widen_u8_u16 expects 1 argument (u8x16 vector)",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 16 } if matches!(elem.as_ref(), Type::U8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::U16),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("widen_u8_u16 expects u8x16, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    /// shuffle_bytes(u8x16, u8x16) -> u8x16  (SSSE3 pshufb / NEON tbl)
    /// shuffle_bytes(u8x32, u8x32) -> u8x32  (AVX2 vpshufb, x86-only)
    pub(super) fn check_shuffle_bytes(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "shuffle_bytes expects 2 arguments: (u8xN, u8xN) where N=16 or 32",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (
                Type::Vector {
                    elem: ea,
                    width: wa,
                },
                Type::Vector {
                    elem: eb,
                    width: wb,
                },
            ) if matches!(ea.as_ref(), Type::U8)
                && matches!(eb.as_ref(), Type::U8)
                && wa == wb
                && (*wa == 16 || *wa == 32) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::U8),
                    width: *wa,
                })
            }
            _ => Err(CompileError::type_error(
                format!("shuffle_bytes expects (u8x16, u8x16) or (u8x32, u8x32), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }
}
