use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// abs_diff(a, b) -> same type.
    /// ARM-only: supported types are i8x16, u8x16, i16x8, u16x8, i32x4, u32x4.
    pub(super) fn check_abs_diff(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "abs_diff expects 2 arguments",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        if a != b {
            return Err(CompileError::type_error(
                format!("abs_diff arguments must have the same type, got ({a}, {b})"),
                span.clone(),
            ));
        }
        match &a {
            Type::Vector { elem, width }
                if matches!(
                    (elem.as_ref(), *width),
                    (Type::I8, 16)
                        | (Type::U8, 16)
                        | (Type::I16, 8)
                        | (Type::U16, 8)
                        | (Type::I32, 4)
                        | (Type::U32, 4)
                ) =>
            {
                Ok(a)
            }
            _ => Err(CompileError::type_error(
                format!("abs_diff supports i8x16, u8x16, i16x8, u16x8, i32x4, u32x4; got {a}"),
                span.clone(),
            )),
        }
    }

    /// wmul_i16(a: i8x8, b: i8x8) -> i16x8. ARM-only widening multiply.
    pub(super) fn check_wmul_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "wmul_i16 expects 2 arguments: (i8x8, i8x8)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (Type::Vector { elem: ea, width: 8 }, Type::Vector { elem: eb, width: 8 })
                if matches!(ea.as_ref(), Type::I8) && matches!(eb.as_ref(), Type::I8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("wmul_i16 expects (i8x8, i8x8), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// wmul_u16(a: u8x8, b: u8x8) -> u16x8. ARM-only widening multiply.
    pub(super) fn check_wmul_u16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "wmul_u16 expects 2 arguments: (u8x8, u8x8)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (Type::Vector { elem: ea, width: 8 }, Type::Vector { elem: eb, width: 8 })
                if matches!(ea.as_ref(), Type::U8) && matches!(eb.as_ref(), Type::U8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::U16),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("wmul_u16 expects (u8x8, u8x8), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// wmul_i32(a: i16x4, b: i16x4) -> i32x4. ARM-only widening multiply.
    pub(super) fn check_wmul_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "wmul_i32 expects 2 arguments: (i16x4, i16x4)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (Type::Vector { elem: ea, width: 4 }, Type::Vector { elem: eb, width: 4 })
                if matches!(ea.as_ref(), Type::I16) && matches!(eb.as_ref(), Type::I16) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("wmul_i32 expects (i16x4, i16x4), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// wmul_u32(a: u16x4, b: u16x4) -> u32x4. ARM-only widening multiply.
    pub(super) fn check_wmul_u32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "wmul_u32 expects 2 arguments: (u16x4, u16x4)",
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        match (&a, &b) {
            (Type::Vector { elem: ea, width: 4 }, Type::Vector { elem: eb, width: 4 })
                if matches!(ea.as_ref(), Type::U16) && matches!(eb.as_ref(), Type::U16) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::U32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("wmul_u32 expects (u16x4, u16x4), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }
}
