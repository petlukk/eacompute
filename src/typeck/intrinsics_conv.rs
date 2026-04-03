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

    pub(super) fn check_maddubs_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "maddubs_i16 expects 2 arguments: (u8x16, i8x16)",
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
            ) if matches!(ea.as_ref(), Type::U8) && matches!(eb.as_ref(), Type::I8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("maddubs_i16 expects (u8x16, i8x16), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// madd_i16(i16x8, i16x8) -> i32x4  (SSE2 pmaddwd)
    /// madd_i16(i16x16, i16x16) -> i32x8  (AVX2 vpmaddwd)
    /// Multiply i16 pairs, add adjacent products -> i32.
    pub(super) fn check_madd_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "madd_i16 expects 2 arguments: (i16xN, i16xN) where N=8 or 16",
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
            ) if matches!(ea.as_ref(), Type::I16)
                && matches!(eb.as_ref(), Type::I16)
                && wa == wb
                && (*wa == 8 || *wa == 16) =>
            {
                // i16x8,i16x8 → i32x4 (SSE2)
                // i16x16,i16x16 → i32x8 (AVX2)
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: wa / 2,
                })
            }
            _ => Err(CompileError::type_error(
                format!(
                    "madd_i16 expects (i16x8, i16x8) or (i16x16, i16x16), \
                     got ({a}, {b})"
                ),
                span.clone(),
            )),
        }
    }

    pub(super) fn check_vdot_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "vdot_i32 expects 2 arguments: (i8x16, i8x16)",
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
            ) if matches!(ea.as_ref(), Type::I8) && matches!(eb.as_ref(), Type::I8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("vdot_i32 expects (i8x16, i8x16), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// smmla_i32(acc: i32x4, a: i8x16, b: i8x16) -> i32x4
    /// ARM I8MM: signed x signed matrix multiply-accumulate.
    pub(super) fn check_smmla_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "smmla_i32 expects 3 arguments: (i32x4, i8x16, i8x16)",
                span.clone(),
            ));
        }
        let acc = self.check_expr(&args[0], locals)?;
        let a = self.check_expr(&args[1], locals)?;
        let b = self.check_expr(&args[2], locals)?;
        match (&acc, &a, &b) {
            (
                Type::Vector {
                    elem: eacc,
                    width: 4,
                },
                Type::Vector {
                    elem: ea,
                    width: 16,
                },
                Type::Vector {
                    elem: eb,
                    width: 16,
                },
            ) if matches!(eacc.as_ref(), Type::I32)
                && matches!(ea.as_ref(), Type::I8)
                && matches!(eb.as_ref(), Type::I8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("smmla_i32 expects (i32x4, i8x16, i8x16), got ({acc}, {a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// ummla_i32(acc: i32x4, a: u8x16, b: u8x16) -> i32x4
    /// ARM I8MM: unsigned x unsigned matrix multiply-accumulate.
    pub(super) fn check_ummla_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "ummla_i32 expects 3 arguments: (i32x4, u8x16, u8x16)",
                span.clone(),
            ));
        }
        let acc = self.check_expr(&args[0], locals)?;
        let a = self.check_expr(&args[1], locals)?;
        let b = self.check_expr(&args[2], locals)?;
        match (&acc, &a, &b) {
            (
                Type::Vector {
                    elem: eacc,
                    width: 4,
                },
                Type::Vector {
                    elem: ea,
                    width: 16,
                },
                Type::Vector {
                    elem: eb,
                    width: 16,
                },
            ) if matches!(eacc.as_ref(), Type::I32)
                && matches!(ea.as_ref(), Type::U8)
                && matches!(eb.as_ref(), Type::U8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("ummla_i32 expects (i32x4, u8x16, u8x16), got ({acc}, {a}, {b})"),
                span.clone(),
            )),
        }
    }

    /// usmmla_i32(acc: i32x4, a: u8x16, b: i8x16) -> i32x4
    /// ARM I8MM: unsigned x signed matrix multiply-accumulate.
    pub(super) fn check_usmmla_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "usmmla_i32 expects 3 arguments: (i32x4, u8x16, i8x16)",
                span.clone(),
            ));
        }
        let acc = self.check_expr(&args[0], locals)?;
        let a = self.check_expr(&args[1], locals)?;
        let b = self.check_expr(&args[2], locals)?;
        match (&acc, &a, &b) {
            (
                Type::Vector {
                    elem: eacc,
                    width: 4,
                },
                Type::Vector {
                    elem: ea,
                    width: 16,
                },
                Type::Vector {
                    elem: eb,
                    width: 16,
                },
            ) if matches!(eacc.as_ref(), Type::I32)
                && matches!(ea.as_ref(), Type::U8)
                && matches!(eb.as_ref(), Type::I8) =>
            {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!("usmmla_i32 expects (i32x4, u8x16, i8x16), got ({acc}, {a}, {b})"),
                span.clone(),
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

    pub(super) fn check_shuffle_bytes(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "shuffle_bytes expects 2 arguments: (u8x16, u8x16)",
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
            ) if matches!(ea.as_ref(), Type::U8) && matches!(eb.as_ref(), Type::U8) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::U8),
                    width: 16,
                })
            }
            _ => Err(CompileError::type_error(
                format!("shuffle_bytes expects (u8x16, u8x16), got ({a}, {b})"),
                span.clone(),
            )),
        }
    }
}
