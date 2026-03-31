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

    pub(super) fn check_maddubs_i32(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "maddubs_i32 expects 2 arguments: (u8xN, i8xN) where N=16 or 32",
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
                && matches!(eb.as_ref(), Type::I8)
                && wa == wb
                && (*wa == 16 || *wa == 32) =>
            {
                // u8x16,i8x16 → i32x4 (SSE)
                // u8x32,i8x32 → i32x8 (AVX2)
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: wa / 4,
                })
            }
            _ => Err(CompileError::type_error(
                format!(
                    "maddubs_i32 expects (u8x16, i8x16) or (u8x32, i8x32), \
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
