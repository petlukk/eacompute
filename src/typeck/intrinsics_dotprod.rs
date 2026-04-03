use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// maddubs_i16(u8x16, i8x16) -> i16x8   (SSSE3 pmaddubsw)
    /// maddubs_i16(u8x32, i8x32) -> i16x16  (AVX2 vpmaddubsw)
    pub(super) fn check_maddubs_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "maddubs_i16 expects 2 arguments: (u8xN, i8xN) where N=16 or 32",
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
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: wa / 2,
                })
            }
            _ => Err(CompileError::type_error(
                format!("maddubs_i16 expects (u8x16, i8x16) or (u8x32, i8x32), got ({a}, {b})"),
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

    /// hadd_i16(i16x8, i16x8) -> i16x8   (SSSE3 phaddw)
    /// hadd_i16(i16x16, i16x16) -> i16x16  (AVX2 vphaddw)
    /// Horizontal add: adjacent i16 pairs from both inputs.
    pub(super) fn check_hadd_i16(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "hadd_i16 expects 2 arguments: (i16xN, i16xN) where N=8 or 16",
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
                Ok(Type::Vector {
                    elem: Box::new(Type::I16),
                    width: *wa,
                })
            }
            _ => Err(CompileError::type_error(
                format!(
                    "hadd_i16 expects (i16x8, i16x8) or (i16x16, i16x16), \
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
}
