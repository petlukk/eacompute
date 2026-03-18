use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::{self, Type};

impl TypeChecker {
    pub(super) fn check_reduction(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, .. } => Ok(*elem.clone()),
            _ => Err(CompileError::type_error(
                format!("{name} expects vector argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_float_reduction(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, .. } if elem.is_float() => Ok(*elem.clone()),
            Type::Vector { .. } => Err(CompileError::type_error(
                format!(
                    "{name} expects float vector argument, got {arg_type}. \
                     Integer reduce_add is already unordered — use reduce_add instead"
                ),
                args[0].span().clone(),
            )),
            _ => Err(CompileError::type_error(
                format!("{name} expects float vector argument, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_shuffle(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "shuffle expects 2 arguments (vector, indices)",
                span.clone(),
            ));
        }
        let vec_type = self.check_expr(&args[0], locals)?;
        let width = match &vec_type {
            Type::Vector { width, .. } => *width,
            _ => {
                return Err(CompileError::type_error(
                    format!("shuffle first argument must be vector, got {vec_type}"),
                    args[0].span().clone(),
                ));
            }
        };

        match &args[1] {
            Expr::ArrayLiteral(indices, arr_span) => {
                if indices.len() != width {
                    return Err(CompileError::type_error(
                        format!(
                            "shuffle indices length {} != vector width {width}",
                            indices.len()
                        ),
                        arr_span.clone(),
                    ));
                }
                for (i, idx) in indices.iter().enumerate() {
                    match idx {
                        Expr::Literal(crate::ast::Literal::Integer(n), idx_span) => {
                            if *n < 0 || *n >= width as i64 {
                                return Err(CompileError::type_error(
                                    format!("shuffle index {i} out of range: {n} (width {width})"),
                                    idx_span.clone(),
                                ));
                            }
                        }
                        _ => {
                            return Err(CompileError::type_error(
                                format!("shuffle index {i} must be integer literal"),
                                idx.span().clone(),
                            ));
                        }
                    }
                }
            }
            _ => {
                return Err(CompileError::type_error(
                    "shuffle second argument must be [index, ...] array literal",
                    args[1].span().clone(),
                ));
            }
        }
        Ok(vec_type)
    }

    pub(super) fn check_select(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "select expects 3 arguments (mask, a, b)",
                span.clone(),
            ));
        }
        let mask_type = self.check_expr(&args[0], locals)?;
        let a_type = self.check_expr(&args[1], locals)?;
        let b_type = self.check_expr(&args[2], locals)?;
        types::unify_vector(&a_type, &b_type, span.clone())?;

        match (&mask_type, &a_type) {
            (
                Type::Vector {
                    elem: mask_elem,
                    width: mask_w,
                },
                Type::Vector { width: val_w, .. },
            ) if mask_elem.is_bool() && mask_w == val_w => Ok(a_type),
            _ => Err(CompileError::type_error(
                format!(
                    "select mask must be bool vector matching operand width, got {mask_type}. Use comparison operators (.>, .==) to create a mask"
                ),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_min_max(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                format!("{name} expects 2 arguments"),
                span.clone(),
            ));
        }
        let t1 = self.check_expr(&args[0], locals)?;
        let t2 = self.check_expr(&args[1], locals)?;
        match (&t1, &t2) {
            (Type::I32, Type::I32)
            | (Type::I32, Type::IntLiteral)
            | (Type::IntLiteral, Type::I32) => Ok(Type::I32),
            (Type::F32, Type::F32)
            | (Type::F32, Type::FloatLiteral)
            | (Type::FloatLiteral, Type::F32) => Ok(Type::F32),
            (Type::F64, Type::F64)
            | (Type::F64, Type::FloatLiteral)
            | (Type::FloatLiteral, Type::F64) => Ok(Type::F64),
            (
                Type::Vector {
                    elem: e1,
                    width: w1,
                },
                Type::Vector {
                    elem: e2,
                    width: w2,
                },
            ) if e1 == e2 && w1 == w2 && e1.is_float() => Ok(t1.clone()),
            _ => Err(CompileError::type_error(
                format!("{name} expects matching numeric types, got ({t1}, {t2})"),
                span.clone(),
            )),
        }
    }

    pub(super) fn check_movemask(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "movemask expects 1 argument (byte or bool vector)",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;

        match &arg_type {
            Type::Vector { elem, width } => {
                let valid_elem = matches!(elem.as_ref(), Type::U8 | Type::I8 | Type::Bool);
                if !valid_elem {
                    return Err(CompileError::type_error(
                        format!("movemask requires u8/i8/bool vector, got {arg_type}"),
                        args[0].span().clone(),
                    ));
                }
                if *width != 16 && *width != 32 {
                    return Err(CompileError::type_error(
                        format!("movemask requires width 16 or 32, got {width}"),
                        args[0].span().clone(),
                    ));
                }
                Ok(Type::I32)
            }
            _ => Err(CompileError::type_error(
                format!("movemask expects a vector, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }
}
