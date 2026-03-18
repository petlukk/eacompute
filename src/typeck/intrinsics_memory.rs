use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::{self, Type};

impl TypeChecker {
    pub(super) fn check_load(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "load expects 2 arguments (ptr, index)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "load index must be integer",
                args[1].span().clone(),
            ));
        }

        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width,
            _ => 4,
        };

        match ptr_type {
            Type::Pointer { inner, .. } => {
                if inner.is_numeric() {
                    Ok(Type::Vector { elem: inner, width })
                } else {
                    Err(CompileError::type_error(
                        "load expects pointer to numeric type",
                        args[0].span().clone(),
                    ))
                }
            }
            _ => Err(CompileError::type_error(
                format!("load expects a pointer, got {ptr_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_store(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "store expects 3 arguments (ptr, index, vector)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "store index must be integer",
                args[1].span().clone(),
            ));
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!(
                            "store type mismatch: pointer element type is {inner}, but vector element type is {elem}"
                        ),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "store requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "store expects (mut ptr, index, vector)",
                span.clone(),
            )),
        }
    }

    pub(super) fn check_gather(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "gather expects 2 arguments (ptr, indices)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let inner = match &ptr_type {
            Type::Pointer { inner, .. } if inner.is_numeric() => inner.clone(),
            Type::Pointer { .. } => {
                return Err(CompileError::type_error(
                    "gather expects pointer to numeric type",
                    args[0].span().clone(),
                ));
            }
            _ => {
                return Err(CompileError::type_error(
                    format!("gather expects a pointer, got {ptr_type}"),
                    args[0].span().clone(),
                ));
            }
        };
        let width = match &idx_type {
            Type::Vector { elem, width } if matches!(elem.as_ref(), Type::I32) => *width,
            _ => {
                return Err(CompileError::type_error(
                    format!("gather indices must be i32 vector, got {idx_type}"),
                    args[1].span().clone(),
                ));
            }
        };
        Ok(Type::Vector { elem: inner, width })
    }

    pub(super) fn check_scatter(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "scatter expects 3 arguments (ptr, indices, values)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;
        match &idx_type {
            Type::Vector { elem, .. } if matches!(elem.as_ref(), Type::I32) => {}
            _ => {
                return Err(CompileError::type_error(
                    format!("scatter indices must be i32 vector, got {idx_type}"),
                    args[1].span().clone(),
                ));
            }
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!(
                            "scatter type mismatch: pointer element type is {inner}, but vector element type is {elem}"
                        ),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "scatter requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "scatter expects (mut ptr, indices, values)",
                span.clone(),
            )),
        }
    }

    pub(super) fn check_load_masked(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        type_hint: Option<&Type>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 3 {
            return Err(CompileError::type_error(
                "load_masked expects 3 arguments (ptr, offset, count)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let count_type = self.check_expr(&args[2], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "load_masked offset must be integer",
                args[1].span().clone(),
            ));
        }
        if !count_type.is_integer() {
            return Err(CompileError::type_error(
                "load_masked count must be integer",
                args[2].span().clone(),
            ));
        }

        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width,
            _ => 4,
        };

        match ptr_type {
            Type::Pointer { inner, .. } => {
                if inner.is_numeric() {
                    Ok(Type::Vector { elem: inner, width })
                } else {
                    Err(CompileError::type_error(
                        "load_masked expects pointer to numeric type",
                        args[0].span().clone(),
                    ))
                }
            }
            _ => Err(CompileError::type_error(
                format!("load_masked expects a pointer, got {ptr_type}"),
                args[0].span().clone(),
            )),
        }
    }

    pub(super) fn check_store_masked(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 4 {
            return Err(CompileError::type_error(
                "store_masked expects 4 arguments (ptr, offset, vector, count)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        let idx_type = self.check_expr(&args[1], locals)?;
        let val_type = self.check_expr(&args[2], locals)?;
        let count_type = self.check_expr(&args[3], locals)?;

        if !idx_type.is_integer() {
            return Err(CompileError::type_error(
                "store_masked offset must be integer",
                args[1].span().clone(),
            ));
        }
        if !count_type.is_integer() {
            return Err(CompileError::type_error(
                "store_masked count must be integer",
                args[3].span().clone(),
            ));
        }
        match (ptr_type, val_type) {
            (
                Type::Pointer {
                    mutable: true,
                    inner,
                    ..
                },
                Type::Vector { elem, .. },
            ) => {
                if !types::types_compatible(&elem, &inner) {
                    return Err(CompileError::type_error(
                        format!(
                            "store_masked type mismatch: pointer element type is {inner}, but vector element type is {elem}"
                        ),
                        span.clone(),
                    ));
                }
                Ok(Type::Void)
            }
            (Type::Pointer { mutable: false, .. }, _) => Err(CompileError::type_error(
                "store_masked requires mutable pointer. Declare as *mut to allow writes",
                args[0].span().clone(),
            )),
            (_, _) => Err(CompileError::type_error(
                "store_masked expects (mut ptr, offset, vector, count)",
                span.clone(),
            )),
        }
    }
}
