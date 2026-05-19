//! Type check for the `prefetch` / `prefetch_write` / `prefetch_nta` family.
//!
//! All three intrinsics share the same `(ptr, integer-offset) -> void`
//! shape. The dispatcher in `intrinsics.rs` routes all three names to this
//! single `check_prefetch` because the rw / locality / cache-type fields
//! are constant-baked into the codegen, not part of the typed signature.

use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    pub(super) fn check_prefetch(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                "prefetch expects 2 arguments: (ptr, offset)",
                span.clone(),
            ));
        }
        let ptr_type = self.check_expr(&args[0], locals)?;
        if !matches!(ptr_type, Type::Pointer { .. }) {
            return Err(CompileError::type_error(
                format!("prefetch first argument must be a pointer, got {ptr_type}"),
                args[0].span().clone(),
            ));
        }
        let offset_type = self.check_expr(&args[1], locals)?;
        if !offset_type.is_integer() {
            return Err(CompileError::type_error(
                format!("prefetch offset must be integer, got {offset_type}"),
                args[1].span().clone(),
            ));
        }
        Ok(Type::Void)
    }
}
