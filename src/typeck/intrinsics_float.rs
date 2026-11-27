use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.21.0 `floor_f32x4` / `floor_f32x8`: round each lane down (x86
    /// `vroundps` with the floor mode, ARM `frintm`).
    pub(super) fn check_float_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        let width = match name {
            "floor_f32x4" => 4,
            "floor_f32x8" => 8,
            _ => return None,
        };
        let want = Type::Vector { elem: Box::new(Type::F32), width };
        if args.len() != 1 {
            return Some(Err(CompileError::type_error(format!("{name} expects 1 argument"), span.clone())));
        }
        Some(self.check_expr(&args[0], locals).and_then(|t| {
            if t == want {
                Ok(want.clone())
            } else {
                Err(CompileError::type_error(format!("{name} expects {want}, got {t}"), args[0].span().clone()))
            }
        }))
    }
}
