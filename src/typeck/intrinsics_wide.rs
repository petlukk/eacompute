use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.21.0 `widen_u8x16_u16x16(v: u8x16) -> u16x16`: all 16 bytes
    /// zero-extended into one 256-bit vector (x86 `vpmovzxbw` ymm; ARM has
    /// no 256-bit vectors, so codegen gives an error there).
    pub(super) fn check_wide_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        if name != "widen_u8x16_u16x16" {
            return None;
        }
        if args.len() != 1 {
            return Some(Err(CompileError::type_error(format!("{name} expects 1 argument"), span.clone())));
        }
        let want = Type::Vector { elem: Box::new(Type::U8), width: 16 };
        Some(self.check_expr(&args[0], locals).and_then(|t| {
            if t == want {
                Ok(Type::Vector { elem: Box::new(Type::U16), width: 16 })
            } else {
                Err(CompileError::type_error(format!("{name} expects u8x16, got {t}"), args[0].span().clone()))
            }
        }))
    }
}
