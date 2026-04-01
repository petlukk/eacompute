use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// round_f32x8_i32x8(f32x8) -> i32x8
    /// Round-to-nearest-even (banker's rounding) on both x86 and ARM.
    pub(super) fn check_round_f32x8_i32x8(
        &self,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                "round_f32x8_i32x8 expects 1 argument (f32x8 vector)",
                span.clone(),
            ));
        }
        let arg_type = self.check_expr(&args[0], locals)?;
        match &arg_type {
            Type::Vector { elem, width: 8 } if matches!(elem.as_ref(), Type::F32) => {
                Ok(Type::Vector {
                    elem: Box::new(Type::I32),
                    width: 8,
                })
            }
            _ => Err(CompileError::type_error(
                format!("round_f32x8_i32x8 expects f32x8, got {arg_type}"),
                args[0].span().clone(),
            )),
        }
    }
}
