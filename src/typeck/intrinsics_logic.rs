use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.21.0 typed andnot: `andnot_<t>(a, b)` is `a & !b`, both operands
    /// and the result of the type in the name (x86 `vpandn`, ARM `bic`).
    pub(super) fn check_logic_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        let (elem, width) = match name {
            "andnot_u8x16" => (Type::U8, 16),
            "andnot_u8x32" => (Type::U8, 32),
            "andnot_u64x2" => (Type::U64, 2),
            "andnot_u64x4" => (Type::U64, 4),
            _ => return None,
        };
        Some(self.check_andnot(name, elem, width, args, locals, span))
    }

    fn check_andnot(
        &self,
        name: &str,
        elem: Type,
        width: usize,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(format!("{name} expects 2 arguments"), span.clone()));
        }
        let want = Type::Vector { elem: Box::new(elem), width };
        for a in args {
            let t = self.check_expr(a, locals)?;
            if t != want {
                return Err(CompileError::type_error(
                    format!("{name} expects {want} operands, got {t}"),
                    a.span().clone(),
                ));
            }
        }
        Ok(want)
    }
}
