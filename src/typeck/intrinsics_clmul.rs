use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.17.0 carry-less multiply: `clmul_lo_u64x2(a, b)` multiplies lane 0
    /// of `a` by lane 0 of `b`, `clmul_hi_u64x2` lane 1 by lane 1, each as
    /// polynomials over GF(2), into a 128-bit `u64x2` (low half in lane 0).
    /// The target feature (x86 `--pclmul`, ARM `--aes`) is checked at
    /// codegen, like the other native-or-error intrinsics.
    pub(super) fn check_clmul_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        if !matches!(name, "clmul_lo_u64x2" | "clmul_hi_u64x2") {
            return None;
        }
        Some(self.check_clmul(name, args, locals, span))
    }

    fn check_clmul(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                format!("{name} expects 2 arguments (u64x2, u64x2)"),
                span.clone(),
            ));
        }
        let u64x2 = Type::Vector {
            elem: Box::new(Type::U64),
            width: 2,
        };
        for arg in args {
            let t = self.check_expr(arg, locals)?;
            if t != u64x2 {
                return Err(CompileError::type_error(
                    format!("{name} expects u64x2 arguments, got {t}"),
                    arg.span().clone(),
                ));
            }
        }
        Ok(u64x2)
    }
}
