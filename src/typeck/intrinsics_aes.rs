use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.18.0 AES rounds, one native instruction each:
    /// `aesenc_u8x16(x, k)` (x86 `aesenc`) is
    /// MixColumns(ShiftRows(SubBytes(x))) ^ k; `aese_u8x16(x, k)` (ARM
    /// `aese`) is SubBytes(ShiftRows(x ^ k)) and `aesmc_u8x16(x)` (ARM
    /// `aesmc`) is MixColumns(x). The architecture and the `--aes` flag are
    /// checked at codegen, like the other native-or-error intrinsics.
    pub(super) fn check_aes_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        let arity = match name {
            "aesenc_u8x16" | "aese_u8x16" => 2,
            "aesmc_u8x16" => 1,
            _ => return None,
        };
        Some(self.check_aes(name, arity, args, locals, span))
    }

    fn check_aes(
        &self,
        name: &str,
        arity: usize,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> crate::error::Result<Type> {
        if args.len() != arity {
            let want = if arity == 2 {
                "2 arguments (u8x16, u8x16)"
            } else {
                "1 argument (u8x16)"
            };
            return Err(CompileError::type_error(
                format!("{name} expects {want}"),
                span.clone(),
            ));
        }
        let u8x16 = Type::Vector {
            elem: Box::new(Type::U8),
            width: 16,
        };
        for arg in args {
            let t = self.check_expr(arg, locals)?;
            if t != u8x16 {
                return Err(CompileError::type_error(
                    format!("{name} expects u8x16 arguments, got {t}"),
                    arg.span().clone(),
                ));
            }
        }
        Ok(u8x16)
    }
}
