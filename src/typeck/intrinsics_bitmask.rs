use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// v1.16.0 typed bitmask / bit-scan intrinsics. Platform restrictions
    /// (x86-only movemask_*, ARM-only nibble_mask_u8x16) are enforced at
    /// codegen time, like the other native-or-error intrinsics.
    pub(super) fn check_bitmask_intrinsic(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
    ) -> Option<crate::error::Result<Type>> {
        let (width, raw_bytes_ok, ret) = match name {
            "movemask_u8x16" => (16, true, Type::U32),
            "movemask_u8x32" => (32, true, Type::U32),
            "movemask_u64x4" => (4, false, Type::U32),
            "nibble_mask_u8x16" => (16, false, Type::U64),
            "ctz_u32" => return Some(self.check_ctz(name, Type::U32, args, locals, span)),
            "ctz_u64" => return Some(self.check_ctz(name, Type::U64, args, locals, span)),
            _ => return None,
        };
        Some(self.check_lane_mask(name, width, raw_bytes_ok, ret, args, locals, span))
    }

    /// Argument: a comparison result (`bool` vector) of exactly `width`
    /// lanes, or — for the pmovmskb forms — raw u8/i8 bytes (MSB per lane).
    #[allow(clippy::too_many_arguments)]
    fn check_lane_mask(
        &self,
        name: &str,
        width: usize,
        raw_bytes_ok: bool,
        ret: Type,
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
        let arg = self.check_expr(&args[0], locals)?;
        let ok = match &arg {
            Type::Vector { elem, width: w } if *w == width => match elem.as_ref() {
                Type::Bool => true,
                Type::U8 | Type::I8 => raw_bytes_ok,
                _ => false,
            },
            _ => false,
        };
        if ok {
            return Ok(ret);
        }
        let accepted = if raw_bytes_ok {
            format!("a {width}-lane comparison result or u8x{width}/i8x{width}")
        } else {
            format!("a {width}-lane comparison result")
        };
        Err(CompileError::type_error(
            format!("{name} expects {accepted}, got {arg}"),
            args[0].span().clone(),
        ))
    }

    fn check_ctz(
        &self,
        name: &str,
        expected: Type,
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
        let arg = self.check_expr(&args[0], locals)?;
        if arg == expected || arg == Type::IntLiteral {
            Ok(Type::I32)
        } else {
            Err(CompileError::type_error(
                format!("{name} expects {expected}, got {arg}"),
                args[0].span().clone(),
            ))
        }
    }
}
