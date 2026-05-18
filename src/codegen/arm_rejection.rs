//! ARM-rejection pre-scan for intrinsics with no NEON equivalent.
//!
//! Some intrinsics (currently: `permute_runtime`) lower to x86-specific
//! instructions and have no clean NEON path. Their codegen-time `is_arm`
//! guard is unreachable for the common case because
//! `validate_type_for_target` rejects the required 256-bit vector types
//! on ARM before codegen reaches the intrinsic, producing a generic
//! "f32x8 requires AVX2" error instead of the helpful idiom-doc message.
//!
//! `check_body` walks a function body before `validate_type_for_target`
//! runs and returns the intrinsic-specific rejection eagerly, so users
//! see the canonical "see docs/idioms/neon-runtime-permute.md" guidance.
//!
//! Called from `compile_function` in `statements.rs` gated by `is_arm`.

use crate::ast::{Expr, Stmt};
use crate::error::{CompileError, Result};

/// Walk a function body and return an error if it contains an ARM-rejected
/// intrinsic call. Currently rejects: `permute_runtime`.
pub(super) fn check_body(body: &[Stmt]) -> Result<()> {
    for stmt in body {
        check_stmt(stmt)?;
    }
    Ok(())
}

fn check_stmt(stmt: &Stmt) -> Result<()> {
    match stmt {
        Stmt::Let { value, .. }
        | Stmt::ExprStmt(value, _)
        | Stmt::Assign { value, .. }
        | Stmt::FieldAssign { value, .. } => check_expr(value),
        Stmt::IndexAssign { index, value, .. } => {
            check_expr(index)?;
            check_expr(value)
        }
        Stmt::Return(Some(expr), _)
        | Stmt::StaticAssert {
            condition: expr, ..
        } => check_expr(expr),
        // Stmt::Kernel is desugared to Stmt::Function by desugar_kernels
        // (lib.rs:144) before codegen runs; this arm is unreachable at runtime.
        Stmt::Return(None, _)
        | Stmt::Struct { .. }
        | Stmt::Const { .. }
        | Stmt::Function { .. }
        | Stmt::Kernel { .. } => Ok(()),
        Stmt::If {
            condition,
            then_body,
            else_body,
            ..
        } => {
            check_expr(condition)?;
            check_body(then_body)?;
            if let Some(eb) = else_body {
                check_body(eb)?;
            }
            Ok(())
        }
        Stmt::While {
            condition, body, ..
        } => {
            check_expr(condition)?;
            check_body(body)
        }
        Stmt::For { body, .. } | Stmt::ForEach { body, .. } => check_body(body),
        Stmt::Unroll { body, .. } => check_stmt(body),
    }
}

/// Recurse into every sub-expression. Exhaustive over `Expr` variants so
/// future additions are compiler-checked.
fn check_expr(expr: &Expr) -> Result<()> {
    match expr {
        Expr::Literal(_, _) | Expr::Variable(_, _) => Ok(()),
        Expr::Call { name, args, .. } => {
            if name == "permute_runtime" {
                return Err(CompileError::codegen_error(
                    "permute_runtime has no NEON equivalent on ARM. For small \
                     runtime LUTs (<= 8 entries) use a compare-and-select chain. \
                     For byte-domain LUTs use shuffle_bytes. See \
                     docs/idioms/neon-runtime-permute.md for canonical patterns.",
                ));
            }
            for arg in args {
                check_expr(arg)?;
            }
            Ok(())
        }
        Expr::Binary(lhs, _, rhs, _) => {
            check_expr(lhs)?;
            check_expr(rhs)
        }
        Expr::Not(inner, _) | Expr::Negate(inner, _) => check_expr(inner),
        Expr::Index { object, index, .. } => {
            check_expr(object)?;
            check_expr(index)
        }
        Expr::Vector { elements, .. } | Expr::ArrayLiteral(elements, _) => {
            for e in elements {
                check_expr(e)?;
            }
            Ok(())
        }
        Expr::FieldAccess { object, .. } => check_expr(object),
        Expr::StructLiteral { fields, .. } => {
            for (_, e) in fields {
                check_expr(e)?;
            }
            Ok(())
        }
    }
}
