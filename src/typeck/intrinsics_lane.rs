//! Lane-movement intrinsics: concat, lo/hi extract, and per-sublane 32-bit
//! broadcasts. All primitives in this file are pure shufflevector emissions
//! with fixed compile-time masks — no arithmetic, no I/O, no runtime imm.
//!
//! See docs/superpowers/plans/2026-04-08-avx512-lane-intrinsics.md for the
//! full rationale and the llama.cpp Q4K/Q8K gemm path this enables.

use std::collections::HashMap;

use crate::ast::Expr;
use crate::error::CompileError;
use crate::lexer::Span;

use super::TypeChecker;
use super::types::Type;

impl TypeChecker {
    /// concat_*(lo: VecN, hi: VecN) -> Vec(2N)
    /// Pure linear concatenation: result = [lo_elements, hi_elements].
    pub(super) fn check_concat(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        expected_elem: Type,
        expected_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 2 {
            return Err(CompileError::type_error(
                format!("{name} expects 2 arguments: (lo, hi)"),
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        let b = self.check_expr(&args[1], locals)?;
        let matches_expected = |t: &Type| -> bool {
            matches!(t, Type::Vector { elem, width }
                if **elem == expected_elem && *width == expected_width)
        };
        if !matches_expected(&a) || !matches_expected(&b) {
            return Err(CompileError::type_error(
                format!(
                    "{name} expects two {expected_elem}x{expected_width} arguments, got ({a}, {b})"
                ),
                span.clone(),
            ));
        }
        Ok(Type::Vector {
            elem: Box::new(expected_elem),
            width: expected_width * 2,
        })
    }

    /// lo_extract: Vec(2N) -> VecN, result lanes = input[0..N].
    pub(super) fn check_lo_extract(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        expected_elem: Type,
        input_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument"),
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        if !matches!(&a, Type::Vector { elem, width }
            if **elem == expected_elem && *width == input_width)
        {
            return Err(CompileError::type_error(
                format!("{name} expects {expected_elem}x{input_width}, got {a}"),
                span.clone(),
            ));
        }
        Ok(Type::Vector {
            elem: Box::new(expected_elem),
            width: input_width / 2,
        })
    }

    /// hi_extract: Vec(2N) -> VecN, result lanes = input[N..2N].
    /// Type rules are identical to check_lo_extract; only the emitted
    /// shufflevector mask differs (handled in codegen).
    pub(super) fn check_hi_extract(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        expected_elem: Type,
        input_width: usize,
    ) -> crate::error::Result<Type> {
        self.check_lo_extract(name, args, locals, span, expected_elem, input_width)
    }

    /// Per-sublane 32-bit broadcast: {bcast_even_pairs,bcast_odd_pairs}_i32x{8,16}.
    /// Accepts an i32 vector of width 8 or 16, returns the same type.
    /// The even/odd distinction is handled at codegen time; type-check is identical.
    pub(super) fn check_bcast_pairs(
        &self,
        name: &str,
        args: &[Expr],
        locals: &HashMap<String, (Type, bool)>,
        span: &Span,
        expected_width: usize,
    ) -> crate::error::Result<Type> {
        if args.len() != 1 {
            return Err(CompileError::type_error(
                format!("{name} expects 1 argument: i32x{expected_width}"),
                span.clone(),
            ));
        }
        let a = self.check_expr(&args[0], locals)?;
        if !matches!(&a, Type::Vector { elem, width }
            if matches!(**elem, Type::I32) && *width == expected_width)
        {
            return Err(CompileError::type_error(
                format!("{name} expects i32x{expected_width}, got {a}"),
                span.clone(),
            ));
        }
        Ok(a)
    }
}
