//! Lane-movement intrinsics: concat, lo/hi extract, and per-sublane 32-bit
//! broadcasts. All primitives in this file are pure shufflevector emissions
//! with fixed compile-time masks — no arithmetic, no I/O, no runtime imm.
//!
//! See docs/superpowers/plans/2026-04-08-avx512-lane-intrinsics.md for the
//! full rationale and the llama.cpp Q4K/Q8K gemm path this enables.

use super::TypeChecker;

impl TypeChecker {
    // Type checkers for concat/extract/bcast families are added in Tasks 2-4.
}
