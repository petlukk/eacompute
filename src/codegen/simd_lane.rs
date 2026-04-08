//! Codegen for lane-movement intrinsics. See intrinsics_lane.rs for the
//! type-checker side and the plan document for the full rationale.
//!
//! All helpers in this file emit LLVM `shufflevector` IR with compile-time
//! constant masks. No x86-specific LLVM intrinsics are used — LLVM 18
//! pattern-matches the shufflevectors to vinserti32x8 / vextracti128 /
//! vpshufd automatically at ISEL time.

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    // Codegen helpers for concat/extract/bcast families are added in Tasks 2-4.
}
