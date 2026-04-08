//! Codegen for lane-movement intrinsics. See intrinsics_lane.rs for the
//! type-checker side and the plan document for the full rationale.
//!
//! All helpers in this file emit LLVM `shufflevector` IR with compile-time
//! constant masks. No x86-specific LLVM intrinsics are used — LLVM 18
//! pattern-matches the shufflevectors to vinserti32x8 / vextracti128 /
//! vpshufd automatically at ISEL time.

use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Emit a concat shufflevector: result = [a_lanes..., b_lanes...].
    /// Works for any vector type; the LLVM shufflevector mask is a linear
    /// sequence 0..2N which LLVM 18 lowers to vinserti128 or vinserti32x8.
    pub(super) fn emit_concat(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let n = a.get_type().get_size();
        // Mask: [0, 1, ..., 2N-1] — pick a[0..N] then b[0..N]
        let mask_vals: Vec<_> = (0..(n * 2))
            .map(|i| self.context.i32_type().const_int(i as u64, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let result = self
            .builder
            .build_shuffle_vector(a, b, mask, "concat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }
}
