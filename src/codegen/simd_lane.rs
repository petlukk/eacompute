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

    /// Emit a low-half extract: result[i] = input[i] for i in 0..N/2.
    pub(super) fn emit_lo_extract(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let n = a.get_type().get_size();
        let half = n / 2;
        let mask_vals: Vec<_> = (0..half)
            .map(|i| self.context.i32_type().const_int(i as u64, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let undef = a.get_type().get_undef();
        let result = self
            .builder
            .build_shuffle_vector(a, undef, mask, "lo_extract")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// Emit a high-half extract: result[i] = input[i + N/2] for i in 0..N/2.
    pub(super) fn emit_hi_extract(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let n = a.get_type().get_size();
        let half = n / 2;
        let mask_vals: Vec<_> = (0..half)
            .map(|i| self.context.i32_type().const_int((i + half) as u64, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let undef = a.get_type().get_undef();
        let result = self
            .builder
            .build_shuffle_vector(a, undef, mask, "hi_extract")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// Emit a per-sublane 32-bit broadcast shufflevector.
    /// `odd = false` produces [l0, l0, l2, l2] per 4-lane sublane (lowers to vpshufd imm=0xA0).
    /// `odd = true`  produces [l1, l1, l3, l3] per 4-lane sublane (lowers to vpshufd imm=0xF5).
    pub(super) fn emit_bcast_pairs(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        odd: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let n = a.get_type().get_size() as usize;
        assert!(
            n.is_multiple_of(4),
            "bcast_pairs requires width divisible by 4"
        );
        let offset = if odd { 1 } else { 0 };
        let mask_vals: Vec<_> = (0..n)
            .map(|i| {
                let sublane = i / 4;
                let within = i % 4;
                // within in [0,1] -> sublane*4 + offset + 0
                // within in [2,3] -> sublane*4 + offset + 2
                let src_lane = sublane * 4 + offset + if within >= 2 { 2 } else { 0 };
                self.context.i32_type().const_int(src_lane as u64, false)
            })
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let undef = a.get_type().get_undef();
        let name = if odd {
            "bcast_odd_pairs"
        } else {
            "bcast_even_pairs"
        };
        let result = self
            .builder
            .build_shuffle_vector(a, undef, mask, name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }
}
