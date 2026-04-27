//! Codegen for lane-movement intrinsics. See intrinsics_lane.rs for the
//! type-checker side and the plan document for the full rationale.
//!
//! All helpers in this file emit LLVM `shufflevector` IR with compile-time
//! constant masks. No x86-specific LLVM intrinsics are used — LLVM 18
//! pattern-matches the shufflevectors to vinserti32x8 / vextracti128 /
//! vpshufd automatically at ISEL time.

use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::{Expr, Literal};
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Extract a u8 immediate from an AST expression. The type-checker has
    /// already validated that this is an integer literal in 0..=255.
    pub(super) fn extract_imm8(expr: &Expr) -> crate::error::Result<u8> {
        match expr {
            Expr::Literal(Literal::Integer(n), _) => Ok(*n as u8),
            _ => Err(CompileError::codegen_error(
                "expected integer literal for immediate argument",
            )),
        }
    }

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

    /// Emit a per-sublane 32-bit shuffle with immediate (vpshufd).
    /// `imm` is an 8-bit value with four 2-bit selectors: bits [1:0] control
    /// lane 0, [3:2] control lane 1, [5:4] control lane 2, [7:6] control lane 3.
    /// The pattern repeats per 128-bit sublane.
    pub(super) fn emit_shuffle_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let n = a.get_type().get_size() as usize;
        assert!(n.is_multiple_of(4));
        let sel: [usize; 4] = [
            (imm & 3) as usize,
            ((imm >> 2) & 3) as usize,
            ((imm >> 4) & 3) as usize,
            ((imm >> 6) & 3) as usize,
        ];
        let mask_vals: Vec<_> = (0..n)
            .map(|i| {
                let sublane_base = (i / 4) * 4;
                let src = sublane_base + sel[i % 4];
                self.context.i32_type().const_int(src as u64, false)
            })
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let undef = a.get_type().get_undef();
        let result = self
            .builder
            .build_shuffle_vector(a, undef, mask, "shuffle_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// Emit a per-element blend from two i32x8 vectors with immediate mask.
    /// Bit N of `imm` selects b[N], else a[N]. Maps to vpblendd.
    pub(super) fn emit_blend_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let n = 8usize;
        // Mask: lane i from a → index i, lane i from b → index i+N
        let mask_vals: Vec<_> = (0..n)
            .map(|i| {
                let src = if (imm >> i) & 1 == 1 { i + n } else { i };
                self.context.i32_type().const_int(src as u64, false)
            })
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let result = self
            .builder
            .build_shuffle_vector(a, b, mask, "blend_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// Emit a vector built from N scalar values via `insertelement` chain.
    /// LLVM 18 folds this to `ins v.s[i], wn` on NEON and `vinsertps` on x86.
    pub(super) fn emit_f32_from_scalars(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        width: u32,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let f32_ty = self.context.f32_type();
        let vec_ty = f32_ty.vec_type(width);
        let mut v: BasicValueEnum<'ctx> = vec_ty.get_undef().into();
        for (i, arg) in args.iter().enumerate() {
            let scalar = self.compile_expr(arg, function)?.into_float_value();
            let lane = self.context.i32_type().const_int(i as u64, false);
            v = self
                .builder
                .build_insert_element(v.into_vector_value(), scalar, lane, "ins")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .into();
        }
        Ok(v)
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
