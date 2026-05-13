use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// bsrli_i8x16(v, imm) -> i8x16. Byte shift right, zeros shifted in.
    /// Cross-platform via shufflevector.
    pub(super) fn compile_bsrli_i8x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let zero = self.context.i8_type().vec_type(16).const_zero();
        let indices: Vec<_> = (0u64..16)
            .map(|i| {
                let idx = i + imm as u64;
                let sel = if idx < 16 { idx } else { 16 + (idx - 16) };
                self.context.i32_type().const_int(sel, false)
            })
            .collect();
        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(v, zero, mask, "bsrli")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// bsrli_i8x32(v, imm) -> i8x32. Per-lane byte shift right.
    /// x86-only (AVX2). Each 128-bit lane shifts independently.
    pub(super) fn compile_bsrli_i8x32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "bsrli_i8x32 is x86-only (AVX2); use bsrli_i8x16 on ARM",
            ));
        }
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let zero = self.context.i8_type().vec_type(32).const_zero();
        let indices: Vec<_> = (0u64..32)
            .map(|i| {
                let lane_base = (i / 16) * 16;
                let lane_offset = i % 16;
                let src = lane_offset + imm as u64;
                let sel = if src < 16 {
                    lane_base + src
                } else {
                    32 + lane_base + (src - 16)
                };
                self.context.i32_type().const_int(sel, false)
            })
            .collect();
        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(v, zero, mask, "bsrli")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// bslli_i8x16(v, imm) -> i8x16. Byte shift left, zeros shifted in.
    /// Cross-platform via shufflevector.
    pub(super) fn compile_bslli_i8x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let zero = self.context.i8_type().vec_type(16).const_zero();
        let indices: Vec<_> = (0u64..16)
            .map(|i| {
                let sel = if i < imm as u64 {
                    16 + i
                } else {
                    i - imm as u64
                };
                self.context.i32_type().const_int(sel, false)
            })
            .collect();
        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(v, zero, mask, "bslli")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// bslli_i8x32(v, imm) -> i8x32. Per-lane byte shift left.
    /// x86-only (AVX2). Each 128-bit lane shifts independently.
    pub(super) fn compile_bslli_i8x32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        imm: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "bslli_i8x32 is x86-only (AVX2); use bslli_i8x16 on ARM",
            ));
        }
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let zero = self.context.i8_type().vec_type(32).const_zero();
        let indices: Vec<_> = (0u64..32)
            .map(|i| {
                let lane_base = (i / 16) * 16;
                let lane_offset = i % 16;
                let sel = if lane_offset < imm as u64 {
                    32 + lane_base + lane_offset
                } else {
                    lane_base + lane_offset - imm as u64
                };
                self.context.i32_type().const_int(sel, false)
            })
            .collect();
        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(v, zero, mask, "bslli")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }
}
