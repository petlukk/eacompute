use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// round_f32x8_i32x8(a: f32x8) -> i32x8.
    /// Round-to-nearest (banker's rounding) converting float to int.
    /// x86-only: vcvtps2dq (llvm.x86.avx.cvt.ps2dq.256).
    pub(super) fn compile_round_f32x8_i32x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "round_f32x8_i32x8 is x86-only (AVX2); use round_f32x4_i32x4 on ARM",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let f32x8_ty = self.context.f32_type().vec_type(8);
        let i32x8_ty = self.context.i32_type().vec_type(8);
        let intrinsic_name = "llvm.x86.avx.cvt.ps2dq.256";
        let fn_type = i32x8_ty.fn_type(&[f32x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into()], "round")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("cvtps2dq did not return a value"))?;

        Ok(result)
    }

    /// pack_sat_i32x8(a: i32x8, b: i32x8) -> i16x16
    /// Saturating narrow. Per-128-bit-lane packing on x86 (no fixup shuffle).
    /// x86-only: vpackssdw (llvm.x86.avx2.packssdw).
    pub(super) fn compile_pack_sat_i32x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "pack_sat_i32x8 is x86-only (AVX2); use pack_sat_i32x4 on ARM",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i32x8_ty = self.context.i32_type().vec_type(8);
        let i16x16_ty = self.context.i16_type().vec_type(16);
        let intrinsic_name = "llvm.x86.avx2.packssdw";
        let fn_type = i16x16_ty.fn_type(&[i32x8_ty.into(), i32x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "pack_sat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("packssdw failed"))?;

        Ok(result)
    }

    /// pack_sat_i16x16(a: i16x16, b: i16x16) -> i8x32
    /// Saturating narrow. Per-128-bit-lane packing on x86 (no fixup shuffle).
    /// x86-only: vpacksswb (llvm.x86.avx2.packsswb).
    pub(super) fn compile_pack_sat_i16x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "pack_sat_i16x16 is x86-only (AVX2); use pack_sat_i16x8 on ARM",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x16_ty = self.context.i16_type().vec_type(16);
        let i8x32_ty = self.context.i8_type().vec_type(32);
        let intrinsic_name = "llvm.x86.avx2.packsswb";
        let fn_type = i8x32_ty.fn_type(&[i16x16_ty.into(), i16x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "pack_sat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("packsswb failed"))?;

        Ok(result)
    }

    /// round_f32x4_i32x4(a: f32x4) -> i32x4.
    /// Round-to-nearest-even. Cross-platform.
    /// x86: cvtps2dq (SSE2). ARM: fcvtns (NEON).
    pub(super) fn compile_round_f32x4_i32x4(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let f32x4_ty = self.context.f32_type().vec_type(4);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        if self.is_arm {
            let intrinsic_name = "llvm.aarch64.neon.fcvtns.v4i32.v4f32";
            let fn_type = i32x4_ty.fn_type(&[f32x4_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into()], "round")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("fcvtns did not return a value"))?;

            Ok(result)
        } else {
            let intrinsic_name = "llvm.x86.sse2.cvtps2dq";
            let fn_type = i32x4_ty.fn_type(&[f32x4_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into()], "round")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("cvtps2dq did not return a value"))?;

            Ok(result)
        }
    }

    /// pack_sat_i32x4(a: i32x4, b: i32x4) -> i16x8.
    /// Saturating narrow. Cross-platform.
    /// x86: packssdw (SSE2). ARM: sqxtn on each input, concat.
    pub(super) fn compile_pack_sat_i32x4(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i32x4_ty = self.context.i32_type().vec_type(4);
        let i16x8_ty = self.context.i16_type().vec_type(8);

        if self.is_arm {
            let i16x4_ty = self.context.i16_type().vec_type(4);
            let sqxtn_name = "llvm.aarch64.neon.sqxtn.v4i16";
            let sqxtn_ty = i16x4_ty.fn_type(&[i32x4_ty.into()], false);
            let sqxtn = self
                .module
                .get_function(sqxtn_name)
                .unwrap_or_else(|| self.module.add_function(sqxtn_name, sqxtn_ty, None));

            let na = self
                .builder
                .build_call(sqxtn, &[a.into()], "na")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let nb = self
                .builder
                .build_call(sqxtn, &[b.into()], "nb")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let cat8: Vec<_> = (0u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat8_mask = VectorType::const_vector(&cat8);
            let result = self
                .builder
                .build_shuffle_vector(na, nb, cat8_mask, "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
            let intrinsic_name = "llvm.x86.sse2.packssdw.128";
            let fn_type = i16x8_ty.fn_type(&[i32x4_ty.into(), i32x4_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into(), b.into()], "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("packssdw.128 failed"))?;

            Ok(result)
        }
    }

    /// pack_sat_i16x8(a: i16x8, b: i16x8) -> i8x16.
    /// Saturating narrow. Cross-platform.
    /// x86: packsswb (SSE2). ARM: sqxtn on each input, concat.
    pub(super) fn compile_pack_sat_i16x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let i8x16_ty = self.context.i8_type().vec_type(16);

        if self.is_arm {
            let i8x8_ty = self.context.i8_type().vec_type(8);
            let sqxtn_name = "llvm.aarch64.neon.sqxtn.v8i8";
            let sqxtn_ty = i8x8_ty.fn_type(&[i16x8_ty.into()], false);
            let sqxtn = self
                .module
                .get_function(sqxtn_name)
                .unwrap_or_else(|| self.module.add_function(sqxtn_name, sqxtn_ty, None));

            let na = self
                .builder
                .build_call(sqxtn, &[a.into()], "na")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let nb = self
                .builder
                .build_call(sqxtn, &[b.into()], "nb")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let cat16: Vec<_> = (0u64..16)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat16_mask = VectorType::const_vector(&cat16);
            let result = self
                .builder
                .build_shuffle_vector(na, nb, cat16_mask, "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
            let intrinsic_name = "llvm.x86.sse2.packsswb.128";
            let fn_type = i8x16_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into(), b.into()], "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("packsswb.128 failed"))?;

            Ok(result)
        }
    }
}
