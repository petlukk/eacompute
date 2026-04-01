use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// round_f32x8_i32x8(a: f32x8) -> i32x8.
    /// Round-to-nearest (banker's rounding) converting float to int.
    /// x86: vcvtps2dq (llvm.x86.avx.cvt.ps2dq.256).
    pub(super) fn compile_round_f32x8_i32x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let f32x8_ty = self.context.f32_type().vec_type(8);
        let i32x8_ty = self.context.i32_type().vec_type(8);

        if self.is_arm {
            // ARM: split f32x8 -> two f32x4 halves, call fcvtns on each, concat
            let f32x4_ty = self.context.f32_type().vec_type(4);
            let i32x4_ty = self.context.i32_type().vec_type(4);

            // shufflevector masks for split and concat
            let lo_indices: Vec<_> = (0u64..4)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let hi_indices: Vec<_> = (4u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let concat_indices: Vec<_> = (0u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();

            let lo_mask = VectorType::const_vector(&lo_indices);
            let hi_mask = VectorType::const_vector(&hi_indices);
            let concat_mask = VectorType::const_vector(&concat_indices);

            let undef_f32x8 = f32x8_ty.get_undef();

            let lo_f32 = self
                .builder
                .build_shuffle_vector(a, undef_f32x8, lo_mask, "lo_f32")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let hi_f32 = self
                .builder
                .build_shuffle_vector(a, undef_f32x8, hi_mask, "hi_f32")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let fcvtns_name = "llvm.aarch64.neon.fcvtns.v4i32.v4f32";
            let fcvtns_ty = i32x4_ty.fn_type(&[f32x4_ty.into()], false);
            let fcvtns = self
                .module
                .get_function(fcvtns_name)
                .unwrap_or_else(|| self.module.add_function(fcvtns_name, fcvtns_ty, None));

            let lo_i32 = self
                .builder
                .build_call(fcvtns, &[lo_f32.into()], "lo_i32")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("fcvtns lo did not return a value"))?
                .into_vector_value();

            let hi_i32 = self
                .builder
                .build_call(fcvtns, &[hi_f32.into()], "hi_i32")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("fcvtns hi did not return a value"))?
                .into_vector_value();

            let result = self
                .builder
                .build_shuffle_vector(lo_i32, hi_i32, concat_mask, "i32x8")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
            // x86: single vcvtps2dq instruction
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
    }

    /// pack_sat_i32x8(a: i32x8, b: i32x8) -> i16x16
    /// Saturating narrow. Per-128-bit-lane packing on x86 (no fixup shuffle).
    /// x86: vpackssdw (llvm.x86.avx2.packssdw).
    /// ARM: split -> sqxtn -> concat.
    pub(super) fn compile_pack_sat_i32x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        if self.is_arm {
            let i32x4_ty = self.context.i32_type().vec_type(4);
            let i16x4_ty = self.context.i16_type().vec_type(4);

            let lo4: Vec<_> = (0u64..4)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let hi4: Vec<_> = (4u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let lo_mask = VectorType::const_vector(&lo4);
            let hi_mask = VectorType::const_vector(&hi4);

            let i32x8_undef = self.context.i32_type().vec_type(8).get_undef();
            let a_lo = self
                .builder
                .build_shuffle_vector(a, i32x8_undef, lo_mask, "a_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let a_hi = self
                .builder
                .build_shuffle_vector(a, i32x8_undef, hi_mask, "a_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_lo = self
                .builder
                .build_shuffle_vector(b, i32x8_undef, lo_mask, "b_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_hi = self
                .builder
                .build_shuffle_vector(b, i32x8_undef, hi_mask, "b_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let sqxtn_name = "llvm.aarch64.neon.sqxtn.v4i16";
            let sqxtn_ty = i16x4_ty.fn_type(&[i32x4_ty.into()], false);
            let sqxtn = self
                .module
                .get_function(sqxtn_name)
                .unwrap_or_else(|| self.module.add_function(sqxtn_name, sqxtn_ty, None));

            let na_lo = self
                .builder
                .build_call(sqxtn, &[a_lo.into()], "na_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let na_hi = self
                .builder
                .build_call(sqxtn, &[a_hi.into()], "na_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let nb_lo = self
                .builder
                .build_call(sqxtn, &[b_lo.into()], "nb_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let nb_hi = self
                .builder
                .build_call(sqxtn, &[b_hi.into()], "nb_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let cat8: Vec<_> = (0u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat8_mask = VectorType::const_vector(&cat8);
            let a_i16x8 = self
                .builder
                .build_shuffle_vector(na_lo, na_hi, cat8_mask, "a_i16x8")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_i16x8 = self
                .builder
                .build_shuffle_vector(nb_lo, nb_hi, cat8_mask, "b_i16x8")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let cat16: Vec<_> = (0u64..16)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat16_mask = VectorType::const_vector(&cat16);
            let result = self
                .builder
                .build_shuffle_vector(a_i16x8, b_i16x8, cat16_mask, "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
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
    }

    /// pack_sat_i16x16(a: i16x16, b: i16x16) -> i8x32
    /// Saturating narrow. Per-128-bit-lane packing on x86 (no fixup shuffle).
    /// x86: vpacksswb (llvm.x86.avx2.packsswb).
    /// ARM: split -> sqxtn -> concat.
    pub(super) fn compile_pack_sat_i16x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        if self.is_arm {
            let i16x8_ty = self.context.i16_type().vec_type(8);
            let i8x8_ty = self.context.i8_type().vec_type(8);

            let lo8: Vec<_> = (0u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let hi8: Vec<_> = (8u64..16)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let lo_mask = VectorType::const_vector(&lo8);
            let hi_mask = VectorType::const_vector(&hi8);

            let i16x16_undef = self.context.i16_type().vec_type(16).get_undef();
            let a_lo = self
                .builder
                .build_shuffle_vector(a, i16x16_undef, lo_mask, "a_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let a_hi = self
                .builder
                .build_shuffle_vector(a, i16x16_undef, hi_mask, "a_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_lo = self
                .builder
                .build_shuffle_vector(b, i16x16_undef, lo_mask, "b_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_hi = self
                .builder
                .build_shuffle_vector(b, i16x16_undef, hi_mask, "b_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let sqxtn_name = "llvm.aarch64.neon.sqxtn.v8i8";
            let sqxtn_ty = i8x8_ty.fn_type(&[i16x8_ty.into()], false);
            let sqxtn = self
                .module
                .get_function(sqxtn_name)
                .unwrap_or_else(|| self.module.add_function(sqxtn_name, sqxtn_ty, None));

            let na_lo = self
                .builder
                .build_call(sqxtn, &[a_lo.into()], "na_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let na_hi = self
                .builder
                .build_call(sqxtn, &[a_hi.into()], "na_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let nb_lo = self
                .builder
                .build_call(sqxtn, &[b_lo.into()], "nb_lo")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();
            let nb_hi = self
                .builder
                .build_call(sqxtn, &[b_hi.into()], "nb_hi")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtn failed"))?
                .into_vector_value();

            let cat16: Vec<_> = (0u64..16)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat16_mask = VectorType::const_vector(&cat16);
            let a_i8x16 = self
                .builder
                .build_shuffle_vector(na_lo, na_hi, cat16_mask, "a_i8x16")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_i8x16 = self
                .builder
                .build_shuffle_vector(nb_lo, nb_hi, cat16_mask, "b_i8x16")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            let cat32: Vec<_> = (0u64..32)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat32_mask = VectorType::const_vector(&cat32);
            let result = self
                .builder
                .build_shuffle_vector(a_i8x16, b_i8x16, cat32_mask, "pack_sat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
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
    }
}
