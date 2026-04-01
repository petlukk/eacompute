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
}
