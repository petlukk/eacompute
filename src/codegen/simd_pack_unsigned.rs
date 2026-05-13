use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// pack_usat_i32x8(a: i32x8, b: i32x8) -> u16x16
    /// Unsigned saturating narrow with lane fixup.
    /// x86-only: vpackusdw + vpermq 0xD8.
    pub(super) fn compile_pack_usat_i32x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "pack_usat_i32x8 is x86-only (AVX2); use pack_usat_i32x4 on ARM",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i32x8_ty = self.context.i32_type().vec_type(8);
        let i16x16_ty = self.context.i16_type().vec_type(16);
        let intrinsic_name = "llvm.x86.avx2.packusdw";
        let fn_type = i16x16_ty.fn_type(&[i32x8_ty.into(), i32x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let packed = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "pack_usat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("packusdw failed"))?;

        let result = self.permq_fixup(packed.into_vector_value())?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// pack_usat_i16x16(a: i16x16, b: i16x16) -> u8x32
    /// Unsigned saturating narrow with lane fixup.
    /// x86-only: vpackuswb + vpermq 0xD8.
    pub(super) fn compile_pack_usat_i16x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "pack_usat_i16x16 is x86-only (AVX2); use pack_usat_i16x8 on ARM",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x16_ty = self.context.i16_type().vec_type(16);
        let i8x32_ty = self.context.i8_type().vec_type(32);
        let intrinsic_name = "llvm.x86.avx2.packuswb";
        let fn_type = i8x32_ty.fn_type(&[i16x16_ty.into(), i16x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let packed = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "pack_usat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("packuswb failed"))?;

        let result = self.permq_fixup(packed.into_vector_value())?;
        Ok(BasicValueEnum::VectorValue(result))
    }

    /// pack_usat_i32x4(a: i32x4, b: i32x4) -> u16x8
    /// Unsigned saturating narrow. Cross-platform.
    /// x86: packusdw (SSE4.1). ARM: sqxtun + concat.
    pub(super) fn compile_pack_usat_i32x4(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i32x4_ty = self.context.i32_type().vec_type(4);
        let i16x8_ty = self.context.i16_type().vec_type(8);

        if self.is_arm {
            let u16x4_ty = self.context.i16_type().vec_type(4);
            let sqxtun_name = "llvm.aarch64.neon.sqxtun.v4i16";
            let sqxtun_ty = u16x4_ty.fn_type(&[i32x4_ty.into()], false);
            let sqxtun = self
                .module
                .get_function(sqxtun_name)
                .unwrap_or_else(|| self.module.add_function(sqxtun_name, sqxtun_ty, None));

            let na = self
                .builder
                .build_call(sqxtun, &[a.into()], "na")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtun failed"))?
                .into_vector_value();

            let nb = self
                .builder
                .build_call(sqxtun, &[b.into()], "nb")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtun failed"))?
                .into_vector_value();

            let cat8: Vec<_> = (0u64..8)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat8_mask = VectorType::const_vector(&cat8);
            let result = self
                .builder
                .build_shuffle_vector(na, nb, cat8_mask, "pack_usat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
            let intrinsic_name = "llvm.x86.sse41.packusdw";
            let fn_type = i16x8_ty.fn_type(&[i32x4_ty.into(), i32x4_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into(), b.into()], "pack_usat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("packusdw.128 failed"))?;

            Ok(result)
        }
    }

    /// pack_usat_i16x8(a: i16x8, b: i16x8) -> u8x16
    /// Unsigned saturating narrow. Cross-platform.
    /// x86: packuswb (SSE2). ARM: sqxtun + concat.
    pub(super) fn compile_pack_usat_i16x8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let i8x16_ty = self.context.i8_type().vec_type(16);

        if self.is_arm {
            let u8x8_ty = self.context.i8_type().vec_type(8);
            let sqxtun_name = "llvm.aarch64.neon.sqxtun.v8i8";
            let sqxtun_ty = u8x8_ty.fn_type(&[i16x8_ty.into()], false);
            let sqxtun = self
                .module
                .get_function(sqxtun_name)
                .unwrap_or_else(|| self.module.add_function(sqxtun_name, sqxtun_ty, None));

            let na = self
                .builder
                .build_call(sqxtun, &[a.into()], "na")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtun failed"))?
                .into_vector_value();

            let nb = self
                .builder
                .build_call(sqxtun, &[b.into()], "nb")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("sqxtun failed"))?
                .into_vector_value();

            let cat16: Vec<_> = (0u64..16)
                .map(|i| self.context.i32_type().const_int(i, false))
                .collect();
            let cat16_mask = VectorType::const_vector(&cat16);
            let result = self
                .builder
                .build_shuffle_vector(na, nb, cat16_mask, "pack_usat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;

            Ok(BasicValueEnum::VectorValue(result))
        } else {
            let intrinsic_name = "llvm.x86.sse2.packuswb.128";
            let fn_type = i8x16_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

            let result = self
                .builder
                .build_call(intrinsic, &[a.into(), b.into()], "pack_usat")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("packuswb.128 failed"))?;

            Ok(result)
        }
    }
}
