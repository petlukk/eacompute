use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// addp_i32(a: i32x4, b: i32x4) -> i32x4. ARM NEON pairwise add.
    /// Lowers to llvm.aarch64.neon.addp.v4i32.
    pub(super) fn compile_addp_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "addp_i32 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i32x4_ty = self.context.i32_type().vec_type(4);
        let fn_type = i32x4_ty.fn_type(&[i32x4_ty.into(), i32x4_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.addp.v4i32")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.addp.v4i32", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "addp_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("addp_i32 did not return a value"))?;
        Ok(result)
    }

    /// addp_i16(a: i16x8, b: i16x8) -> i16x8. ARM NEON pairwise add.
    /// Lowers to llvm.aarch64.neon.addp.v8i16.
    pub(super) fn compile_addp_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "addp_i16 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let fn_type = i16x8_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.addp.v8i16")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.addp.v8i16", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "addp_i16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("addp_i16 did not return a value"))?;
        Ok(result)
    }

    /// wmul_i16(a: i8x8, b: i8x8) -> i16x8. ARM NEON signed widening multiply.
    /// Lowers to llvm.aarch64.neon.smull.v8i16.
    pub(super) fn compile_wmul_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "wmul_i16 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i8x8_ty = self.context.i8_type().vec_type(8);
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let fn_type = i16x8_ty.fn_type(&[i8x8_ty.into(), i8x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.smull.v8i16")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.smull.v8i16", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "wmul_i16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("wmul_i16 did not return a value"))?;
        Ok(result)
    }

    /// wmul_u16(a: u8x8, b: u8x8) -> u16x8. ARM NEON unsigned widening multiply.
    /// Lowers to llvm.aarch64.neon.umull.v8i16.
    pub(super) fn compile_wmul_u16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "wmul_u16 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let u8x8_ty = self.context.i8_type().vec_type(8);
        let u16x8_ty = self.context.i16_type().vec_type(8);
        let fn_type = u16x8_ty.fn_type(&[u8x8_ty.into(), u8x8_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.umull.v8i16")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.umull.v8i16", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "wmul_u16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("wmul_u16 did not return a value"))?;
        Ok(result)
    }

    /// wmul_i32(a: i16x4, b: i16x4) -> i32x4. ARM NEON signed widening multiply.
    /// Lowers to llvm.aarch64.neon.smull.v4i32.
    pub(super) fn compile_wmul_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "wmul_i32 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let i16x4_ty = self.context.i16_type().vec_type(4);
        let i32x4_ty = self.context.i32_type().vec_type(4);
        let fn_type = i32x4_ty.fn_type(&[i16x4_ty.into(), i16x4_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.smull.v4i32")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.smull.v4i32", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "wmul_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("wmul_i32 did not return a value"))?;
        Ok(result)
    }

    /// wmul_u32(a: u16x4, b: u16x4) -> u32x4. ARM NEON unsigned widening multiply.
    /// Lowers to llvm.aarch64.neon.umull.v4i32.
    pub(super) fn compile_wmul_u32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "wmul_u32 is ARM-only (NEON); no x86 equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let u16x4_ty = self.context.i16_type().vec_type(4);
        let u32x4_ty = self.context.i32_type().vec_type(4);
        let fn_type = u32x4_ty.fn_type(&[u16x4_ty.into(), u16x4_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.umull.v4i32")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.umull.v4i32", fn_type, None)
            });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "wmul_u32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("wmul_u32 did not return a value"))?;
        Ok(result)
    }
}
