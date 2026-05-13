use inkwell::types::VectorType;
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

    /// wmul_u64_lo(a: u32x4, b: u32x4) -> u64x2 — widening multiply of the
    /// low half (logical lanes 0,1) of each input. Cross-platform.
    pub(super) fn compile_wmul_u64_lo(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_wmul_u64_half(args, function, true)
    }

    /// wmul_u64_hi(a: u32x4, b: u32x4) -> u64x2 — widening multiply of the
    /// high half (logical lanes 2,3) of each input. Cross-platform.
    pub(super) fn compile_wmul_u64_hi(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_wmul_u64_half(args, function, false)
    }

    fn compile_wmul_u64_half(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        is_lo: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let name = if is_lo { "wmul_u64_lo" } else { "wmul_u64_hi" };
        let i32_ty = self.context.i32_type();

        if self.is_arm {
            // ARM: shuffle u32x4 -> u32x2 (low: [0,1], high: [2,3]) then
            // call llvm.aarch64.neon.umull.v2i64. For the high half LLVM 18
            // pattern-matches the extract+umull sequence to a single
            // `umull2 v.2d, v.4s, v.4s` instruction.
            let mask_vals: Vec<_> = if is_lo {
                vec![i32_ty.const_int(0, false), i32_ty.const_int(1, false)]
            } else {
                vec![i32_ty.const_int(2, false), i32_ty.const_int(3, false)]
            };
            let mask = VectorType::const_vector(&mask_vals);
            let undef = a.get_type().get_undef();
            let a_half = self
                .builder
                .build_shuffle_vector(a, undef, mask, &format!("{name}_a_half"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_half = self
                .builder
                .build_shuffle_vector(b, undef, mask, &format!("{name}_b_half"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let u32x2_ty = self.context.i32_type().vec_type(2);
            let u64x2_ty = self.context.i64_type().vec_type(2);
            let fn_type = u64x2_ty.fn_type(&[u32x2_ty.into(), u32x2_ty.into()], false);
            let intrinsic = self
                .module
                .get_function("llvm.aarch64.neon.umull.v2i64")
                .unwrap_or_else(|| {
                    self.module
                        .add_function("llvm.aarch64.neon.umull.v2i64", fn_type, None)
                });
            let result = self
                .builder
                .build_call(intrinsic, &[a_half.into(), b_half.into()], name)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| {
                    CompileError::codegen_error(format!("{name} did not return a value"))
                })?;
            Ok(result)
        } else {
            // x86: use the canonical IR pattern `mul(zext, zext)`. LLVM 7+
            // removed `llvm.x86.sse2.pmulu.dq` because it's representable
            // this way; the x86 backend pattern-matches the pattern to a
            // single `pmuludq` (or `vpmuludq` with AVX2). Same shape as
            // the ARM path: shuffle u32x4 → u32x2 for the appropriate half.
            let mask_vals: Vec<_> = if is_lo {
                vec![i32_ty.const_int(0, false), i32_ty.const_int(1, false)]
            } else {
                vec![i32_ty.const_int(2, false), i32_ty.const_int(3, false)]
            };
            let mask = VectorType::const_vector(&mask_vals);
            let undef = a.get_type().get_undef();
            let a_half = self
                .builder
                .build_shuffle_vector(a, undef, mask, &format!("{name}_a_half"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_half = self
                .builder
                .build_shuffle_vector(b, undef, mask, &format!("{name}_b_half"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let u64x2_ty = self.context.i64_type().vec_type(2);
            let a_64 = self
                .builder
                .build_int_z_extend(a_half, u64x2_ty, &format!("{name}_a_zext"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let b_64 = self
                .builder
                .build_int_z_extend(b_half, u64x2_ty, &format!("{name}_b_zext"))
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let result = self
                .builder
                .build_int_mul(a_64, b_64, name)
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(BasicValueEnum::VectorValue(result))
        }
    }
}
