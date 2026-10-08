use inkwell::values::{BasicValueEnum, FunctionValue, IntValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

/// v1.16.0 typed bitmask / bit-scan intrinsics. Each lowers to the native
/// idiom of one architecture, or is a compile error pointing at the other
/// architecture's idiom — never an emulation with a hidden cost.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_bitmask_intrinsic(name: &str) -> bool {
        matches!(
            name,
            "movemask_u8x16"
                | "movemask_u8x32"
                | "movemask_u64x4"
                | "nibble_mask_u8x16"
                | "ctz_u32"
                | "ctz_u64"
        )
    }

    pub(super) fn compile_bitmask_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        match name {
            // Same pmovmskb lowering as the deprecated polymorphic movemask.
            "movemask_u8x16" | "movemask_u8x32" => self.compile_movemask(args, function),
            "movemask_u64x4" => self.compile_movemask_u64x4(args, function),
            "nibble_mask_u8x16" => self.compile_nibble_mask_u8x16(args, function),
            "ctz_u32" => self.compile_ctz(args, Type::U32, function),
            _ => self.compile_ctz(args, Type::U64, function),
        }
    }

    /// x86: <4 x i1> -> i4 -> i32, which LLVM lowers to `vmovmskpd`.
    fn compile_movemask_u64x4(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "movemask_u64x4 is x86-only (AVX2 vmovmskpd); in a *_arm.ea kernel compare \
                 u64x2 vectors and read the two lanes with c[0] / c[1]",
            ));
        }
        let mask = self.compile_expr(&args[0], function)?.into_vector_value();
        let bits = self
            .builder
            .build_bit_cast(mask, self.context.custom_width_int_type(4), "mask_bits")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into_int_value();
        self.zext_to(bits, self.context.i32_type(), "movemask_u64x4")
    }

    /// ARM: cmeq result -> `shrn #4` -> `fmov`, giving 4 bits per byte lane
    /// (lane i occupies bits 4i..4i+3; first match index = ctz / 4).
    fn compile_nibble_mask_u8x16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "nibble_mask_u8x16 is ARM-only (NEON shrn); on x86 use movemask_u8x16 \
                 (1 bit per lane)",
            ));
        }
        let err = |e: inkwell::builder::BuilderError| CompileError::codegen_error(e.to_string());
        let (i8t, i16t) = (self.context.i8_type(), self.context.i16_type());
        let mask = self.compile_expr(&args[0], function)?.into_vector_value();
        let bytes = self
            .builder
            .build_int_s_extend(mask, i8t.vec_type(16), "nib_sext")
            .map_err(err)?;
        let halves = self
            .builder
            .build_bit_cast(bytes, i16t.vec_type(8), "nib_halves")
            .map_err(err)?
            .into_vector_value();
        let four = i16t.const_int(4, false);
        let shift = inkwell::types::VectorType::const_vector(&[four; 8]);
        let shifted = self
            .builder
            .build_right_shift(halves, shift, false, "nib_shr")
            .map_err(err)?;
        let narrowed = self
            .builder
            .build_int_truncate(shifted, i8t.vec_type(8), "nib_trunc")
            .map_err(err)?;
        self.builder
            .build_bit_cast(narrowed, self.context.i64_type(), "nibble_mask")
            .map_err(err)
    }

    /// `llvm.cttz` with zero defined (returns the bit width): `tzcnt` on
    /// x86-64-v3, `rbit` + `clz` on AArch64. Result is i32.
    fn compile_ctz(
        &mut self,
        args: &[Expr],
        ty: Type,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let x = self
            .compile_expr_typed(&args[0], Some(&ty), function)?
            .into_int_value();
        let int_ty = x.get_type();
        let bits = int_ty.get_bit_width();
        let fn_name = format!("llvm.cttz.i{bits}");
        let bool_ty = self.context.bool_type();
        let fn_type = int_ty.fn_type(&[int_ty.into(), bool_ty.into()], false);
        let cttz = self
            .module
            .get_function(&fn_name)
            .unwrap_or_else(|| self.module.add_function(&fn_name, fn_type, None));
        let zero_is_poison = bool_ty.const_int(0, false);
        let count = self
            .builder
            .build_call(cttz, &[x.into(), zero_is_poison.into()], "ctz")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("cttz did not return a value"))?
            .into_int_value();
        if bits == 32 {
            return Ok(count.into());
        }
        Ok(self
            .builder
            .build_int_truncate(count, self.context.i32_type(), "ctz_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into())
    }

    fn zext_to(
        &self,
        v: IntValue<'ctx>,
        ty: inkwell::types::IntType<'ctx>,
        name: &str,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        Ok(self
            .builder
            .build_int_z_extend(v, ty, name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .into())
    }
}
