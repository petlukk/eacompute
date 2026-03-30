use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// maddubs_i16(u8x16, i8x16) -> i16x8
    /// Multiplies unsigned 8-bit × signed 8-bit pairs, adds adjacent products → signed 16-bit.
    /// Maps to SSSE3 pmaddubsw (_mm_maddubs_epi16). Fast but accumulator overflows at i16.
    pub(super) fn compile_maddubs_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "maddubs_i16 is x86-only (SSSE3); no NEON equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value(); // u8x16
        let b = self.compile_expr(&args[1], function)?.into_vector_value(); // i8x16

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i16x8_ty = self.context.i16_type().vec_type(8);

        let fn_type = i16x8_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.x86.ssse3.pmadd.ub.sw.128")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.x86.ssse3.pmadd.ub.sw.128", fn_type, None)
            });

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "maddubs_i16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("maddubs_i16 did not return a value"))?;

        Ok(result)
    }

    /// maddubs_i32(u8xN, i8xN) -> i32x(N/4)
    /// Two-intrinsic chain: pmaddubsw → pmaddwd(ones).
    /// N=16: SSE path (SSSE3 + SSE2) → i32x4.
    /// N=32: AVX2 path (vpmaddubsw + vpmaddwd) → i32x8.
    pub(super) fn compile_maddubs_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "maddubs_i32 is x86-only (SSSE3+SSE2); no NEON equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        let width = a.get_type().get_size();
        match width {
            16 => self.compile_maddubs_i32_128(a, b),
            32 => self.compile_maddubs_i32_256(a, b),
            _ => Err(CompileError::codegen_error(
                format!("maddubs_i32: unsupported width {width}"),
            )),
        }
    }

    /// SSE path: u8x16 × i8x16 → i32x4 (SSSE3 + SSE2).
    fn compile_maddubs_i32_128(
        &mut self,
        a: inkwell::values::VectorValue<'ctx>,
        b: inkwell::values::VectorValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i16x8_ty = self.context.i16_type().vec_type(8);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        // Step 1: pmaddubsw
        let ubsw_ty = i16x8_ty.fn_type(
            &[i8x16_ty.into(), i8x16_ty.into()], false,
        );
        let ubsw = self.module
            .get_function("llvm.x86.ssse3.pmadd.ub.sw.128")
            .unwrap_or_else(|| {
                self.module.add_function(
                    "llvm.x86.ssse3.pmadd.ub.sw.128", ubsw_ty, None,
                )
            });
        let t = self.builder
            .build_call(ubsw, &[a.into(), b.into()], "maddubs_i32_step1")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value().basic()
            .ok_or_else(|| CompileError::codegen_error(
                "pmaddubsw128 did not return a value",
            ))?
            .into_vector_value();

        // Step 2: pmaddwd with ones
        let one = self.context.i16_type().const_int(1, false);
        let ones = VectorType::const_vector(&[one; 8]);
        let wd_ty = i32x4_ty.fn_type(
            &[i16x8_ty.into(), i16x8_ty.into()], false,
        );
        let wd = self.module
            .get_function("llvm.x86.sse2.pmadd.wd")
            .unwrap_or_else(|| {
                self.module.add_function(
                    "llvm.x86.sse2.pmadd.wd", wd_ty, None,
                )
            });
        self.builder
            .build_call(wd, &[t.into(), ones.into()], "maddubs_i32_step2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value().basic()
            .ok_or_else(|| CompileError::codegen_error(
                "pmaddwd128 did not return a value",
            ))
    }

    /// AVX2 path: u8x32 × i8x32 → i32x8 (vpmaddubsw + vpmaddwd).
    fn compile_maddubs_i32_256(
        &mut self,
        a: inkwell::values::VectorValue<'ctx>,
        b: inkwell::values::VectorValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let i8x32_ty = self.context.i8_type().vec_type(32);
        let i16x16_ty = self.context.i16_type().vec_type(16);
        let i32x8_ty = self.context.i32_type().vec_type(8);

        // Step 1: vpmaddubsw (AVX2)
        let ubsw_ty = i16x16_ty.fn_type(
            &[i8x32_ty.into(), i8x32_ty.into()], false,
        );
        let ubsw = self.module
            .get_function("llvm.x86.avx2.pmadd.ub.sw")
            .unwrap_or_else(|| {
                self.module.add_function(
                    "llvm.x86.avx2.pmadd.ub.sw", ubsw_ty, None,
                )
            });
        let t = self.builder
            .build_call(ubsw, &[a.into(), b.into()], "maddubs_i32_avx2_step1")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value().basic()
            .ok_or_else(|| CompileError::codegen_error(
                "vpmaddubsw did not return a value",
            ))?
            .into_vector_value();

        // Step 2: vpmaddwd with ones (AVX2)
        let one = self.context.i16_type().const_int(1, false);
        let ones = VectorType::const_vector(&[one; 16]);
        let wd_ty = i32x8_ty.fn_type(
            &[i16x16_ty.into(), i16x16_ty.into()], false,
        );
        let wd = self.module
            .get_function("llvm.x86.avx2.pmadd.wd")
            .unwrap_or_else(|| {
                self.module.add_function(
                    "llvm.x86.avx2.pmadd.wd", wd_ty, None,
                )
            });
        self.builder
            .build_call(wd, &[t.into(), ones.into()], "maddubs_i32_avx2_step2")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value().basic()
            .ok_or_else(|| CompileError::codegen_error(
                "vpmaddwd did not return a value",
            ))
    }

    /// vdot_i32(i8x16, i8x16) -> i32x4
    /// ARM NEON dot product: signed × signed, groups of 4 products per i32 lane.
    /// Maps to llvm.aarch64.neon.sdot.v4i32.v16i8 with a zero accumulator.
    pub(super) fn compile_vdot_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "vdot_i32 is ARM-only (NEON dot product); no x86 equivalent",
            ));
        }
        if !self.dotprod {
            return Err(CompileError::codegen_error(
                "vdot_i32 requires ARMv8.2-A dot product extension; compile with --dotprod",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value(); // i8x16
        let b = self.compile_expr(&args[1], function)?.into_vector_value(); // i8x16

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        // Zero accumulator — caller does explicit .+ for accumulation
        let zero_acc = i32x4_ty.const_zero();

        let sdot_fn_type =
            i32x4_ty.fn_type(&[i32x4_ty.into(), i8x16_ty.into(), i8x16_ty.into()], false);
        let sdot = self
            .module
            .get_function("llvm.aarch64.neon.sdot.v4i32.v16i8")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.sdot.v4i32.v16i8", sdot_fn_type, None)
            });

        let result = self
            .builder
            .build_call(sdot, &[zero_acc.into(), a.into(), b.into()], "vdot_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("vdot_i32 did not return a value"))?;

        Ok(result)
    }

    /// shuffle_bytes(table: u8x16, indices: u8x16) -> u8x16
    /// Byte-level table lookup: output[i] = table[indices[i]].
    /// x86: SSSE3 pshufb. ARM: NEON tbl.
    /// Indices should be 0-15; out-of-range zeroes the lane on both platforms.
    pub(super) fn compile_shuffle_bytes(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let table = self.compile_expr(&args[0], function)?.into_vector_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();

        let i8x16_ty = self.context.i8_type().vec_type(16);

        let (intrinsic_name, fn_type, call_args): (
            _,
            _,
            Vec<inkwell::values::BasicMetadataValueEnum>,
        ) = if self.is_arm {
            (
                "llvm.aarch64.neon.tbl1.v16i8",
                i8x16_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false),
                vec![table.into(), indices.into()],
            )
        } else {
            (
                "llvm.x86.ssse3.pshuf.b.128",
                i8x16_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false),
                vec![table.into(), indices.into()],
            )
        };

        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &call_args, "shuffle_bytes")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("shuffle_bytes did not return a value"))?;

        Ok(result)
    }
}
