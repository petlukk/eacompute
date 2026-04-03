use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// maddubs_i16(u8x16, i8x16) -> i16x8   (SSSE3 pmaddubsw)
    /// maddubs_i16(u8x32, i8x32) -> i16x16  (AVX2 vpmaddubsw)
    pub(super) fn compile_maddubs_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "maddubs_i16 is x86-only (SSSE3/AVX2 pmaddubsw); no NEON equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        let width = a.get_type().get_size();
        match width {
            16 => {
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
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "maddubs_i16")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("maddubs_i16 did not return a value")
                    })
            }
            32 => {
                let i8x32_ty = self.context.i8_type().vec_type(32);
                let i16x16_ty = self.context.i16_type().vec_type(16);
                let fn_type = i16x16_ty.fn_type(&[i8x32_ty.into(), i8x32_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx2.pmadd.ub.sw")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx2.pmadd.ub.sw", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "maddubs_i16_avx2")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("maddubs_i16 AVX2 did not return a value")
                    })
            }
            _ => Err(CompileError::codegen_error(format!(
                "maddubs_i16: unsupported width {width}"
            ))),
        }
    }

    /// madd_i16(i16x8, i16x8) -> i32x4   (SSE2 pmaddwd)
    /// madd_i16(i16x16, i16x16) -> i32x8  (AVX2 vpmaddwd)
    /// Multiply i16 pairs, add adjacent products -> i32.
    pub(super) fn compile_madd_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "madd_i16 is x86-only (SSE2/AVX2 pmaddwd); no NEON equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        let width = a.get_type().get_size();
        match width {
            8 => {
                let i16x8_ty = self.context.i16_type().vec_type(8);
                let i32x4_ty = self.context.i32_type().vec_type(4);
                let fn_type = i32x4_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.sse2.pmadd.wd")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.sse2.pmadd.wd", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "madd_i16")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("madd_i16 did not return a value"))
            }
            16 => {
                let i16x16_ty = self.context.i16_type().vec_type(16);
                let i32x8_ty = self.context.i32_type().vec_type(8);
                let fn_type = i32x8_ty.fn_type(&[i16x16_ty.into(), i16x16_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx2.pmadd.wd")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx2.pmadd.wd", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "madd_i16_avx2")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("madd_i16 AVX2 did not return a value")
                    })
            }
            _ => Err(CompileError::codegen_error(format!(
                "madd_i16: unsupported width {width}"
            ))),
        }
    }

    /// hadd_i16(i16x8, i16x8) -> i16x8   (SSSE3 phaddw)
    /// hadd_i16(i16x16, i16x16) -> i16x16  (AVX2 vphaddw)
    pub(super) fn compile_hadd_i16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "hadd_i16 is x86-only (SSSE3/AVX2 phaddw); no NEON equivalent",
            ));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();

        let width = a.get_type().get_size();
        match width {
            8 => {
                let i16x8_ty = self.context.i16_type().vec_type(8);
                let fn_type = i16x8_ty.fn_type(&[i16x8_ty.into(), i16x8_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.ssse3.phadd.w.128")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.ssse3.phadd.w.128", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "hadd_i16")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("hadd_i16 did not return a value"))
            }
            16 => {
                let i16x16_ty = self.context.i16_type().vec_type(16);
                let fn_type = i16x16_ty.fn_type(&[i16x16_ty.into(), i16x16_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx2.phadd.w")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx2.phadd.w", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "hadd_i16_avx2")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("hadd_i16 AVX2 did not return a value")
                    })
            }
            _ => Err(CompileError::codegen_error(format!(
                "hadd_i16: unsupported width {width}"
            ))),
        }
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

    /// smmla_i32(acc: i32x4, a: i8x16, b: i8x16) -> i32x4
    /// ARM I8MM: signed x signed 2x8 x 8x2 matrix multiply-accumulate.
    /// Maps to llvm.aarch64.neon.smmla.v4i32.v16i8
    pub(super) fn compile_smmla_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "smmla_i32 is ARM-only (I8MM); no x86 equivalent",
            ));
        }
        if !self.i8mm {
            return Err(CompileError::codegen_error(
                "smmla_i32 requires ARMv8.6-A I8MM extension; compile with --i8mm",
            ));
        }
        let acc = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        let fn_type = i32x4_ty.fn_type(&[i32x4_ty.into(), i8x16_ty.into(), i8x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.smmla.v4i32.v16i8")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.smmla.v4i32.v16i8", fn_type, None)
            });

        let result = self
            .builder
            .build_call(intrinsic, &[acc.into(), a.into(), b.into()], "smmla_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("smmla_i32 did not return a value"))?;

        Ok(result)
    }

    /// ummla_i32(acc: i32x4, a: u8x16, b: u8x16) -> i32x4
    /// ARM I8MM: unsigned x unsigned 2x8 x 8x2 matrix multiply-accumulate.
    /// Maps to llvm.aarch64.neon.ummla.v4i32.v16i8
    pub(super) fn compile_ummla_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "ummla_i32 is ARM-only (I8MM); no x86 equivalent",
            ));
        }
        if !self.i8mm {
            return Err(CompileError::codegen_error(
                "ummla_i32 requires ARMv8.6-A I8MM extension; compile with --i8mm",
            ));
        }
        let acc = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        let fn_type = i32x4_ty.fn_type(&[i32x4_ty.into(), i8x16_ty.into(), i8x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.ummla.v4i32.v16i8")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.ummla.v4i32.v16i8", fn_type, None)
            });

        let result = self
            .builder
            .build_call(intrinsic, &[acc.into(), a.into(), b.into()], "ummla_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("ummla_i32 did not return a value"))?;

        Ok(result)
    }

    /// usmmla_i32(acc: i32x4, a: u8x16, b: i8x16) -> i32x4
    /// ARM I8MM: unsigned x signed 2x8 x 8x2 matrix multiply-accumulate.
    /// Maps to llvm.aarch64.neon.usmmla.v4i32.v16i8
    pub(super) fn compile_usmmla_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "usmmla_i32 is ARM-only (I8MM); no x86 equivalent",
            ));
        }
        if !self.i8mm {
            return Err(CompileError::codegen_error(
                "usmmla_i32 requires ARMv8.6-A I8MM extension; compile with --i8mm",
            ));
        }
        let acc = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i32x4_ty = self.context.i32_type().vec_type(4);

        let fn_type = i32x4_ty.fn_type(&[i32x4_ty.into(), i8x16_ty.into(), i8x16_ty.into()], false);
        let intrinsic = self
            .module
            .get_function("llvm.aarch64.neon.usmmla.v4i32.v16i8")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.aarch64.neon.usmmla.v4i32.v16i8", fn_type, None)
            });

        let result = self
            .builder
            .build_call(intrinsic, &[acc.into(), a.into(), b.into()], "usmmla_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("usmmla_i32 did not return a value"))?;

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
