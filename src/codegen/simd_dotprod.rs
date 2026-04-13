use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
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

    /// vdot_lane_i32(acc: i32x4, a: i8x16, b: i8x16, lane: 0..3) -> i32x4
    /// ARM NEON dot product by element: each i32 lane of acc gets
    /// dot(a[4i..4i+3], b[4*lane..4*lane+3]) added to it.
    /// Emits shuffle + sdot; LLVM folds this into SDOT (by element).
    pub(super) fn compile_vdot_lane_i32(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        lane: u8,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "vdot_lane_i32 is ARM-only (NEON dot product by element); no x86 equivalent",
            ));
        }
        if !self.dotprod {
            return Err(CompileError::codegen_error(
                "vdot_lane_i32 requires ARMv8.2-A dot product extension; compile with --dotprod",
            ));
        }
        let acc = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let i8x16_ty = self.context.i8_type().vec_type(16);
        let i32x4_ty = self.context.i32_type().vec_type(4);
        let i32_ty = self.context.i32_type();

        // Shuffle b to replicate the selected 4-byte group across all lanes.
        // lane=N → mask [4N, 4N+1, 4N+2, 4N+3] repeated 4 times.
        let base = (lane as u32) * 4;
        let mask: Vec<_> = (0..16u32)
            .map(|i| i32_ty.const_int((base + (i % 4)) as u64, false))
            .collect();
        let mask_vec = VectorType::const_vector(&mask);
        let b_splat = self
            .builder
            .build_shuffle_vector(b, b, mask_vec, "vdot_lane_splat")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

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
            .build_call(
                sdot,
                &[acc.into(), a.into(), b_splat.into()],
                "vdot_lane_i32",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("vdot_lane_i32 did not return a value"))?;

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

    /// shuffle_bytes(u8x16, u8x16) -> u8x16  (SSSE3 pshufb / NEON tbl)
    /// shuffle_bytes(u8x32, u8x32) -> u8x32  (AVX2 vpshufb, x86-only)
    /// Byte-level table lookup: output[i] = table[indices[i]].
    /// Indices should be 0-15 (128-bit) or 0-15 per lane (256-bit).
    /// Out-of-range (high bit set) zeroes the lane.
    pub(super) fn compile_shuffle_bytes(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let table = self.compile_expr(&args[0], function)?.into_vector_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();

        let width = table.get_type().get_size();
        match width {
            16 => {
                let i8x16_ty = self.context.i8_type().vec_type(16);
                let (intrinsic_name, fn_type): (&str, _) = if self.is_arm {
                    (
                        "llvm.aarch64.neon.tbl1.v16i8",
                        i8x16_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false),
                    )
                } else {
                    (
                        "llvm.x86.ssse3.pshuf.b.128",
                        i8x16_ty.fn_type(&[i8x16_ty.into(), i8x16_ty.into()], false),
                    )
                };
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                self.builder
                    .build_call(intrinsic, &[table.into(), indices.into()], "shuffle_bytes")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("shuffle_bytes did not return a value")
                    })
            }
            32 => {
                if self.is_arm {
                    return Err(CompileError::codegen_error(
                        "shuffle_bytes(u8x32) is x86-only (AVX2 vpshufb); \
                         use u8x16 on ARM",
                    ));
                }
                let i8x32_ty = self.context.i8_type().vec_type(32);
                let fn_type = i8x32_ty.fn_type(&[i8x32_ty.into(), i8x32_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx2.pshuf.b")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx2.pshuf.b", fn_type, None)
                    });
                self.builder
                    .build_call(
                        intrinsic,
                        &[table.into(), indices.into()],
                        "shuffle_bytes_avx2",
                    )
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("shuffle_bytes AVX2 did not return a value")
                    })
            }
            _ => Err(CompileError::codegen_error(format!(
                "shuffle_bytes: unsupported width {width}"
            ))),
        }
    }
}
