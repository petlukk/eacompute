use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// maddubs_i16(u8x16, i8x16) -> i16x8   (SSSE3 pmaddubsw)
    /// maddubs_i16(u8x32, i8x32) -> i16x16  (AVX2 vpmaddubsw)
    /// maddubs_i16(u8x64, i8x64) -> i16x32  (AVX-512BW vpmaddubsw)
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
            64 => {
                let i8x64_ty = self.context.i8_type().vec_type(64);
                let i16x32_ty = self.context.i16_type().vec_type(32);
                let fn_type = i16x32_ty.fn_type(&[i8x64_ty.into(), i8x64_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx512.pmaddubs.w.512")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx512.pmaddubs.w.512", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "maddubs_i16_avx512")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("maddubs_i16 AVX-512 did not return a value")
                    })
            }
            _ => Err(CompileError::codegen_error(format!(
                "maddubs_i16: unsupported width {width}"
            ))),
        }
    }

    /// madd_i16(i16x8, i16x8)   -> i32x4   (SSE2 pmaddwd)
    /// madd_i16(i16x16, i16x16) -> i32x8   (AVX2 vpmaddwd)
    /// madd_i16(i16x32, i16x32) -> i32x16  (AVX-512BW vpmaddwd)
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
            32 => {
                let i16x32_ty = self.context.i16_type().vec_type(32);
                let i32x16_ty = self.context.i32_type().vec_type(16);
                let fn_type = i32x16_ty.fn_type(&[i16x32_ty.into(), i16x32_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function("llvm.x86.avx512.pmaddw.d.512")
                    .unwrap_or_else(|| {
                        self.module
                            .add_function("llvm.x86.avx512.pmaddw.d.512", fn_type, None)
                    });
                self.builder
                    .build_call(intrinsic, &[a.into(), b.into()], "madd_i16_avx512")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error("madd_i16 AVX-512 did not return a value")
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
}
