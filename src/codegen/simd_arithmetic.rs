use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::{BinaryOp, Expr};
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_vector_binary(
        &mut self,
        l: VectorValue<'ctx>,
        r: VectorValue<'ctx>,
        op: &BinaryOp,
        is_unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_float = l.get_type().get_element_type().is_float_type();
        let result = match op {
            BinaryOp::AddDot => {
                if is_float {
                    self.builder.build_float_add(l, r, "vadd")
                } else {
                    self.builder.build_int_add(l, r, "vadd")
                }
            }
            BinaryOp::SubDot => {
                if is_float {
                    self.builder.build_float_sub(l, r, "vsub")
                } else {
                    self.builder.build_int_sub(l, r, "vsub")
                }
            }
            BinaryOp::MulDot => {
                if is_float {
                    self.builder.build_float_mul(l, r, "vmul")
                } else {
                    self.builder.build_int_mul(l, r, "vmul")
                }
            }
            BinaryOp::DivDot => {
                if is_float {
                    self.builder.build_float_div(l, r, "vdiv")
                } else if is_unsigned {
                    self.builder.build_int_unsigned_div(l, r, "vdiv")
                } else {
                    self.builder.build_int_signed_div(l, r, "vdiv")
                }
            }
            BinaryOp::AndDot => {
                // Type checker guarantees integer vectors only
                self.builder.build_and(l, r, "vand")
            }
            BinaryOp::OrDot => self.builder.build_or(l, r, "vor"),
            BinaryOp::XorDot => self.builder.build_xor(l, r, "vxor"),
            BinaryOp::ShiftLeftDot => self.builder.build_left_shift(l, r, "vshl"),
            BinaryOp::ShiftRightDot => self.builder.build_right_shift(l, r, !is_unsigned, "vshr"),
            BinaryOp::LessDot
            | BinaryOp::GreaterDot
            | BinaryOp::LessEqualDot
            | BinaryOp::GreaterEqualDot
            | BinaryOp::EqualDot
            | BinaryOp::NotEqualDot => {
                return self.compile_vector_compare(l, r, op, is_unsigned);
            }
            _ => {
                return Err(CompileError::codegen_error(
                    "unsupported vector binary operation (internal error)",
                ));
            }
        };
        Ok(BasicValueEnum::VectorValue(
            result.map_err(|e| CompileError::codegen_error(e.to_string()))?,
        ))
    }

    fn compile_vector_compare(
        &self,
        l: VectorValue<'ctx>,
        r: VectorValue<'ctx>,
        op: &BinaryOp,
        is_unsigned: bool,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_float = l.get_type().get_element_type().is_float_type();
        if is_float {
            use inkwell::FloatPredicate;
            let pred = match op {
                BinaryOp::LessDot => FloatPredicate::OLT,
                BinaryOp::GreaterDot => FloatPredicate::OGT,
                BinaryOp::LessEqualDot => FloatPredicate::OLE,
                BinaryOp::GreaterEqualDot => FloatPredicate::OGE,
                BinaryOp::EqualDot => FloatPredicate::OEQ,
                BinaryOp::NotEqualDot => FloatPredicate::ONE,
                _ => unreachable!(),
            };
            let cmp = self
                .builder
                .build_float_compare(pred, l, r, "vcmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(cmp.into())
        } else {
            use inkwell::IntPredicate;
            let pred = match op {
                BinaryOp::LessDot => {
                    if is_unsigned {
                        IntPredicate::ULT
                    } else {
                        IntPredicate::SLT
                    }
                }
                BinaryOp::GreaterDot => {
                    if is_unsigned {
                        IntPredicate::UGT
                    } else {
                        IntPredicate::SGT
                    }
                }
                BinaryOp::LessEqualDot => {
                    if is_unsigned {
                        IntPredicate::ULE
                    } else {
                        IntPredicate::SLE
                    }
                }
                BinaryOp::GreaterEqualDot => {
                    if is_unsigned {
                        IntPredicate::UGE
                    } else {
                        IntPredicate::SGE
                    }
                }
                BinaryOp::EqualDot => IntPredicate::EQ,
                BinaryOp::NotEqualDot => IntPredicate::NE,
                _ => unreachable!(),
            };
            let cmp = self
                .builder
                .build_int_compare(pred, l, r, "vcmp")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            Ok(cmp.into())
        }
    }

    pub(super) fn compile_select(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let mask = self.compile_expr(&args[0], function)?.into_vector_value();
        let a = self.compile_expr(&args[1], function)?.into_vector_value();
        let b = self.compile_expr(&args[2], function)?.into_vector_value();

        let result = self
            .builder
            .build_select(mask, a, b, "vselect")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(result)
    }

    /// Widen lower N bytes of i8x16/u8x16 to f32xN (N = 4, 8, or 16).
    /// unsigned=true uses zero-extension (for u8); false uses sign-extension (for i8).
    pub(super) fn compile_widen_i8_f32(
        &mut self,
        args: &[Expr],
        unsigned: bool,
        output_width: u32,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec16 = self.compile_expr(&args[0], function)?.into_vector_value();

        // Extract lower N bytes via shufflevector: <16 x i8> → <N x i8>
        let undef16 = vec16.get_type().get_undef();
        let mask_vals: Vec<_> = (0u64..output_width as u64)
            .map(|i| self.context.i32_type().const_int(i, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let lower_n = self
            .builder
            .build_shuffle_vector(vec16, undef16, mask, "widen_lower")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Extend <N x i8> to <N x i32>
        let i32xn_type = self.context.i32_type().vec_type(output_width);
        let i32xn = if unsigned {
            self.builder
                .build_int_z_extend(lower_n, i32xn_type, "zext_i8_i32")
        } else {
            self.builder
                .build_int_s_extend(lower_n, i32xn_type, "sext_i8_i32")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Convert <N x i32> to <N x float>
        let f32xn_type = self.context.f32_type().vec_type(output_width);
        let f32xn = self
            .builder
            .build_signed_int_to_float(i32xn, f32xn_type, "sitofp_i32_f32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::VectorValue(f32xn))
    }

    /// Widen lower N bytes of u8x16 to i32xN (N = 4, 8, or 16).
    /// Zero-extends only — no float conversion. Useful for gather indices.
    pub(super) fn compile_widen_u8_i32(
        &mut self,
        args: &[Expr],
        output_width: u32,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec16 = self.compile_expr(&args[0], function)?.into_vector_value();

        // Extract lower N bytes via shufflevector: <16 x i8> → <N x i8>
        let undef16 = vec16.get_type().get_undef();
        let mask_vals: Vec<_> = (0u64..output_width as u64)
            .map(|i| self.context.i32_type().const_int(i, false))
            .collect();
        let mask = VectorType::const_vector(&mask_vals);
        let lower_n = self
            .builder
            .build_shuffle_vector(vec16, undef16, mask, "widen_lower")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Zero-extend <N x i8> to <N x i32>
        let i32xn_type = self.context.i32_type().vec_type(output_width);
        let i32xn = self
            .builder
            .build_int_z_extend(lower_n, i32xn_type, "zext_u8_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::VectorValue(i32xn))
    }

    /// Narrow f32x4 to i8x4: fptosi + trunc, no padding.
    pub(super) fn compile_narrow_f32x4_i8(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let f32x4 = self.compile_expr(&args[0], function)?.into_vector_value();

        // fptosi <4 x float> to <4 x i32>
        let i32x4_type = self.context.i32_type().vec_type(4);
        let i32x4 = self
            .builder
            .build_float_to_signed_int(f32x4, i32x4_type, "fptosi_f32_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // trunc <4 x i32> to <4 x i8>
        let i8x4_type = self.context.i8_type().vec_type(4);
        let i8x4 = self
            .builder
            .build_int_truncate(i32x4, i8x4_type, "trunc_i32_i8")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::VectorValue(i8x4))
    }

    /// movemask(vec) -> i32
    /// Extracts the MSB of each byte into a scalar bitmask.
    /// Accepts <N x i1> (from comparisons — sext to i8 first) or <N x i8> (raw byte vectors).
    /// Maps to SSE2 pmovmskb.128 (width 16) or AVX2 pmovmskb.256 (width 32).
    pub(super) fn compile_movemask(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "movemask is x86-only (SSE2/AVX2); no NEON equivalent",
            ));
        }

        let arg_val = self.compile_expr(&args[0], function)?;
        let vec_val = arg_val.into_vector_value();
        let vec_ty = vec_val.get_type();
        let width = vec_ty.get_size();

        // If the vector is <N x i1> (from a comparison), sext to <N x i8>
        let i8_vec_ty = self.context.i8_type().vec_type(width);
        let byte_vec = if vec_ty.get_element_type().into_int_type().get_bit_width() == 1 {
            self.builder
                .build_int_s_extend(vec_val, i8_vec_ty, "movemask_sext")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
        } else {
            vec_val
        };

        let i32_ty = self.context.i32_type();
        let (intrinsic_name, fn_type) = match width {
            16 => (
                "llvm.x86.sse2.pmovmskb.128",
                i32_ty.fn_type(&[i8_vec_ty.into()], false),
            ),
            32 => (
                "llvm.x86.avx2.pmovmskb",
                i32_ty.fn_type(&[i8_vec_ty.into()], false),
            ),
            _ => {
                return Err(CompileError::codegen_error(format!(
                    "movemask unsupported for width {width}"
                )));
            }
        };

        let intrinsic = self
            .module
            .get_function(intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[byte_vec.into()], "movemask")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("movemask did not return a value"))?;

        Ok(result)
    }
}
