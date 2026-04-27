use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Type conversion intrinsics: to_f16, to_f32, to_f64, to_i16, to_i32, to_i64.
    pub(super) fn compile_conversion(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_unsigned_src = self.arg_is_unsigned(&args[0]);
        let val = self.compile_expr(&args[0], function)?;

        match name {
            "to_f16" => {
                let target = self.context.f16_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder
                                .build_unsigned_int_to_float(iv, target, "uitofp_f16")
                        } else {
                            self.builder
                                .build_signed_int_to_float(iv, target, "sitofp_f16")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        let src_ty = fv.get_type();
                        if src_ty == self.context.f16_type() {
                            Ok(val)
                        } else {
                            let result = self
                                .builder
                                .build_float_trunc(fv, target, "fptrunc_f16")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_f16: unsupported source type",
                    )),
                }
            }
            "to_f32" => {
                let target = self.context.f32_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder
                                .build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        let src_ty = fv.get_type();
                        if src_ty == self.context.f64_type() {
                            let result = self
                                .builder
                                .build_float_trunc(fv, target, "fptrunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else if src_ty == self.context.f16_type() {
                            let result = self
                                .builder
                                .build_float_ext(fv, target, "fpext_f16_f32")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    BasicValueEnum::VectorValue(vv) => {
                        // Vector i32xN -> f32xN. LLVM build_signed_int_to_float is width-polymorphic.
                        let width = vv.get_type().get_size();
                        let f32xn_ty = self.context.f32_type().vec_type(width);
                        let result = self
                            .builder
                            .build_signed_int_to_float(vv, f32xn_ty, "sitofp_vec_i32_f32")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::VectorValue(result))
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_f32: unsupported source type",
                    )),
                }
            }
            "to_f64" => {
                let target = self.context.f64_type();
                match val {
                    BasicValueEnum::IntValue(iv) => {
                        let result = if is_unsigned_src {
                            self.builder
                                .build_unsigned_int_to_float(iv, target, "uitofp")
                        } else {
                            self.builder.build_signed_int_to_float(iv, target, "sitofp")
                        }
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::FloatValue(result))
                    }
                    BasicValueEnum::FloatValue(fv) => {
                        if fv.get_type() == self.context.f32_type() {
                            let result = self
                                .builder
                                .build_float_ext(fv, target, "fpext")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::FloatValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_f64: unsupported source type",
                    )),
                }
            }
            "to_i16" => {
                let target = self.context.i16_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width > 16 {
                            let result = self
                                .builder
                                .build_int_truncate(iv, target, "trunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else if src_width < 16 {
                            let result = if is_unsigned_src {
                                self.builder.build_int_z_extend(iv, target, "zext")
                            } else {
                                self.builder.build_int_s_extend(iv, target, "sext")
                            }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_i16: unsupported source type",
                    )),
                }
            }
            "to_i32" => {
                let target = self.context.i32_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width > 32 {
                            let result = self
                                .builder
                                .build_int_truncate(iv, target, "trunc")
                                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else if src_width < 32 {
                            let result = if is_unsigned_src {
                                self.builder.build_int_z_extend(iv, target, "zext")
                            } else {
                                self.builder.build_int_s_extend(iv, target, "sext")
                            }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_i32: unsupported source type",
                    )),
                }
            }
            "to_i64" => {
                let target = self.context.i64_type();
                match val {
                    BasicValueEnum::FloatValue(fv) => {
                        let result = self
                            .builder
                            .build_float_to_signed_int(fv, target, "fptosi")
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                        Ok(BasicValueEnum::IntValue(result))
                    }
                    BasicValueEnum::IntValue(iv) => {
                        let src_width = iv.get_type().get_bit_width();
                        if src_width < 64 {
                            let result = if is_unsigned_src {
                                self.builder.build_int_z_extend(iv, target, "zext")
                            } else {
                                self.builder.build_int_s_extend(iv, target, "sext")
                            }
                            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                            Ok(BasicValueEnum::IntValue(result))
                        } else {
                            Ok(val)
                        }
                    }
                    _ => Err(CompileError::codegen_error(
                        "to_i64: unsupported source type",
                    )),
                }
            }
            _ => unreachable!(),
        }
    }
}
