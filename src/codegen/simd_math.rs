use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// sqrt(x) for scalar f32/f64 and float vectors.
    pub(super) fn compile_sqrt(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        match val {
            BasicValueEnum::FloatValue(fv) => {
                let float_ty = fv.get_type();
                let intrinsic_name = if float_ty == self.context.f32_type() {
                    "llvm.sqrt.f32"
                } else {
                    "llvm.sqrt.f64"
                };
                let fn_type = float_ty.fn_type(&[float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[fv.into()], "sqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("sqrt did not return a value"))?;
                Ok(result)
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.sqrt", vec_ty);
                let fn_type = vec_ty.fn_type(&[vec_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[vv.into()], "vsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("sqrt did not return a value"))?;
                Ok(result)
            }
            _ => Err(CompileError::codegen_error(
                "sqrt expects float or float vector",
            )),
        }
    }

    /// abs(x) — cross-platform absolute value.
    /// Float: llvm.fabs.{type}. Signed integer vector: llvm.abs.{type}.
    pub(super) fn compile_abs(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        match val {
            BasicValueEnum::FloatValue(fv) => {
                let float_ty = fv.get_type();
                let intrinsic_name = if float_ty == self.context.f32_type() {
                    "llvm.fabs.f32"
                } else {
                    "llvm.fabs.f64"
                };
                let fn_type = float_ty.fn_type(&[float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[fv.into()], "abs")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("abs did not return a value"))?;
                Ok(result)
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let elem = vec_ty.get_element_type();
                if elem.is_float_type() {
                    let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.fabs", vec_ty);
                    let fn_type = vec_ty.fn_type(&[vec_ty.into()], false);
                    let intrinsic =
                        self.module
                            .get_function(&intrinsic_name)
                            .unwrap_or_else(|| {
                                self.module.add_function(&intrinsic_name, fn_type, None)
                            });
                    let result = self
                        .builder
                        .build_call(intrinsic, &[vv.into()], "vfabs")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CompileError::codegen_error("abs did not return a value"))?;
                    Ok(result)
                } else {
                    // Signed integer vector: llvm.abs.v{N}i{bits}
                    let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.abs", vec_ty);
                    let i1_false = self.context.bool_type().const_int(0, false);
                    let fn_type =
                        vec_ty.fn_type(&[vec_ty.into(), self.context.bool_type().into()], false);
                    let intrinsic =
                        self.module
                            .get_function(&intrinsic_name)
                            .unwrap_or_else(|| {
                                self.module.add_function(&intrinsic_name, fn_type, None)
                            });
                    let result = self
                        .builder
                        .build_call(intrinsic, &[vv.into(), i1_false.into()], "vabs")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                        .try_as_basic_value()
                        .basic()
                        .ok_or_else(|| CompileError::codegen_error("abs did not return a value"))?;
                    Ok(result)
                }
            }
            _ => Err(CompileError::codegen_error(
                "abs expects float or signed integer vector",
            )),
        }
    }

    /// exp(x) for scalar f32/f64 and float vectors.
    pub(super) fn compile_exp(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let val = self.compile_expr(&args[0], function)?;
        match val {
            BasicValueEnum::FloatValue(fv) => {
                let float_ty = fv.get_type();
                let intrinsic_name = if float_ty == self.context.f32_type() {
                    "llvm.exp.f32"
                } else {
                    "llvm.exp.f64"
                };
                let fn_type = float_ty.fn_type(&[float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[fv.into()], "exp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("exp did not return a value"))?;
                Ok(result)
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.exp", vec_ty);
                let fn_type = vec_ty.fn_type(&[vec_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[vv.into()], "vexp")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("exp did not return a value"))?;
                Ok(result)
            }
            _ => Err(CompileError::codegen_error(
                "exp expects float or float vector",
            )),
        }
    }

    /// rsqrt(x) = 1.0 / sqrt(x). Accurate; LLVM may lower to vrsqrtps + refinement.
    pub(super) fn compile_rsqrt(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let sqrt_val = self.compile_sqrt(args, function)?;
        match sqrt_val {
            BasicValueEnum::FloatValue(fv) => {
                let one = fv.get_type().const_float(1.0);
                let result = self
                    .builder
                    .build_float_div(one, fv, "rsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::FloatValue(result))
            }
            BasicValueEnum::VectorValue(vv) => {
                let vec_ty = vv.get_type();
                let elem_ty = vec_ty.get_element_type().into_float_type();
                let one_scalar = elem_ty.const_float(1.0);
                let one_vec =
                    self.build_splat(BasicValueEnum::FloatValue(one_scalar), vec_ty.get_size())?;
                let result = self
                    .builder
                    .build_float_div(one_vec, vv, "vrsqrt")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?;
                Ok(BasicValueEnum::VectorValue(result))
            }
            _ => Err(CompileError::codegen_error(
                "rsqrt expects float or float vector",
            )),
        }
    }

    /// min(a, b) / max(a, b) for scalar i32, f32, f64.
    pub(super) fn compile_min_max(
        &mut self,
        args: &[Expr],
        name: &str,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?;
        let b = self.compile_expr(&args[1], function)?;
        match (a, b) {
            (BasicValueEnum::IntValue(av), BasicValueEnum::IntValue(bv)) => {
                let int_ty = av.get_type();
                let intrinsic_name = if name == "min" {
                    "llvm.smin.i32"
                } else {
                    "llvm.smax.i32"
                };
                let fn_type = int_ty.fn_type(&[int_ty.into(), int_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[av.into(), bv.into()], name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error(format!("{name} did not return a value"))
                    })?;
                Ok(result)
            }
            (BasicValueEnum::FloatValue(av), BasicValueEnum::FloatValue(bv)) => {
                let float_ty = av.get_type();
                let type_suffix = if float_ty == self.context.f32_type() {
                    "f32"
                } else {
                    "f64"
                };
                let op = if name == "min" { "minnum" } else { "maxnum" };
                let intrinsic_name = format!("llvm.{op}.{type_suffix}");
                let fn_type = float_ty.fn_type(&[float_ty.into(), float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[av.into(), bv.into()], name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error(format!("{name} did not return a value"))
                    })?;
                Ok(result)
            }
            (BasicValueEnum::VectorValue(av), BasicValueEnum::VectorValue(bv)) => {
                let vec_ty = av.get_type();
                let base = if name == "min" {
                    "llvm.minnum"
                } else {
                    "llvm.maxnum"
                };
                let intrinsic_name = self.llvm_vector_intrinsic_name(base, vec_ty);
                let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[av.into(), bv.into()], name)
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| {
                        CompileError::codegen_error(format!("{name} did not return a value"))
                    })?;
                Ok(result)
            }
            _ => Err(CompileError::codegen_error(format!(
                "{name} expects matching numeric types"
            ))),
        }
    }
}
