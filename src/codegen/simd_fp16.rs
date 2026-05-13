//! Native f16 codegen — splat, load, store, fma, reductions.
//! Gated on `self.fp16` (set by `--fp16` flag, which appends
//! `+fullfp16` to LLVM target features).
//!
//! Element-wise arithmetic on f16 vectors does NOT live here: it
//! flows through the existing `compile_vector_binary` path because
//! LLVM's `is_float_type()` is true for `<N x half>`.

use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Build an f16 splat: `<N x half>` with all lanes equal to `value`.
    /// Width is inferred from the type_hint or defaults to 4.
    pub(super) fn compile_splat_f16(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width as u32,
            _ => 4,
        };

        // Compile the scalar arg without a type hint — float literals become f64.
        // Then fptrunc to f16 so build_splat gets the right element type.
        let raw = self.compile_expr(&args[0], function)?;
        let f16_ty = self.context.f16_type();
        let scalar = self
            .builder
            .build_float_trunc(raw.into_float_value(), f16_ty, "f16_scalar")
            .map_err(|e| crate::error::CompileError::codegen_error(e.to_string()))?;

        let vec = self.build_splat(BasicValueEnum::FloatValue(scalar), width)?;
        Ok(BasicValueEnum::VectorValue(vec))
    }

    /// Native f16 vector load: `<N x half>` aligned load from a `*f16` pointer.
    /// Width is inferred from the type_hint or defaults to 4.
    pub(super) fn compile_load_f16(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let width = match type_hint {
            Some(Type::Vector { width, .. }) => *width as u32,
            _ => 4,
        };
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx = self.compile_expr(&args[1], function)?.into_int_value();
        let f16_ty = self.context.f16_type();
        let vec_ty = f16_ty.vec_type(width);
        let elem_ptr = unsafe {
            self.builder
                .build_gep(f16_ty, base, &[idx], "f16_gep")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
        };
        let load = self
            .builder
            .build_load(vec_ty, elem_ptr, "f16_load")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        if let BasicValueEnum::VectorValue(vec_val) = load {
            let load_inst = vec_val.as_instruction_value().ok_or_else(|| {
                CompileError::codegen_error("f16 vector load did not produce an instruction")
            })?;
            load_inst.set_alignment(2).map_err(|e| {
                CompileError::codegen_error(format!("failed to set alignment: {e}"))
            })?;
        }
        Ok(load)
    }

    /// Native f16 vector store: store `<N x half>` to a `*mut f16` pointer.
    pub(super) fn compile_store_f16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx = self.compile_expr(&args[1], function)?.into_int_value();
        let val = self.compile_expr(&args[2], function)?.into_vector_value();
        let f16_ty = self.context.f16_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(f16_ty, base, &[idx], "f16_store_gep")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
        };
        let store_inst = self
            .builder
            .build_store(elem_ptr, val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        store_inst.set_alignment(2).map_err(|e| {
            CompileError::codegen_error(format!("failed to set store alignment: {e}"))
        })?;
        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    /// Native f16 reductions: `llvm.vector.reduce.{fadd,fmax,fmin}.v{N}f16`.
    /// With +fullfp16 the backend emits `faddv h0, v0.8h` / `fmaxv` / `fminv` directly.
    pub(super) fn compile_reduce_f16(
        &mut self,
        args: &[Expr],
        name: &str,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let width = v.get_type().get_size();
        let f16_ty = self.context.f16_type();
        let vec_ty = f16_ty.vec_type(width);

        let intrinsic_name = match name {
            "reduce_add" | "reduce_add_fast" => format!("llvm.vector.reduce.fadd.v{width}f16"),
            "reduce_max" => format!("llvm.vector.reduce.fmax.v{width}f16"),
            "reduce_min" => format!("llvm.vector.reduce.fmin.v{width}f16"),
            _ => {
                return Err(CompileError::codegen_error(format!(
                    "unsupported f16 reduction: {name}"
                )));
            }
        };

        if name.starts_with("reduce_add") {
            // fadd reduction takes a start (seed) value
            let zero = f16_ty.const_zero();
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| {
                    let fn_ty = f16_ty.fn_type(&[f16_ty.into(), vec_ty.into()], false);
                    self.module.add_function(&intrinsic_name, fn_ty, None)
                });
            let result = self
                .builder
                .build_call(intrinsic, &[zero.into(), v.into()], "reduce_add_f16")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("reduce_f16 returned no value"))?;
            Ok(result)
        } else {
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| {
                    let fn_ty = f16_ty.fn_type(&[vec_ty.into()], false);
                    self.module.add_function(&intrinsic_name, fn_ty, None)
                });
            let result = self
                .builder
                .build_call(intrinsic, &[v.into()], "reduce_minmax_f16")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("reduce_f16 returned no value"))?;
            Ok(result)
        }
    }

    /// Native f16 fused multiply-add: `llvm.fma.v{N}f16(a, b, c)`.
    /// With +fullfp16 the backend emits `fmla v.8h, v.8h, v.8h` directly.
    pub(super) fn compile_fma_f16(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let c = self.compile_expr(&args[2], function)?.into_vector_value();
        let width = a.get_type().get_size();
        let f16_ty = self.context.f16_type();
        let vec_ty = f16_ty.vec_type(width);
        let name = format!("llvm.fma.v{width}f16");
        let intrinsic = self.module.get_function(&name).unwrap_or_else(|| {
            let fn_ty = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into(), vec_ty.into()], false);
            self.module.add_function(&name, fn_ty, None)
        });
        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into(), c.into()], "fma_f16")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("fma_f16 returned no value"))?;
        Ok(result)
    }
}
