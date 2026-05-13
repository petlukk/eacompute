use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::{Expr, TypeAnnotation};
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_vector_literal(
        &mut self,
        elements: &[Expr],
        ty: &TypeAnnotation,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec_type = Self::resolve_annotation(ty);
        let elem_hint = match &vec_type {
            Type::Vector { elem, .. } => Some(elem.as_ref()),
            _ => None,
        };
        let llvm_vec_type = self.llvm_type(&vec_type).into_vector_type();
        let mut vec_val = llvm_vec_type.get_undef();

        for (i, elem) in elements.iter().enumerate() {
            let elem_val = self.compile_expr_typed(elem, elem_hint, function)?;
            let idx = self.context.i32_type().const_int(i as u64, false);
            vec_val = self
                .builder
                .build_insert_element(vec_val, elem_val, idx, "vec_ins")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }
        Ok(BasicValueEnum::VectorValue(vec_val))
    }

    pub(crate) fn compile_fma(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?;
        let b = self.compile_expr(&args[1], function)?;
        let c = self.compile_expr(&args[2], function)?;

        match a {
            BasicValueEnum::FloatValue(fa) => {
                let float_ty = fa.get_type();
                // Cast args to match first arg's type (FloatLiteral defaults to f64)
                let raw_b = b.into_float_value();
                let fb = if raw_b.get_type() != float_ty {
                    self.builder
                        .build_float_cast(raw_b, float_ty, "fma_cast_b")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                } else {
                    raw_b
                };
                let raw_c = c.into_float_value();
                let fc = if raw_c.get_type() != float_ty {
                    self.builder
                        .build_float_cast(raw_c, float_ty, "fma_cast_c")
                        .map_err(|e| CompileError::codegen_error(e.to_string()))?
                } else {
                    raw_c
                };
                let intrinsic_name = if float_ty == self.context.f32_type() {
                    "llvm.fma.f32"
                } else {
                    "llvm.fma.f64"
                };
                let fn_type =
                    float_ty.fn_type(&[float_ty.into(), float_ty.into(), float_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[fa.into(), fb.into(), fc.into()], "fma")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("fma did not return a value"))?;
                Ok(result)
            }
            BasicValueEnum::VectorValue(va) => {
                let vb = b.into_vector_value();
                let vc = c.into_vector_value();
                let vec_ty = va.get_type();
                let intrinsic_name = self.llvm_vector_intrinsic_name("llvm.fma", vec_ty);
                let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into(), vec_ty.into()], false);
                let intrinsic = self
                    .module
                    .get_function(&intrinsic_name)
                    .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
                let result = self
                    .builder
                    .build_call(intrinsic, &[va.into(), vb.into(), vc.into()], "fma")
                    .map_err(|e| CompileError::codegen_error(e.to_string()))?
                    .try_as_basic_value()
                    .basic()
                    .ok_or_else(|| CompileError::codegen_error("fma did not return a value"))?;
                Ok(result)
            }
            _ => Err(CompileError::codegen_error(
                "fma: unexpected argument type (expected float or float vector)",
            )),
        }
    }

    pub(crate) fn compile_reduce(
        &mut self,
        args: &[Expr],
        op: &str,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_unsigned_elem = self.infer_unsigned_elem_from_arg(&args[0]);
        let vec = self.compile_expr(&args[0], function)?.into_vector_value();
        let vec_ty = vec.get_type();
        let elem_ty = vec_ty.get_element_type();
        let is_float = elem_ty.is_float_type();

        let (intrinsic_base, needs_start_value, set_reassoc) =
            match (op, is_float, is_unsigned_elem) {
                ("reduce_add_fast", true, _) => ("llvm.vector.reduce.fadd", true, true),
                ("reduce_add", true, _) => ("llvm.vector.reduce.fadd", true, false),
                ("reduce_add", false, _) => ("llvm.vector.reduce.add", false, false),
                ("reduce_max", true, _) => ("llvm.vector.reduce.fmax", false, false),
                ("reduce_max", false, true) => ("llvm.vector.reduce.umax", false, false),
                ("reduce_max", false, false) => ("llvm.vector.reduce.smax", false, false),
                ("reduce_min", true, _) => ("llvm.vector.reduce.fmin", false, false),
                ("reduce_min", false, true) => ("llvm.vector.reduce.umin", false, false),
                ("reduce_min", false, false) => ("llvm.vector.reduce.smin", false, false),
                _ => {
                    return Err(CompileError::codegen_error(format!(
                        "unknown reduction {op}"
                    )));
                }
            };

        let intrinsic_name = self.llvm_vector_intrinsic_name(intrinsic_base, vec_ty);

        if needs_start_value {
            let zero = elem_ty.into_float_type().const_float(0.0);
            let fn_type = elem_ty
                .into_float_type()
                .fn_type(&[elem_ty.into_float_type().into(), vec_ty.into()], false);
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
            let call = self
                .builder
                .build_call(intrinsic, &[zero.into(), vec.into()], "reduce")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            if set_reassoc {
                use inkwell::values::AsValueRef;
                unsafe {
                    inkwell::llvm_sys::core::LLVMSetFastMathFlags(
                        call.as_value_ref(),
                        inkwell::llvm_sys::LLVMFastMathAllowReassoc,
                    );
                }
            }
            let result = call
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("reduce did not return a value"))?;
            Ok(result)
        } else {
            let fn_type = if is_float {
                elem_ty.into_float_type().fn_type(&[vec_ty.into()], false)
            } else {
                elem_ty.into_int_type().fn_type(&[vec_ty.into()], false)
            };
            let intrinsic = self
                .module
                .get_function(&intrinsic_name)
                .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
            let result = self
                .builder
                .build_call(intrinsic, &[vec.into()], "reduce")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?
                .try_as_basic_value()
                .basic()
                .ok_or_else(|| CompileError::codegen_error("reduce did not return a value"))?;
            Ok(result)
        }
    }

    /// Returns `(width, elem_name)` for a vector type, e.g. `(8, "f32")`.
    pub(crate) fn vector_type_parts(&self, vec_ty: VectorType<'ctx>) -> (u32, &'static str) {
        let width = vec_ty.get_size();
        let elem = vec_ty.get_element_type();
        let elem_name = if elem.is_float_type() {
            let ft = elem.into_float_type();
            if ft == self.context.f32_type() {
                "f32"
            } else {
                "f64"
            }
        } else {
            let it = elem.into_int_type();
            match it.get_bit_width() {
                8 => "i8",
                16 => "i16",
                32 => "i32",
                _ => "i64",
            }
        };
        (width, elem_name)
    }

    pub(crate) fn llvm_vector_intrinsic_name(
        &self,
        base: &str,
        vec_ty: VectorType<'ctx>,
    ) -> String {
        let (width, elem_name) = self.vector_type_parts(vec_ty);
        format!("{base}.v{width}{elem_name}")
    }

    /// Formats a vector type suffix with opaque pointer for masked intrinsics.
    /// E.g. `"v8f32.p0"`, `"v4i32.p0"`.
    pub(crate) fn llvm_vector_type_suffix(&self, vec_ty: VectorType<'ctx>) -> String {
        let (width, elem_name) = self.vector_type_parts(vec_ty);
        format!("v{width}{elem_name}.p0")
    }

    pub(crate) fn compile_prefetch(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if args.len() != 2 {
            return Err(CompileError::codegen_error(
                "prefetch requires 2 arguments: (ptr, offset)",
            ));
        }
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let offset_val = self.compile_expr(&args[1], function)?.into_int_value();

        let elem_llvm = if let Expr::Variable(name, _) = &args[0] {
            if let Some((_, Type::Pointer { inner, .. })) = self.variables.get(name) {
                self.llvm_type(inner)
            } else {
                self.context.i8_type().into()
            }
        } else {
            self.context.i8_type().into()
        };

        let gep = unsafe {
            self.builder
                .build_gep(elem_llvm, ptr_val, &[offset_val], "prefetch_ptr")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let i32_type = self.context.i32_type();
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let prefetch_type = self.context.void_type().fn_type(
            &[
                ptr_type.into(),
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
            ],
            false,
        );
        let prefetch_fn = self
            .module
            .get_function("llvm.prefetch.p0")
            .unwrap_or_else(|| {
                self.module
                    .add_function("llvm.prefetch.p0", prefetch_type, None)
            });

        self.builder
            .build_call(
                prefetch_fn,
                &[
                    gep.into(),
                    i32_type.const_int(0, false).into(),
                    i32_type.const_int(3, false).into(),
                    i32_type.const_int(1, false).into(),
                ],
                "",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    pub(crate) fn compile_shuffle(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let vec = self.compile_expr(&args[0], function)?.into_vector_value();

        let indices = match &args[1] {
            Expr::ArrayLiteral(elems, _) => elems
                .iter()
                .map(|e| match e {
                    Expr::Literal(crate::ast::Literal::Integer(n), _) => {
                        Ok(self.context.i32_type().const_int(*n as u64, false))
                    }
                    _ => Err(CompileError::codegen_error(
                        "shuffle mask must contain only integer literals",
                    )),
                })
                .collect::<crate::error::Result<Vec<_>>>()?,
            _ => {
                return Err(CompileError::codegen_error(
                    "shuffle requires array literal",
                ));
            }
        };

        let mask = VectorType::const_vector(&indices);
        let result = self
            .builder
            .build_shuffle_vector(vec, vec, mask, "shuffle")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::VectorValue(result))
    }
}
