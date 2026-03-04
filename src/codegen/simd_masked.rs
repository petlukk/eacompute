use inkwell::IntPredicate;
use inkwell::types::{BasicTypeEnum, VectorType};
use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Builds a `<width x i1>` mask where lane `i` is true iff `i < count_val`.
    fn build_count_mask(
        &mut self,
        count_val: inkwell::values::IntValue<'ctx>,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let i32_ty = self.context.i32_type();
        let lane_indices: Vec<_> = (0..width)
            .map(|i| i32_ty.const_int(i as u64, false))
            .collect();
        let indices_vec = VectorType::const_vector(&lane_indices);

        let count_i32 = self
            .builder
            .build_int_z_extend_or_bit_cast(count_val, i32_ty, "count_i32")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let count_splat = self.build_splat(count_i32.into(), width)?;

        let mask = self
            .builder
            .build_int_compare(IntPredicate::ULT, indices_vec, count_splat, "mask")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(mask)
    }

    pub(super) fn compile_load_masked(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let count_val = self.compile_expr(&args[2], function)?.into_int_value();

        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();

        // GEP to offset position (same as compile_load)
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "masked_load_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Build mask: lane_indices < count_splat -> <N x i1>
        let mask = self.build_count_mask(count_val, width)?;

        // Build zero passthrough vector
        let passthrough = vec_ty.const_zero();

        // Element alignment
        let element_alignment = self.element_alignment(elem_ty);

        // Declare @llvm.masked.load.vNTy.p0
        let i32_ty = self.context.i32_type();
        let suffix = self.llvm_vector_type_suffix(vec_ty);
        let intrinsic_name = format!("llvm.masked.load.{suffix}");
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = vec_ty.fn_type(
            &[
                ptr_type.into(),
                i32_ty.into(),
                mask_ty.into(),
                vec_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let align_val = i32_ty.const_int(element_alignment as u64, false);

        let result = self
            .builder
            .build_call(
                intrinsic,
                &[
                    elem_ptr.into(),
                    align_val.into(),
                    mask.into(),
                    passthrough.into(),
                ],
                "masked_load",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("masked load did not return a value"))?;

        Ok(result)
    }

    pub(super) fn compile_store_masked(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();
        let count_val = self.compile_expr(&args[3], function)?.into_int_value();

        let vec_ty = vec_val.get_type();
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();

        // GEP to offset position (same as compile_store)
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "masked_store_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        // Build mask: lane_indices < count_splat -> <N x i1>
        let mask = self.build_count_mask(count_val, width)?;

        // Element alignment
        let element_alignment = self.element_alignment(elem_ty);

        // Declare @llvm.masked.store.vNTy.p0
        let i32_ty = self.context.i32_type();
        let suffix = self.llvm_vector_type_suffix(vec_ty);
        let intrinsic_name = format!("llvm.masked.store.{suffix}");
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = self.context.void_type().fn_type(
            &[
                vec_ty.into(),
                ptr_type.into(),
                i32_ty.into(),
                mask_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let align_val = i32_ty.const_int(element_alignment as u64, false);

        self.builder
            .build_call(
                intrinsic,
                &[
                    vec_val.into(),
                    elem_ptr.into(),
                    align_val.into(),
                    mask.into(),
                ],
                "",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    fn build_gather_ptrs(
        &mut self,
        base: inkwell::values::PointerValue<'ctx>,
        indices: VectorValue<'ctx>,
        elem_ty: BasicTypeEnum<'ctx>,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let width = indices.get_type().get_size();
        let ptr_ty = self.context.ptr_type(inkwell::AddressSpace::default());
        let mut vec = ptr_ty.vec_type(width).get_undef();
        for i in 0..width {
            let lane = self.context.i32_type().const_int(i as u64, false);
            let idx = self
                .builder
                .build_extract_element(indices, lane, "idx")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            let gep = unsafe {
                self.builder
                    .build_gep(elem_ty, base, &[idx.into_int_value()], "gep")
            }
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
            vec = self
                .builder
                .build_insert_element(vec, gep, lane, "ins")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }
        Ok(vec)
    }

    pub(super) fn compile_gather(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "gather has no NEON equivalent; use a scalar loop on ARM",
            ));
        }
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();
        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let width = indices.get_type().get_size();
        let vec_ty = match elem_ty {
            BasicTypeEnum::FloatType(ft) => ft.vec_type(width),
            BasicTypeEnum::IntType(it) => it.vec_type(width),
            _ => {
                return Err(CompileError::codegen_error(
                    "unsupported gather element type",
                ));
            }
        };
        let ptr_vec = self.build_gather_ptrs(base, indices, elem_ty)?;
        let align = self.element_alignment(elem_ty);
        let i32_ty = self.context.i32_type();
        let true_val = self.context.bool_type().const_int(1, false);
        let mask = VectorType::const_vector(&vec![true_val; width as usize]);
        let passthrough = vec_ty.const_zero();
        let (w, en) = self.vector_type_parts(vec_ty);
        let name = format!("llvm.masked.gather.v{w}{en}.v{w}p0");
        let ptr_vec_ty = self
            .context
            .ptr_type(inkwell::AddressSpace::default())
            .vec_type(width);
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = vec_ty.fn_type(
            &[
                ptr_vec_ty.into(),
                i32_ty.into(),
                mask_ty.into(),
                vec_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| self.module.add_function(&name, fn_type, None));
        let result = self
            .builder
            .build_call(
                intrinsic,
                &[
                    ptr_vec.into(),
                    i32_ty.const_int(align as u64, false).into(),
                    mask.into(),
                    passthrough.into(),
                ],
                "gather",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .left()
            .ok_or_else(|| CompileError::codegen_error("gather did not return a value"))?;
        Ok(result)
    }

    pub(super) fn compile_scatter(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(
                "scatter has no NEON equivalent; use a scalar loop on ARM",
            ));
        }
        if !self.avx512 {
            return Err(CompileError::codegen_error(
                "scatter requires AVX-512 for hardware support; compile with --avx512 or write a scalar loop explicitly",
            ));
        }
        let base = self.compile_expr(&args[0], function)?.into_pointer_value();
        let indices = self.compile_expr(&args[1], function)?.into_vector_value();
        let values = self.compile_expr(&args[2], function)?.into_vector_value();
        let vec_ty = values.get_type();
        let elem_ty = vec_ty.get_element_type();
        let width = vec_ty.get_size();
        let ptr_vec = self.build_gather_ptrs(base, indices, elem_ty)?;
        let align = self.element_alignment(elem_ty);
        let i32_ty = self.context.i32_type();
        let true_val = self.context.bool_type().const_int(1, false);
        let mask = VectorType::const_vector(&vec![true_val; width as usize]);
        let (w, en) = self.vector_type_parts(vec_ty);
        let name = format!("llvm.masked.scatter.v{w}{en}.v{w}p0");
        let ptr_vec_ty = self
            .context
            .ptr_type(inkwell::AddressSpace::default())
            .vec_type(width);
        let mask_ty = self.context.bool_type().vec_type(width);
        let fn_type = self.context.void_type().fn_type(
            &[
                vec_ty.into(),
                ptr_vec_ty.into(),
                i32_ty.into(),
                mask_ty.into(),
            ],
            false,
        );
        let intrinsic = self
            .module
            .get_function(&name)
            .unwrap_or_else(|| self.module.add_function(&name, fn_type, None));
        self.builder
            .build_call(
                intrinsic,
                &[
                    values.into(),
                    ptr_vec.into(),
                    i32_ty.const_int(align as u64, false).into(),
                    mask.into(),
                ],
                "",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(BasicValueEnum::IntValue(i32_ty.const_int(0, false)))
    }
}
