use inkwell::types::{BasicTypeEnum, VectorType};
use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// Returns the natural alignment in bytes for an element type.
    pub(super) fn element_alignment(&self, elem_ty: BasicTypeEnum<'ctx>) -> u32 {
        match elem_ty {
            BasicTypeEnum::FloatType(ft) => {
                if ft == self.context.f64_type() {
                    8
                } else if ft == self.context.f16_type() {
                    2
                } else {
                    4
                }
            }
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 8 => 1,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 16 => 2,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 32 => 4,
            BasicTypeEnum::IntType(it) if it.get_bit_width() == 64 => 8,
            _ => 1,
        }
    }

    pub(super) fn infer_load_vector_type(
        &self,
        ptr_arg: &Expr,
        type_hint: Option<&Type>,
    ) -> VectorType<'ctx> {
        if let Some(Type::Vector { elem, width }) = type_hint {
            let elem_llvm = self.llvm_type(elem);
            match elem_llvm {
                BasicTypeEnum::FloatType(ft) => return ft.vec_type(*width as u32),
                BasicTypeEnum::IntType(it) => return it.vec_type(*width as u32),
                _ => {}
            }
        }
        if let Expr::Variable(name, _) = ptr_arg
            && let Some((_, Type::Pointer { inner, .. })) = self.variables.get(name)
        {
            let elem_llvm = self.llvm_type(inner);
            match elem_llvm {
                BasicTypeEnum::FloatType(ft) => return ft.vec_type(4),
                BasicTypeEnum::IntType(it) => return it.vec_type(4),
                _ => {}
            }
        }
        self.context.f32_type().vec_type(4)
    }

    pub(super) fn compile_load(
        &mut self,
        args: &[Expr],
        type_hint: Option<&Type>,
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();

        let vec_ty = self.infer_load_vector_type(&args[0], type_hint);
        let elem_ty = vec_ty.get_element_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "load_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let val = self
            .builder
            .build_load(vec_ty, elem_ptr, "vec_load")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let element_alignment = self.element_alignment(vec_ty.get_element_type());

        if let BasicValueEnum::VectorValue(vec_val) = val {
            let load_inst = vec_val.as_instruction_value().ok_or_else(|| {
                CompileError::codegen_error("vector load did not produce an instruction")
            })?;
            load_inst.set_alignment(element_alignment).map_err(|e| {
                CompileError::codegen_error(format!("failed to set alignment: {e}"))
            })?;
        }

        Ok(val)
    }

    pub(super) fn compile_store(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();

        let vec_ty = vec_val.get_type();
        let elem_ty = vec_ty.get_element_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "store_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let store_inst = self
            .builder
            .build_store(elem_ptr, vec_val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let element_alignment = self.element_alignment(vec_ty.get_element_type());

        store_inst.set_alignment(element_alignment).map_err(|e| {
            CompileError::codegen_error(format!("failed to set store alignment: {e}"))
        })?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    pub(super) fn compile_stream_store(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let ptr_val = self.compile_expr(&args[0], function)?.into_pointer_value();
        let idx_val = self.compile_expr(&args[1], function)?.into_int_value();
        let vec_val = self.compile_expr(&args[2], function)?.into_vector_value();

        let vec_ty = vec_val.get_type();
        let elem_ty = vec_ty.get_element_type();
        let elem_ptr = unsafe {
            self.builder
                .build_gep(elem_ty, ptr_val, &[idx_val], "nt_store_gep")
        }
        .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let store_inst = self
            .builder
            .build_store(elem_ptr, vec_val)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let element_alignment = self.element_alignment(vec_ty.get_element_type());
        store_inst.set_alignment(element_alignment).map_err(|e| {
            CompileError::codegen_error(format!("failed to set store alignment: {e}"))
        })?;

        let kind_id = self.context.get_kind_id("nontemporal");
        let i32_one = self.context.i32_type().const_int(1, false);
        let md_node = self.context.metadata_node(&[i32_one.into()]);
        store_inst.set_metadata(md_node, kind_id).map_err(|e| {
            CompileError::codegen_error(format!("failed to set nontemporal metadata: {e}"))
        })?;

        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }

    pub(crate) fn build_splat(
        &self,
        val: BasicValueEnum<'ctx>,
        width: u32,
    ) -> crate::error::Result<VectorValue<'ctx>> {
        let first_ty = val.get_type();
        let vec_ty: BasicTypeEnum = match first_ty {
            BasicTypeEnum::IntType(t) => t.vec_type(width).into(),
            BasicTypeEnum::FloatType(t) => t.vec_type(width).into(),
            BasicTypeEnum::PointerType(t) => t.vec_type(width).into(),
            _ => {
                return Err(CompileError::codegen_error(format!(
                    "cannot splat type {first_ty:?}"
                )));
            }
        };

        let vec_ty_vector = vec_ty.into_vector_type();
        let undef = vec_ty_vector.get_undef();

        let vec0 = self
            .builder
            .build_insert_element(
                undef,
                val,
                self.context.i32_type().const_int(0, false),
                "splat_ins",
            )
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        let zero = self.context.i32_type().const_int(0, false);
        let mask: Vec<_> = (0..width).map(|_| zero).collect();
        let mask_val = VectorType::const_vector(&mask);

        let res = self
            .builder
            .build_shuffle_vector(vec0, vec0, mask_val, "splat_shuf")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;

        Ok(res)
    }
}
