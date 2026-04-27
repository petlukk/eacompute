//! Native f16 codegen — splat, load, store, fma, reductions.
//! Gated on `self.fp16` (set by `--fp16` flag, which appends
//! `+fullfp16` to LLVM target features).
//!
//! Element-wise arithmetic on f16 vectors does NOT live here: it
//! flows through the existing `compile_vector_binary` path because
//! LLVM's `is_float_type()` is true for `<N x half>`.

use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
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
}
