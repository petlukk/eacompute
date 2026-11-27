use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

/// v1.21.0 `floor_f32x4` / `floor_f32x8` through `llvm.floor`: x86
/// `vroundps` (floor mode), ARM `frintm`; the tests check that no libm call
/// is left.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_float_intrinsic(name: &str) -> bool {
        matches!(name, "floor_f32x4" | "floor_f32x8")
    }

    pub(super) fn compile_float_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let width = v.get_type().get_size();
        let intrinsic_name = format!("llvm.floor.v{width}f32");
        let fn_type = v.get_type().fn_type(&[v.get_type().into()], false);
        let f = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));
        self.builder
            .build_call(f, &[v.into()], name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error(format!("{name} did not return a value")))
    }
}
