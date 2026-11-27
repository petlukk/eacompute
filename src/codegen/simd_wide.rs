use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

/// v1.21.0 `widen_u8x16_u16x16`: `zext <16 x i8>` to `<16 x i16>`, x86
/// `vpmovzxbw ymm, xmm` (checked in the disassembly). ARM: error.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_wide_intrinsic(name: &str) -> bool {
        name == "widen_u8x16_u16x16"
    }

    pub(super) fn compile_wide_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(format!(
                "{name}: ARM has no 256-bit vectors; use widen_u8_u16 and widen_u8_u16_hi (two u16x8)"
            )));
        }
        let v = self.compile_expr(&args[0], function)?.into_vector_value();
        let target = self.context.i16_type().vec_type(16);
        let r = self
            .builder
            .build_int_z_extend(v, target, name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(r.into())
    }
}
