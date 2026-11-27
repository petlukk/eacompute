use inkwell::types::VectorType;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

/// v1.21.0 typed andnot: `a & !b`, one instruction on both targets (x86
/// `vpandn` with the operands swapped, ARM `bic`); the tests check the
/// disassembly, since the IR is `and(a, xor(b, -1))`.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_logic_intrinsic(name: &str) -> bool {
        matches!(name, "andnot_u8x16" | "andnot_u8x32" | "andnot_u64x2" | "andnot_u64x4")
    }

    pub(super) fn compile_logic_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let lanes = b.get_type().get_size() as usize;
        let elem = b.get_type().get_element_type().into_int_type();
        let ones = VectorType::const_vector(&vec![elem.const_all_ones(); lanes]);
        let not_b = self
            .builder
            .build_xor(b, ones, "not")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        let r = self
            .builder
            .build_and(a, not_b, name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        Ok(r.into())
    }
}
