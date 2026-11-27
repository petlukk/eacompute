use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;
use crate::typeck::Type;

use super::CodeGenerator;

/// v1.21.0 typed masked loads and stores. For 32- and 64-bit lanes LLVM's
/// masked load/store is AVX2's own instruction (vpmaskmovd/q, vmaskmovps;
/// the tests check the disassembly). NEON has no masked load at all.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_maskmem_intrinsic(name: &str) -> bool {
        crate::typeck::intrinsics_maskmem::typed_masked(name).is_some()
    }

    pub(super) fn compile_maskmem_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if self.is_arm {
            return Err(CompileError::codegen_error(format!(
                "{name}: NEON has no masked load or store; copy the tail into a buffer padded \
                 to a whole vector and use plain load/store (see docs/migrations/v1.21.0.md)"
            )));
        }
        let (store, elem, width) =
            crate::typeck::intrinsics_maskmem::typed_masked(name).expect("checked by is_maskmem_intrinsic");
        if store {
            self.compile_store_masked(args, function, true)
        } else {
            let hint = Type::Vector { elem: Box::new(elem), width };
            self.compile_load_masked(args, Some(&hint), function, true)
        }
    }
}
