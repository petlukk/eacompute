use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    /// sat_add(a, b) -> same type. Cross-platform saturating addition.
    ///
    /// Lowers to LLVM's target-independent `llvm.{s,u}add.sat` intrinsic.
    /// Backend emits `padds`/`paddus` on x86 and `sqadd`/`uqadd` on ARM.
    /// The platform-specific `llvm.x86.sse2.padds.*` names are deprecated
    /// in LLVM 7+; using them produces an unresolved symbol at link time
    /// (caught by tests/monomorphic_sat_diff_tests.rs).
    pub(super) fn compile_sat_add(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_sat_op(args, function, "add", "sat_add")
    }

    /// sat_sub(a, b) -> same type. Cross-platform saturating subtraction.
    /// See [`compile_sat_add`] for the lowering rationale.
    pub(super) fn compile_sat_sub(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        self.compile_sat_op(args, function, "sub", "sat_sub")
    }

    fn compile_sat_op(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
        op: &str, // "add" or "sub"
        call_name: &str,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let is_unsigned = self.infer_unsigned_elem_from_arg(&args[0]);
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let vec_ty = a.get_type();
        let (width, elem_name) = self.vector_type_parts(vec_ty);

        let sign = if is_unsigned { "u" } else { "s" };
        let intrinsic_name = format!("llvm.{sign}{op}.sat.v{width}{elem_name}");

        let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], call_name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| {
                CompileError::codegen_error(format!("{call_name} did not return a value"))
            })?;

        Ok(result)
    }

    /// abs_diff(a, b) -> same type. ARM NEON absolute difference.
    /// Lowers to llvm.aarch64.neon.sabd / uabd.
    pub(super) fn compile_abs_diff(
        &mut self,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !self.is_arm {
            return Err(CompileError::codegen_error(
                "abs_diff is ARM-only (NEON); no x86 equivalent — use max(a .- b, b .- a) explicitly",
            ));
        }
        let is_unsigned = self.infer_unsigned_from_arg_full(&args[0]);
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        let vec_ty = a.get_type();
        let (width, elem_name) = self.vector_type_parts(vec_ty);

        let sign = if is_unsigned { "u" } else { "s" };
        let intrinsic_name = format!("llvm.aarch64.neon.{sign}abd.v{width}{elem_name}");

        let fn_type = vec_ty.fn_type(&[vec_ty.into(), vec_ty.into()], false);
        let intrinsic = self
            .module
            .get_function(&intrinsic_name)
            .unwrap_or_else(|| self.module.add_function(&intrinsic_name, fn_type, None));

        let result = self
            .builder
            .build_call(intrinsic, &[a.into(), b.into()], "abs_diff")
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error("abs_diff did not return a value"))?;

        Ok(result)
    }

    /// Returns true if the first argument is a vector with an unsigned element type.
    /// Broader than infer_unsigned_elem_from_arg (covers U8, U16, U32, U64).
    fn infer_unsigned_from_arg_full(&self, arg: &Expr) -> bool {
        use crate::ast::Expr;
        use crate::typeck::Type;
        if let Expr::Variable(name, _) = arg
            && let Some((_, Type::Vector { elem, .. })) = self.variables.get(name)
        {
            return matches!(elem.as_ref(), Type::U8 | Type::U16 | Type::U32 | Type::U64);
        }
        false
    }
}
