//! `fence_nt()` — non-temporal store fence.
//!
//! Orders `stream_store` writes relative to each other and relative to
//! subsequent regular stores within the same kernel (store-store ordering
//! only). It does *not* order stores relative to subsequent loads — for a
//! write-then-read-back pattern, a full barrier (`mfence` / `dmb sy`) is
//! needed instead. Cross-thread visibility typically comes from the host-
//! side sync primitive after the kernel returns.
//!
//! Lowering:
//! - x86: `call void @llvm.x86.sse.sfence()` — emits a single `sfence`.
//! - aarch64: `call void @llvm.aarch64.dmb(i32 10)` — emits `dmb ishst`
//!   (inner-shareable store-store barrier, the tightest matching ARM
//!   barrier for NT-store ordering).
//!
//! These are explicit target intrinsics rather than the IR-level
//! `fence release`, because `fence release` lowers to `mfence` on x86
//! (heavier than needed) and `dmb ish` on aarch64 (also heavier).

use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn compile_fence_nt(
        &mut self,
        args: &[Expr],
        _function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        if !args.is_empty() {
            return Err(CompileError::codegen_error(
                "fence_nt takes 0 arguments",
            ));
        }

        if self.is_arm {
            // aarch64: llvm.aarch64.dmb(i32 10) -> dmb ishst
            let i32_type = self.context.i32_type();
            let fn_type = self.context.void_type().fn_type(&[i32_type.into()], false);
            let dmb_fn = self
                .module
                .get_function("llvm.aarch64.dmb")
                .unwrap_or_else(|| self.module.add_function("llvm.aarch64.dmb", fn_type, None));
            // 10 = ishst encoding per AArch64 architectural manual
            let ishst = i32_type.const_int(10, false);
            self.builder
                .build_call(dmb_fn, &[ishst.into()], "")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        } else {
            // x86: llvm.x86.sse.sfence() -> sfence
            let fn_type = self.context.void_type().fn_type(&[], false);
            let sfence_fn = self
                .module
                .get_function("llvm.x86.sse.sfence")
                .unwrap_or_else(|| {
                    self.module
                        .add_function("llvm.x86.sse.sfence", fn_type, None)
                });
            self.builder
                .build_call(sfence_fn, &[], "")
                .map_err(|e| CompileError::codegen_error(e.to_string()))?;
        }

        // fence_nt is void; return the same placeholder i32(0) as other
        // void-returning intrinsics like compile_store and compile_prefetch.
        Ok(BasicValueEnum::IntValue(
            self.context.i32_type().const_int(0, false),
        ))
    }
}
