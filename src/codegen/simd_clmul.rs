use inkwell::values::{BasicValueEnum, FunctionValue, VectorValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

/// v1.17.0 carry-less multiply of one 64-bit lane of each operand into a
/// 128-bit product. Native on both architectures behind a feature flag:
/// x86 `pclmulqdq` (`--pclmul`; not part of x86-64-v3), ARM `pmull` /
/// `pmull2` (`--aes`). Without the flag it is a compile error, never an
/// emulation.
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_clmul_intrinsic(name: &str) -> bool {
        matches!(name, "clmul_lo_u64x2" | "clmul_hi_u64x2")
    }

    pub(super) fn compile_clmul_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let lane = u64::from(name == "clmul_hi_u64x2");
        if self.is_arm && !self.aes {
            return Err(CompileError::codegen_error(format!(
                "{name} lowers to PMULL{}, which needs the crypto extension; compile with --aes",
                if lane == 1 { "2" } else { "" }
            )));
        }
        if !self.is_arm && !self.pclmul {
            return Err(CompileError::codegen_error(format!(
                "{name} lowers to PCLMULQDQ, which is not part of x86-64-v3; compile with --pclmul"
            )));
        }
        let a = self.compile_expr(&args[0], function)?.into_vector_value();
        let b = self.compile_expr(&args[1], function)?.into_vector_value();
        if self.is_arm {
            self.pmull(a, b, lane, name)
        } else {
            self.pclmulqdq(a, b, lane, name)
        }
    }

    /// x86: `llvm.x86.pclmulqdq(a, b, imm)`, imm 0x00 (lanes 0 x 0) or
    /// 0x11 (lanes 1 x 1).
    fn pclmulqdq(
        &mut self,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        lane: u64,
        name: &str,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let v2i64 = self.context.i64_type().vec_type(2);
        let i8t = self.context.i8_type();
        let f = self
            .module
            .get_function("llvm.x86.pclmulqdq")
            .unwrap_or_else(|| {
                let ty = v2i64.fn_type(&[v2i64.into(), v2i64.into(), i8t.into()], false);
                self.module.add_function("llvm.x86.pclmulqdq", ty, None)
            });
        let imm = i8t.const_int(lane * 0x11, false);
        self.builder
            .build_call(f, &[a.into(), b.into(), imm.into()], name)
            .map_err(|e| CompileError::codegen_error(e.to_string()))?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error(format!("{name} did not return a value")))
    }

    /// ARM: `llvm.aarch64.neon.pmull64(a[lane], b[lane])`, a 128-bit
    /// product as <16 x i8>, viewed as u64x2 (LLVM folds the lane 1
    /// extracts into `pmull2`).
    fn pmull(
        &mut self,
        a: VectorValue<'ctx>,
        b: VectorValue<'ctx>,
        lane: u64,
        name: &str,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let i64t = self.context.i64_type();
        let v16i8 = self.context.i8_type().vec_type(16);
        let f = self
            .module
            .get_function("llvm.aarch64.neon.pmull64")
            .unwrap_or_else(|| {
                let ty = v16i8.fn_type(&[i64t.into(), i64t.into()], false);
                self.module
                    .add_function("llvm.aarch64.neon.pmull64", ty, None)
            });
        let idx = self.context.i32_type().const_int(lane, false);
        let err = |e: inkwell::builder::BuilderError| CompileError::codegen_error(e.to_string());
        let x = self
            .builder
            .build_extract_element(a, idx, "clmul_a")
            .map_err(err)?;
        let y = self
            .builder
            .build_extract_element(b, idx, "clmul_b")
            .map_err(err)?;
        let p = self
            .builder
            .build_call(f, &[x.into(), y.into()], name)
            .map_err(err)?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error(format!("{name} did not return a value")))?;
        self.builder
            .build_bit_cast(p, i64t.vec_type(2), name)
            .map_err(err)
    }
}
