use inkwell::types::BasicTypeEnum;
use inkwell::values::{BasicValueEnum, FunctionValue};

use crate::ast::Expr;
use crate::error::CompileError;

use super::CodeGenerator;

/// v1.18.0 AES rounds. The two architectures split a round differently, so
/// each instruction is its own intrinsic and the other architecture's is a
/// compile error naming the local idiom, never an emulation: x86
/// `aesenc_u8x16` (`aesenc`), ARM `aese_u8x16` (`aese`) and `aesmc_u8x16`
/// (`aesmc`). Both need `--aes` (x86 AES-NI is not part of x86-64-v3).
impl<'ctx> CodeGenerator<'ctx> {
    pub(crate) fn is_aes_intrinsic(name: &str) -> bool {
        matches!(name, "aesenc_u8x16" | "aese_u8x16" | "aesmc_u8x16")
    }

    pub(super) fn compile_aes_intrinsic(
        &mut self,
        name: &str,
        args: &[Expr],
        function: FunctionValue<'ctx>,
    ) -> crate::error::Result<BasicValueEnum<'ctx>> {
        let x86 = name == "aesenc_u8x16";
        if x86 && self.is_arm {
            return Err(CompileError::codegen_error(
                "aesenc_u8x16 is x86-only (AES-NI aesenc); on AArch64 a round is aesmc_u8x16(aese_u8x16(x, zero)) .^ k".to_string(),
            ));
        }
        if !x86 && !self.is_arm {
            return Err(CompileError::codegen_error(format!(
                "{name} is AArch64-only ({}); on x86-64 a round is aesenc_u8x16(x, k)",
                &name[..name.len() - 6]
            )));
        }
        if !self.aes {
            let what = if self.is_arm {
                "the crypto extension"
            } else {
                "AES-NI, which is not part of x86-64-v3"
            };
            return Err(CompileError::codegen_error(format!(
                "{name} needs {what}; compile with --aes"
            )));
        }
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.compile_expr(a, function)?.into_vector_value());
        }
        let v16i8 = self.context.i8_type().vec_type(16);
        let v2i64 = self.context.i64_type().vec_type(2);
        let err = |e: inkwell::builder::BuilderError| CompileError::codegen_error(e.to_string());
        // x86's intrinsic works on <2 x i64>, ARM's on <16 x i8>.
        let (llvm_name, ty): (&str, BasicTypeEnum) = match name {
            "aesenc_u8x16" => ("llvm.x86.aesni.aesenc", v2i64.into()),
            "aese_u8x16" => ("llvm.aarch64.crypto.aese", v16i8.into()),
            _ => ("llvm.aarch64.crypto.aesmc", v16i8.into()),
        };
        let mut call_args = Vec::with_capacity(vals.len());
        for v in &vals {
            call_args.push(
                self.builder
                    .build_bit_cast(*v, ty, "aes_in")
                    .map_err(err)?
                    .into(),
            );
        }
        let f = self.module.get_function(llvm_name).unwrap_or_else(|| {
            let params = vec![ty.into(); vals.len()];
            let fn_ty = ty.into_vector_type().fn_type(&params, false);
            self.module.add_function(llvm_name, fn_ty, None)
        });
        let r = self
            .builder
            .build_call(f, &call_args, name)
            .map_err(err)?
            .try_as_basic_value()
            .basic()
            .ok_or_else(|| CompileError::codegen_error(format!("{name} did not return a value")))?;
        self.builder.build_bit_cast(r, v16i8, name).map_err(err)
    }
}
