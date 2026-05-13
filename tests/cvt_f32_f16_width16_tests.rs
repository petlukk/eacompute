//! `cvt_f32_f16(f32x16) -> i16x16` — AVX-512 width-16 form. Closes the
//! asymmetry where `cvt_f16_f32` already accepted widths {4, 8, 16} but
//! `cvt_f32_f16` only {4, 8}.
//!
//! IR-level tests follow the existing AVX-512 test pattern in
//! `tests/phase_b_avx512_dotprod.rs` — compile to IR with `+avx512f` and
//! assert the lowering. Avoids depending on the CI runner having AVX-512.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode, compile_with_options};
    use tempfile::TempDir;

    fn avx512_opts() -> CompileOptions {
        CompileOptions {
            opt_level: 3,
            extra_features: "+avx512f".to_string(),
            ..CompileOptions::default()
        }
    }

    fn compile_to_ir(source: &str, name: &str) -> String {
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join(format!("{name}.ll"));
        compile_with_options(source, &ir_path, OutputMode::LlvmIr, &avx512_opts())
            .unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        std::fs::read_to_string(&ir_path).unwrap_or_default()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cvt_f32_f16_width16_compiles_and_lowers_to_fptrunc() {
        let ir = compile_to_ir(
            r#"
export func f(a: f32x16) -> i16x16 {
    return cvt_f32_f16(a)
}
"#,
            "cvt_f32_f16_w16",
        );
        // The codegen path is: fptrunc <16 x float> to <16 x half>, then
        // bitcast to <16 x i16>. The host x86 backend will emit vcvtps2ph
        // (zmm form) from this IR at the right -mattr settings; the IR
        // itself is what we control.
        assert!(
            ir.contains("fptrunc <16 x float>") && ir.contains("to <16 x half>"),
            "expected fptrunc to <16 x half> in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i16>"),
            "expected <16 x i16> return type in IR, got:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cvt_f16_f32_width16_still_works() {
        // Sanity: the partner direction was already at width 16; pin the
        // expected lowering alongside.
        let ir = compile_to_ir(
            r#"
export func f(a: i16x16) -> f32x16 {
    return cvt_f16_f32(a)
}
"#,
            "cvt_f16_f32_w16",
        );
        assert!(
            ir.contains("fpext <16 x half>") && ir.contains("to <16 x float>"),
            "expected fpext to <16 x float> in IR, got:\n{ir}"
        );
    }

    // --- Type-checker rejections ---

    fn try_compile(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &avx512_opts())
    }

    #[test]
    fn cvt_f32_f16_error_message_lists_width16() {
        // A non-supported width should fail with an error message that now
        // includes f32x16 (was: only "f32x4 or f32x8").
        let err = try_compile("export func f(a: i32x4) -> i16x4 { return cvt_f32_f16(a) }\n")
            .expect_err("i32x4 input should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("f32x4, f32x8 or f32x16"),
            "expected updated error message, got: {msg}"
        );
    }

    // --- ARM rejection: f32x16 itself isn't a valid type on ARM ---

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile_arm(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &arm_opts())
    }

    #[test]
    fn cvt_f32_f16_width16_rejected_on_arm() {
        // f32x16 is 512-bit; NEON max is 128-bit. Caught at the
        // vector-type validation site, not at the cvt intrinsic itself.
        let err = try_compile_arm("export func f(a: f32x16) -> i16x16 { return cvt_f32_f16(a) }\n")
            .expect_err("f32x16 input must fail on ARM");
        assert!(
            format!("{err}").contains("f32x16 requires AVX-512"),
            "expected AVX-512 narrowing hint, got: {err}"
        );
    }
}
