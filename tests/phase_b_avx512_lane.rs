//! AVX-512 lane-movement intrinsic tests: concat, lo/hi extractors,
//! per-sublane 32-bit broadcasts. IR-based verification following the
//! pattern established in tests/phase_b_avx512_dotprod.rs.
//!
//! Tests added incrementally by Tasks 2-4 of the lane intrinsics plan.

#[cfg(all(feature = "llvm", target_arch = "x86_64"))]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode, compile_with_options};
    use tempfile::TempDir;

    fn avx512_opts() -> CompileOptions {
        CompileOptions {
            opt_level: 0,
            extra_features: "+avx512f,+avx512vl,+avx512bw".to_string(),
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
    fn concat_i8x32_ymm_to_zmm() {
        let source = r#"
export func f(a: i8x32, b: i8x32) -> i8x64 {
    return concat_i8x32(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_i8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shufflevector of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<64 x i8>"),
            "expected <64 x i8> result type, got:\n{ir}"
        );
    }

    #[test]
    fn concat_u8x32_ymm_to_zmm() {
        let source = r#"
export func f(a: u8x32, b: u8x32) -> u8x64 {
    return concat_u8x32(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_u8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shuffle of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<64 x i8>"),
            "expected <64 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn concat_i32x8_ymm_to_zmm() {
        let source = r#"
export func f(a: i32x8, b: i32x8) -> i32x16 {
    return concat_i32x8(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_i32x8");
        assert!(
            ir.contains("shufflevector <8 x i32>"),
            "expected shuffle of <8 x i32>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i32>"),
            "expected <16 x i32> result, got:\n{ir}"
        );
    }

    #[test]
    fn concat_f32x8_ymm_to_zmm() {
        let source = r#"
export func f(a: f32x8, b: f32x8) -> f32x16 {
    return concat_f32x8(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_f32x8");
        assert!(
            ir.contains("shufflevector <8 x float>"),
            "expected shuffle of <8 x float>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x float>"),
            "expected <16 x float> result, got:\n{ir}"
        );
    }

    #[test]
    fn concat_i8x16_xmm_to_ymm() {
        let source = r#"
export func f(a: i8x16, b: i8x16) -> i8x32 {
    return concat_i8x16(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_i8x16");
        assert!(
            ir.contains("shufflevector <16 x i8>"),
            "expected shuffle of <16 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn concat_u8x16_xmm_to_ymm() {
        let source = r#"
export func f(a: u8x16, b: u8x16) -> u8x32 {
    return concat_u8x16(a, b)
}
"#;
        let ir = compile_to_ir(source, "concat_u8x16");
        assert!(
            ir.contains("shufflevector <16 x i8>"),
            "expected shuffle of <16 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }
}
