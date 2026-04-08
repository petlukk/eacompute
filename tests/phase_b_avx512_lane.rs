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

    #[test]
    fn lo128_i8x32_extract() {
        let source = r#"
export func f(a: i8x32) -> i8x16 {
    return lo128_i8x32(a)
}
"#;
        let ir = compile_to_ir(source, "lo128_i8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shuffle of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i8>"),
            "expected <16 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi128_i8x32_extract() {
        let source = r#"
export func f(a: i8x32) -> i8x16 {
    return hi128_i8x32(a)
}
"#;
        let ir = compile_to_ir(source, "hi128_i8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shuffle of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i8>"),
            "expected <16 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn lo128_u8x32_extract() {
        let source = r#"
export func f(a: u8x32) -> u8x16 {
    return lo128_u8x32(a)
}
"#;
        let ir = compile_to_ir(source, "lo128_u8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shuffle of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i8>"),
            "expected <16 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi128_u8x32_extract() {
        let source = r#"
export func f(a: u8x32) -> u8x16 {
    return hi128_u8x32(a)
}
"#;
        let ir = compile_to_ir(source, "hi128_u8x32");
        assert!(
            ir.contains("shufflevector <32 x i8>"),
            "expected shuffle of <32 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i8>"),
            "expected <16 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn lo256_i8x64_extract() {
        let source = r#"
export func f(a: i8x64) -> i8x32 {
    return lo256_i8x64(a)
}
"#;
        let ir = compile_to_ir(source, "lo256_i8x64");
        assert!(
            ir.contains("shufflevector <64 x i8>"),
            "expected shuffle of <64 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi256_i8x64_extract() {
        let source = r#"
export func f(a: i8x64) -> i8x32 {
    return hi256_i8x64(a)
}
"#;
        let ir = compile_to_ir(source, "hi256_i8x64");
        assert!(
            ir.contains("shufflevector <64 x i8>"),
            "expected shuffle of <64 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn lo256_u8x64_extract() {
        let source = r#"
export func f(a: u8x64) -> u8x32 {
    return lo256_u8x64(a)
}
"#;
        let ir = compile_to_ir(source, "lo256_u8x64");
        assert!(
            ir.contains("shufflevector <64 x i8>"),
            "expected shuffle of <64 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi256_u8x64_extract() {
        let source = r#"
export func f(a: u8x64) -> u8x32 {
    return hi256_u8x64(a)
}
"#;
        let ir = compile_to_ir(source, "hi256_u8x64");
        assert!(
            ir.contains("shufflevector <64 x i8>"),
            "expected shuffle of <64 x i8>, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i8>"),
            "expected <32 x i8> result, got:\n{ir}"
        );
    }

    #[test]
    fn lo256_i32x16_extract() {
        let source = r#"
export func f(a: i32x16) -> i32x8 {
    return lo256_i32x16(a)
}
"#;
        let ir = compile_to_ir(source, "lo256_i32x16");
        assert!(
            ir.contains("shufflevector <16 x i32>"),
            "expected shuffle of <16 x i32>, got:\n{ir}"
        );
        assert!(
            ir.contains("<8 x i32>"),
            "expected <8 x i32> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi256_i32x16_extract() {
        let source = r#"
export func f(a: i32x16) -> i32x8 {
    return hi256_i32x16(a)
}
"#;
        let ir = compile_to_ir(source, "hi256_i32x16");
        assert!(
            ir.contains("shufflevector <16 x i32>"),
            "expected shuffle of <16 x i32>, got:\n{ir}"
        );
        assert!(
            ir.contains("<8 x i32>"),
            "expected <8 x i32> result, got:\n{ir}"
        );
    }

    #[test]
    fn lo256_f32x16_extract() {
        let source = r#"
export func f(a: f32x16) -> f32x8 {
    return lo256_f32x16(a)
}
"#;
        let ir = compile_to_ir(source, "lo256_f32x16");
        assert!(
            ir.contains("shufflevector <16 x float>"),
            "expected shuffle of <16 x float>, got:\n{ir}"
        );
        assert!(
            ir.contains("<8 x float>"),
            "expected <8 x float> result, got:\n{ir}"
        );
    }

    #[test]
    fn hi256_f32x16_extract() {
        let source = r#"
export func f(a: f32x16) -> f32x8 {
    return hi256_f32x16(a)
}
"#;
        let ir = compile_to_ir(source, "hi256_f32x16");
        assert!(
            ir.contains("shufflevector <16 x float>"),
            "expected shuffle of <16 x float>, got:\n{ir}"
        );
        assert!(
            ir.contains("<8 x float>"),
            "expected <8 x float> result, got:\n{ir}"
        );
    }

    #[test]
    fn bcast_even_pairs_i32x8_emits_shuffle_a0() {
        let source = r#"
export func f(a: i32x8) -> i32x8 {
    return bcast_even_pairs_i32x8(a)
}
"#;
        let ir = compile_to_ir(source, "bcast_even_pairs_i32x8");
        assert!(
            ir.contains("shufflevector <8 x i32>"),
            "expected shufflevector of <8 x i32>, got:\n{ir}"
        );
        // Mask: [0,0,2,2,4,4,6,6] — LLVM IR serializes as "i32 0, i32 0, i32 2, i32 2"
        assert!(
            ir.contains("i32 0, i32 0, i32 2, i32 2"),
            "expected even-pair broadcast mask (first sublane), got:\n{ir}"
        );
        assert!(
            ir.contains("i32 4, i32 4, i32 6, i32 6"),
            "expected even-pair broadcast mask (second sublane), got:\n{ir}"
        );
    }

    #[test]
    fn bcast_odd_pairs_i32x8_emits_shuffle_f5() {
        let source = r#"
export func f(a: i32x8) -> i32x8 {
    return bcast_odd_pairs_i32x8(a)
}
"#;
        let ir = compile_to_ir(source, "bcast_odd_pairs_i32x8");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 1, i32 1, i32 3, i32 3"),
            "expected odd-pair mask (first sublane), got:\n{ir}"
        );
        assert!(
            ir.contains("i32 5, i32 5, i32 7, i32 7"),
            "expected odd-pair mask (second sublane), got:\n{ir}"
        );
    }

    #[test]
    fn bcast_even_pairs_i32x16_emits_shuffle_a0() {
        let source = r#"
export func f(a: i32x16) -> i32x16 {
    return bcast_even_pairs_i32x16(a)
}
"#;
        let ir = compile_to_ir(source, "bcast_even_pairs_i32x16");
        assert!(ir.contains("shufflevector <16 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 0, i32 0, i32 2, i32 2"),
            "expected even-pair mask (first sublane), got:\n{ir}"
        );
        assert!(
            ir.contains("i32 12, i32 12, i32 14, i32 14"),
            "expected even-pair mask (fourth sublane), got:\n{ir}"
        );
    }

    #[test]
    fn bcast_odd_pairs_i32x16_emits_shuffle_f5() {
        let source = r#"
export func f(a: i32x16) -> i32x16 {
    return bcast_odd_pairs_i32x16(a)
}
"#;
        let ir = compile_to_ir(source, "bcast_odd_pairs_i32x16");
        assert!(ir.contains("shufflevector <16 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 1, i32 1, i32 3, i32 3"),
            "expected odd-pair mask (first sublane), got:\n{ir}"
        );
        assert!(
            ir.contains("i32 13, i32 13, i32 15, i32 15"),
            "expected odd-pair mask (fourth sublane), got:\n{ir}"
        );
    }
}
