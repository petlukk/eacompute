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

    // ── shuffle_i32 ──────────────────────────────────────────────

    #[test]
    fn shuffle_i32x8_imm_136() {
        // imm=136 = 0b10_00_10_00 → selectors [0,2,0,2] per sublane
        // This is _MM_SHUFFLE(2,0,2,0) — used in llama.cpp Q4K
        let source = r#"
export func f(v: i32x8) -> i32x8 {
    return shuffle_i32x8(v, 136)
}
"#;
        let ir = compile_to_ir(source, "shuffle_i32x8_136");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        // sublane 0: [0,2,0,2], sublane 1: [4,6,4,6]
        assert!(
            ir.contains("i32 0, i32 2, i32 0, i32 2, i32 4, i32 6, i32 4, i32 6"),
            "expected imm=136 mask, got:\n{ir}"
        );
    }

    #[test]
    fn shuffle_i32x16_imm_136() {
        let source = r#"
export func f(v: i32x16) -> i32x16 {
    return shuffle_i32x16(v, 136)
}
"#;
        let ir = compile_to_ir(source, "shuffle_i32x16_136");
        assert!(ir.contains("shufflevector <16 x i32>"), "got:\n{ir}");
        // First sublane [0,2,0,2], last sublane [12,14,12,14]
        assert!(
            ir.contains("i32 0, i32 2, i32 0, i32 2"),
            "expected first sublane, got:\n{ir}"
        );
        assert!(
            ir.contains("i32 12, i32 14, i32 12, i32 14"),
            "expected fourth sublane, got:\n{ir}"
        );
    }

    #[test]
    fn shuffle_i32x8_identity() {
        // imm=228 = 0b11_10_01_00 → selectors [0,1,2,3] = identity
        let source = r#"
export func f(v: i32x8) -> i32x8 {
    return shuffle_i32x8(v, 228)
}
"#;
        let ir = compile_to_ir(source, "shuffle_i32x8_228");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7"),
            "expected identity mask, got:\n{ir}"
        );
    }

    // ── blend_i32 ────────────────────────────────────────────────

    #[test]
    fn blend_i32_imm_240() {
        // imm=240 = 0xF0 → bits [0,0,0,0,1,1,1,1] → a[0..4], b[4..8]
        // This is _mm256_blend_epi32(..., 240) from llama.cpp
        let source = r#"
export func f(a: i32x8, b: i32x8) -> i32x8 {
    return blend_i32(a, b, 240)
}
"#;
        let ir = compile_to_ir(source, "blend_i32_240");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        // a[0..4] then b[4..7] → indices 0,1,2,3,12,13,14,15
        assert!(
            ir.contains("i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15"),
            "expected blend 0xF0 mask, got:\n{ir}"
        );
    }

    #[test]
    fn blend_i32_imm_0() {
        // imm=0 → all from a
        let source = r#"
export func f(a: i32x8, b: i32x8) -> i32x8 {
    return blend_i32(a, b, 0)
}
"#;
        let ir = compile_to_ir(source, "blend_i32_0");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7"),
            "expected all-from-a mask, got:\n{ir}"
        );
    }

    #[test]
    fn blend_i32_imm_255() {
        // imm=255 → all from b
        let source = r#"
export func f(a: i32x8, b: i32x8) -> i32x8 {
    return blend_i32(a, b, 255)
}
"#;
        let ir = compile_to_ir(source, "blend_i32_255");
        assert!(ir.contains("shufflevector <8 x i32>"), "got:\n{ir}");
        assert!(
            ir.contains("i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15"),
            "expected all-from-b mask, got:\n{ir}"
        );
    }

    // ── error cases ──────────────────────────────────────────────

    #[test]
    fn shuffle_i32x8_rejects_non_const() {
        let source = r#"
export func f(v: i32x8, imm: i32) -> i32x8 {
    return shuffle_i32x8(v, imm)
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("compile-time"),
            "expected compile-time constant error, got: {msg}"
        );
    }

    #[test]
    fn shuffle_i32x8_rejects_out_of_range() {
        let source = r#"
export func f(v: i32x8) -> i32x8 {
    return shuffle_i32x8(v, 256)
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("0..=255"), "expected range error, got: {msg}");
    }

    #[test]
    fn blend_i32_rejects_wrong_type() {
        let source = r#"
export func f(a: i32x16, b: i32x16) -> i32x16 {
    return blend_i32(a, b, 240)
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("i32x8"),
            "expected i32x8 type error, got: {msg}"
        );
    }
}
