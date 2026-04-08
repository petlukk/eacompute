//! AVX-512 int-dot chain tests: maddubs_i16 / madd_i16 / to_f32 / cvt_f16_f32
//! at the 512-bit widths (u8x64, i16x32, i32x16, f32x16).
//!
//! These tests compile to LLVM IR and assert that the correct LLVM intrinsics
//! and types appear, rather than running the binary (AVX-512 hardware may not
//! be available on dev/CI machines). This mirrors the pattern in phase_b_ext.rs.

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
    fn madd_i16_avx512_i16x32_to_i32x16() {
        let source = r#"
export func f(a: i16x32, b: i16x32) -> i32x16 {
    return madd_i16(a, b)
}
"#;
        let ir = compile_to_ir(source, "madd_avx512");
        assert!(
            ir.contains("llvm.x86.avx512.pmaddw.d.512"),
            "expected avx512 pmaddwd intrinsic in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x i32>"),
            "expected <16 x i32> return type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i16>"),
            "expected <32 x i16> argument type in IR, got:\n{ir}"
        );
    }

    #[test]
    fn maddubs_i16_avx512_u8x64_to_i16x32() {
        let source = r#"
export func f(a: u8x64, b: i8x64) -> i16x32 {
    return maddubs_i16(a, b)
}
"#;
        let ir = compile_to_ir(source, "maddubs_avx512");
        assert!(
            ir.contains("llvm.x86.avx512.pmaddubs.w.512"),
            "expected avx512 pmaddubs intrinsic in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<32 x i16>"),
            "expected <32 x i16> return type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<64 x i8>"),
            "expected <64 x i8> argument type in IR, got:\n{ir}"
        );
    }

    #[test]
    fn to_f32_avx512_i32x16_to_f32x16() {
        let source = r#"
export func f(a: i32x16) -> f32x16 {
    return to_f32(a)
}
"#;
        let ir = compile_to_ir(source, "to_f32_avx512");
        assert!(
            ir.contains("sitofp <16 x i32>") && ir.contains("to <16 x float>"),
            "expected vector sitofp i32x16 -> f32x16 in IR, got:\n{ir}"
        );
    }

    #[test]
    fn to_f32_avx2_i32x8_to_f32x8() {
        let source = r#"
export func f(a: i32x8) -> f32x8 {
    return to_f32(a)
}
"#;
        let ir = compile_to_ir(source, "to_f32_avx2");
        assert!(
            ir.contains("sitofp <8 x i32>") && ir.contains("to <8 x float>"),
            "expected vector sitofp i32x8 -> f32x8 in IR, got:\n{ir}"
        );
    }

    #[test]
    fn cvt_f16_f32_avx512_i16x16_to_f32x16() {
        let source = r#"
export func f(a: i16x16) -> f32x16 {
    return cvt_f16_f32(a)
}
"#;
        let ir = compile_to_ir(source, "cvt_f16_f32_avx512");
        assert!(
            ir.contains("<16 x float>"),
            "expected <16 x float> return type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("<16 x half>"),
            "expected <16 x half> bitcast in IR, got:\n{ir}"
        );
        assert!(ir.contains("fpext"), "expected fpext in IR, got:\n{ir}");
    }

    #[test]
    fn u8x64_bitwise_and_shift() {
        let source = r#"
export func lo_nibbles(a: u8x64) -> u8x64 {
    let mask: u8x64 = splat(15)
    return a .& mask
}
export func hi_nibbles(a: u8x64) -> u8x64 {
    let shift: u8x64 = splat(4)
    return a .>> shift
}
"#;
        let ir = compile_to_ir(source, "u8x64_bitwise");
        assert!(
            ir.contains("<64 x i8>"),
            "expected <64 x i8> type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains(" and ") || ir.contains("and <64"),
            "expected bitwise AND in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("lshr") || ir.contains("ashr"),
            "expected right shift in IR, got:\n{ir}"
        );
    }

    #[test]
    fn i16x32_add() {
        let source = r#"
export func f(a: i16x32, b: i16x32) -> i16x32 {
    return a .+ b
}
"#;
        let ir = compile_to_ir(source, "i16x32_add");
        assert!(
            ir.contains("<32 x i16>"),
            "expected <32 x i16> type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("add <32 x i16>") || ir.contains("add nsw <32 x i16>"),
            "expected vector add in IR, got:\n{ir}"
        );
    }

    #[test]
    fn i32x16_add() {
        let source = r#"
export func f(a: i32x16, b: i32x16) -> i32x16 {
    return a .+ b
}
"#;
        let ir = compile_to_ir(source, "i32x16_add");
        assert!(
            ir.contains("<16 x i32>"),
            "expected <16 x i32> type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("add <16 x i32>") || ir.contains("add nsw <16 x i32>"),
            "expected vector add in IR, got:\n{ir}"
        );
    }

    #[test]
    fn f32x16_arith_unfused() {
        let source = r#"
export func f(a: f32x16, b: f32x16, c: f32x16) -> f32x16 {
    return a .* b .+ c
}
"#;
        let ir = compile_to_ir(source, "f32x16_arith");
        assert!(
            ir.contains("<16 x float>"),
            "expected <16 x float> type in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("fmul"),
            "expected separate fmul in IR (no auto-fusion), got:\n{ir}"
        );
        assert!(
            ir.contains("fadd"),
            "expected separate fadd in IR, got:\n{ir}"
        );
        assert!(
            !ir.contains("llvm.fma"),
            "unexpected llvm.fma intrinsic (auto-fusion violates explicit-cost rule), got:\n{ir}"
        );
    }
}
