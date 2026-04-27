//! ARM-safety regression tests for the AVX-512 wide vector types
//! (u8x64, i8x64, i16x32). These types MUST produce a clean compile
//! error when targeting aarch64, because NEON is 128-bit. If this test
//! ever starts passing (i.e., compilation succeeds), a silent wide-vector
//! emulation has leaked into the ARM path — STOP and investigate.

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode, compile_with_options};
    use tempfile::TempDir;

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        compile_with_options(source, &obj_path, OutputMode::ObjectFile, &arm_opts())
    }

    fn assert_rejects_wide_type(source: &str, type_name: &str) {
        let err = try_compile(source).expect_err(&format!(
            "{type_name} must error on ARM target, but compilation succeeded"
        ));
        let msg = format!("{err:?}");
        assert!(
            msg.contains("128")
                || msg.contains("NEON")
                || msg.contains("narrower")
                || msg.contains("AVX2"),
            "error for {type_name} should mention 128-bit/NEON/AVX2 limit, got: {msg}"
        );
    }

    #[test]
    fn u8x64_rejected_on_arm() {
        let source = r#"
export func f(a: u8x64) -> u8 {
    return a[0]
}
"#;
        assert_rejects_wide_type(source, "u8x64");
    }

    #[test]
    fn i8x64_rejected_on_arm() {
        let source = r#"
export func f(a: i8x64) -> i8 {
    return a[0]
}
"#;
        assert_rejects_wide_type(source, "i8x64");
    }

    #[test]
    fn i16x32_rejected_on_arm() {
        let source = r#"
export func f(a: i16x32) -> i16 {
    return a[0]
}
"#;
        assert_rejects_wide_type(source, "i16x32");
    }

    #[test]
    fn concat_i8x32_rejected_on_arm() {
        let source = r#"
export func f(a: i8x32, b: i8x32) -> i8x64 {
    return concat_i8x32(a, b)
}
"#;
        assert_rejects_wide_type(source, "concat_i8x32");
    }

    #[test]
    fn hi256_i8x64_rejected_on_arm() {
        let source = r#"
export func f(a: i8x64) -> i8x32 {
    return hi256_i8x64(a)
}
"#;
        assert_rejects_wide_type(source, "hi256_i8x64");
    }

    #[test]
    fn bcast_even_pairs_i32x16_rejected_on_arm() {
        let source = r#"
export func f(a: i32x16) -> i32x16 {
    return bcast_even_pairs_i32x16(a)
}
"#;
        assert_rejects_wide_type(source, "bcast_even_pairs_i32x16");
    }

    #[test]
    fn shuffle_i32x16_rejected_on_arm() {
        let source = r#"
export func f(a: i32x16) -> i32x16 {
    return shuffle_i32x16(a, 136)
}
"#;
        assert_rejects_wide_type(source, "shuffle_i32x16");
    }

    #[test]
    fn blend_i32_rejected_on_arm() {
        // i32x8 is 256-bit — must be rejected on NEON.
        let source = r#"
export func f(a: i32x8, b: i32x8) -> i32x8 {
    return blend_i32(a, b, 240)
}
"#;
        assert_rejects_wide_type(source, "blend_i32");
    }

    #[test]
    fn lo128_i8x32_rejected_on_arm() {
        // Input is i8x32 (256 bits, wider than NEON's 128), result is i8x16.
        // The ARM gate must reject on the INPUT type, not just the result.
        // If this test ever fails, narrowing extractors bypass the gate —
        // that's a real safety gap, STOP and report.
        let source = r#"
export func f(a: i8x32) -> i8x16 {
    return lo128_i8x32(a)
}
"#;
        assert_rejects_wide_type(source, "lo128_i8x32");
    }

    #[test]
    fn test_fp16_on_x86_is_rejected() {
        // --fp16 appends +fullfp16 to target features; it is ARM-only.
        // Passing it with an x86 target triple must produce a cross-arch error.
        // This test is expected to FAIL until B2 wires the --fp16 flag and
        // adds the cross-arch guard in main.rs / compile_with_options.
        let source = r#"
export func main() {}
"#;
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        let opts = CompileOptions {
            target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
            extra_features: "+fullfp16".to_string(),
            ..CompileOptions::default()
        };
        let result = compile_with_options(source, &obj_path, OutputMode::ObjectFile, &opts);
        let err = result.expect_err(
            "--fp16 (+fullfp16) must be rejected on x86 target, but compilation succeeded",
        );
        let msg = format!("{err:?}");
        assert!(
            msg.contains("--fp16 is incompatible with non-ARM target")
                || msg.contains("--fp16 is only valid for AArch64"),
            "error should explain the cross-arch restriction, got: {msg}"
        );
    }
}
