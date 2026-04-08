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
            msg.contains("128") || msg.contains("NEON") || msg.contains("narrower"),
            "error for {type_name} should mention 128-bit/NEON limit, got: {msg}"
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
}
