#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile(
        source: &str,
        opts: &CompileOptions,
    ) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("out.o");
        ea_compiler::compile_with_options(source, &out, OutputMode::ObjectFile, opts)
    }

    // --- type error tests ---

    #[test]
    fn test_bsrli_i8x16_wrong_type() {
        let source = r#"export func f(a: i32x4) -> i8x16 { return bsrli_i8x16(a, 4) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("bsrli_i8x16") && msg.contains("i8x16"),
            "expected type error mentioning i8x16, got: {msg}"
        );
    }

    #[test]
    fn test_bsrli_i8x16_imm_out_of_range() {
        let source = r#"export func f(a: i8x16) -> i8x16 { return bsrli_i8x16(a, 16) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("0..=15"), "expected range error, got: {msg}");
    }

    #[test]
    fn test_bslli_i8x32_wrong_arg_count() {
        let source = r#"export func f(a: i8x32) -> i8x32 { return bslli_i8x32(a) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("2 arguments"),
            "expected arg count error, got: {msg}"
        );
    }

    // --- 128-bit byte shift (cross-platform) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_bsrli_i8x16_x86() {
        try_compile(
            r#"export func f(a: i8x16) -> i8x16 { return bsrli_i8x16(a, 4) }"#,
            &CompileOptions::default(),
        )
        .expect("bsrli_i8x16 should compile on x86");
    }

    #[test]
    fn test_bsrli_i8x16_arm() {
        try_compile(
            r#"export func f(a: i8x16) -> i8x16 { return bsrli_i8x16(a, 4) }"#,
            &arm_opts(),
        )
        .expect("bsrli_i8x16 should compile on ARM");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_bslli_i8x16_x86() {
        try_compile(
            r#"export func f(a: i8x16) -> i8x16 { return bslli_i8x16(a, 4) }"#,
            &CompileOptions::default(),
        )
        .expect("bslli_i8x16 should compile on x86");
    }

    #[test]
    fn test_bslli_i8x16_arm() {
        try_compile(
            r#"export func f(a: i8x16) -> i8x16 { return bslli_i8x16(a, 4) }"#,
            &arm_opts(),
        )
        .expect("bslli_i8x16 should compile on ARM");
    }

    // --- 256-bit byte shift (x86-only) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_bsrli_i8x32_x86() {
        try_compile(
            r#"export func f(a: i8x32) -> i8x32 { return bsrli_i8x32(a, 4) }"#,
            &CompileOptions::default(),
        )
        .expect("bsrli_i8x32 should compile on x86");
    }

    #[test]
    fn test_bsrli_i8x32_arm_rejects() {
        let err = try_compile(
            r#"export func f(a: i8x32) -> i8x32 { return bsrli_i8x32(a, 4) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("x86-only") || msg.contains("i8x16"),
            "expected ARM rejection, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_bslli_i8x32_x86() {
        try_compile(
            r#"export func f(a: i8x32) -> i8x32 { return bslli_i8x32(a, 4) }"#,
            &CompileOptions::default(),
        )
        .expect("bslli_i8x32 should compile on x86");
    }

    #[test]
    fn test_bslli_i8x32_arm_rejects() {
        let err = try_compile(
            r#"export func f(a: i8x32) -> i8x32 { return bslli_i8x32(a, 4) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("x86-only") || msg.contains("i8x16"),
            "expected ARM rejection, got: {msg}"
        );
    }

    // --- edge case: imm=0 (no-op) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_bsrli_i8x16_zero_shift() {
        try_compile(
            r#"export func f(a: i8x16) -> i8x16 { return bsrli_i8x16(a, 0) }"#,
            &CompileOptions::default(),
        )
        .expect("bsrli_i8x16 with imm=0 should compile");
    }
}
