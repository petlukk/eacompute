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

    // --- round_f32x8_i32x8 codegen tests ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_round_f32x8_i32x8_x86() {
        try_compile(
            r#"export func f(a: f32x8) -> i32x8 { return round_f32x8_i32x8(a) }"#,
            &CompileOptions::default(),
        )
        .expect("round_f32x8_i32x8 should compile on x86");
    }

    #[test]
    fn test_round_f32x8_i32x8_arm_rejects_wide_vector() {
        // f32x8 is AVX2-only; on ARM codegen rejects the parameter before reaching the intrinsic
        let err = try_compile(
            r#"export func f(a: f32x8) -> i32x8 { return round_f32x8_i32x8(a) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("f32x4"),
            "expected ARM rejection of f32x8, got: {msg}"
        );
    }

    // --- round_f32x8_i32x8 type error tests ---

    #[test]
    fn test_round_wrong_type() {
        let source = r#"export func f(a: i32x8) -> i32x8 { return round_f32x8_i32x8(a) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("round_f32x8_i32x8") && msg.contains("f32x8"),
            "expected type error mentioning f32x8, got: {msg}"
        );
    }

    #[test]
    fn test_pack_sat_wrong_type() {
        let source =
            r#"export func f(a: f32x8, b: f32x8) -> i16x16 { return pack_sat_i32x8(a, b) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pack_sat_i32x8") && msg.contains("i32x8"),
            "expected type error mentioning i32x8, got: {msg}"
        );
    }

    #[test]
    fn test_pack_sat_wrong_arg_count() {
        let source = r#"export func f(a: i32x8) -> i16x16 { return pack_sat_i32x8(a) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pack_sat_i32x8") && msg.contains("2 arguments"),
            "expected arg count error, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_sat_i32x8_x86() {
        try_compile(
            r#"export func f(a: i32x8, b: i32x8) -> i16x16 { return pack_sat_i32x8(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_sat_i32x8 should compile on x86");
    }

    #[test]
    fn test_pack_sat_i32x8_arm_rejects_wide_vector() {
        let err = try_compile(
            r#"export func f(a: i32x8, b: i32x8) -> i16x16 { return pack_sat_i32x8(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("i32x4"),
            "expected ARM rejection of i32x8, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_sat_i16x16_x86() {
        try_compile(
            r#"export func f(a: i16x16, b: i16x16) -> i8x32 { return pack_sat_i16x16(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_sat_i16x16 should compile on x86");
    }

    #[test]
    fn test_pack_sat_i16x16_arm_rejects_wide_vector() {
        let err = try_compile(
            r#"export func f(a: i16x16, b: i16x16) -> i8x32 { return pack_sat_i16x16(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("i16x8"),
            "expected ARM rejection of i16x16, got: {msg}"
        );
    }
}
