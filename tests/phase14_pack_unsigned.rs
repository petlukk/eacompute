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
    fn test_pack_usat_i32x8_wrong_type() {
        let source =
            r#"export func f(a: f32x8, b: f32x8) -> u16x16 { return pack_usat_i32x8(a, b) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pack_usat_i32x8") && msg.contains("i32x8"),
            "expected type error mentioning i32x8, got: {msg}"
        );
    }

    #[test]
    fn test_pack_usat_i32x8_wrong_arg_count() {
        let source = r#"export func f(a: i32x8) -> u16x16 { return pack_usat_i32x8(a) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pack_usat_i32x8") && msg.contains("2 arguments"),
            "expected arg count error, got: {msg}"
        );
    }

    #[test]
    fn test_pack_usat_i16x16_wrong_type() {
        let source =
            r#"export func f(a: i32x16, b: i32x16) -> u8x32 { return pack_usat_i16x16(a, b) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pack_usat_i16x16") && msg.contains("i16x16"),
            "expected type error mentioning i16x16, got: {msg}"
        );
    }

    // --- x86 codegen ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_usat_i32x8_x86() {
        try_compile(
            r#"export func f(a: i32x8, b: i32x8) -> u16x16 { return pack_usat_i32x8(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_usat_i32x8 should compile on x86");
    }

    #[test]
    fn test_pack_usat_i32x8_arm_rejects() {
        let err = try_compile(
            r#"export func f(a: i32x8, b: i32x8) -> u16x16 { return pack_usat_i32x8(a, b) }"#,
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
    fn test_pack_usat_i16x16_x86() {
        try_compile(
            r#"export func f(a: i16x16, b: i16x16) -> u8x32 { return pack_usat_i16x16(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_usat_i16x16 should compile on x86");
    }

    #[test]
    fn test_pack_usat_i16x16_arm_rejects() {
        let err = try_compile(
            r#"export func f(a: i16x16, b: i16x16) -> u8x32 { return pack_usat_i16x16(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX2") || msg.contains("i16x8"),
            "expected ARM rejection of i16x16, got: {msg}"
        );
    }

    // --- 128-bit cross-platform ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_usat_i32x4_x86() {
        try_compile(
            r#"export func f(a: i32x4, b: i32x4) -> u16x8 { return pack_usat_i32x4(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_usat_i32x4 should compile on x86");
    }

    #[test]
    fn test_pack_usat_i32x4_arm() {
        try_compile(
            r#"export func f(a: i32x4, b: i32x4) -> u16x8 { return pack_usat_i32x4(a, b) }"#,
            &arm_opts(),
        )
        .expect("pack_usat_i32x4 should compile on ARM");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pack_usat_i16x8_x86() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> u8x16 { return pack_usat_i16x8(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("pack_usat_i16x8 should compile on x86");
    }

    #[test]
    fn test_pack_usat_i16x8_arm() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> u8x16 { return pack_usat_i16x8(a, b) }"#,
            &arm_opts(),
        )
        .expect("pack_usat_i16x8 should compile on ARM");
    }
}
