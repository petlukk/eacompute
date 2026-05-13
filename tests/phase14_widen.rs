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
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, opts)
    }

    // widen_u8_f32x4 lane offsets
    #[test]
    fn test_widen_u8_f32x4_4_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_4(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_u8_f32x4_8_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_8(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_u8_f32x4_12_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_12(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_widen_u8_f32x4_4_x86() {
        try_compile(
            r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_4(v) }"#,
            &CompileOptions::default(),
        )
        .expect("should compile");
    }

    // widen_i8_f32x4 lane offsets
    #[test]
    fn test_widen_i8_f32x4_4_arm() {
        try_compile(
            r#"export func f(v: i8x16) -> f32x4 { return widen_i8_f32x4_4(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_i8_f32x4_8_arm() {
        try_compile(
            r#"export func f(v: i8x16) -> f32x4 { return widen_i8_f32x4_8(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_i8_f32x4_12_arm() {
        try_compile(
            r#"export func f(v: i8x16) -> f32x4 { return widen_i8_f32x4_12(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    // widen_u8_i32x4 lane offsets
    #[test]
    fn test_widen_u8_i32x4_4_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> i32x4 { return widen_u8_i32x4_4(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_u8_i32x4_8_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> i32x4 { return widen_u8_i32x4_8(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    #[test]
    fn test_widen_u8_i32x4_12_arm() {
        try_compile(
            r#"export func f(v: u8x16) -> i32x4 { return widen_u8_i32x4_12(v) }"#,
            &arm_opts(),
        )
        .expect("should compile");
    }

    // IR verification
    #[test]
    fn test_widen_u8_f32x4_4_ir_has_offset() {
        let source = r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_4(v) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("widen4.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("i32 4, i32 5, i32 6, i32 7"),
            "expected indices 4,5,6,7 in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_widen_u8_f32x4_12_ir_has_offset() {
        let source = r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4_12(v) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("widen12.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("i32 12, i32 13, i32 14, i32 15"),
            "expected indices 12,13,14,15 in IR, got:\n{ir}"
        );
    }

    // Type error
    #[test]
    fn test_widen_u8_f32x4_4_rejects_f32x4() {
        let err = try_compile(
            r#"export func f(v: f32x4) -> f32x4 { return widen_u8_f32x4_4(v) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("u8x16") || format!("{err}").contains("widen"),
            "expected error mentioning u8x16 or widen, got: {err}"
        );
    }

    // Original still works
    #[test]
    fn test_original_widen_u8_f32x4_still_works() {
        try_compile(
            r#"export func f(v: u8x16) -> f32x4 { return widen_u8_f32x4(v) }"#,
            &arm_opts(),
        )
        .expect("should still work");
    }
}
