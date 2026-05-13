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

    // --- sat_add ARM tests ---

    #[test]
    fn test_arm_sat_add_i8x16() {
        try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_add(i8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_add_u8x16() {
        try_compile(
            r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_add(u8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_add_i16x8() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i16x8 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_add(i16x8) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_add_u16x8() {
        try_compile(
            r#"export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_add(u16x8) should compile on ARM");
    }

    // --- sat_add x86 tests ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_add_i8x16() {
        try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_add(i8x16) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_add_u8x16() {
        try_compile(
            r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_add(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_add(u8x16) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_add_i16x8() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i16x8 { return sat_add(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_add(i16x8) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_add_u16x8() {
        try_compile(
            r#"export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_add(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_add(u16x8) should compile on x86");
    }

    // --- sat_add IR checks ---

    #[test]
    fn test_arm_sat_add_ir() {
        let source = r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("sat.ll");
        let opts = CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.sqadd") || ir.contains("sadd.sat"),
            "expected sqadd or sadd.sat in IR, got:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_add_ir() {
        let source = r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("sat.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("x86.sse2.padds.b") || ir.contains("sadd.sat"),
            "expected padds.b or sadd.sat in IR, got:\n{ir}"
        );
    }

    // --- sat_add type error tests ---

    #[test]
    fn test_sat_add_rejects_f32x4() {
        let err = try_compile(
            r#"export func f(a: f32x4, b: f32x4) -> f32x4 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("sat_add"));
    }

    #[test]
    fn test_sat_add_rejects_mismatched() {
        let err = try_compile(
            r#"export func f(a: i8x16, b: u8x16) -> i8x16 { return sat_add(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("sat_add") || msg.contains("mismatch"),
            "unexpected error: {msg}"
        );
    }

    // --- sat_sub ARM tests ---

    #[test]
    fn test_arm_sat_sub_i8x16() {
        try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_sub(i8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_sub_u8x16() {
        try_compile(
            r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_sub(u8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_sub_i16x8() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i16x8 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_sub(i16x8) should compile on ARM");
    }

    #[test]
    fn test_arm_sat_sub_u16x8() {
        try_compile(
            r#"export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .expect("sat_sub(u16x8) should compile on ARM");
    }

    // --- sat_sub x86 tests ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_sub_i8x16() {
        try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_sub(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_sub(i8x16) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_sub_u8x16() {
        try_compile(
            r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_sub(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_sub(u8x16) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_sub_i16x8() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i16x8 { return sat_sub(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_sub(i16x8) should compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_sub_u16x8() {
        try_compile(
            r#"export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_sub(a, b) }"#,
            &CompileOptions::default(),
        )
        .expect("sat_sub(u16x8) should compile on x86");
    }

    // --- sat_sub IR checks ---

    #[test]
    fn test_arm_sat_sub_ir() {
        let source = r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_sub(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("satsub.ll");
        let opts = CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.uqsub") || ir.contains("usub.sat"),
            "expected uqsub or usub.sat in IR, got:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_sat_sub_ir() {
        let source = r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_sub(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("satsub.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("x86.sse2.psubs.b") || ir.contains("ssub.sat"),
            "expected psubs.b or ssub.sat in IR, got:\n{ir}"
        );
    }

    // --- sat_sub type error tests ---

    #[test]
    fn test_sat_sub_rejects_f32x4() {
        let err = try_compile(
            r#"export func f(a: f32x4, b: f32x4) -> f32x4 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("sat_sub"));
    }

    #[test]
    fn test_sat_sub_rejects_mismatched() {
        let err = try_compile(
            r#"export func f(a: i16x8, b: u16x8) -> i16x8 { return sat_sub(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("sat_sub") || msg.contains("mismatch"),
            "unexpected error: {msg}"
        );
    }
}
