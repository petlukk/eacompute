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

    fn arm_i8mm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            extra_features: "+i8mm".to_string(),
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

    // === ARM: I8MM ummla_i32 ===

    #[test]
    fn test_arm_accepts_ummla_i32() {
        let source = r#"
            export func f(acc: i32x4, a: u8x16, b: u8x16) -> i32x4 {
                return ummla_i32(acc, a, b)
            }
        "#;
        try_compile(source, &arm_i8mm_opts()).expect("ummla_i32 should compile on ARM with --i8mm");
    }

    #[test]
    fn test_arm_ummla_i32_ir_contains_ummla() {
        let source = r#"
            export func f(acc: i32x4, a: u8x16, b: u8x16) -> i32x4 {
                return ummla_i32(acc, a, b)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("ummla.ll");
        let mut opts = arm_i8mm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("ummla_i32 IR compilation failed");
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.ummla"),
            "expected aarch64.neon.ummla in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_rejects_ummla_i32_without_i8mm() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: u8x16, b: u8x16) -> i32x4 {
                return ummla_i32(acc, a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("--i8mm"), "expected --i8mm hint, got: {msg}");
    }

    #[test]
    fn test_ummla_i32_type_error_wrong_args() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: u8x16) -> i32x4 {
                return ummla_i32(acc, a, b)
            }
        "#;
        let err = try_compile(source, &arm_i8mm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("u8x16") || msg.contains("ummla_i32"),
            "expected type error, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_ummla_i32() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: u8x16, b: u8x16) -> i32x4 {
                return ummla_i32(acc, a, b)
            }
            "#,
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("ARM") || msg.contains("I8MM"),
            "expected ARM/I8MM mention, got: {msg}"
        );
    }

    // === ARM: I8MM usmmla_i32 ===

    #[test]
    fn test_arm_accepts_usmmla_i32() {
        let source = r#"
            export func f(acc: i32x4, a: u8x16, b: i8x16) -> i32x4 {
                return usmmla_i32(acc, a, b)
            }
        "#;
        try_compile(source, &arm_i8mm_opts())
            .expect("usmmla_i32 should compile on ARM with --i8mm");
    }

    #[test]
    fn test_arm_usmmla_i32_ir_contains_usmmla() {
        let source = r#"
            export func f(acc: i32x4, a: u8x16, b: i8x16) -> i32x4 {
                return usmmla_i32(acc, a, b)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("usmmla.ll");
        let mut opts = arm_i8mm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("usmmla_i32 IR compilation failed");
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.usmmla"),
            "expected aarch64.neon.usmmla in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_rejects_usmmla_i32_without_i8mm() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: u8x16, b: i8x16) -> i32x4 {
                return usmmla_i32(acc, a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("--i8mm"), "expected --i8mm hint, got: {msg}");
    }

    #[test]
    fn test_usmmla_i32_type_error_wrong_args() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return usmmla_i32(acc, a, b)
            }
        "#;
        let err = try_compile(source, &arm_i8mm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("u8x16") || msg.contains("usmmla_i32"),
            "expected type error, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_usmmla_i32() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: u8x16, b: i8x16) -> i32x4 {
                return usmmla_i32(acc, a, b)
            }
            "#,
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("ARM") || msg.contains("I8MM"),
            "expected ARM/I8MM mention, got: {msg}"
        );
    }
}
