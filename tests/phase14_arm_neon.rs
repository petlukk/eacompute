#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
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

    // --- abs_diff tests ---

    #[test]
    fn test_arm_abs_diff_i8x16() {
        try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return abs_diff(a, b) }"#,
            &arm_opts(),
        )
        .expect("abs_diff(i8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_diff_u8x16() {
        try_compile(
            r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff(a, b) }"#,
            &arm_opts(),
        )
        .expect("abs_diff(u8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_diff_i32x4() {
        try_compile(
            r#"export func f(a: i32x4, b: i32x4) -> i32x4 { return abs_diff(a, b) }"#,
            &arm_opts(),
        )
        .expect("abs_diff(i32x4) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_diff_u32x4() {
        try_compile(
            r#"export func f(a: u32x4, b: u32x4) -> u32x4 { return abs_diff(a, b) }"#,
            &arm_opts(),
        )
        .expect("abs_diff(u32x4) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_diff_ir_sabd() {
        let source = r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return abs_diff(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("abd.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.sabd"),
            "expected sabd in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_abs_diff_ir_uabd() {
        let source = r#"export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("abd.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.uabd"),
            "expected uabd in IR, got:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_abs_diff() {
        let err = try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i8x16 { return abs_diff(a, b) }"#,
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("ARM") || msg.contains("NEON"),
            "expected ARM/NEON, got: {msg}"
        );
    }

    #[test]
    fn test_abs_diff_rejects_f32x4() {
        let err = try_compile(
            r#"export func f(a: f32x4, b: f32x4) -> f32x4 { return abs_diff(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("abs_diff"));
    }

    // --- wmul_i16 / wmul_u16 tests ---

    #[test]
    fn test_arm_wmul_i16() {
        try_compile(
            r#"export func f(a: i8x8, b: i8x8) -> i16x8 { return wmul_i16(a, b) }"#,
            &arm_opts(),
        )
        .expect("wmul_i16 should compile on ARM");
    }

    #[test]
    fn test_arm_wmul_u16() {
        try_compile(
            r#"export func f(a: u8x8, b: u8x8) -> u16x8 { return wmul_u16(a, b) }"#,
            &arm_opts(),
        )
        .expect("wmul_u16 should compile on ARM");
    }

    #[test]
    fn test_arm_wmul_i16_ir() {
        let source = r#"export func f(a: i8x8, b: i8x8) -> i16x8 { return wmul_i16(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("wmul.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.smull"),
            "expected smull in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_wmul_i16_rejects_i8x16() {
        let err = try_compile(
            r#"export func f(a: i8x16, b: i8x16) -> i16x8 { return wmul_i16(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("i8x8") || msg.contains("wmul_i16"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_wmul_i16() {
        // On x86, i8x8 itself is rejected (64-bit NEON type), so this errors at type level
        let err = try_compile(
            r#"export func f(a: i8x8, b: i8x8) -> i16x8 { return wmul_i16(a, b) }"#,
            &CompileOptions::default(),
        );
        assert!(err.is_err());
    }

    // --- wmul_i32 / wmul_u32 tests ---

    #[test]
    fn test_arm_wmul_i32() {
        try_compile(
            r#"export func f(a: i16x4, b: i16x4) -> i32x4 { return wmul_i32(a, b) }"#,
            &arm_opts(),
        )
        .expect("wmul_i32 should compile on ARM");
    }

    #[test]
    fn test_arm_wmul_u32() {
        try_compile(
            r#"export func f(a: u16x4, b: u16x4) -> u32x4 { return wmul_u32(a, b) }"#,
            &arm_opts(),
        )
        .expect("wmul_u32 should compile on ARM");
    }

    #[test]
    fn test_arm_wmul_i32_ir() {
        let source = r#"export func f(a: i16x4, b: i16x4) -> i32x4 { return wmul_i32(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("wmul32.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.smull"),
            "expected smull in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_wmul_i32_rejects_i16x8() {
        let err = try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i32x4 { return wmul_i32(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("i16x4") || msg.contains("wmul_i32"),
            "unexpected error: {msg}"
        );
    }

    // --- addp (pairwise add) tests ---

    #[test]
    fn test_arm_addp_i32() {
        try_compile(
            r#"export func f(a: i32x4, b: i32x4) -> i32x4 { return addp_i32(a, b) }"#,
            &arm_opts(),
        )
        .expect("addp_i32 should compile on ARM");
    }

    #[test]
    fn test_arm_addp_i16() {
        try_compile(
            r#"export func f(a: i16x8, b: i16x8) -> i16x8 { return addp_i16(a, b) }"#,
            &arm_opts(),
        )
        .expect("addp_i16 should compile on ARM");
    }

    #[test]
    fn test_arm_addp_i32_ir() {
        let source = r#"export func f(a: i32x4, b: i32x4) -> i32x4 { return addp_i32(a, b) }"#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("addp.ll");
        let mut opts = arm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.addp"),
            "expected addp in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_addp_i32_rejected_on_x86() {
        let err = try_compile(
            r#"export func f(a: i32x4, b: i32x4) -> i32x4 { return addp_i32(a, b) }"#,
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("ARM-only"),
            "expected ARM-only error, got: {msg}"
        );
    }

    #[test]
    fn test_arm_addp_i32_wrong_type() {
        let err = try_compile(
            r#"export func f(a: f32x4, b: f32x4) -> f32x4 { return addp_i32(a, b) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("i32x4"),
            "expected type error mentioning i32x4, got: {msg}"
        );
    }

    #[test]
    fn test_f32x4_from_scalars() {
        assert_output(
            r#"
        export func main() {
            let a: f32 = 1.0
            let b: f32 = 2.0
            let c: f32 = 3.0
            let d: f32 = 4.0
            let v: f32x4 = f32x4_from_scalars(a, b, c, d)
            let s: f32 = reduce_add(v)
            println(s)
        }
        "#,
            "10",
        );
    }

    #[test]
    fn test_f32x8_from_scalars() {
        assert_output(
            r#"
        export func main() {
            let v: f32x8 = f32x8_from_scalars(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            )
            let s: f32 = reduce_add(v)
            println(s)
        }
        "#,
            "36",
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_gather_on_arm_points_to_compose() {
        use ea_compiler::{CompileOptions, OutputMode};
        let src = r#"
            export func k(lut: *f32, idx: i32x4, out: *mut f32) {
                let v: f32x4 = gather(lut, idx)
                store(out, 0, v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = tempfile::TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("gather on ARM should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("f32x4_from_scalars"),
            "error must mention the new intrinsic, got: {msg}"
        );
        assert!(
            msg.contains("neon-gather.md"),
            "error must mention the docs file, got: {msg}"
        );
    }
}
