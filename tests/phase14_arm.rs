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

    // === ARM: >128-bit vectors should error ===

    #[test]
    fn test_arm_rejects_f32x8() {
        let err =
            try_compile("export func f(v: f32x8) -> f32x8 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
        assert!(msg.contains("f32x4"), "expected f32x4 hint, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_f32x16() {
        let err = try_compile(
            "export func f(v: f32x16) -> f32x16 { return v }",
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("AVX-512"),
            "expected AVX-512 mention, got: {msg}"
        );
    }

    #[test]
    fn test_arm_rejects_i32x8() {
        let err =
            try_compile("export func f(v: i32x8) -> i32x8 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_i8x32() {
        let err =
            try_compile("export func f(v: i8x32) -> i8x32 { return v }", &arm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_i16x16() {
        let err = try_compile(
            "export func f(v: i16x16) -> i16x16 { return v }",
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_f32x8_local() {
        let err = try_compile(
            r#"
            export func f() {
                let v: f32x8 = splat(1.0)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("AVX2"), "expected AVX2 mention, got: {msg}");
    }

    // === ARM: x86-only intrinsics should error ===

    #[test]
    fn test_arm_rejects_maddubs_i16() {
        let err = try_compile(
            r#"
            export func f(a: u8x16, b: i8x16) -> i16x8 {
                return maddubs_i16(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x86"), "expected x86 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_madd_i16() {
        let err = try_compile(
            r#"
            export func f(a: i16x8, b: i16x8) -> i32x4 {
                return madd_i16(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x86"), "expected x86 mention, got: {msg}");
    }

    #[test]
    fn test_arm_rejects_hadd_i16() {
        let err = try_compile(
            r#"
            export func f(a: i16x8, b: i16x8) -> i16x8 {
                return hadd_i16(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x86"), "expected x86 mention, got: {msg}");
    }

    // === ARM: vdot_i32 should compile ===

    fn arm_dotprod_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            extra_features: "+dotprod".to_string(),
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

    #[test]
    fn test_arm_accepts_vdot_i32() {
        try_compile(
            r#"
            export func f(a: i8x16, b: i8x16) -> i32x4 {
                return vdot_i32(a, b)
            }
            "#,
            &arm_dotprod_opts(),
        )
        .expect("vdot_i32 should compile on ARM with --dotprod");
    }

    #[test]
    fn test_arm_rejects_vdot_i32_without_dotprod() {
        let err = try_compile(
            r#"
            export func f(a: i8x16, b: i8x16) -> i32x4 {
                return vdot_i32(a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--dotprod"),
            "expected --dotprod hint, got: {msg}"
        );
    }

    #[test]
    fn test_arm_vdot_i32_ir_contains_sdot() {
        let source = r#"
            export func f(a: i8x16, b: i8x16) -> i32x4 {
                return vdot_i32(a, b)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("vdot.ll");
        let opts = CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            extra_features: "+dotprod".to_string(),
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("vdot_i32 IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.sdot"),
            "expected aarch64.neon.sdot in IR, got:\n{ir}"
        );
    }

    // === ARM: vdot_lane_i32 (dot product by element) ===

    #[test]
    fn test_arm_accepts_vdot_lane_i32() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return vdot_lane_i32(acc, a, b, 2)
            }
        "#;
        try_compile(source, &arm_dotprod_opts())
            .expect("vdot_lane_i32 should compile on ARM with --dotprod");
    }

    #[test]
    fn test_arm_vdot_lane_i32_ir_contains_sdot() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return vdot_lane_i32(acc, a, b, 1)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("vdot_lane.ll");
        let mut opts = arm_dotprod_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("vdot_lane_i32 IR compilation failed");
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.sdot"),
            "expected aarch64.neon.sdot in IR, got:\n{ir}"
        );
        assert!(
            ir.contains("shufflevector"),
            "expected shufflevector for lane splat in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_rejects_vdot_lane_i32_without_dotprod() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return vdot_lane_i32(acc, a, b, 0)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--dotprod"),
            "expected --dotprod hint, got: {msg}"
        );
    }

    #[test]
    fn test_arm_vdot_lane_i32_rejects_out_of_range_lane() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return vdot_lane_i32(acc, a, b, 4)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0..3"),
            "expected lane range error, got: {msg}"
        );
    }

    #[test]
    fn test_arm_vdot_lane_i32_rejects_wrong_types() {
        let source = r#"
            export func f(acc: i32x4, a: f32x4, b: i8x16) -> i32x4 {
                return vdot_lane_i32(acc, a, b, 0)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("i8x16"),
            "expected type mismatch error, got: {msg}"
        );
    }

    // === ARM: I8MM smmla_i32 ===

    #[test]
    fn test_arm_accepts_smmla_i32() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return smmla_i32(acc, a, b)
            }
        "#;
        try_compile(source, &arm_i8mm_opts()).expect("smmla_i32 should compile on ARM with --i8mm");
    }

    #[test]
    fn test_arm_smmla_i32_ir_contains_smmla() {
        let source = r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return smmla_i32(acc, a, b)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("smmla.ll");
        let mut opts = arm_i8mm_opts();
        opts.opt_level = 0;
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("smmla_i32 IR compilation failed");
        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.smmla"),
            "expected aarch64.neon.smmla in IR, got:\n{ir}"
        );
    }

    #[test]
    fn test_arm_rejects_smmla_i32_without_i8mm() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return smmla_i32(acc, a, b)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("--i8mm"), "expected --i8mm hint, got: {msg}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_smmla_i32() {
        let err = try_compile(
            r#"
            export func f(acc: i32x4, a: i8x16, b: i8x16) -> i32x4 {
                return smmla_i32(acc, a, b)
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

    #[test]
    fn test_smmla_i32_type_error_wrong_args() {
        let source = r#"
            export func f(acc: i32x4, a: u8x16, b: i8x16) -> i32x4 {
                return smmla_i32(acc, a, b)
            }
        "#;
        let err = try_compile(source, &arm_i8mm_opts()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("i8x16") || msg.contains("smmla_i32"),
            "expected type error mentioning i8x16, got: {msg}"
        );
    }

    // === ARM: shuffle_bytes should compile ===

    #[test]
    fn test_arm_accepts_shuffle_bytes() {
        try_compile(
            r#"
            export func f(table: u8x16, idx: u8x16) -> u8x16 {
                return shuffle_bytes(table, idx)
            }
            "#,
            &arm_opts(),
        )
        .expect("shuffle_bytes should compile on ARM");
    }

    #[test]
    fn test_arm_shuffle_bytes_ir_contains_tbl() {
        let source = r#"
            export func f(table: u8x16, idx: u8x16) -> u8x16 {
                return shuffle_bytes(table, idx)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("shuf.ll");
        let opts = CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("shuffle_bytes IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("aarch64.neon.tbl1"),
            "expected aarch64.neon.tbl1 in IR, got:\n{ir}"
        );
    }

    // === ARM: gather/scatter should error ===

    #[test]
    fn test_arm_rejects_gather() {
        let err = try_compile(
            r#"
            export func f(base: *f32, idx: i32x4) -> f32x4 {
                return gather(base, idx)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("scalar"),
            "expected ARM/NEON mention, got: {msg}"
        );
    }

    #[test]
    fn test_arm_rejects_scatter() {
        let err = try_compile(
            r#"
            export func f(base: *mut f32, idx: i32x4, vals: f32x4) {
                scatter(base, idx, vals)
            }
            "#,
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("scalar"),
            "expected ARM/NEON mention, got: {msg}"
        );
    }

    // === ARM: 128-bit vectors should compile OK ===

    #[test]
    fn test_arm_accepts_f32x4() {
        try_compile("export func f(v: f32x4) -> f32x4 { return v }", &arm_opts())
            .expect("f32x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i32x4() {
        try_compile("export func f(v: i32x4) -> i32x4 { return v }", &arm_opts())
            .expect("i32x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i8x16() {
        try_compile("export func f(v: i8x16) -> i8x16 { return v }", &arm_opts())
            .expect("i8x16 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i16x8() {
        try_compile("export func f(v: i16x8) -> i16x8 { return v }", &arm_opts())
            .expect("i16x8 should compile on ARM");
    }
}
