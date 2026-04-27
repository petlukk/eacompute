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

    // === ARM: 64-bit vector types ===

    #[test]
    fn test_arm_accepts_i8x8() {
        try_compile("export func f(v: i8x8) -> i8x8 { return v }", &arm_opts())
            .expect("i8x8 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_u8x8() {
        try_compile("export func f(v: u8x8) -> u8x8 { return v }", &arm_opts())
            .expect("u8x8 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i16x4() {
        try_compile("export func f(v: i16x4) -> i16x4 { return v }", &arm_opts())
            .expect("i16x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_u16x4() {
        try_compile("export func f(v: u16x4) -> u16x4 { return v }", &arm_opts())
            .expect("u16x4 should compile on ARM");
    }

    #[test]
    fn test_arm_accepts_i32x2() {
        try_compile("export func f(v: i32x2) -> i32x2 { return v }", &arm_opts())
            .expect("i32x2 should compile on ARM");
    }

    // === x86: 64-bit vectors rejected ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_i8x8() {
        let err = try_compile(
            "export func f(v: i8x8) -> i8x8 { return v }",
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("64-bit"),
            "expected NEON/ARM/64-bit mention, got: {msg}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_u16x4() {
        let err = try_compile(
            "export func f(v: u16x4) -> u16x4 { return v }",
            &CompileOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON") || msg.contains("ARM") || msg.contains("64-bit"),
            "expected NEON/ARM/64-bit mention, got: {msg}"
        );
    }

    // === ARM: cvt_f16_f32 / cvt_f32_f16 ===

    #[test]
    fn test_cvt_f16_f32_arm_compiles() {
        try_compile(
            "export func convert(v: i16x4) -> f32x4 { return cvt_f16_f32(v) }",
            &arm_opts(),
        )
        .expect("cvt_f16_f32 with i16x4 should compile on ARM");
    }

    #[test]
    fn test_cvt_f32_f16_arm_compiles() {
        try_compile(
            "export func convert(v: f32x4) -> i16x4 { return cvt_f32_f16(v) }",
            &arm_opts(),
        )
        .expect("cvt_f32_f16 with f32x4 should compile on ARM");
    }

    #[test]
    fn test_cvt_f16_f32_arm_rejects_i16x8() {
        let err = try_compile(
            "export func convert(v: i16x8) -> f32x8 { return cvt_f16_f32(v) }",
            &arm_opts(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("x86-only") || msg.contains("256-bit") || msg.contains("AVX2"),
            "got: {msg}"
        );
    }

    // === x86: no regression — wider vectors still work ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_still_accepts_f32x8() {
        try_compile(
            "export func f(v: f32x8) -> f32x8 { return v }",
            &CompileOptions::default(),
        )
        .expect("f32x8 should still compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_still_accepts_i32x8() {
        try_compile(
            "export func f(v: i32x8) -> i32x8 { return v }",
            &CompileOptions::default(),
        )
        .expect("i32x8 should still compile on x86");
    }
}
