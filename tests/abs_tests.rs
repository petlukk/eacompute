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

    // --- scalar float ---

    #[test]
    fn test_abs_f32_positive() {
        assert_output(
            r#"
            export func main() {
                let x: f32 = 3.14
                println(abs(x))
            }
            "#,
            "3.14",
        );
    }

    #[test]
    fn test_abs_f32_negative() {
        assert_output(
            r#"
            export func main() {
                let x: f32 = -3.14
                println(abs(x))
            }
            "#,
            "3.14",
        );
    }

    #[test]
    fn test_abs_f32_zero() {
        assert_output(
            r#"
            export func main() {
                let x: f32 = 0.0
                println(abs(x))
            }
            "#,
            "0",
        );
    }

    #[test]
    fn test_abs_f64_negative() {
        assert_output(
            r#"
            export func main() {
                let x: f64 = -2.718281828
                println(abs(x))
            }
            "#,
            "2.71828",
        );
    }

    // --- IR uses llvm.fabs ---

    #[test]
    fn test_abs_f32_ir_uses_fabs() {
        let ir = compile_to_ir(r#"export func test(x: f32, out: *mut f32) { out[0] = abs(x) }"#);
        assert!(
            ir.contains("llvm.fabs.f32"),
            "IR should use llvm.fabs.f32: {ir}"
        );
    }

    // --- float vector ---

    #[test]
    fn test_abs_f32x4_compiles() {
        assert_c_interop(
            r#"
            export func test(a: *f32, out: *mut f32) {
                let v: f32x4 = load(a, 0)
                let r: f32x4 = abs(v)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float a[] = {-1.0f, 2.5f, -3.0f, 0.0f};
                float out[4];
                test(a, out);
                for (int i = 0; i < 4; i++) printf("%.1f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "1.0 2.5 3.0 0.0",
        );
    }

    // --- signed integer vector (ARM cross-compile) ---

    #[test]
    fn test_arm_abs_f32x4() {
        try_compile(
            r#"export func f(v: f32x4) -> f32x4 { return abs(v) }"#,
            &arm_opts(),
        )
        .expect("abs(f32x4) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_i8x16() {
        try_compile(
            r#"export func f(v: i8x16) -> i8x16 { return abs(v) }"#,
            &arm_opts(),
        )
        .expect("abs(i8x16) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_i16x8() {
        try_compile(
            r#"export func f(v: i16x8) -> i16x8 { return abs(v) }"#,
            &arm_opts(),
        )
        .expect("abs(i16x8) should compile on ARM");
    }

    #[test]
    fn test_arm_abs_i32x4() {
        try_compile(
            r#"export func f(v: i32x4) -> i32x4 { return abs(v) }"#,
            &arm_opts(),
        )
        .expect("abs(i32x4) should compile on ARM");
    }

    // --- type errors ---

    #[test]
    fn test_abs_rejects_unsigned_vector() {
        let err = try_compile(
            r#"export func f(v: u8x16) -> u8x16 { return abs(v) }"#,
            &arm_opts(),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("abs"),
            "error should mention abs: {err}"
        );
    }

    #[test]
    fn test_abs_rejects_wrong_arg_count() {
        let tokens =
            ea_compiler::tokenize(r#"export func f(a: f32, b: f32) -> f32 { return abs(a, b) }"#)
                .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err}").contains("abs"),
            "error should mention abs: {err}"
        );
    }
}
