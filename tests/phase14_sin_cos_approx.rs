#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::assert_transcendental_accuracy;
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    /// Smoke test: sin_approx_f32(splat(0.0)) should give all 0.0s; the
    /// range reduction yields q=0, d'=0, and the quadrant-0 branch picks
    /// sin_val = d' = 0.
    #[test]
    fn test_sin_approx_f32x4_at_zero() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(0.0)
                let r: f32x4 = sin_approx_f32(z)
                store(out, 0, r)
            }
        "#;
        run_smoke(ea, "0\n0\n0\n0");
    }

    /// Smoke test: cos_approx_f32(splat(0.0)) should give all 1.0s; the
    /// range reduction yields q=0, the cos-shift gives k=1, swap=true →
    /// pick cos_val = 1.0.
    #[test]
    fn test_cos_approx_f32x4_at_zero() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(0.0)
                let r: f32x4 = cos_approx_f32(z)
                store(out, 0, r)
            }
        "#;
        run_smoke(ea, "1\n1\n1\n1");
    }

    fn run_smoke(ea: &str, expected: &str) {
        let c = r#"
            #include <stdio.h>
            extern void k(float *out);
            int main(void) {
                float out[4] = {-99, -99, -99, -99};
                k(out);
                for (int i = 0; i < 4; ++i) printf("%g\n", out[i]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), expected);
    }

    /// Helper wrapping `common::assert_transcendental_accuracy` with the
    /// sin/cos shared parameters. Absolute error tolerance because sin/cos
    /// pass through zero at the quadrant boundaries — relative error blows
    /// up there.
    fn accuracy_test_impl(inputs: &[f32], vector_type: &str, intrinsic: &str, ref_fn: &str) {
        // padding = 0.0 because sin(0) = 0 and cos(0) = 1 are both within
        // tolerance of the libm reference for the padding lane.
        assert_transcendental_accuracy(inputs, vector_type, intrinsic, ref_fn, 3.0e-6, 0.0);
    }

    fn boundary_points() -> Vec<f32> {
        use std::f32::consts::PI;
        vec![
            -2.0 * PI,
            -3.0 * PI / 2.0,
            -PI,
            -PI / 2.0,
            -PI / 4.0,
            -0.1,
            0.0,
            0.1,
            PI / 4.0,
            PI / 2.0,
            PI,
            3.0 * PI / 2.0,
            2.0 * PI,
            // Moderate range
            10.0,
            100.0,
            // Within the documented defined range [-1e7, 1e7]
            1.0e6,
        ]
    }

    #[test]
    fn test_sin_approx_f32x4_boundary_points() {
        accuracy_test_impl(&boundary_points(), "f32x4", "sin_approx_f32", "sinf");
    }

    #[test]
    fn test_cos_approx_f32x4_boundary_points() {
        accuracy_test_impl(&boundary_points(), "f32x4", "cos_approx_f32", "cosf");
    }

    /// Pythagorean identity: sin²(x) + cos²(x) ≈ 1.
    /// Pin-tests that the two intrinsics share consistent range reduction
    /// and quadrant logic — a regression in either's q-handling would
    /// surface here even if individual tolerance still passes.
    #[test]
    fn test_sin_cos_pythagorean_identity() {
        let ea = r#"
            export func k(input: *f32, output: *mut f32) {
                let x: f32x4 = load(input, 0)
                let s: f32x4 = sin_approx_f32(x)
                let c: f32x4 = cos_approx_f32(x)
                let result: f32x4 = fma(s, s, c .* c)
                store(output, 0, result)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <math.h>
            extern void k(const float *input, float *output);
            int main(void) {
                float in[4] = {0.1f, 1.2f, 3.4f, 5.6f};
                float out[4] = {0};
                k(in, out);
                for (int i = 0; i < 4; ++i) {
                    float err = fabsf(out[i] - 1.0f);
                    if (err > 6.0e-6f) {
                        printf("FAIL i=%d in=%g s²+c²=%g err=%g\n", i, in[i], out[i], err);
                        return 1;
                    }
                }
                printf("OK\n");
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), "OK");
    }

    #[test]
    fn test_sin_approx_f32x4_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Sample [-10π, 10π] — exercises range reduction over multiple
            // quadrants without hitting the documented range boundary.
            let u = state as f32 / u32::MAX as f32;
            let x = (u - 0.5) * 20.0 * std::f32::consts::PI;
            points.push(x);
        }
        accuracy_test_impl(&points, "f32x4", "sin_approx_f32", "sinf");
    }

    #[test]
    fn test_cos_approx_f32x4_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xCAFE_F00D;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = state as f32 / u32::MAX as f32;
            let x = (u - 0.5) * 20.0 * std::f32::consts::PI;
            points.push(x);
        }
        accuracy_test_impl(&points, "f32x4", "cos_approx_f32", "cosf");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sin_approx_f32x8_boundary_points() {
        accuracy_test_impl(&boundary_points(), "f32x8", "sin_approx_f32", "sinf");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cos_approx_f32x8_boundary_points() {
        accuracy_test_impl(&boundary_points(), "f32x8", "cos_approx_f32", "cosf");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sin_approx_f32x8_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xFACE_BEEF;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = state as f32 / u32::MAX as f32;
            let x = (u - 0.5) * 20.0 * std::f32::consts::PI;
            points.push(x);
        }
        accuracy_test_impl(&points, "f32x8", "sin_approx_f32", "sinf");
    }

    /// CRITICAL regression guard: future "simplifications" that delegate
    /// to @llvm.sin / @llvm.cos would scalarize to per-lane libm sinf/cosf
    /// and silently undo the vectorization that motivates the intrinsics.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sin_cos_approx_does_not_emit_llvm_sin_cos() {
        let src = r#"
            export func sk(v: f32x8) -> f32x8 { return sin_approx_f32(v) }
            export func ck(v: f32x8) -> f32x8 { return cos_approx_f32(v) }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("compile failed");
        let ir = std::fs::read_to_string(&ir_path).expect("read IR");
        assert!(
            !ir.contains("@llvm.sin") && !ir.contains("@llvm.cos"),
            "sin/cos_approx_f32 must NOT lower to @llvm.sin/@llvm.cos:\n{ir}"
        );
    }

    /// Confirm the range-reduction + polynomial pattern is emitted for f32x8.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sin_approx_f32x8_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return sin_approx_f32(v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("compile failed");
        let ir = std::fs::read_to_string(&ir_path).expect("read IR");

        // Horner for sin (3 FMA) + Horner for cos (4 FMA) + Cody-Waite (≥2 FMA)
        let fma_count = ir.matches("@llvm.fma.v8f32").count();
        assert!(
            fma_count >= 6,
            "expected at least 6 @llvm.fma.v8f32 calls; got {fma_count}\nIR:\n{ir}"
        );
        assert!(
            ir.contains("@llvm.nearbyint"),
            "expected @llvm.nearbyint for quadrant rounding:\n{ir}"
        );
        assert!(
            ir.contains("fptosi <8 x float>") && ir.contains("to <8 x i32>"),
            "expected fptosi <8 x float> → <8 x i32> for quadrant index:\n{ir}"
        );
    }

    #[test]
    fn test_sin_approx_f32x4_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x4) -> f32x4 {
                return sin_approx_f32(v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("compile failed");
        let ir = std::fs::read_to_string(&ir_path).expect("read IR");
        assert!(!ir.contains("@llvm.sin"), "no @llvm.sin expected:\n{ir}");
        let fma_count = ir.matches("@llvm.fma.v4f32").count();
        assert!(
            fma_count >= 6,
            "expected at least 6 @llvm.fma.v4f32 calls; got {fma_count}\nIR:\n{ir}"
        );
    }

    // --- Typeck rejections (sin) ---

    #[test]
    fn test_sin_approx_f32_rejects_scalar_f32() {
        let src = r#"
            export func k(x: f32) -> f32 {
                return sin_approx_f32(x)
            }
        "#;
        let err = compile_err(src);
        assert!(
            err.contains("f32 vector"),
            "error must mention f32 vector requirement, got: {err}"
        );
    }

    #[test]
    fn test_sin_approx_f32_rejects_f64_vector() {
        let src = r#"
            export func k(v: f64x2) -> f64x2 {
                return sin_approx_f32(v)
            }
        "#;
        let err = compile_err(src);
        assert!(
            err.contains("f32") && err.contains("element type"),
            "error must mention f32 element type, got: {err}"
        );
    }

    #[test]
    fn test_sin_approx_f32_rejects_integer_vector() {
        let src = r#"
            export func k(v: i32x4) -> i32x4 {
                return sin_approx_f32(v)
            }
        "#;
        let err = compile_err(src);
        assert!(
            err.contains("float") || err.contains("f32"),
            "error must mention float requirement, got: {err}"
        );
    }

    #[test]
    fn test_sin_approx_f32_rejects_wrong_arity() {
        let src = r#"
            export func k(a: f32x4, b: f32x4) -> f32x4 {
                return sin_approx_f32(a, b)
            }
        "#;
        let err = compile_err(src);
        assert!(
            err.contains("1 argument") || err.contains("expects 1"),
            "error must mention arity, got: {err}"
        );
    }

    // --- Typeck rejections (cos) — same shape as sin ---

    #[test]
    fn test_cos_approx_f32_rejects_scalar_f32() {
        let src = r#"
            export func k(x: f32) -> f32 {
                return cos_approx_f32(x)
            }
        "#;
        let err = compile_err(src);
        assert!(err.contains("f32 vector"), "got: {err}");
    }

    #[test]
    fn test_cos_approx_f32_rejects_f64_vector() {
        let src = r#"
            export func k(v: f64x2) -> f64x2 {
                return cos_approx_f32(v)
            }
        "#;
        let err = compile_err(src);
        assert!(
            err.contains("f32") && err.contains("element type"),
            "got: {err}"
        );
    }

    fn compile_err(src: &str) -> String {
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("expected compile failure");
        format!("{err}")
    }
}
