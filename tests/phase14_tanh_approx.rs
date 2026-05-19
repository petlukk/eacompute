#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::assert_transcendental_accuracy;
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    /// Smoke test: tanh_approx_f32(splat(0.0)) should give all 0.0s.
    /// Exercises the full pipeline — clamp passthrough, polynomial at zero
    /// returns 0 in the numerator, denominator returns β₀, result = 0.
    #[test]
    fn test_tanh_approx_f32x4_at_zero() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(0.0)
                let r: f32x4 = tanh_approx_f32(z)
                store(out, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void k(float *out);
            int main(void) {
                float out[4] = {1, 1, 1, 1};
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
        assert_eq!(stdout.trim(), "0\n0\n0\n0");
    }

    /// Helper wrapping `common::assert_transcendental_accuracy` with the
    /// tanh-specific parameters. Absolute (not relative) error tolerance
    /// because tanh is bounded in [-1, 1] and tanh(x)→0 as x→0 makes
    /// relative error blow up near the origin.
    fn accuracy_test_impl(inputs: &[f32], vector_type: &str) {
        // padding = 0.0 because tanh(0) = 0 — safe to pad input arrays with.
        assert_transcendental_accuracy(
            inputs,
            vector_type,
            "tanh_approx_f32",
            "tanhf",
            5.0e-6,
            0.0,
        );
    }

    #[test]
    fn test_tanh_approx_f32x4_boundary_points() {
        accuracy_test_impl(
            &[
                -50.0, -9.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 9.0, 50.0,
            ],
            "f32x4",
        );
    }

    /// Verify odd symmetry tanh(-x) = -tanh(x) holds within tight tolerance.
    /// This pins the property that the numerator is x · (even polynomial in x²)
    /// and the denominator is even in x² — algebraic guarantee, but worth
    /// testing because compiler reassociation could break it.
    #[test]
    fn test_tanh_approx_f32x4_odd_symmetry() {
        let ea = r#"
            export func k(input: *f32, output: *mut f32) {
                let v: f32x4 = load(input, 0)
                let r: f32x4 = tanh_approx_f32(v)
                store(output, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <math.h>
            extern void k(const float *input, float *output);
            int main(void) {
                float pos[4] = {0.5f, 1.0f, 2.0f, 3.5f};
                float neg[4] = {-0.5f, -1.0f, -2.0f, -3.5f};
                float pos_out[4] = {0}, neg_out[4] = {0};
                k(pos, pos_out);
                k(neg, neg_out);
                for (int i = 0; i < 4; ++i) {
                    float asym = fabsf(pos_out[i] + neg_out[i]);
                    if (asym > 1.0e-6f) {
                        printf("FAIL i=%d pos=%g neg=%g asym=%g\n",
                               i, pos_out[i], neg_out[i], asym);
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

    /// Saturation: tanh(±large) must be exactly ±1 (or within ULPs) — confirms
    /// clamp is wired in and that the rational polynomial doesn't overshoot.
    #[test]
    fn test_tanh_approx_f32x4_saturates() {
        let ea = r#"
            export func k(out: *mut f32) {
                let big_pos: f32x4 = splat(50.0)
                let big_neg: f32x4 = splat(-50.0)
                let r_pos: f32x4 = tanh_approx_f32(big_pos)
                let r_neg: f32x4 = tanh_approx_f32(big_neg)
                store(out, 0, r_pos)
                store(out, 4, r_neg)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <math.h>
            extern void k(float *out);
            int main(void) {
                float out[8] = {0};
                k(out);
                for (int i = 0; i < 4; ++i) {
                    if (fabsf(out[i] - 1.0f) > 1.0e-6f) {
                        printf("FAIL pos i=%d got=%g\n", i, out[i]);
                        return 1;
                    }
                }
                for (int i = 4; i < 8; ++i) {
                    if (fabsf(out[i] + 1.0f) > 1.0e-6f) {
                        printf("FAIL neg i=%d got=%g\n", i, out[i]);
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
    fn test_tanh_approx_f32x4_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Sample [-10, 10] — covers body + saturation transition.
            let f = (state as f32 / u32::MAX as f32) * 20.0 - 10.0;
            points.push(f);
        }
        accuracy_test_impl(&points, "f32x4");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tanh_approx_f32x8_boundary_points() {
        accuracy_test_impl(
            &[
                -50.0, -9.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 9.0, 50.0,
            ],
            "f32x8",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tanh_approx_f32x8_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xCAFE_F00D;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = (state as f32 / u32::MAX as f32) * 20.0 - 10.0;
            points.push(f);
        }
        accuracy_test_impl(&points, "f32x8");
    }

    /// CRITICAL regression guard: a future "simplification" of
    /// compile_tanh_approx_f32 to delegate to compile_tanh (which doesn't
    /// exist yet but would scalarize via libm) would silently undo the
    /// vectorization that motivates the intrinsic. Pins absence of @llvm.tanh.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tanh_approx_f32_does_not_emit_llvm_tanh() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return tanh_approx_f32(v)
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
        assert!(
            !ir.contains("@llvm.tanh"),
            "tanh_approx_f32 must NOT lower to @llvm.tanh; found scalarization-prone intrinsic in IR:\n{ir}"
        );
        assert!(
            !ir.contains("@llvm.exp"),
            "tanh_approx_f32 should not delegate to @llvm.exp either:\n{ir}"
        );
    }

    /// Confirm the rational polynomial pattern is actually emitted for f32x8:
    /// FMA chain (for Horner) + a division (rational form, P/Q).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tanh_approx_f32x8_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return tanh_approx_f32(v)
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

        let fma_count = ir.matches("@llvm.fma.v8f32").count();
        assert!(
            fma_count >= 8,
            "expected at least 8 @llvm.fma.v8f32 calls (6 numerator + 3 denominator Horner steps); got {fma_count}\nIR:\n{ir}"
        );
        assert!(
            ir.contains("fdiv <8 x float>"),
            "expected fdiv <8 x float> for the rational form's P/Q division\nIR:\n{ir}"
        );
        // Clamp emits min + max intrinsics or fcmp + select.
        assert!(
            ir.contains("@llvm.minnum.v8f32")
                || ir.contains("@llvm.maxnum.v8f32")
                || ir.contains("fcmp"),
            "expected clamp pattern (minnum/maxnum or fcmp+select)\nIR:\n{ir}"
        );
    }

    #[test]
    fn test_tanh_approx_f32x4_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x4) -> f32x4 {
                return tanh_approx_f32(v)
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
        assert!(!ir.contains("@llvm.tanh"), "no @llvm.tanh expected:\n{ir}");
        let fma_count = ir.matches("@llvm.fma.v4f32").count();
        assert!(
            fma_count >= 8,
            "expected at least 8 @llvm.fma.v4f32 calls; got {fma_count}\nIR:\n{ir}"
        );
        assert!(
            ir.contains("fdiv <4 x float>"),
            "expected fdiv <4 x float>\nIR:\n{ir}"
        );
    }

    /// Scalar f32 should fail with helpful message pointing at tanh().
    #[test]
    fn test_tanh_approx_f32_rejects_scalar_f32() {
        let src = r#"
            export func k(x: f32) -> f32 {
                return tanh_approx_f32(x)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("scalar f32 should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("f32 vector"),
            "error must mention f32 vector requirement, got: {msg}"
        );
    }

    #[test]
    fn test_tanh_approx_f32_rejects_f64_vector() {
        let src = r#"
            export func k(v: f64x2) -> f64x2 {
                return tanh_approx_f32(v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("f64x2 should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("f32") && msg.contains("element type"),
            "error must mention f32 element type, got: {msg}"
        );
    }

    #[test]
    fn test_tanh_approx_f32_rejects_integer_vector() {
        let src = r#"
            export func k(v: i32x4) -> i32x4 {
                return tanh_approx_f32(v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("i32x4 should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("float") || msg.contains("f32"),
            "error must mention float requirement, got: {msg}"
        );
    }

    #[test]
    fn test_tanh_approx_f32_rejects_wrong_arity() {
        let src = r#"
            export func k(a: f32x4, b: f32x4) -> f32x4 {
                return tanh_approx_f32(a, b)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("wrong arity should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("1 argument") || msg.contains("expects 1"),
            "error must mention arity, got: {msg}"
        );
    }
}
