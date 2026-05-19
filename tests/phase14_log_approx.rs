#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    /// Smoke test: log_approx_f32(splat(1.0)) should give all 0.0s.
    /// Exercises the full pipeline — bit extraction gives e=0, m=1.0
    /// (after exponent reset), √2/2 rebalance doesn't fire (1.0 > √2/2),
    /// u = m - 1 = 0, polynomial returns 0, recombine returns 0.
    #[test]
    fn test_log_approx_f32x4_at_one() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(1.0)
                let r: f32x4 = log_approx_f32(z)
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

    /// Compile a kernel that runs log_approx_f32 over `inputs` lane-by-lane,
    /// link with C harness that calls logf, assert absolute error ≤ 3e-6.
    ///
    /// Absolute (not relative) error because log(x) → 0 as x → 1, making
    /// relative error blow up near 1. Across the rest of the input range
    /// the magnitude of log(x) is bounded modestly, so absolute is the
    /// natural metric.
    fn accuracy_test_impl(inputs: &[f32], vector_type: &str) {
        let lanes = if vector_type == "f32x4" { 4 } else { 8 };

        let ea = format!(
            r#"
            export func k(input: *f32, output: *mut f32, n: i32) {{
                let mut i: i32 = 0
                while i + {lanes} <= n {{
                    let v: {vector_type} = load(input, i)
                    let r: {vector_type} = log_approx_f32(v)
                    store(output, i, r)
                    i = i + {lanes}
                }}
            }}
            "#
        );

        let mut padded = inputs.to_vec();
        while !padded.len().is_multiple_of(lanes) {
            padded.push(1.0); // pad with 1.0 so log_approx_f32 returns 0 on padding
        }
        let n = padded.len();
        let original_n = inputs.len();

        let inputs_str = padded
            .iter()
            .map(|f| format!("{f:.10e}f"))
            .collect::<Vec<_>>()
            .join(", ");

        let c = format!(
            r#"
            #include <stdio.h>
            #include <math.h>
            extern void k(const float *input, float *output, int n);
            int main(void) {{
                float in[{n}] = {{{inputs_str}}};
                float out[{n}] = {{0}};
                k(in, out, {n});
                for (int i = 0; i < {original_n}; ++i) {{
                    float ref = logf(in[i]);
                    float got = out[i];
                    float abs_err = fabsf(got - ref);
                    if (abs_err > 3.0e-6f) {{
                        printf("FAIL i=%d in=%g got=%g ref=%g abs=%g\n", i, in[i], got, ref, abs_err);
                        return 1;
                    }}
                }}
                printf("OK\n");
                return 0;
            }}
            "#
        );

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
        ea_compiler::compile_with_options(&ea, &obj, OutputMode::ObjectFile, &opts)
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
        assert_eq!(
            stdout.trim(),
            "OK",
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn test_log_approx_f32x4_boundary_points() {
        accuracy_test_impl(
            &[
                // Powers of 2 — exercise the e_int extraction across the full
                // f32 exponent range
                1.0 / 1024.0,
                1.0 / 8.0,
                0.5,
                std::f32::consts::FRAC_1_SQRT_2, // √2/2 — rebalance boundary
                1.0,
                std::f32::consts::SQRT_2, // √2 — opposite boundary
                2.0,
                8.0,
                1024.0,
                // Transcendental constants
                std::f32::consts::E,
                std::f32::consts::PI,
                // Larger values
                100.0,
                1.0e6,
                1.0e9,
            ],
            "f32x4",
        );
    }

    /// e^log(x) ≈ x roundtrip — combined exp_poly + log_approx accuracy guard.
    /// The two intrinsics use compatible Cody-Waite ln(2) splits, so the
    /// combined error should stay within ~3e-5 (each contributes ~3e-6).
    #[test]
    fn test_log_approx_then_exp_poly_roundtrip() {
        let ea = r#"
            export func k(input: *f32, output: *mut f32) {
                let x: f32x4 = load(input, 0)
                let l: f32x4 = log_approx_f32(x)
                let r: f32x4 = exp_poly_f32(l)
                store(output, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <math.h>
            extern void k(const float *input, float *output);
            int main(void) {
                float in[4] = {0.5f, 2.0f, 7.5f, 42.0f};
                float out[4] = {0};
                k(in, out);
                for (int i = 0; i < 4; ++i) {
                    float rel = fabsf(out[i] - in[i]) / in[i];
                    if (rel > 1.0e-4f) {
                        printf("FAIL i=%d in=%g out=%g rel=%g\n", i, in[i], out[i], rel);
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
    fn test_log_approx_f32x4_random() {
        // Sample log-uniformly over [0.01, 100] — i.e., 2^[-7, 7] roughly.
        // Avoids the x→0 region where the approximation degrades.
        let mut points = Vec::new();
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to [log(0.01), log(100)] = [-4.6, 4.6], then exp.
            let u = state as f32 / u32::MAX as f32;
            let logx = -4.6 + 9.2 * u;
            let x = logx.exp();
            points.push(x);
        }
        accuracy_test_impl(&points, "f32x4");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_log_approx_f32x8_boundary_points() {
        accuracy_test_impl(
            &[
                1.0 / 1024.0,
                1.0 / 8.0,
                0.5,
                std::f32::consts::FRAC_1_SQRT_2,
                1.0,
                std::f32::consts::SQRT_2,
                2.0,
                8.0,
                1024.0,
                std::f32::consts::E,
                std::f32::consts::PI,
                100.0,
                1.0e6,
                1.0e9,
            ],
            "f32x8",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_log_approx_f32x8_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xCAFE_F00D;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = state as f32 / u32::MAX as f32;
            let logx = -4.6 + 9.2 * u;
            let x = logx.exp();
            points.push(x);
        }
        accuracy_test_impl(&points, "f32x8");
    }

    /// CRITICAL regression guard: a future "simplification" of
    /// compile_log_approx_f32 to delegate to compile_log (which doesn't
    /// exist yet but would scalarize via libm) would silently undo the
    /// vectorization that motivates the intrinsic. Pins absence of @llvm.log.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_log_approx_f32_does_not_emit_llvm_log() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return log_approx_f32(v)
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
            !ir.contains("@llvm.log"),
            "log_approx_f32 must NOT lower to @llvm.log; found scalarization-prone intrinsic:\n{ir}"
        );
    }

    /// Confirm the bit-manipulation + Horner pattern is emitted for f32x8.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_log_approx_f32x8_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return log_approx_f32(v)
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

        // Horner needs ≥8 FMAs for the polynomial; plus the recombine
        // (e·c1, e·c2) FMAs add ~2 more.
        let fma_count = ir.matches("@llvm.fma.v8f32").count();
        assert!(
            fma_count >= 8,
            "expected at least 8 @llvm.fma.v8f32 calls (8 polynomial + 2 recombine); got {fma_count}\nIR:\n{ir}"
        );
        // Bit manipulation: bitcast float↔int, lshr (right shift), and/or
        assert!(
            ir.contains("bitcast <8 x float>") && ir.contains("to <8 x i32>"),
            "expected float→int bitcast for exponent extraction\nIR:\n{ir}"
        );
        assert!(
            ir.contains("bitcast <8 x i32>") && ir.contains("to <8 x float>"),
            "expected int→float bitcast for mantissa reconstruction\nIR:\n{ir}"
        );
        assert!(
            ir.contains("lshr <8 x i32>"),
            "expected lshr <8 x i32> for exponent extraction\nIR:\n{ir}"
        );
        // sitofp to convert exponent to f32
        assert!(
            ir.contains("sitofp <8 x i32>"),
            "expected sitofp <8 x i32> for exponent → f32 conversion\nIR:\n{ir}"
        );
    }

    #[test]
    fn test_log_approx_f32x4_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x4) -> f32x4 {
                return log_approx_f32(v)
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
        assert!(!ir.contains("@llvm.log"), "no @llvm.log expected:\n{ir}");
        let fma_count = ir.matches("@llvm.fma.v4f32").count();
        assert!(
            fma_count >= 8,
            "expected at least 8 @llvm.fma.v4f32 calls; got {fma_count}\nIR:\n{ir}"
        );
        assert!(
            ir.contains("sitofp <4 x i32>"),
            "expected sitofp <4 x i32>:\n{ir}"
        );
    }

    /// Scalar f32 should fail with helpful message pointing at log().
    #[test]
    fn test_log_approx_f32_rejects_scalar_f32() {
        let src = r#"
            export func k(x: f32) -> f32 {
                return log_approx_f32(x)
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
    fn test_log_approx_f32_rejects_f64_vector() {
        let src = r#"
            export func k(v: f64x2) -> f64x2 {
                return log_approx_f32(v)
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
    fn test_log_approx_f32_rejects_integer_vector() {
        let src = r#"
            export func k(v: i32x4) -> i32x4 {
                return log_approx_f32(v)
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
    fn test_log_approx_f32_rejects_wrong_arity() {
        let src = r#"
            export func k(a: f32x4, b: f32x4) -> f32x4 {
                return log_approx_f32(a, b)
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
