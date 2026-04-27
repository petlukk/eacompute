#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    /// Smoke test: exp_poly_f32(splat(0.0)) should give all 1.0s.
    /// Exercises the full pipeline (range reduction with n=0, polynomial
    /// returns 1, ldexp adds 0 to exponent — round trip).
    #[test]
    fn test_exp_poly_f32x4_at_zero() {
        let ea = r#"
            export func k(out: *mut f32) {
                let z: f32x4 = splat(0.0)
                let r: f32x4 = exp_poly_f32(z)
                store(out, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void k(float *out);
            int main(void) {
                float out[4] = {0};
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
        assert_eq!(stdout.trim(), "1\n1\n1\n1");
    }

    /// Compile a kernel that runs exp_poly_f32 over `inputs` lane-by-lane,
    /// link with C harness that calls expf, assert relative error ≤ 4e-6.
    fn accuracy_test_impl(inputs: &[f32], vector_type: &str) {
        let lanes = if vector_type == "f32x4" { 4 } else { 8 };

        let ea = format!(
            r#"
            export func k(input: *f32, output: *mut f32, n: i32) {{
                let mut i: i32 = 0
                while i + {lanes} <= n {{
                    let v: {vector_type} = load(input, i)
                    let r: {vector_type} = exp_poly_f32(v)
                    store(output, i, r)
                    i = i + {lanes}
                }}
            }}
            "#
        );

        // Pad inputs up to a multiple of `lanes` with zeros
        let mut padded = inputs.to_vec();
        while !padded.len().is_multiple_of(lanes) {
            padded.push(0.0);
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
                    float ref = expf(in[i]);
                    float got = out[i];
                    float rel;
                    if (ref > 1e-30f) {{
                        rel = fabsf(got - ref) / ref;
                    }} else {{
                        rel = fabsf(got - ref);
                    }}
                    if (rel > 4.0e-6f) {{
                        printf("FAIL i=%d in=%g got=%g ref=%g rel=%g\n", i, in[i], got, ref, rel);
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
    fn test_exp_poly_f32x4_boundary_points() {
        accuracy_test_impl(
            &[
                -50.0, -25.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 25.0, 50.0,
            ],
            "f32x4",
        );
    }

    #[test]
    fn test_exp_poly_f32x4_reduction_stress() {
        use std::f32::consts::LN_2;
        accuracy_test_impl(
            &[
                -LN_2,
                -LN_2 / 2.0,
                0.0,
                LN_2 / 2.0,
                LN_2,
                2.0 * LN_2,
                3.0 * LN_2,
            ],
            "f32x4",
        );
    }

    #[test]
    fn test_exp_poly_f32x4_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xDEAD_BEEF;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = (state as f32 / u32::MAX as f32) * 100.0 - 50.0;
            points.push(f);
        }
        accuracy_test_impl(&points, "f32x4");
    }

    #[test]
    fn test_exp_poly_f32x8_boundary_points() {
        accuracy_test_impl(
            &[
                -50.0, -25.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 25.0, 50.0,
            ],
            "f32x8",
        );
    }

    #[test]
    fn test_exp_poly_f32x8_reduction_stress() {
        use std::f32::consts::LN_2;
        accuracy_test_impl(
            &[
                -LN_2,
                -LN_2 / 2.0,
                0.0,
                LN_2 / 2.0,
                LN_2,
                2.0 * LN_2,
                3.0 * LN_2,
            ],
            "f32x8",
        );
    }

    #[test]
    fn test_exp_poly_f32x8_random() {
        let mut points = Vec::new();
        let mut state: u32 = 0xCAFE_F00D;
        for _ in 0..256 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = (state as f32 / u32::MAX as f32) * 100.0 - 50.0;
            points.push(f);
        }
        accuracy_test_impl(&points, "f32x8");
    }

    /// CRITICAL regression guard: a future "simplification" of compile_exp_poly_f32
    /// to delegate to compile_exp would silently undo the entire feature. This
    /// test pins the property that motivated the whole spec.
    #[test]
    fn test_exp_poly_f32_does_not_emit_llvm_exp() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return exp_poly_f32(v)
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
            !ir.contains("@llvm.exp"),
            "exp_poly_f32 must NOT lower to @llvm.exp; found scalarization-prone intrinsic in IR:\n{ir}"
        );
    }

    /// Confirm the polynomial / ldexp pattern is actually emitted for f32x8.
    #[test]
    fn test_exp_poly_f32x8_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x8) -> f32x8 {
                return exp_poly_f32(v)
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
            fma_count >= 6,
            "expected at least 6 @llvm.fma.v8f32 calls (8 total: 6 polynomial + 2 range reduction); got {fma_count}\nIR:\n{ir}"
        );
        assert!(
            ir.contains("@llvm.nearbyint"),
            "expected @llvm.nearbyint for round-to-nearest in range reduction\nIR:\n{ir}"
        );
        assert!(
            ir.contains("bitcast <8 x float>") && ir.contains("to <8 x i32>"),
            "expected float->int bitcast for ldexp\nIR:\n{ir}"
        );
        assert!(
            ir.contains("bitcast <8 x i32>") && ir.contains("to <8 x float>"),
            "expected int->float bitcast for ldexp\nIR:\n{ir}"
        );
        assert!(
            ir.contains("shl <8 x i32>"),
            "expected shl <8 x i32> for shifting n into exponent field\nIR:\n{ir}"
        );
        assert!(
            ir.contains("add <8 x i32>"),
            "expected add <8 x i32> for ldexp exponent add\nIR:\n{ir}"
        );
    }

    /// Same checks for f32x4.
    #[test]
    fn test_exp_poly_f32x4_emits_expected_pattern() {
        let src = r#"
            export func k(v: f32x4) -> f32x4 {
                return exp_poly_f32(v)
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
        assert!(!ir.contains("@llvm.exp"), "no @llvm.exp expected:\n{ir}");
        let fma_count = ir.matches("@llvm.fma.v4f32").count();
        assert!(
            fma_count >= 6,
            "expected at least 6 @llvm.fma.v4f32 calls; got {fma_count}\nIR:\n{ir}"
        );
    }

    /// Scalar f32 should fail with helpful message pointing at exp().
    #[test]
    fn test_exp_poly_f32_rejects_scalar_f32() {
        let src = r#"
            export func k(x: f32) -> f32 {
                return exp_poly_f32(x)
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

    /// f64x2 vector should fail with f64-not-supported message.
    #[test]
    fn test_exp_poly_f32_rejects_f64_vector() {
        let src = r#"
            export func k(v: f64x2) -> f64x2 {
                return exp_poly_f32(v)
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

    /// f16x8 vector should fail (cut #1 is f32-only).
    #[test]
    fn test_exp_poly_f32_rejects_f16_vector() {
        let src = r#"
            export func k(v: f16x8) -> f16x8 {
                return exp_poly_f32(v)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(), // enable f16 types
            target_triple: None,
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("f16x8 should fail");
        let msg = format!("{err}");
        // On x86-64 hosts the +fullfp16 feature triggers the ARM-only gate
        // before exp_poly_f32 ever sees the operand — that's still a correct
        // rejection. Accept any of the plausible error forms.
        assert!(
            msg.contains("f32") && msg.contains("element type")
                || msg.contains("f16")
                || msg.contains("not supported")
                || msg.contains("float")
                || msg.contains("incompatible")
                || msg.contains("ARM"),
            "error must clarify f16 rejection, got: {msg}"
        );
    }

    /// i32x4 (integer vector) should fail.
    #[test]
    fn test_exp_poly_f32_rejects_integer_vector() {
        let src = r#"
            export func k(v: i32x4) -> i32x4 {
                return exp_poly_f32(v)
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
        // Error class can be either codegen "expects f32 element type" or typeck
        // sqrt-family "requires float" — both acceptable.
        assert!(
            msg.contains("f32") || msg.contains("float") || msg.contains("integer"),
            "error must clarify why integer vector is rejected, got: {msg}"
        );
    }

    /// End-to-end softmax using exp_poly_f32. Compares against an expf-based
    /// reference. Tolerance 1e-3 relative (~2^-10 — softmax's normalize-by-sum
    /// absorbs the polynomial error).
    #[test]
    fn test_exp_poly_f32_softmax_integration() {
        let ea = r#"
            export func softmax(x: *f32, out: *mut f32) {
                let v: f32x8 = load(x, 0)
                let mx: f32 = reduce_max(v)
                let mxv: f32x8 = splat(mx)
                let shifted: f32x8 = v .- mxv
                let ev: f32x8 = exp_poly_f32(shifted)
                let s: f32 = reduce_add(ev)
                let inv: f32 = 1.0 / s
                let invv: f32x8 = splat(inv)
                let r: f32x8 = ev .* invv
                store(out, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <math.h>
            extern void softmax(const float *x, float *out);
            int main(void) {
                float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                float out[8] = {0};
                softmax(x, out);

                // Reference softmax
                float mx = x[0];
                for (int i = 1; i < 8; ++i) if (x[i] > mx) mx = x[i];
                float ref[8];
                float s = 0.0f;
                for (int i = 0; i < 8; ++i) {
                    ref[i] = expf(x[i] - mx);
                    s += ref[i];
                }
                for (int i = 0; i < 8; ++i) ref[i] /= s;

                // Compare with relative tolerance 1e-3 (~2^-10)
                for (int i = 0; i < 8; ++i) {
                    float rel = fabsf(out[i] - ref[i]) / ref[i];
                    if (rel > 1.0e-3f) {
                        printf("FAIL i=%d got=%g ref=%g rel=%g\n", i, out[i], ref[i], rel);
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
        assert_eq!(
            stdout.trim(),
            "OK",
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
