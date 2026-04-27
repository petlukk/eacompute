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
}
