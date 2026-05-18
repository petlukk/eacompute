//! `permute_runtime` — runtime SIMD permute (x86-only, AVX2).
//!
//! Lowers to `vpermps` (f32x8) / `vpermd` (i32x8). ARM is rejected
//! with a codegen_error pointing at docs/idioms/neon-runtime-permute.md.
//!
//! Index semantics: `result[k] = table[indices[k] & 0x7]` — hardware
//! ignores the upper 29 bits of each index lane.
//!
//! All x86 numeric tests `assert_c_interop`; the codegen-assertion
//! tests additionally objdump and grep for the expected mnemonic.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[allow(unused_imports)]
    use super::common::*;
    use ea_compiler::OutputMode;
    use tempfile::TempDir;

    // --- Type-check happy paths (compile to object, no run) ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn typecheck_permute_runtime_f32x8_compiles() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        ea_compiler::compile(src, &obj, OutputMode::ObjectFile)
            .expect("f32x8 permute_runtime should typecheck");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn typecheck_permute_runtime_i32x8_compiles() {
        let src = r#"
            export func k(t: *i32, idx: *i32, out: *mut i32) {
                let tv: i32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: i32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        ea_compiler::compile(src, &obj, OutputMode::ObjectFile)
            .expect("i32x8 permute_runtime should typecheck");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_width_4() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x4 = load(t, 0)
                let iv: i32x4 = load(idx, 0)
                let r: f32x4 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "table must have width 8");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_u32_indices() {
        let src = r#"
            export func k(tv: f32x8, iv: u32x8, out: *mut f32) {
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "i32 vector");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_wrong_arity() {
        let src = r#"
            export func k(t: *f32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let r: f32x8 = permute_runtime(tv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "2 arguments");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_f64_table() {
        let src = r#"
            export func k(tv: f64x4, iv: i32x4, out: *mut f64) {
                let r: f64x4 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "f32 or i32");
    }

    // --- Functional correctness (x86, host runner) ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_identity() {
        assert_c_interop(
            r#"
            export func test(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float t[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
                int idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                float out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "10 11 12 13 14 15 16 17",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_reverse() {
        assert_c_interop(
            r#"
            export func test(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float t[8] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
                int idx[8] = {7, 6, 5, 4, 3, 2, 1, 0};
                float out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "17 16 15 14 13 12 11 10",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_broadcast_lane0() {
        assert_c_interop(
            r#"
            export func test(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float t[8] = {42.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                int idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                float out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "42 42 42 42 42 42 42 42",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_small_lut_six_active() {
        // Models the kernel_v113.ea pattern: 6 active LUT entries
        // (lanes 0..5), indices in [0..5], lanes 6 and 7 of the table
        // are zero/don't-care.
        assert_c_interop(
            r#"
            export func test(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float t[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f};
                int idx[8] = {5, 4, 3, 2, 1, 0, 1, 2};
                float out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "6 5 4 3 2 1 2 3",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_high_bits_ignored() {
        // Hardware masks index to low 3 bits. Index 8 wraps to 0, 9 to 1, etc.
        assert_c_interop(
            r#"
            export func test(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const int*, float*);
            int main() {
                float t[8] = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f};
                int idx[8] = {8, 9, 16, 17, 0, 1, 2, 3};
                float out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "100 101 100 101 100 101 102 103",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_f32x8_emits_vpermps() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        // LLVM 18 may emit either vpermps or vpermd for f32 permute
        // (shared opcode, verified in Task 0 spike).
        assert_intrinsic_in_disassembly(src, &["vpermps", "vpermd"]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_i32x8_identity() {
        assert_c_interop(
            r#"
            export func test(t: *i32, idx: *i32, out: *mut i32) {
                let tv: i32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: i32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const int*, const int*, int*);
            int main() {
                int t[8] = {100, 101, 102, 103, 104, 105, 106, 107};
                int idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
                int out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "100 101 102 103 104 105 106 107",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_i32x8_reverse() {
        assert_c_interop(
            r#"
            export func test(t: *i32, idx: *i32, out: *mut i32) {
                let tv: i32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: i32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const int*, const int*, int*);
            int main() {
                int t[8] = {100, 101, 102, 103, 104, 105, 106, 107};
                int idx[8] = {7, 6, 5, 4, 3, 2, 1, 0};
                int out[8];
                test(t, idx, out);
                for (int i = 0; i < 8; ++i) printf("%d ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "107 106 105 104 103 102 101 100",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn permute_runtime_i32x8_emits_vpermd() {
        let src = r#"
            export func k(t: *i32, idx: *i32, out: *mut i32) {
                let tv: i32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: i32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        // LLVM 18 emits vpermps for both .permps and .permd intrinsics
        // (shared opcode, verified in Task 0 spike). Accept either.
        assert_intrinsic_in_disassembly(src, &["vpermps", "vpermd"]);
    }

    #[test]
    fn permute_runtime_arm_rejected() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let opts = ea_compiler::CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("permute_runtime on ARM should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("NEON"),
            "error should mention NEON, got: {msg}"
        );
        assert!(
            msg.contains("neon-runtime-permute.md"),
            "error must mention the docs file, got: {msg}"
        );
    }

    #[test]
    fn permute_runtime_arm_rejected_in_negate() {
        // Regression: permute_runtime wrapped in a Negate expression must
        // still be rejected on ARM (walker must recurse into Negate).
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = -permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let opts = ea_compiler::CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("permute_runtime inside negate should still fail on ARM");
        let msg = format!("{err}");
        assert!(
            msg.contains("neon-runtime-permute.md"),
            "expected idiom-doc reference, got: {msg}"
        );
    }

    #[test]
    fn permute_runtime_arm_rejected_in_call_arg() {
        // Regression: permute_runtime nested inside another call's argument
        // must still be rejected (walker must recurse into Call.args).
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                store(out, 0, permute_runtime(tv, iv))
            }
        "#;
        let opts = ea_compiler::CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("permute_runtime inside store() arg should still fail on ARM");
        let msg = format!("{err}");
        assert!(
            msg.contains("neon-runtime-permute.md"),
            "expected idiom-doc reference, got: {msg}"
        );
    }
}
