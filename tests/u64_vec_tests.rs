//! `u64xN` vector type — lexer, parser, and codegen sanity.
//!
//! These types are the prerequisite for the `wmul_u64_lo` / `wmul_u64_hi`
//! intrinsics (Poly1305 unblocker). Tests verify:
//!   - tokens lex correctly
//!   - 128-bit `u64x2` works on both platforms
//!   - 256-bit `u64x4` and 512-bit `u64x8` compile on x86 and produce
//!     correct numeric results
//!   - ARM rejects `u64x4` / `u64x8` with the canonical "use narrower" hint

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    #[test]
    fn lex_u64_vec_types() {
        use ea_compiler::tokenize;
        let toks = tokenize("u64x2 u64x4 u64x8").unwrap();
        let kinds: Vec<String> = toks.iter().map(|t| format!("{:?}", t.kind)).collect();
        for expected in ["U64x2", "U64x4", "U64x8"] {
            assert!(
                kinds.iter().any(|k| k == expected),
                "{expected} token not found in {kinds:?}"
            );
        }
    }

    #[test]
    fn test_u64x2_basic_add() {
        assert_c_interop(
            r#"
            export func test(a: *u64, b: *u64, out: *mut u64) {
                let va: u64x2 = load(a, 0)
                let vb: u64x2 = load(b, 0)
                let result: u64x2 = va .+ vb
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint64_t*, const uint64_t*, uint64_t*);
            int main() {
                uint64_t a[2] = {1ULL << 40, 2ULL << 40};
                uint64_t b[2] = {3ULL << 40, 5ULL << 40};
                uint64_t out[2];
                test(a, b, out);
                printf("%lu %lu\n", out[0], out[1]);
                return 0;
            }
            "#,
            "4398046511104 7696581394432",
        );
    }

    #[test]
    fn test_u64x2_load_store_roundtrip() {
        assert_c_interop(
            r#"
            export func test(input: *u64, out: *mut u64) {
                let v: u64x2 = load(input, 0)
                store(out, 0, v)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint64_t*, uint64_t*);
            int main() {
                uint64_t input[2] = {0xCAFEBABEDEADBEEFULL, 0x0123456789ABCDEFULL};
                uint64_t out[2];
                test(input, out);
                printf("%lx %lx\n", out[0], out[1]);
                return 0;
            }
            "#,
            "cafebabedeadbeef 123456789abcdef",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_u64x4_add_x86() {
        assert_c_interop(
            r#"
            export func test(a: *u64, b: *u64, out: *mut u64) {
                let va: u64x4 = load(a, 0)
                let vb: u64x4 = load(b, 0)
                let result: u64x4 = va .+ vb
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint64_t*, const uint64_t*, uint64_t*);
            int main() {
                uint64_t a[4] = {1, 10, 100, 1000};
                uint64_t b[4] = {2, 20, 200, 2000};
                uint64_t out[4];
                test(a, b, out);
                printf("%lu %lu %lu %lu\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "3 30 300 3000",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_u64x8_add_x86() {
        assert_c_interop(
            r#"
            export func test(a: *u64, b: *u64, out: *mut u64) {
                let va: u64x8 = load(a, 0)
                let vb: u64x8 = load(b, 0)
                let result: u64x8 = va .+ vb
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint64_t*, const uint64_t*, uint64_t*);
            int main() {
                uint64_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                uint64_t b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
                uint64_t out[8];
                test(a, b, out);
                printf("%lu %lu %lu %lu %lu %lu %lu %lu\n",
                    out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "11 22 33 44 55 66 77 88",
        );
    }

    #[test]
    fn test_u64x2_splat() {
        assert_c_interop(
            r#"
            export func test(out: *mut u64) {
                let v: u64x2 = splat(42)
                store(out, 0, v)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(uint64_t*);
            int main() {
                uint64_t out[2];
                test(out);
                printf("%lu %lu\n", out[0], out[1]);
                return 0;
            }
            "#,
            "42 42",
        );
    }

    // --- ARM rejection of wider variants ---

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile_arm(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &arm_opts())
    }

    #[test]
    fn u64x2_works_on_arm() {
        try_compile_arm("export func f(x: u64x2) -> u64x2 { return x .+ x }\n")
            .expect("u64x2 (128-bit) must compile on ARM");
    }

    #[test]
    fn u64x4_rejected_on_arm_with_avx2_hint() {
        let err = try_compile_arm("export func f(x: u64x4) -> u64x4 { return x .+ x }\n")
            .expect_err("u64x4 (256-bit) must fail on ARM");
        let msg = format!("{err}");
        assert!(
            msg.contains("u64x4 requires AVX2"),
            "expected AVX2 hint, got: {msg}"
        );
        assert!(msg.contains("u64x2 on ARM"), "expected narrowing recipe");
    }

    #[test]
    fn u64x8_rejected_on_arm_with_avx512_hint() {
        let err = try_compile_arm("export func f(x: u64x8) -> u64x8 { return x .+ x }\n")
            .expect_err("u64x8 (512-bit) must fail on ARM");
        let msg = format!("{err}");
        assert!(
            msg.contains("u64x8 requires AVX-512"),
            "expected AVX-512 hint, got: {msg}"
        );
        assert!(msg.contains("u64x2 on ARM"), "expected narrowing recipe");
    }
}
