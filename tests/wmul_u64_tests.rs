//! `wmul_u64_lo` / `wmul_u64_hi` / `wmul_u64` — u32×u32 → u64 widening
//! multiply family (originally Poly1305 unblocker, v1.12.0).
//!
//! Two shapes are available:
//!   - `wmul_u64_lo(u32x4, u32x4) -> u64x2` operates on lanes 0,1 (cross-platform)
//!   - `wmul_u64_hi(u32x4, u32x4) -> u64x2` operates on lanes 2,3 (cross-platform)
//!   - `wmul_u64(u32x4, u32x4) -> u64x4` widens all four lanes (x86-only,
//!     because the u64x4 return type is 256-bit and rejected by the
//!     existing ARM >128-bit guard — v1.14.0)
//!
//! Numeric tests use values near the u32 boundary so a 32-bit truncation
//! would be visible. ARM is verified by cross-compile to object; numeric
//! x86 tests use the host runner. Same kernels both targets — exact lane
//! semantics are contract.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    // --- Numeric correctness on host (x86) ---

    #[test]
    fn test_wmul_u64_lo_basic() {
        // u32 lanes [3, 5, 7, 11] × [10, 20, 30, 40]
        // lo widens lanes 0,1 → u64x2: [30, 100]
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x2 = wmul_u64_lo(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {3, 5, 7, 11};
                uint32_t b[4] = {10, 20, 30, 40};
                uint64_t out[2];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 "\n", out[0], out[1]);
                return 0;
            }
            "#,
            "30 100",
        );
    }

    #[test]
    fn test_wmul_u64_hi_basic() {
        // u32 lanes [3, 5, 7, 11] × [10, 20, 30, 40]
        // hi widens lanes 2,3 → u64x2: [210, 440]
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x2 = wmul_u64_hi(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {3, 5, 7, 11};
                uint32_t b[4] = {10, 20, 30, 40};
                uint64_t out[2];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 "\n", out[0], out[1]);
                return 0;
            }
            "#,
            "210 440",
        );
    }

    #[test]
    fn test_wmul_u64_no_32bit_truncation() {
        // (2^31) * 3 = 6442450944 — overflows u32 by ~50%, so a truncated
        // 32-bit multiply would produce 0x80000000 * 3 mod 2^32 =
        // 0x80000000 = 2147483648. The correct u64 result is 6442450944.
        // Tests that the widening genuinely uses 64-bit arithmetic.
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x2 = wmul_u64_lo(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {0x80000000u, 0xFFFFFFFFu, 0, 0};
                uint32_t b[4] = {3, 2, 0, 0};
                uint64_t out[2];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 "\n", out[0], out[1]);
                return 0;
            }
            "#,
            // 0x80000000 * 3        = 6442450944
            // 0xFFFFFFFF * 2        = 8589934590
            "6442450944 8589934590",
        );
    }

    #[test]
    fn test_wmul_u64_max_u32_squared() {
        // (2^32 - 1)^2 = 18446744065119617025 = 0xFFFFFFFE00000001
        // — the maximum unsigned 32×32→64 product. Anything smaller would
        // be wrong.
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x2 = wmul_u64_lo(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0};
                uint32_t b[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0, 0};
                uint64_t out[2];
                test(a, b, out);
                printf("%" PRIx64 " %" PRIx64 "\n", out[0], out[1]);
                return 0;
            }
            "#,
            "fffffffe00000001 fffffffe00000001",
        );
    }

    #[test]
    fn test_wmul_u64_lo_hi_pair_full_widening() {
        // Use both lo+hi to widen all 4 lanes of u32x4 in two calls.
        // Verifies the lane semantics are consistent across the pair.
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let lo: u64x2 = wmul_u64_lo(va, vb)
                let hi: u64x2 = wmul_u64_hi(va, vb)
                store(out, 0, lo)
                store(out, 2, hi)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {100000, 200000, 300000, 400000};
                uint32_t b[4] = {  9999,  9998,  9997,  9996};
                uint64_t out[4];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            // 100000 * 9999  = 999900000
            // 200000 * 9998  = 1999600000
            // 300000 * 9997  = 2999100000
            // 400000 * 9996  = 3998400000
            "999900000 1999600000 2999100000 3998400000",
        );
    }

    // --- Fused full-widening form: wmul_u64(u32x4, u32x4) -> u64x4 ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_wmul_u64_fused_all_lanes() {
        // Single call widens all 4 lanes — same numeric expectation as the
        // lo_hi_pair test above, but emitted as one intrinsic instead of two.
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x4 = wmul_u64(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {100000, 200000, 300000, 400000};
                uint32_t b[4] = {  9999,  9998,  9997,  9996};
                uint64_t out[4];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "999900000 1999600000 2999100000 3998400000",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_wmul_u64_fused_max_u32_squared() {
        // (2^32 - 1)^2 = 0xFFFFFFFE00000001 = 18446744065119617025
        // Any 32-bit truncation would show 1 instead.
        assert_c_interop(
            r#"
            export func test(a: *u32, b: *u32, out: *mut u64) {
                let va: u32x4 = load(a, 0)
                let vb: u32x4 = load(b, 0)
                let prod: u64x4 = wmul_u64(va, vb)
                store(out, 0, prod)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <inttypes.h>
            extern void test(const uint32_t*, const uint32_t*, uint64_t*);
            int main() {
                uint32_t a[4] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
                uint32_t b[4] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
                uint64_t out[4];
                test(a, b, out);
                printf("%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
                    out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "18446744065119617025 18446744065119617025 18446744065119617025 18446744065119617025",
        );
    }

    /// IR-pattern guard: the fused form lowers to zext + mul on a full u64x4
    /// vector, which the x86 backend pattern-matches to two `vpmuludq` +
    /// interleave. A future "simplification" that decomposed this into the
    /// existing `wmul_u64_lo` + `wmul_u64_hi` pair followed by a shuffle
    /// would still be correct but would defeat LLVM's clean pattern-match.
    /// Pin the canonical form.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_wmul_u64_fused_emits_zext_mul_pattern() {
        let src = r#"
            export func k(a: u32x4, b: u32x4) -> u64x4 {
                return wmul_u64(a, b)
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
            ir.contains("zext <4 x i32>") && ir.contains("to <4 x i64>"),
            "expected zext <4 x i32> → <4 x i64>:\n{ir}"
        );
        assert!(
            ir.contains("mul <4 x i64>"),
            "expected mul <4 x i64> (the full-width multiply):\n{ir}"
        );
        assert!(
            !ir.contains("@llvm.x86.sse2.pmulu.dq"),
            "must not call the deprecated pmuludq intrinsic; LLVM pattern-matches the IR form:\n{ir}"
        );
    }

    // --- Type-checker rejections ---

    fn try_compile(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile(source, &obj_path, OutputMode::ObjectFile)
    }

    #[test]
    fn wmul_u64_lo_rejects_wrong_arity() {
        let err = try_compile("export func f(a: u32x4) -> u64x2 { return wmul_u64_lo(a) }\n")
            .expect_err("missing second argument should fail");
        assert!(format!("{err}").contains("expects 2 arguments"));
    }

    #[test]
    fn wmul_u64_lo_rejects_wrong_elem_type() {
        let err = try_compile(
            "export func f(a: i32x4, b: i32x4) -> u64x2 { return wmul_u64_lo(a, b) }\n",
        )
        .expect_err("i32x4 input should fail");
        assert!(format!("{err}").contains("expects (u32x4, u32x4)"));
    }

    #[test]
    fn wmul_u64_hi_rejects_wrong_width() {
        // u32x8 (AVX2 width) — not the documented contract.
        let err = try_compile(
            "export func f(a: u32x8, b: u32x8) -> u64x2 { return wmul_u64_hi(a, b) }\n",
        )
        .expect_err("u32x8 input should fail");
        assert!(format!("{err}").contains("expects (u32x4, u32x4)"));
    }

    #[test]
    fn wmul_u64_fused_rejects_wrong_arity() {
        let err = try_compile("export func f(a: u32x4) -> u64x4 { return wmul_u64(a) }\n")
            .expect_err("missing second argument should fail");
        assert!(format!("{err}").contains("expects 2 arguments"));
    }

    #[test]
    fn wmul_u64_fused_rejects_signed() {
        let err =
            try_compile("export func f(a: i32x4, b: i32x4) -> u64x4 { return wmul_u64(a, b) }\n")
                .expect_err("i32x4 input should fail");
        assert!(format!("{err}").contains("expects (u32x4, u32x4)"));
    }

    #[test]
    fn wmul_u64_fused_rejects_wrong_width() {
        // Wider input widths require new lexer tokens — explicitly deferred
        // in v1.14.0. Until then, u32x8 falls through to a named type and
        // typeck rejects with the contract message.
        let err =
            try_compile("export func f(a: u32x8, b: u32x8) -> u64x4 { return wmul_u64(a, b) }\n")
                .expect_err("u32x8 input should fail");
        assert!(
            format!("{err}").contains("expects (u32x4, u32x4)")
                || format!("{err}").contains("u32x4"),
            "got: {err}"
        );
    }

    // --- ARM cross-compile sanity ---

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
    fn wmul_u64_lo_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: u32x4, b: u32x4) -> u64x2 { return wmul_u64_lo(a, b) }\n",
        )
        .expect("wmul_u64_lo must compile on ARM");
    }

    #[test]
    fn wmul_u64_hi_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: u32x4, b: u32x4) -> u64x2 { return wmul_u64_hi(a, b) }\n",
        )
        .expect("wmul_u64_hi must compile on ARM");
    }

    /// The fused full-widening form returns u64x4 (256-bit), which trips the
    /// existing ARM >128-bit type-validation guard before the intrinsic
    /// dispatcher even runs. NEON's widest u64 vector is u64x2; supporting
    /// u64x4 on ARM would need two umull/umull2 calls and either a struct
    /// return or two output pointers — a meaningful API choice deferred
    /// until a consumer asks. The existing `wmul_u64_lo` / `wmul_u64_hi`
    /// pair (returning u64x2 each) remains the ARM-compatible path.
    #[test]
    fn wmul_u64_fused_rejected_on_arm() {
        let err = try_compile_arm(
            "export func f(a: u32x4, b: u32x4) -> u64x4 { return wmul_u64(a, b) }\n",
        )
        .expect_err("u64x4 must fail on ARM via the >128-bit guard");
        let msg = format!("{err}");
        assert!(
            msg.contains("u64x4") && (msg.contains("AVX2") || msg.contains("ARM")),
            "expected u64x4/AVX2/ARM message, got: {msg}"
        );
    }
}
