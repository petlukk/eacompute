//! `lo128_i16x16` / `hi128_i16x16` / `lo128_u16x16` / `hi128_u16x16` /
//! `lo256_i16x32` / `hi256_i16x32` / `lo256_u16x32` / `hi256_u16x32` —
//! fills the i16/u16 gap in the existing lo*/hi* lane-extractor family
//! (which previously covered only i8/u8/i32/f32). The `u16x32` token
//! itself was added in v1.14.0 alongside the unsigned 512-bit lane pair.
//!
//! Motivating use: the Phase 3 `madd_i16` ARM-recipe currently has to walk
//! around the missing i16 lane extractors. ROADMAP.md entry under
//! "API consistency (v1.12.0 batch)".
//!
//! No new codegen — these dispatch into the existing generic
//! `check_lo_extract` / `check_hi_extract` and `compile_lo_extract` /
//! `compile_hi_extract`. Pure dispatch additions.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    // --- Numeric correctness on host (x86) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn lo128_i16x16_extracts_low_8_lanes() {
        assert_c_interop(
            r#"
            export func test(input: *i16, out: *mut i16) {
                let v: i16x16 = load(input, 0)
                let lo: i16x8 = lo128_i16x16(v)
                store(out, 0, lo)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int16_t*, int16_t*);
            int main() {
                int16_t input[16] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                       40,  50,  60,  70,  80,  90, 100, 110};
                int16_t out[8];
                test(input, out);
                printf("%d %d %d %d %d %d %d %d\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }
            "#,
            "-100 -90 -80 -70 -60 -50 -40 -30",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn hi128_i16x16_extracts_high_8_lanes() {
        assert_c_interop(
            r#"
            export func test(input: *i16, out: *mut i16) {
                let v: i16x16 = load(input, 0)
                let hi: i16x8 = hi128_i16x16(v)
                store(out, 0, hi)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int16_t*, int16_t*);
            int main() {
                int16_t input[16] = {-100, -90, -80, -70, -60, -50, -40, -30,
                                       40,  50,  60,  70,  80,  90, 100, 110};
                int16_t out[8];
                test(input, out);
                printf("%d %d %d %d %d %d %d %d\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }
            "#,
            "40 50 60 70 80 90 100 110",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn lo128_u16x16_extracts_low_8_unsigned() {
        assert_c_interop(
            r#"
            export func test(input: *u16, out: *mut u16) {
                let v: u16x16 = load(input, 0)
                let lo: u16x8 = lo128_u16x16(v)
                store(out, 0, lo)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint16_t*, uint16_t*);
            int main() {
                uint16_t input[16] = {  1,  2,  3,  4,  5,  6,  7,  8,
                                      100,200,300,400,500,600,700,800};
                uint16_t out[8];
                test(input, out);
                printf("%u %u %u %u %u %u %u %u\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }
            "#,
            "1 2 3 4 5 6 7 8",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn hi128_u16x16_extracts_high_8_unsigned() {
        assert_c_interop(
            r#"
            export func test(input: *u16, out: *mut u16) {
                let v: u16x16 = load(input, 0)
                let hi: u16x8 = hi128_u16x16(v)
                store(out, 0, hi)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint16_t*, uint16_t*);
            int main() {
                uint16_t input[16] = {  1,  2,  3,  4,  5,  6,  7,  8,
                                      100,200,300,400,500,600,700,800};
                uint16_t out[8];
                test(input, out);
                printf("%u %u %u %u %u %u %u %u\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }
            "#,
            "100 200 300 400 500 600 700 800",
        );
    }

    // --- 512-bit input (AVX-512BW): kernel is compile-only here because the
    //     host runner may not have AVX-512. Correctness of the generic
    //     extract helper is already covered above; this just checks the
    //     dispatch entry is wired correctly. ---

    fn try_compile(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile(source, &obj_path, OutputMode::ObjectFile)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn lo256_i16x32_compiles_on_x86() {
        try_compile("export func f(a: i16x32) -> i16x16 { return lo256_i16x32(a) }\n")
            .expect("lo256_i16x32 must compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn hi256_i16x32_compiles_on_x86() {
        try_compile("export func f(a: i16x32) -> i16x16 { return hi256_i16x32(a) }\n")
            .expect("hi256_i16x32 must compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn u16x32_parses_as_vector_type() {
        // Smoke: u16x32 must be a recognized vector type annotation. Pure
        // parser exercise — codegen needs AVX-512 to actually run a u16x32
        // load/store, so the function body just passes the value through.
        try_compile("export func f(a: u16x32) -> u16x32 { return a }\n")
            .expect("u16x32 must parse as a vector type");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn lo256_u16x32_compiles_on_x86() {
        try_compile("export func f(a: u16x32) -> u16x16 { return lo256_u16x32(a) }\n")
            .expect("lo256_u16x32 must compile on x86");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn hi256_u16x32_compiles_on_x86() {
        try_compile("export func f(a: u16x32) -> u16x16 { return hi256_u16x32(a) }\n")
            .expect("hi256_u16x32 must compile on x86");
    }

    #[test]
    fn lo256_u16x32_rejects_signed() {
        let err = try_compile("export func f(a: i16x32) -> i16x16 { return lo256_u16x32(a) }\n")
            .expect_err("i16x32 input should fail u16x32 typeck");
        assert!(
            format!("{err}").contains("lo256_u16x32 expects u16x32"),
            "got: {err}"
        );
    }

    #[test]
    fn hi256_u16x32_rejects_signed() {
        let err = try_compile("export func f(a: i16x32) -> i16x16 { return hi256_u16x32(a) }\n")
            .expect_err("i16x32 input should fail u16x32 typeck");
        assert!(
            format!("{err}").contains("hi256_u16x32 expects u16x32"),
            "got: {err}"
        );
    }

    // --- Type-mismatch rejections ---

    #[test]
    fn lo128_i16x16_rejects_wrong_width() {
        let err = try_compile("export func f(a: i16x8) -> i16x4 { return lo128_i16x16(a) }\n")
            .expect_err("i16x8 input should fail");
        assert!(
            format!("{err}").contains("lo128_i16x16 expects i16x16"),
            "got: {err}"
        );
    }

    #[test]
    fn lo128_u16x16_rejects_signed() {
        let err = try_compile("export func f(a: i16x16) -> i16x8 { return lo128_u16x16(a) }\n")
            .expect_err("i16x16 input should fail u16x16 typeck");
        assert!(
            format!("{err}").contains("lo128_u16x16 expects u16x16"),
            "got: {err}"
        );
    }

    // --- ARM rejection: i16x16 / i16x32 are wider than 128 bits, NEON-rejected ---

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
    fn lo128_i16x16_rejected_on_arm() {
        // i16x16 is 256-bit; ARM NEON tops out at 128-bit. Caught at the
        // vector-type validation site in codegen/mod.rs.
        let err = try_compile_arm("export func f(a: i16x16) -> i16x8 { return lo128_i16x16(a) }\n")
            .expect_err("i16x16 input must fail on ARM");
        assert!(
            format!("{err}").contains("i16x16 requires AVX2"),
            "expected AVX2/narrowing hint, got: {err}"
        );
    }

    #[test]
    fn lo256_i16x32_rejected_on_arm() {
        let err =
            try_compile_arm("export func f(a: i16x32) -> i16x16 { return lo256_i16x32(a) }\n")
                .expect_err("i16x32 input must fail on ARM");
        assert!(
            format!("{err}").contains("i16x32") && format!("{err}").contains("ARM"),
            "expected ARM/narrowing message, got: {err}"
        );
    }

    #[test]
    fn lo256_u16x32_rejected_on_arm() {
        let err =
            try_compile_arm("export func f(a: u16x32) -> u16x16 { return lo256_u16x32(a) }\n")
                .expect_err("u16x32 input must fail on ARM");
        assert!(
            format!("{err}").contains("u16x32") && format!("{err}").contains("ARM"),
            "expected ARM/narrowing message, got: {err}"
        );
    }
}
