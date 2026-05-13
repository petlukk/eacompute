//! `widen_u8_u16_hi(u8x16) -> u16x8` — partner to existing `widen_u8_u16`
//! (which zero-extends the LOW 8 lanes). Pair them to zero-extend all 16
//! lanes of a u8x16 in two calls without a manual shuffle step.
//!
//! Codegen (verified during development):
//!   - ARM: `ushll v.8h, v.8b, #0` (lo) / `ushll2 v.8h, v.16b, #0` (hi)
//!     — single NEON instruction each.
//!   - x86: `vpmovzxbw` (lo) / `vpxor + vpunpckhbw` (hi) — clean inline ops.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    // --- Numeric correctness on host ---

    #[test]
    fn widen_u8_u16_extracts_low_8_lanes() {
        // Sanity: regression-pin the existing lo behavior alongside the new hi.
        assert_c_interop(
            r#"
            export func test(input: *u8, out: *mut u16) {
                let v: u8x16 = load(input, 0)
                let w: u16x8 = widen_u8_u16(v)
                store(out, 0, w)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, uint16_t*);
            int main() {
                uint8_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                     50, 60, 70, 80, 90, 100, 110, 120};
                uint16_t out[8];
                test(input, out);
                printf("%u %u %u %u %u %u %u %u\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "1 2 3 4 5 6 7 8",
        );
    }

    #[test]
    fn widen_u8_u16_hi_extracts_high_8_lanes() {
        assert_c_interop(
            r#"
            export func test(input: *u8, out: *mut u16) {
                let v: u8x16 = load(input, 0)
                let w: u16x8 = widen_u8_u16_hi(v)
                store(out, 0, w)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, uint16_t*);
            int main() {
                uint8_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                     50, 60, 70, 80, 90, 100, 110, 120};
                uint16_t out[8];
                test(input, out);
                printf("%u %u %u %u %u %u %u %u\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "50 60 70 80 90 100 110 120",
        );
    }

    #[test]
    fn widen_u8_u16_hi_zero_extends_high_bit_set_lanes() {
        // u8 lanes with the high bit set (>= 128) must zero-extend, not
        // sign-extend. e.g. 0xFF → 0x00FF, not 0xFFFF.
        assert_c_interop(
            r#"
            export func test(input: *u8, out: *mut u16) {
                let v: u8x16 = load(input, 0)
                let w: u16x8 = widen_u8_u16_hi(v)
                store(out, 0, w)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, uint16_t*);
            int main() {
                uint8_t input[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                     0xFF, 0x80, 0xC0, 0xE0, 0xF0, 0xF8, 0xFC, 0xFE};
                uint16_t out[8];
                test(input, out);
                printf("%x %x %x %x %x %x %x %x\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "ff 80 c0 e0 f0 f8 fc fe",
        );
    }

    #[test]
    fn widen_u8_u16_lo_hi_pair_full_widening() {
        // Pair the two calls to widen all 16 lanes of a u8x16.
        assert_c_interop(
            r#"
            export func test(input: *u8, out: *mut u16) {
                let v: u8x16 = load(input, 0)
                let lo: u16x8 = widen_u8_u16(v)
                let hi: u16x8 = widen_u8_u16_hi(v)
                store(out, 0, lo)
                store(out, 8, hi)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, uint16_t*);
            int main() {
                uint8_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8,
                                     9, 10, 11, 12, 13, 14, 15, 16};
                uint16_t out[16];
                test(input, out);
                for (int i = 0; i < 16; i++) printf("%u%c", out[i], i==15?'\n':' ');
                return 0;
            }
            "#,
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
        );
    }

    // --- Type-mismatch rejections ---

    fn try_compile(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile(source, &obj_path, OutputMode::ObjectFile)
    }

    #[test]
    fn widen_u8_u16_hi_rejects_wrong_arity() {
        let err = try_compile("export func f() -> u16x8 { return widen_u8_u16_hi() }\n")
            .expect_err("zero args should fail");
        assert!(format!("{err}").contains("expects 1 argument"));
    }

    #[test]
    fn widen_u8_u16_hi_rejects_signed_input() {
        let err = try_compile("export func f(a: i8x16) -> u16x8 { return widen_u8_u16_hi(a) }\n")
            .expect_err("i8x16 should fail (signed)");
        assert!(format!("{err}").contains("widen_u8_u16_hi expects u8x16"));
    }

    #[test]
    fn widen_u8_u16_hi_rejects_wrong_width() {
        let err = try_compile("export func f(a: u8x32) -> u16x8 { return widen_u8_u16_hi(a) }\n")
            .expect_err("u8x32 should fail (width 32)");
        assert!(format!("{err}").contains("widen_u8_u16_hi expects u8x16"));
    }

    // --- ARM cross-compile sanity ---

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    #[test]
    fn widen_u8_u16_hi_compiles_on_arm() {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(
            "export func f(a: u8x16) -> u16x8 { return widen_u8_u16_hi(a) }\n",
            &obj_path,
            OutputMode::ObjectFile,
            &arm_opts(),
        )
        .expect("widen_u8_u16_hi must compile on ARM");
    }
}
