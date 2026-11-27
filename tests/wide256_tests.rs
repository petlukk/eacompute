//! 256-bit integer gaps (v1.21.0) for Tiro's 256-bit row kernels:
//! bitcast_u8x32 and bitcast_i16x16 (no instruction) and
//! widen_u8x16_u16x16 (x86 vpmovzxbw to a ymm register; ARM error).
//!
//! Tests that compile for the x86 triple are gated to x86 hosts.

#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    fn asm(src: &str, triple: &str, cpu: &str) -> Result<String, String> {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions { target_triple: Some(triple.to_string()), target_cpu: Some(cpu.to_string()), ..CompileOptions::default() };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts).map_err(|e| e.to_string())?;
        Ok(std::fs::read_to_string(out).unwrap())
    }

    const BYTES: &str = "export func f(p: *i32, out: *mut u8) {\n    let v: i32x8 = load(p, 0)\n    store(out, 0, bitcast_u8x32(v))\n}\n";
    const HALVES: &str = "export func f(p: *u8, out: *mut i16) {\n    let v: u8x32 = load(p, 0)\n    store(out, 0, bitcast_i16x16(v))\n}\n";
    const WIDEN: &str = "export func f(p: *u8, out: *mut u16) {\n    let v: u8x16 = load(p, 0)\n    store(out, 0, widen_u8x16_u16x16(v))\n}\n";

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bitcast_u8x32_little_endian() {
        assert_c_interop(
            BYTES,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const int32_t *p, uint8_t *out);
            int main(void) {
                int32_t p[8] = {258, -1, 0, 0, 0, 0, 0, 16777216};
                uint8_t o[32];
                f(p, o);
                printf("%d %d %d %d %d %d\n", o[0], o[1], o[2], o[4], o[7], o[31]);
                return 0;
            }
            "#,
            "2 1 0 255 255 1",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bitcast_i16x16_pairs_bytes() {
        assert_c_interop(
            HALVES,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const uint8_t *p, int16_t *out);
            int main(void) {
                uint8_t p[32] = {0};
                p[0] = 1; p[1] = 2; p[30] = 0xff; p[31] = 0xff;
                int16_t o[16];
                f(p, o);
                printf("%d %d\n", o[0], o[15]);
                return 0;
            }
            "#,
            "513 -1",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn widen_u8x16_u16x16_zero_extends() {
        assert_c_interop(
            WIDEN,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const uint8_t *p, uint16_t *out);
            int main(void) {
                uint8_t p[16];
                for (int k = 0; k < 16; k++) p[k] = (uint8_t)(k * 17);
                p[2] = 128;
                uint16_t o[16];
                f(p, o);
                printf("%d %d %d %d\n", o[0], o[1], o[2], o[15]);
                return 0;
            }
            "#,
            "0 17 128 255",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn widen_is_vpmovzxbw_ymm() {
        let s = asm(WIDEN, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpmovzxbw") && s.contains("ymm"), "{s}");
    }

    #[test]
    fn widen_is_an_error_on_arm() {
        let e = asm(WIDEN, "aarch64-unknown-linux-gnu", "generic").unwrap_err();
        assert!(e.contains("widen_u8_u16_hi"), "{e}");
    }

    #[test]
    fn types_are_checked() {
        assert_typecheck_error("export func f(a: i32x4) -> u8x32 { return bitcast_u8x32(a) }", "bit widths");
        assert_typecheck_error("export func f(a: u8x32) -> u16x16 { return widen_u8x16_u16x16(a) }", "u8x16");
    }
}
