//! bitcast_u8x16: any 128-bit vector reinterpreted as u8x16, so lanes
//! loaded or computed as i32 (RGBA pixels, a masked load of whole pixels)
//! can go to the u8x16 intrinsics (`widen_u8_u16`, `shuffle`). A bitcast is
//! no instruction on either architecture.
//!
//! Tests that compile for the x86 triple are gated to x86 hosts: the LLVM on
//! the ubuntu-24.04-arm runner has no x86 backend.

#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    /// The last 1-3 pixels of a row, loaded and stored masked as i32 lanes
    /// and widened as bytes in between: the shape of a pixel blend's tail.
    const TAIL: &str = "export func tail(p: *mut i32, n: i32) {\n    let w: i32x4 = load_masked(p, 0, n)\n    let b: u8x16 = bitcast_u8x16(w)\n    let lo: u16x8 = widen_u8_u16(b)\n    let hi: u16x8 = widen_u8_u16_hi(b)\n    let r: u8x16 = pack_usat_i16x8(bitcast_i16x8(lo), bitcast_i16x8(hi))\n    store_masked(p, 0, bitcast_i32x4(r), n)\n}\n";

    fn asm(src: &str, triple: &str, cpu: &str) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions {
            target_triple: Some(triple.to_string()),
            target_cpu: Some(cpu.to_string()),
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts)
            .expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    #[test]
    fn lanes_are_little_endian_bytes() {
        assert_output_lines(
            r#"
            export func main() {
                let v: i32x4 = [258, -1, 0, 16777216]
                let b: u8x16 = bitcast_u8x16(v)
                println(b[0])
                println(b[1])
                println(b[4])
                println(b[15])
            }
            "#,
            &["2", "1", "255", "1"],
        );
    }

    #[test]
    fn masked_pixels_widen_as_bytes() {
        assert_c_interop(
            TAIL,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void tail(int32_t *p, int32_t n);
            int main(void) {
                uint32_t px[4] = {0x04030201u, 0xfffefdfcu, 0x0, 0xdeadbeefu};
                tail((int32_t *)px, 3);
                printf("%08x %08x %08x %08x\n", px[0], px[1], px[2], px[3]);
                return 0;
            }
            "#,
            "04030201 fffefdfc 00000000 deadbeef",
        );
    }

    #[test]
    fn source_must_be_128_bits() {
        assert_typecheck_error(
            "export func f(a: i32x8) -> u8x16 { return bitcast_u8x16(a) }",
            "bit widths",
        );
    }

    #[test]
    fn source_must_be_a_vector() {
        assert_typecheck_error(
            "export func f(a: i32) -> u8x16 { return bitcast_u8x16(a) }",
            "vector",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn tail_on_x86_is_vpmaskmovd() {
        let s = asm(TAIL, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!(s.contains("vpmaskmovd"), "{s}");
    }

    #[test]
    fn tail_compiles_on_arm() {
        let s = asm(TAIL, "aarch64-unknown-linux-gnu", "generic");
        assert!(s.contains("tail:"), "{s}");
    }
}
