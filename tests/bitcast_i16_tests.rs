//! bitcast_i16x8: any 128-bit vector reinterpreted as i16x8. It is the way
//! back from u16 lanes (`widen_u8_u16`, 8-bit fixed-point math) to
//! `pack_usat_i16x8`, which takes i16x8. A bitcast is no instruction on
//! either architecture.
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

    /// Bytes to u16, `x * a + x >> 8` (Skia's approximate multiply), and
    /// back to bytes: the shape of an 8-bit blend.
    const BLEND: &str = "export func blend(d: *u8, a: *u8, out: *mut u8) {\n    let x: u8x16 = load(d, 0)\n    let m: u8x16 = load(a, 0)\n    let eight: u16x8 = splat(8)\n    let xl: u16x8 = widen_u8_u16(x)\n    let xh: u16x8 = widen_u8_u16_hi(x)\n    let rl: u16x8 = (xl .* widen_u8_u16(m) .+ xl) .>> eight\n    let rh: u16x8 = (xh .* widen_u8_u16_hi(m) .+ xh) .>> eight\n    store(out, 0, pack_usat_i16x8(bitcast_i16x8(rl), bitcast_i16x8(rh)))\n}\n";

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
    fn lanes_are_little_endian_byte_pairs() {
        assert_output_lines(
            r#"
            export func main() {
                let b: u8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255]
                let v: i16x8 = bitcast_i16x8(b)
                println(v[0])
                println(v[7])
            }
            "#,
            &["513", "-241"],
        );
    }

    #[test]
    fn widen_then_pack_gives_the_bytes_back() {
        assert_output_lines(
            r#"
            export func main() {
                let d: u8x16 = [0, 1, 127, 128, 200, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 254]
                let lo: u16x8 = widen_u8_u16(d)
                let hi: u16x8 = widen_u8_u16_hi(d)
                let r: u8x16 = pack_usat_i16x8(bitcast_i16x8(lo), bitcast_i16x8(hi))
                println(r[3])
                println(r[5])
                println(r[15])
            }
            "#,
            &["128", "255", "254"],
        );
    }

    #[test]
    fn source_must_be_128_bits() {
        assert_typecheck_error(
            "export func f(a: i32x8) -> i16x8 { return bitcast_i16x8(a) }",
            "bit widths",
        );
    }

    #[test]
    fn source_must_be_a_vector() {
        assert_typecheck_error(
            "export func f(a: i32) -> i16x8 { return bitcast_i16x8(a) }",
            "vector",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn blend_on_x86_packs_with_packuswb() {
        let s = asm(BLEND, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!(s.contains("vpmullw"), "{s}");
        assert!(s.contains("vpackuswb"), "{s}");
    }

    #[test]
    fn blend_on_arm_packs_with_sqxtun() {
        let s = asm(BLEND, "aarch64-unknown-linux-gnu", "generic");
        assert!(s.contains("mul"), "{s}");
        assert!(s.contains("sqxtun"), "{s}");
    }
}
