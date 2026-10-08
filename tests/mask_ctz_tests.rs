//! v1.16.0 bitmask + bit-scan intrinsics.
//!
//! Typed, native-or-error (same policy as the v1.12.0 monomorphic batch):
//!   movemask_u8x16, movemask_u8x32, movemask_u64x4   x86-only (pmovmskb / vpmovmskb / vmovmskpd)
//!   nibble_mask_u8x16                                 ARM-only (shrn #4 + fmov, 4 bits per lane)
//!   ctz_u32, ctz_u64                                  both (tzcnt / rbit+clz); ctz(0) = bit width
//!
//! The polymorphic `movemask` still compiles but records a deprecation warning.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[allow(unused_imports)]
    use super::common::*;
    use ea_compiler::typeck::TypeChecker;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    const X86: (&str, &str) = ("x86_64-unknown-linux-gnu", "x86-64-v3");
    const ARM: (&str, &str) = ("aarch64-unknown-linux-gnu", "generic");

    fn opts((triple, cpu): (&str, &str)) -> CompileOptions {
        CompileOptions {
            target_triple: Some(triple.to_string()),
            target_cpu: Some(cpu.to_string()),
            ..CompileOptions::default()
        }
    }

    fn compile_for(
        source: &str,
        target: (&str, &str),
    ) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        ea_compiler::compile_with_options(source, &obj, OutputMode::ObjectFile, &opts(target))
    }

    fn asm_for(source: &str, target: (&str, &str)) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        ea_compiler::compile_with_options(source, &out, OutputMode::Asm, &opts(target))
            .expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    fn warnings_for(source: &str) -> Vec<ea_compiler::DeprecationWarning> {
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let mut tc = TypeChecker::new();
        tc.check_program(&stmts).unwrap();
        tc.warnings()
    }

    const MOVEMASK_U8X16: &str = "export func f(p: *u8) -> u32 {\n    let v: u8x16 = load(p, 0)\n    let k: u8x16 = splat(97)\n    return movemask_u8x16(v .== k)\n}\n";
    const MOVEMASK_U8X32: &str = "export func f(p: *u8) -> u32 {\n    let v: u8x32 = load(p, 0)\n    let k: u8x32 = splat(97)\n    return movemask_u8x32(v .== k)\n}\n";
    const MOVEMASK_U64X4: &str = "export func f(p: *u64, m: u64) -> u32 {\n    let v: u64x4 = load(p, 0)\n    let k: u64x4 = splat(m)\n    return movemask_u64x4((v .& k) .== k)\n}\n";
    const NIBBLE_U8X16: &str = "export func f(p: *u8) -> u64 {\n    let v: u8x16 = load(p, 0)\n    let k: u8x16 = splat(97)\n    return nibble_mask_u8x16(v .== k)\n}\n";
    const CTZ_U32: &str = "export func f(x: u32) -> i32 {\n    return ctz_u32(x)\n}\n";
    const CTZ_U64: &str = "export func f(x: u64) -> i32 {\n    return ctz_u64(x)\n}\n";

    // --- Native lowering, checked per target from any host ---

    #[test]
    fn movemask_u8x16_is_pmovmskb() {
        assert!(asm_for(MOVEMASK_U8X16, X86).contains("pmovmskb"));
    }

    #[test]
    fn movemask_u8x32_is_vpmovmskb() {
        assert!(asm_for(MOVEMASK_U8X32, X86).contains("vpmovmskb"));
    }

    #[test]
    fn movemask_u64x4_is_vmovmskpd() {
        assert!(asm_for(MOVEMASK_U64X4, X86).contains("vmovmskpd"));
    }

    #[test]
    fn nibble_mask_u8x16_is_shrn() {
        let asm = asm_for(NIBBLE_U8X16, ARM);
        assert!(asm.contains("shrn"), "{asm}");
    }

    #[test]
    fn ctz_is_tzcnt_on_x86_and_rbit_clz_on_arm() {
        for src in [CTZ_U32, CTZ_U64] {
            assert!(asm_for(src, X86).contains("tzcnt"));
            let arm = asm_for(src, ARM);
            assert!(arm.contains("rbit") && arm.contains("clz"), "{arm}");
        }
    }

    // --- Native-or-error: each arch points at the other's idiom ---

    #[test]
    fn movemask_u8x16_on_arm_points_to_nibble_mask() {
        let err = compile_for(MOVEMASK_U8X16, ARM).expect_err("x86-only");
        let msg = format!("{err}");
        assert!(
            msg.contains("x86-only") && msg.contains("nibble_mask_u8x16"),
            "{msg}"
        );
    }

    #[test]
    fn movemask_u64x4_is_rejected_on_arm() {
        assert!(compile_for(MOVEMASK_U64X4, ARM).is_err());
    }

    #[test]
    fn nibble_mask_on_x86_points_to_movemask() {
        let err = compile_for(NIBBLE_U8X16, X86).expect_err("ARM-only");
        let msg = format!("{err}");
        assert!(
            msg.contains("ARM-only") && msg.contains("movemask_u8x16"),
            "{msg}"
        );
    }

    // --- Type checks ---

    #[test]
    fn movemask_u8x16_rejects_a_four_lane_mask() {
        let src = "export func f(p: *u64) -> i32 {\n    let v: u64x4 = load(p, 0)\n    return movemask_u8x16(v .== v)\n}\n";
        let msg = format!("{}", compile_for(src, X86).expect_err("width mismatch"));
        assert!(msg.contains("movemask_u8x16 expects"), "{msg}");
    }

    #[test]
    fn nibble_mask_rejects_raw_bytes() {
        let src = "export func f(p: *u8) -> u64 {\n    let v: u8x16 = load(p, 0)\n    return nibble_mask_u8x16(v)\n}\n";
        let msg = format!(
            "{}",
            compile_for(src, ARM).expect_err("needs a comparison result")
        );
        assert!(msg.contains("nibble_mask_u8x16 expects"), "{msg}");
    }

    #[test]
    fn ctz_u32_rejects_u64() {
        let src = "export func f(x: u64) -> i32 {\n    return ctz_u32(x)\n}\n";
        let msg = format!("{}", compile_for(src, X86).expect_err("type mismatch"));
        assert!(msg.contains("ctz_u32 expects u32"), "{msg}");
    }

    // --- Runtime results on the host ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn movemask_u8x16_sets_one_bit_per_matching_lane() {
        assert_c_interop(
            MOVEMASK_U8X16,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern uint32_t f(const uint8_t*);
            int main() {
                const uint8_t s[16] = "abcaxxxxxxxxxxxa";
                printf("%u\n", f(s));
                return 0;
            }
            "#,
            // lanes 0, 3 and 15 are 'a'
            "32777",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn movemask_u64x4_reports_bloom_hits() {
        assert_c_interop(
            MOVEMASK_U64X4,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern uint32_t f(const uint64_t*, uint64_t);
            int main() {
                const uint64_t blooms[4] = {0x5, 0x1, 0xF, 0x4};
                printf("%d\n", f(blooms, 0x5));
                return 0;
            }
            "#,
            // lanes 0 and 2 contain both bits of 0x5
            "5",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn nibble_mask_u8x16_sets_four_bits_per_matching_lane() {
        assert_c_interop(
            NIBBLE_U8X16,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern uint64_t f(const uint8_t*);
            int main() {
                const uint8_t s[16] = "xxxaxxxxxxxxxxxa";
                printf("%llx\n", (unsigned long long)f(s));
                return 0;
            }
            "#,
            // lanes 3 and 15 → nibbles 3 and 15
            "f00000000000f000",
        );
    }

    #[test]
    fn ctz_counts_trailing_zeros_and_is_defined_at_zero() {
        assert_c_interop(
            "export func a(x: u32) -> i32 {\n    return ctz_u32(x)\n}\nexport func b(x: u64) -> i32 {\n    return ctz_u64(x)\n}\n",
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t a(uint32_t);
            extern int32_t b(uint64_t);
            int main() {
                printf("%d %d %d %d %d %d\n", a(8), a(1u << 31), a(0), b(1ull << 40), b(1ull << 63), b(0));
                return 0;
            }
            "#,
            "3 31 32 40 63 64",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn movemask_and_ctz_iterate_matching_lanes() {
        assert_c_interop(
            "export func f(p: *u8) -> i32 {\n    let v: u8x32 = load(p, 0)\n    let k: u8x32 = splat(38)\n    let mut m: u32 = movemask_u8x32(v .== k)\n    let mut sum: i32 = 0\n    while m != 0 {\n        sum = sum + ctz_u32(m)\n        m = m & (m - 1)\n    }\n    return sum\n}\n",
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t f(const uint8_t*);
            int main() {
                uint8_t s[32];
                for (int i = 0; i < 32; i++) s[i] = 'x';
                s[1] = '&'; s[17] = '&'; s[31] = '&';
                printf("%d\n", f(s));
                return 0;
            }
            "#,
            "49",
        );
    }

    #[test]
    fn nibble_mask_and_ctz_find_the_first_lane_on_arm() {
        let src = "export func f(p: *u8) -> i32 {\n    let v: u8x16 = load(p, 0)\n    let k: u8x16 = splat(38)\n    let m: u64 = nibble_mask_u8x16(v .== k)\n    return ctz_u64(m) / 4\n}\n";
        compile_for(src, ARM).expect("nibble mask composes with ctz_u64");
    }

    // --- Deprecation of the polymorphic spelling ---

    #[test]
    fn polymorphic_movemask_still_compiles_and_warns() {
        let src = "export func f(p: *u8) -> i32 {\n    let v: u8x16 = load(p, 0)\n    let k: u8x16 = splat(97)\n    return movemask(v .== k)\n}\n";
        compile_for(src, X86).expect("polymorphic movemask must still compile");
        let w = warnings_for(src);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].name, "movemask");
        assert!(
            w[0].advice.contains("movemask_u8x16") && w[0].advice.contains("nibble_mask_u8x16")
        );
    }

    #[test]
    fn typed_spellings_do_not_warn() {
        for src in [
            MOVEMASK_U8X16,
            MOVEMASK_U64X4,
            NIBBLE_U8X16,
            CTZ_U32,
            CTZ_U64,
        ] {
            assert!(warnings_for(src).is_empty(), "{src}");
        }
    }
}
