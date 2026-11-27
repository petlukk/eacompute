//! Masks that record their source type, and .& .| .^ on them (v1.21.0).
//! A comparison of TxN vectors gives a mask<T>xN; two masks of the same
//! source combine with .&, .|, .^ (one vpand/vpor/vpxor, ARM and/orr/eor),
//! where Vingilot's and Linta's kernels went through select(m, ff, 0).
//! Masks of different sources do not combine: that would need a hidden
//! pack or extend.
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

    fn asm(src: &str, triple: &str, cpu: &str) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions { target_triple: Some(triple.to_string()), target_cpu: Some(cpu.to_string()), ..CompileOptions::default() };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts).expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    const WS_X86: &str = "export func ws(p: *u8) -> u32 {\n    let c: u8x16 = load(p, 0)\n    return movemask_u8x16((c .== splat(32)) .| (c .== splat(9)))\n}\n";
    const WS_ARM: &str = "export func ws(p: *u8) -> u64 {\n    let c: u8x16 = load(p, 0)\n    return nibble_mask_u8x16((c .== splat(32)) .| (c .== splat(9)))\n}\n";
    const PICK: &str = "export func pick(a: *i32, b: *i32, out: *mut i32) {\n    let x: i32x8 = load(a, 0)\n    let y: i32x8 = load(b, 0)\n    store(out, 0, select(((x .> splat(0)) .& (y .< splat(10))) .^ (x .== y), x, y))\n}\n";

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn masks_combine_with_and_or_xor() {
        assert_c_interop(
            WS_X86,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            uint32_t ws(const uint8_t *p);
            int main(void) {
                uint8_t s[16] = {'a', ' ', 'b', '\t', 'c', 'd', ' ', ' ', 'e', '\t', 'f', 'g', 'h', 'i', 'j', ' '};
                printf("%x\n", ws(s));
                return 0;
            }
            "#,
            "82ca",
        );
        assert_c_interop(
            PICK,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void pick(const int32_t *a, const int32_t *b, int32_t *out);
            int main(void) {
                int32_t a[8] = {1, -1, 5, 5, 2, 0, 7, 3};
                int32_t b[8] = {2, 2, 5, 20, 9, 9, 7, 30};
                int32_t o[8];
                pick(a, b, o);
                for (int k = 0; k < 8; k++) printf("%d ", o[k]);
                printf("\n");
                return 0;
            }
            "#,
            "1 2 5 20 2 9 7 30",
        );
    }

    #[test]
    fn masks_of_different_sources_do_not_combine() {
        let src = "export func f(a: i32x8, b: i32x8, c: f32x8, d: f32x8) -> i32x8 {\n    return select((a .< b) .& (c .< d), a, b)\n}\n";
        let tokens = ea_compiler::tokenize(src).unwrap();
        let stmts = ea_compiler::desugar(ea_compiler::parse(tokens).unwrap()).unwrap();
        let e = ea_compiler::typeck::TypeChecker::new().check_program(&stmts).unwrap_err().to_string();
        assert!(e.contains("mask<i32>x8") && e.contains("mask<f32>x8"), "{e}");
    }

    #[test]
    fn mask_and_integer_do_not_combine() {
        assert_typecheck_error("export func f(a: i32x4, b: i32x4) -> i32x4 { return select((a .== b) .& splat(1), a, b) }", "mask");
    }

    #[test]
    fn masks_do_not_shift() {
        assert_typecheck_error("export func f(a: i32x4, b: i32x4) -> i32x4 { return select((a .== b) .<< (a .== b), a, b) }", "mask");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn mask_or_is_one_instruction() {
        let s = asm(WS_X86, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!((s.contains("vpor") || s.contains("por")) && !s.contains("blendv"), "{s}");
    }

    #[test]
    fn mask_or_is_orr_on_arm() {
        let s = asm(WS_ARM, "aarch64-unknown-linux-gnu", "generic");
        assert!(s.contains("orr") && !s.contains("bsl") && !s.contains("bif"), "{s}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn existing_mask_users_unchanged() {
        let src = "export func f(a: *i32, out: *mut i32) -> u32 {\n    let x: i32x8 = load(a, 0)\n    let z: i32x8 = splat(0)\n    store(out, 0, select(x .> splat(3), x, z))\n    let b: u8x32 = splat(7)\n    return movemask_u8x32(b .== splat(7))\n}\n";
        assert_c_interop(
            src,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            uint32_t f(const int32_t *a, int32_t *out);
            int main(void) {
                int32_t a[8] = {1, 4, 3, 9, -2, 5, 0, 8};
                int32_t o[8];
                uint32_t m = f(a, o);
                for (int k = 0; k < 8; k++) printf("%d ", o[k]);
                printf("%x\n", m);
                return 0;
            }
            "#,
            "0 4 0 9 0 5 0 8 ffffffff",
        );
    }
}
