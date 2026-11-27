//! Typed andnot (v1.21.0): andnot_<t>(a, b) = a & !b, one instruction on
//! both targets (x86 vpandn, ARM bic). Linta's GC sweep computed
//! alloc & !mark as a ^ (a & m), and its regexp kernels wrote NOT as
//! x .^ ff for LLVM to fold into vpandn/bic behind the programmer's back.
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

    const SWEEP4: &str = "export func f(a: *u64, b: *u64, out: *mut u64) {
    let x: u64x4 = load(a, 0)
    let y: u64x4 = load(b, 0)
    store(out, 0, andnot_u64x4(x, y))
}
";
    const SWEEP2: &str = "export func f(a: *u64, b: *u64, out: *mut u64) {
    let x: u64x2 = load(a, 0)
    let y: u64x2 = load(b, 0)
    store(out, 0, andnot_u64x2(x, y))
}
";
    const BYTES16: &str = "export func f(a: *u8, b: *u8, out: *mut u8) {
    let x: u8x16 = load(a, 0)
    let y: u8x16 = load(b, 0)
    store(out, 0, andnot_u8x16(x, y))
}
";
    const BYTES32: &str = "export func f(a: *u8, b: *u8, out: *mut u8) {
    let x: u8x32 = load(a, 0)
    let y: u8x32 = load(b, 0)
    store(out, 0, andnot_u8x32(x, y))
}
";

    fn asm(src: &str, triple: &str, cpu: &str) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions { target_triple: Some(triple.to_string()), target_cpu: Some(cpu.to_string()), ..CompileOptions::default() };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts).expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn andnot_is_a_and_not_b() {
        assert_c_interop(
            SWEEP4,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const uint64_t *a, const uint64_t *b, uint64_t *out);
            int main(void) {
                uint64_t a[4] = {0xff00, 0xf0f0, UINT64_MAX, 0};
                uint64_t b[4] = {0x0ff0, 0xffff, 1, UINT64_MAX};
                uint64_t o[4];
                f(a, b, o);
                printf("%llx %llx %llx %llx\n", (unsigned long long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], (unsigned long long)o[3]);
                return 0;
            }
            "#,
            "f000 0 fffffffffffffffe 0",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn andnot_is_vpandn() {
        for src in [SWEEP4, SWEEP2, BYTES16, BYTES32] {
            let s = asm(src, "x86_64-unknown-linux-gnu", "x86-64-v3");
            assert!(s.contains("vpandn") || s.contains("vandnps"), "{s}");
            assert!(!s.contains("vpxor") && !s.contains("vxorps"), "{s}");
        }
    }

    #[test]
    fn andnot_is_bic() {
        for src in [SWEEP2, BYTES16] {
            let s = asm(src, "aarch64-unknown-linux-gnu", "generic");
            assert!(s.contains("bic"), "{s}");
            assert!(!s.contains("eor") && !s.contains("mvn"), "{s}");
        }
    }

    #[test]
    fn andnot_wrong_type_errors() {
        assert_typecheck_error("export func f(a: i32x4, b: i32x4) -> u8x16 { return andnot_u8x16(a, b) }", "u8x16");
        assert_typecheck_error("export func f(a: u8x16, b: u8x32) -> u8x16 { return andnot_u8x16(a, b) }", "u8x32");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn andnot_runs_on_arm() {
        assert_c_interop(
            SWEEP2,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const uint64_t *a, const uint64_t *b, uint64_t *out);
            int main(void) {
                uint64_t a[2] = {0xff00, UINT64_MAX};
                uint64_t b[2] = {0x0ff0, 1};
                uint64_t o[2];
                f(a, b, o);
                printf("%llx %llx\n", (unsigned long long)o[0], (unsigned long long)o[1]);
                return 0;
            }
            "#,
            "f000 fffffffffffffffe",
        );
    }
}
