//! Typed native masked loads and stores (v1.21.0): load_masked_<t> /
//! store_masked_<t> for i32x4/x8, f32x4/x8 and u64x4 (AVX2 vpmaskmovd/q,
//! vmaskmovps). They read and write only the first n lanes; the rest of a
//! load is zero. ARM has no masked load: a compile error. The generic
//! load_masked/store_masked (emulated by LLVM for u8/i16 lanes on AVX2 and
//! for every lane type on ARM) are deprecated since 1.21.0.
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

    const TAIL8: &str = "export func tail(p: *mut i32, n: i32) {\n    let v: i32x8 = load_masked_i32x8(p, 0, n)\n    store_masked_i32x8(p, 0, v .+ splat(1), n)\n}\n";
    const F8: &str = "export func f(p: *mut f32, n: i32) {\n    let v: f32x8 = load_masked_f32x8(p, 0, n)\n    store_masked_f32x8(p, 0, v .* splat(2.0), n)\n}\n";
    const Q4: &str = "export func q(p: *mut u64, n: i32) {\n    let v: u64x4 = load_masked_u64x4(p, 0, n)\n    store_masked_u64x4(p, 0, v .+ splat(1), n)\n}\n";
    const I4: &str = "export func f(p: *mut i32, n: i32) {\n    let v: i32x4 = load_masked_i32x4(p, 0, n)\n    store_masked_i32x4(p, 0, v .+ splat(1), n)\n}\n";

    /// The last n ints of a page with an inaccessible page after it:
    /// touching any lane past n faults (POSIX mmap, so not on Windows).
    #[test]
    #[cfg(all(target_arch = "x86_64", unix))]
    fn masked_i32x8_tail_n0_to_n8() {
        assert_c_interop(
            TAIL8,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            #include <sys/mman.h>
            #include <unistd.h>
            void tail(int32_t *p, int32_t n);
            int main(void) {
                long pg = sysconf(_SC_PAGESIZE);
                char *m = mmap(0, 2 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                mprotect(m + pg, pg, PROT_NONE);
                int ok = 1;
                for (int n = 0; n <= 8; n++) {
                    int32_t *p = (int32_t *)(m + pg) - n;
                    for (int k = 0; k < n; k++) p[k] = 10 * k;
                    tail(p, n);
                    for (int k = 0; k < n; k++) ok &= p[k] == 10 * k + 1;
                }
                printf("%d\n", ok);
                return 0;
            }
            "#,
            "1",
        );
    }

    /// A count below zero means no lanes (the generic form took it as an
    /// unsigned count, so every lane).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn negative_count_touches_no_lane() {
        assert_c_interop(
            TAIL8,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void tail(int32_t *p, int32_t n);
            int main(void) {
                int32_t p[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                tail(p, -3);
                int ok = 1;
                for (int k = 0; k < 8; k++) ok &= p[k] == k + 1;
                printf("%d\n", ok);
                return 0;
            }
            "#,
            "1",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn typed_masked_are_native() {
        let s = asm(TAIL8, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpmaskmovd"), "{s}");
        let s = asm(I4, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpmaskmovd"), "{s}");
        let s = asm(F8, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vmaskmovps"), "{s}");
        let s = asm(Q4, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpmaskmovq"), "{s}");
    }

    #[test]
    fn typed_masked_is_an_error_on_arm() {
        let e = asm(I4, "aarch64-unknown-linux-gnu", "generic").unwrap_err();
        assert!(e.contains("padded") && !e.contains('\\'), "{e}");
    }

    #[test]
    fn typed_masked_checks_types() {
        assert_typecheck_error("export func f(p: *f32, n: i32) -> i32x8 { return load_masked_i32x8(p, 0, n) }", "*i32");
        assert_typecheck_error("export func f(p: *mut i32, v: i32x4, n: i32) { store_masked_i32x8(p, 0, v, n) }", "i32x8");
    }

    #[test]
    fn generic_masked_warns() {
        for (src, name) in [
            ("export func f(p: *u8, n: i32) -> u8x32 { let v: u8x32 = load_masked(p, 0, n)\n return v }\n", "load_masked"),
            ("export func f(p: *mut i32, v: i32x4, n: i32) { store_masked(p, 0, v, n) }\n", "store_masked"),
        ] {
            let tokens = ea_compiler::tokenize(src).unwrap();
            let stmts = ea_compiler::desugar(ea_compiler::parse(tokens).unwrap()).unwrap();
            let mut tc = ea_compiler::typeck::TypeChecker::new();
            tc.check_program(&stmts).unwrap();
            let w: Vec<String> = tc.warnings().iter().map(|w| w.to_string()).collect();
            assert!(w.len() == 1 && w[0].contains(&format!("'{name}' is deprecated since 1.21.0")), "{w:?}");
        }
    }
}
