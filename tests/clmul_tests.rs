//! v1.17.0 carry-less multiply: clmul_lo_u64x2 (lanes 0 x 0) and
//! clmul_hi_u64x2 (lanes 1 x 1), u64x2 -> 128-bit product as u64x2.
//!
//! Native or error: x86 `pclmulqdq` with --pclmul (not in x86-64-v3), ARM
//! `pmull` / `pmull2` with --aes. Without the flag, a compile error naming
//! it. The products are checked against a bitwise reference on the host.
//!
//! Tests that compile for the x86 triple are gated to x86 hosts: the LLVM on
//! the ubuntu-24.04-arm runner has no x86 backend.

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    #[allow(dead_code)] // only used by tests gated to x86 hosts
    const X86: (&str, &str, &str) = ("x86_64-unknown-linux-gnu", "x86-64-v3", "+pclmul");
    const ARM: (&str, &str, &str) = ("aarch64-unknown-linux-gnu", "generic", "+aes");

    fn opts((triple, cpu, features): (&str, &str, &str), with_feature: bool) -> CompileOptions {
        CompileOptions {
            target_triple: Some(triple.to_string()),
            target_cpu: Some(cpu.to_string()),
            extra_features: if with_feature {
                features.to_string()
            } else {
                String::new()
            },
            ..CompileOptions::default()
        }
    }

    const KERNEL: &str = "export func clmul2(a: *u64, b: *u64, out: *mut u64) {\n    let x: u64x2 = load(a, 0)\n    let y: u64x2 = load(b, 0)\n    store(out, 0, clmul_lo_u64x2(x, y))\n    store(out, 2, clmul_hi_u64x2(x, y))\n}\n";

    fn asm_for(target: (&str, &str, &str)) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        ea_compiler::compile_with_options(KERNEL, &out, OutputMode::Asm, &opts(target, true))
            .expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    fn error_without_feature(target: (&str, &str, &str)) -> String {
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        match ea_compiler::compile_with_options(
            KERNEL,
            &obj,
            OutputMode::ObjectFile,
            &opts(target, false),
        ) {
            Ok(()) => panic!("compiled without the feature"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_is_pclmulqdq() {
        let asm = asm_for(X86);
        assert_eq!(asm.matches("pclmulqdq").count(), 2, "{asm}");
    }

    #[test]
    fn arm_is_pmull_and_pmull2() {
        let asm = asm_for(ARM);
        assert!(asm.contains("pmull\tv") || asm.contains("pmull v"), "{asm}");
        assert!(asm.contains("pmull2"), "{asm}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_needs_pclmul() {
        let e = error_without_feature(X86);
        assert!(e.contains("--pclmul") && e.contains("x86-64-v3"), "{e}");
    }

    #[test]
    fn arm_needs_aes() {
        let e = error_without_feature(ARM);
        assert!(e.contains("--aes") && e.contains("PMULL"), "{e}");
    }

    #[test]
    fn wrong_argument_type_is_a_type_error() {
        let src = "export func f(a: *u32, out: *mut u64) {\n    let x: u32x4 = load(a, 0)\n    store(out, 0, clmul_lo_u64x2(x, x))\n}\n";
        let dir = TempDir::new().unwrap();
        let e = ea_compiler::compile_with_options(
            src,
            &dir.path().join("t.o"),
            OutputMode::ObjectFile,
            &opts(ARM, true),
        )
        .unwrap_err();
        assert!(
            e.to_string().contains("clmul_lo_u64x2 expects u64x2"),
            "{e}"
        );
    }

    /// Products of random and edge operands on the host, against a bitwise
    /// carry-less multiply in C.
    #[test]
    fn products_match_reference_on_host() {
        let host = if cfg!(target_arch = "x86_64") {
            X86
        } else {
            ARM
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("kernel.o");
        let mut o = opts(host, true);
        o.target_cpu = None; // the host CPU
        ea_compiler::compile_with_options(KERNEL, &obj, OutputMode::ObjectFile, &o)
            .expect("compile");
        let c = r#"
#include <stdint.h>
#include <stdio.h>
void clmul2(const uint64_t *a, const uint64_t *b, uint64_t *out);
static void ref(uint64_t x, uint64_t y, uint64_t *lo, uint64_t *hi) {
    *lo = 0; *hi = 0;
    for (int i = 0; i < 64; i++)
        if (y >> i & 1) { *lo ^= x << i; if (i) *hi ^= x >> (64 - i); }
}
int main(void) {
    uint64_t s = 0x9e3779b97f4a7c15u, bad = 0;
    for (int n = 0; n < 4000; n++) {
        uint64_t v[4];
        for (int k = 0; k < 4; k++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[k] = s; }
        if (n == 0) { v[0] = ~0ull; v[1] = 1; v[2] = ~0ull; v[3] = 0x8000000000000000ull; }
        uint64_t a[2] = {v[0], v[1]}, b[2] = {v[2], v[3]}, out[4], e[4];
        clmul2(a, b, out);
        ref(a[0], b[0], &e[0], &e[1]);
        ref(a[1], b[1], &e[2], &e[3]);
        for (int k = 0; k < 4; k++) bad += out[k] != e[k];
    }
    printf("%llu\n", (unsigned long long)bad);
    return 0;
}
"#;
        let c_path = dir.path().join("h.c");
        let bin = dir.path().join("h");
        std::fs::write(&c_path, c).unwrap();
        let st = Command::new("cc")
            .args([
                c_path.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(st.success());
        let out = Command::new(&bin).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "0");
    }

    fn cli_error(flags: &[&str]) -> String {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("k.ea");
        std::fs::write(&src, "export func f() {}\n").unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_ea"))
            .arg(&src)
            .args(flags)
            .output()
            .unwrap();
        assert!(!out.status.success());
        String::from_utf8_lossy(&out.stderr).into_owned()
    }

    #[test]
    fn pclmul_is_refused_for_aarch64() {
        let e = cli_error(&["--pclmul", "--target-triple=aarch64-unknown-linux-gnu"]);
        assert!(
            e.contains("--pclmul is only valid for x86-64") && e.contains("--aes"),
            "{e}"
        );
    }

    #[test]
    fn aes_is_refused_for_x86_64() {
        let e = cli_error(&["--aes", "--target-triple=x86_64-unknown-linux-gnu"]);
        assert!(
            e.contains("--aes is only valid for AArch64") && e.contains("--pclmul"),
            "{e}"
        );
    }
}
