//! v1.18.0 AES rounds: x86 aesenc_u8x16 (aesenc), ARM aese_u8x16 (aese)
//! and aesmc_u8x16 (aesmc), all u8x16.
//!
//! Native or error: each is a compile error on the other architecture,
//! naming the local idiom, and without --aes a compile error naming it.
//! A round on the host is checked against an AES round in C.
//!
//! Tests that compile for the x86 triple are gated to x86 hosts: the LLVM on
//! the ubuntu-24.04-arm runner has no x86 backend.

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::{CompileOptions, OutputMode};
    use std::process::Command;
    use tempfile::TempDir;

    #[allow(dead_code)] // only used by tests gated to x86 hosts
    const X86: (&str, &str) = ("x86_64-unknown-linux-gnu", "x86-64-v3");
    const ARM: (&str, &str) = ("aarch64-unknown-linux-gnu", "generic");

    /// One AES round, MixColumns(ShiftRows(SubBytes(x))) ^ k, per target.
    const ROUND_X86: &str = "export func aes_round(x: *u8, k: *u8, out: *mut u8) {\n    let a: u8x16 = load(x, 0)\n    let b: u8x16 = load(k, 0)\n    store(out, 0, aesenc_u8x16(a, b))\n}\n";
    const ROUND_ARM: &str = "export func aes_round(x: *u8, k: *u8, out: *mut u8) {\n    let a: u8x16 = load(x, 0)\n    let b: u8x16 = load(k, 0)\n    let z: u8x16 = splat(0)\n    store(out, 0, aesmc_u8x16(aese_u8x16(a, z)) .^ b)\n}\n";

    fn opts((triple, cpu): (&str, &str), aes: bool) -> CompileOptions {
        CompileOptions {
            target_triple: Some(triple.to_string()),
            target_cpu: Some(cpu.to_string()),
            extra_features: if aes {
                "+aes".to_string()
            } else {
                String::new()
            },
            ..CompileOptions::default()
        }
    }

    fn asm(src: &str, target: (&str, &str)) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts(target, true))
            .expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    fn error(src: &str, target: (&str, &str), aes: bool) -> String {
        let dir = TempDir::new().unwrap();
        match ea_compiler::compile_with_options(
            src,
            &dir.path().join("t.o"),
            OutputMode::ObjectFile,
            &opts(target, aes),
        ) {
            Ok(()) => panic!("compiled"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_is_aesenc() {
        let a = asm(ROUND_X86, X86);
        assert_eq!(a.matches("aesenc").count(), 1, "{a}");
    }

    #[test]
    fn arm_is_aese_and_aesmc() {
        let a = asm(ROUND_ARM, ARM);
        assert!(a.contains("aese\t") || a.contains("aese "), "{a}");
        assert!(a.contains("aesmc"), "{a}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_needs_aes() {
        let e = error(ROUND_X86, X86, false);
        assert!(e.contains("--aes") && e.contains("x86-64-v3"), "{e}");
    }

    #[test]
    fn arm_needs_aes() {
        let e = error(ROUND_ARM, ARM, false);
        assert!(e.contains("--aes") && e.contains("crypto extension"), "{e}");
    }

    #[test]
    fn aesenc_on_arm_names_aese_and_aesmc() {
        let e = error(ROUND_X86, ARM, true);
        assert!(
            e.contains("x86-only") && e.contains("aese_u8x16") && e.contains("aesmc_u8x16"),
            "{e}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn aese_on_x86_names_aesenc() {
        let e = error(ROUND_ARM, X86, true);
        assert!(
            e.contains("AArch64-only") && e.contains("aesenc_u8x16"),
            "{e}"
        );
    }

    #[test]
    fn wrong_argument_type_is_a_type_error() {
        let src = "export func f(a: *u32, out: *mut u8) {\n    let x: u32x4 = load(a, 0)\n    store(out, 0, aesmc_u8x16(x))\n}\n";
        let e = error(src, ARM, true);
        assert!(e.contains("aesmc_u8x16 expects u8x16"), "{e}");
    }

    #[test]
    fn wrong_arity_is_a_type_error() {
        let src = "export func f(a: *u8, out: *mut u8) {\n    let x: u8x16 = load(a, 0)\n    store(out, 0, aese_u8x16(x))\n}\n";
        let e = error(src, ARM, true);
        assert!(e.contains("aese_u8x16 expects 2 arguments"), "{e}");
    }

    /// Rounds of random and edge blocks on the host, against an AES round
    /// in C (FIPS-197 S-box, ShiftRows, MixColumns, AddRoundKey).
    #[test]
    fn rounds_match_reference_on_host() {
        let (host, src) = if cfg!(target_arch = "x86_64") {
            (X86, ROUND_X86)
        } else {
            (ARM, ROUND_ARM)
        };
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("kernel.o");
        let mut o = opts(host, true);
        o.target_triple = None; // the host triple: its C ABI
        o.target_cpu = None; // the host CPU
        ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &o).expect("compile");
        let c = r#"
#include <stdint.h>
#include <stdio.h>
void aes_round(const uint8_t *x, const uint8_t *k, uint8_t *out);
static uint8_t sbox[256];
static uint8_t xt(uint8_t a) { return (uint8_t)(a << 1) ^ (a & 0x80 ? 0x1b : 0); }
static uint8_t mul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (; b; b >>= 1, a = xt(a)) if (b & 1) p ^= a;
    return p;
}
static void make_sbox(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t inv = 0;
        for (int j = 1; j < 256 && i; j++) if (mul((uint8_t)i, (uint8_t)j) == 1) { inv = (uint8_t)j; break; }
        uint8_t s = inv, r = inv;
        for (int n = 0; n < 4; n++) { r = (uint8_t)(r << 1 | r >> 7); s ^= r; }
        sbox[i] = s ^ 0x63;
    }
}
static void ref(const uint8_t *x, const uint8_t *k, uint8_t *o) {
    uint8_t t[16];
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++) t[4 * c + r] = sbox[x[4 * ((c + r) % 4) + r]];
    for (int c = 0; c < 4; c++) {
        uint8_t *a = t + 4 * c;
        for (int r = 0; r < 4; r++)
            o[4 * c + r] = mul(a[r], 2) ^ mul(a[(r + 1) % 4], 3) ^ a[(r + 2) % 4] ^ a[(r + 3) % 4] ^ k[4 * c + r];
    }
}
int main(void) {
    make_sbox();
    if (sbox[0] != 0x63 || sbox[0x53] != 0xed) { printf("bad sbox\n"); return 0; }
    uint64_t s = 0x9e3779b97f4a7c15u, bad = 0;
    for (int n = 0; n < 4000; n++) {
        uint8_t x[16], k[16], out[16], e[16];
        for (int i = 0; i < 16; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; x[i] = (uint8_t)s; k[i] = (uint8_t)(s >> 8); }
        if (n == 0) for (int i = 0; i < 16; i++) { x[i] = 0; k[i] = 0xff; }
        aes_round(x, k, out);
        ref(x, k, e);
        for (int i = 0; i < 16; i++) bad += out[i] != e[i];
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

    /// `--aes` is a flag on both architectures (on x86 it was refused in
    /// v1.17.0).
    #[test]
    fn aes_flag_is_accepted_for_x86_64() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("k.ea");
        std::fs::write(&src, ROUND_X86).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_ea"))
            .arg(&src)
            .args([
                "--aes",
                "--target-triple=x86_64-unknown-linux-gnu",
                "--emit-asm",
            ])
            .current_dir(dir.path())
            .output()
            .unwrap();
        if cfg!(target_arch = "x86_64") {
            assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        } else {
            // No x86 backend on the ARM runner: whatever fails, it is not the flag.
            assert!(!String::from_utf8_lossy(&out.stderr).contains("only valid for AArch64"));
        }
    }
}
