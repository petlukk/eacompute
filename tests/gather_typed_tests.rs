//! Typed native gathers (v1.21.0): gather_i32x4/x8 and gather_f32x4/x8
//! lower to AVX2 vpgatherdd/vgatherdps, never to LLVM's generic masked
//! gather (which LLVM scalarizes on x86-64-v3: Linta cells.ea, Tiro's
//! palette rows). ARM has no gather: a compile error. The generic `gather`
//! is deprecated since 1.21.0.
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

    const I32X8: &str = "export func g(t: *i32, idx: *i32, out: *mut i32) {\n    let i: i32x8 = load(idx, 0)\n    store(out, 0, gather_i32x8(t, i))\n}\n";
    const F32X8: &str = "export func g(t: *f32, idx: *i32, out: *mut f32) {\n    let i: i32x8 = load(idx, 0)\n    store(out, 0, gather_f32x8(t, i))\n}\n";
    const I32X4: &str = "export func g(t: *i32, idx: *i32, out: *mut i32) {\n    let i: i32x4 = load(idx, 0)\n    store(out, 0, gather_i32x4(t, i))\n}\n";

    fn asm(src: &str, triple: &str, cpu: &str) -> Result<String, String> {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions { target_triple: Some(triple.to_string()), target_cpu: Some(cpu.to_string()), ..CompileOptions::default() };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts).map_err(|e| e.to_string())?;
        Ok(std::fs::read_to_string(out).unwrap())
    }

    const HARNESS: &str = r#"
        #include <stdio.h>
        #include <stdint.h>
        void g(const int32_t *t, const int32_t *idx, int32_t *out);
        int main(void) {
            int32_t t[64];
            for (int k = 0; k < 64; k++) t[k] = 1000 + k * 7;
            int32_t idx[8] = {3, 0, 63, 7, 7, 1, 62, 5};
            int32_t out[8];
            g(t, idx, out);
            for (int k = 0; k < 8; k++) printf("%d ", out[k] == t[idx[k]]);
            printf("\n");
            return 0;
        }
    "#;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gather_i32x8_equals_scalar_lookup() {
        assert_c_interop(I32X8, HARNESS, "1 1 1 1 1 1 1 1");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn gather_i32x8_is_vpgatherdd() {
        let s = asm(I32X8, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpgatherdd") && !s.contains("vpextrd") && !s.contains("vpinsrd"), "{s}");
        let s = asm(I32X4, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vpgatherdd") && !s.contains("vpextrd") && !s.contains("vpinsrd"), "{s}");
        let s = asm(F32X8, "x86_64-unknown-linux-gnu", "x86-64-v3").unwrap();
        assert!(s.contains("vgatherdps") && !s.contains("vextractps") && !s.contains("vinsertps"), "{s}");
    }

    #[test]
    fn gather_typed_is_an_error_on_arm() {
        let e = asm(I32X4, "aarch64-unknown-linux-gnu", "generic").unwrap_err();
        assert!(e.contains("neon-gather"), "{e}");
    }

    #[test]
    fn gather_typed_checks_its_types() {
        assert_typecheck_error("export func f(t: *f32, i: i32x8) -> i32x8 { return gather_i32x8(t, i) }", "*i32");
        assert_typecheck_error("export func f(t: *i32, i: i32x4) -> i32x8 { return gather_i32x8(t, i) }", "i32x8");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn generic_gather_warns_and_still_runs() {
        let src = I32X8.replace("gather_i32x8(t, i)", "gather(t, i)");
        let tokens = ea_compiler::tokenize(&src).unwrap();
        let stmts = ea_compiler::desugar(ea_compiler::parse(tokens).unwrap()).unwrap();
        let mut tc = ea_compiler::typeck::TypeChecker::new();
        tc.check_program(&stmts).unwrap();
        let w: Vec<String> = tc.warnings().iter().map(|w| w.to_string()).collect();
        assert!(w.len() == 1 && w[0].contains("'gather' is deprecated since 1.21.0") && w[0].contains("gather_i32x8"), "{w:?}");
        assert_c_interop(&src, HARNESS, "1 1 1 1 1 1 1 1");
    }
}
