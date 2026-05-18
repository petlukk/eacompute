//! `permute_runtime` — runtime SIMD permute (x86-only, AVX2).
//!
//! Lowers to `vpermps` (f32x8) / `vpermd` (i32x8). ARM is rejected
//! with a codegen_error pointing at docs/idioms/neon-runtime-permute.md.
//!
//! Index semantics: `result[k] = table[indices[k] & 0x7]` — hardware
//! ignores the upper 29 bits of each index lane.
//!
//! All x86 numeric tests `assert_c_interop`; the codegen-assertion
//! tests additionally objdump and grep for the expected mnemonic.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[allow(unused_imports)]
    use super::common::*;
    use ea_compiler::OutputMode;
    use tempfile::TempDir;

    // --- Type-check happy paths (compile to object, no run) ---

    #[test]
    fn typecheck_permute_runtime_f32x8_compiles() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        ea_compiler::compile(src, &obj, OutputMode::ObjectFile)
            .expect("f32x8 permute_runtime should typecheck");
    }

    #[test]
    fn typecheck_permute_runtime_i32x8_compiles() {
        let src = r#"
            export func k(t: *i32, idx: *i32, out: *mut i32) {
                let tv: i32x8 = load(t, 0)
                let iv: i32x8 = load(idx, 0)
                let r: i32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("t.o");
        ea_compiler::compile(src, &obj, OutputMode::ObjectFile)
            .expect("i32x8 permute_runtime should typecheck");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_width_4() {
        let src = r#"
            export func k(t: *f32, idx: *i32, out: *mut f32) {
                let tv: f32x4 = load(t, 0)
                let iv: i32x4 = load(idx, 0)
                let r: f32x4 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "table must have width 8");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_u32_indices() {
        let src = r#"
            export func k(tv: f32x8, iv: u32x8, out: *mut f32) {
                let r: f32x8 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "i32 vector");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_wrong_arity() {
        let src = r#"
            export func k(t: *f32, out: *mut f32) {
                let tv: f32x8 = load(t, 0)
                let r: f32x8 = permute_runtime(tv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "2 arguments");
    }

    #[test]
    fn typecheck_permute_runtime_rejects_f64_table() {
        let src = r#"
            export func k(tv: f64x4, iv: i32x4, out: *mut f64) {
                let r: f64x4 = permute_runtime(tv, iv)
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "f32 or i32");
    }
}
