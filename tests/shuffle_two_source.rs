//! Two-source `shuffle(a, b, [indices])` — picks lanes from `a` for indices
//! in `[0, width)` and from `b` for indices in `[width, 2*width)`. Lowers via
//! LLVM `shufflevector` two-source mode. Existing single-source `shuffle(v, [...])`
//! is unchanged.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[allow(unused_imports)]
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    // --- two-source codegen: x86 mnemonics ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn interleave_lower_f32x8_emits_shuffle() {
        let src = r#"
            export func k(a: *f32, b: *f32, out: *mut f32) {
                let va: f32x8 = load(a, 0)
                let vb: f32x8 = load(b, 0)
                let r: f32x8 = shuffle(va, vb, [0, 8, 1, 9, 2, 10, 3, 11])
                store(out, 0, r)
            }
        "#;
        // LLVM 18 may emit vunpcklps, vpunpckldq, or vpermi2ps
        // (all valid two-source shuffle instructions).
        assert_intrinsic_in_disassembly(src, &["vunpcklps", "vpunpckldq", "vpermi2ps"]);
    }
}
