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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn blend_i32x4_emits_blend_family() {
        // Pick lanes [a[0], b[1], a[2], b[3]] — lane-by-lane blend.
        let src = r#"
            export func k(a: *i32, b: *i32, out: *mut i32) {
                let va: i32x4 = load(a, 0)
                let vb: i32x4 = load(b, 0)
                let r: i32x4 = shuffle(va, vb, [0, 5, 2, 7])
                store(out, 0, r)
            }
        "#;
        // LLVM may pick pblendw, vpblendd, vblendps, or vpblendw on AVX2,
        // or potentially vpermi2d / vpermt2d on AVX-512. Accept the family.
        assert_intrinsic_in_disassembly(
            src,
            &["pblendw", "vpblendd", "vblendps", "vpblendw", "vpermi2d", "vpermt2d"],
        );
    }

    // --- shape errors ---

    #[test]
    fn two_source_indices_wrong_length_errors() {
        let src = r#"
            export func k(p: *i32, q: *mut i32) {
                let a: i32x4 = load(p, 0)
                let b: i32x4 = load(p, 4)
                let r: i32x4 = shuffle(a, b, [0, 1, 2])
                store(q, 0, r)
            }
        "#;
        assert_typecheck_error(src, "shuffle indices length 3 != vector width 4");
    }

    #[test]
    fn two_source_indices_out_of_range_errors() {
        // width=4, two-source allows indices [0, 8); 8 itself is out of range.
        let src = r#"
            export func k(p: *i32, q: *mut i32) {
                let a: i32x4 = load(p, 0)
                let b: i32x4 = load(p, 4)
                let r: i32x4 = shuffle(a, b, [0, 8, 1, 2])
                store(q, 0, r)
            }
        "#;
        assert_typecheck_error(src, "must be in [0, 8)");
    }

    #[test]
    fn two_source_type_mismatch_errors() {
        let src = r#"
            export func k(p: *i32, q: *f32, out: *mut i32) {
                let a: i32x4 = load(p, 0)
                let b: f32x4 = load(q, 0)
                let r: i32x4 = shuffle(a, b, [0, 4, 1, 5])
                store(out, 0, r)
            }
        "#;
        assert_typecheck_error(src, "must have the same type");
    }

    #[test]
    fn two_source_arg2_must_be_vector_errors() {
        let src = r#"
            export func k(p: *i32, q: *mut i32) {
                let a: i32x4 = load(p, 0)
                let r: i32x4 = shuffle(a, 42, [0, 1, 2, 3])
                store(q, 0, r)
            }
        "#;
        // Non-vector second arg falls into the type-mismatch branch (42 is i64,
        // not a vector).
        assert_typecheck_error(src, "must have the same type");
    }

    #[test]
    fn arg_count_error_message_lists_both_forms() {
        // Single-argument call should produce the new error mentioning both forms.
        let src = r#"
            export func k(p: *i32, q: *mut i32) {
                let v: i32x4 = load(p, 0)
                let r: i32x4 = shuffle(v)
                store(q, 0, r)
            }
        "#;
        assert_typecheck_error(src, "shuffle expects 2 or 3 arguments");
    }
}
