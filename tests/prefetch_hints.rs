//! `prefetch_write` and `prefetch_nta` — hint flavors of llvm.prefetch.p0.
//!
//! prefetch_write → rw=1, locality=3 → x86 `prefetchw` / aarch64 `prfm pstl1keep`
//! prefetch_nta   → rw=0, locality=0 → x86 `prefetchnta` / aarch64 `prfm pldl1strm`
//! plain prefetch → rw=0, locality=3 → x86 `prefetcht0` / aarch64 `prfm pldl1keep`
//!
//! Both cross-platform via LLVM's portable llvm.prefetch.p0 intrinsic.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[allow(unused_imports)]
    use super::common::*;

    // --- prefetch_write: typeck + x86 codegen ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefetch_write_emits_prefetchw_on_x86() {
        let src = r#"
            export func k(p: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    prefetch_write(p, i + 16)
                    i = i + 1
                }
            }
        "#;
        assert_intrinsic_in_disassembly(src, &["prefetchw"]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefetch_nta_emits_prefetchnta_on_x86() {
        let src = r#"
            export func k(p: *i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    prefetch_nta(p, i + 64)
                    i = i + 1
                }
            }
        "#;
        assert_intrinsic_in_disassembly(src, &["prefetchnta"]);
    }

    // --- aarch64 cross-compile sanity (no asm grep) ---

    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile_arm(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("t.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &arm_opts())
    }

    #[test]
    fn prefetch_write_compiles_aarch64() {
        let src = r#"
            export func k(p: *mut i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    prefetch_write(p, i + 16)
                    i = i + 1
                }
            }
        "#;
        try_compile_arm(src).expect("prefetch_write should cross-compile to aarch64");
    }

    #[test]
    fn prefetch_nta_compiles_aarch64() {
        let src = r#"
            export func k(p: *i32, len: i32) {
                let mut i: i32 = 0
                while i < len {
                    prefetch_nta(p, i + 64)
                    i = i + 1
                }
            }
        "#;
        try_compile_arm(src).expect("prefetch_nta should cross-compile to aarch64");
    }

    // --- shape errors (reuses check_prefetch) ---

    #[test]
    fn prefetch_write_too_few_args_errors() {
        let src = r#"
            export func k(p: *i32) { prefetch_write(p) }
        "#;
        assert_typecheck_error(src, "prefetch expects 2 arguments");
    }

    #[test]
    fn prefetch_nta_too_many_args_errors() {
        let src = r#"
            export func k(p: *i32) { prefetch_nta(p, 1, 2) }
        "#;
        assert_typecheck_error(src, "prefetch expects 2 arguments");
    }

    #[test]
    fn prefetch_write_non_pointer_first_arg_errors() {
        let src = r#"
            export func k() { prefetch_write(42, 0) }
        "#;
        assert_typecheck_error(src, "must be a pointer");
    }

    #[test]
    fn prefetch_nta_non_integer_offset_errors() {
        let src = r#"
            export func k(p: *i32) {
                let f: f32 = 1.0
                prefetch_nta(p, f)
            }
        "#;
        assert_typecheck_error(src, "offset must be integer");
    }

    // --- end-to-end semantics (Hard Rule #2) ---

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn prefetch_write_real_kernel_runs() {
        // Write-bound kernel: copy and increment src[i] into dst[i],
        // with prefetch_write hinting the next chunk of writes.
        // Output must match the scalar reference byte-for-byte.
        let ea = r#"
            export func copy_inc(
                src: *restrict i32,
                dst: *restrict mut i32,
                len: i32
            ) {
                let mut i: i32 = 0
                while i + 8 <= len {
                    prefetch_write(dst, i + 32)
                    let v: i32x8 = load(src, i)
                    let one: i32x8 = splat(1)
                    store(dst, i, v .+ one)
                    i = i + 8
                }
                while i < len {
                    dst[i] = src[i] + 1
                    i = i + 1
                }
            }
        "#;
        let c = r#"
            #include <stdio.h>
            #include <string.h>
            void copy_inc(const int*, int*, int);
            int main(void) {
                int src[33];
                int dst[33] = {0};
                int ref[33];
                for (int i = 0; i < 33; i++) {
                    src[i] = i * 7;
                    ref[i] = i * 7 + 1;
                }
                copy_inc(src, dst, 33);
                if (memcmp(dst, ref, sizeof(ref)) == 0) {
                    printf("OK\n");
                } else {
                    for (int i = 0; i < 33; i++) {
                        printf("dst[%d]=%d ref=%d\n", i, dst[i], ref[i]);
                    }
                }
                return 0;
            }
        "#;
        assert_c_interop(ea, c, "OK");
    }
}
