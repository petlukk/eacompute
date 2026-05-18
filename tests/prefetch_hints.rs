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
}
