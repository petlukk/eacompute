//! to_u64 (v1.21.0): the missing member of the to_* conversions, for 64-bit
//! masks built from two u32 halves (Linta lex_masks_arm, Vingilot
//! vg_css_masks), which went through i64 before.

#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    const SRC: &str = "export func join(hi: u32, lo: u32) -> u64 { return to_u64(hi) << to_u64(32) | to_u64(lo) }\nexport func from_i32(x: i32) -> u64 { return to_u64(x) }\nexport func from_f64(x: f64) -> u64 { return to_u64(x) }\n";

    #[test]
    fn to_u64_from_u32_mask_keeps_bit_31() {
        assert_c_interop(
            SRC,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            uint64_t join(uint32_t hi, uint32_t lo);
            uint64_t from_i32(int32_t x);
            uint64_t from_f64(double x);
            int main(void) {
                printf("%llu %llu %llu\n", (unsigned long long)join(0x80000001u, 7u),
                       (unsigned long long)join(1u, 0x80000000u),
                       (unsigned long long)from_f64(1e19));
                return 0;
            }
            "#,
            "9223372041149743111 6442450944 10000000000000000000",
        );
    }

    #[test]
    fn to_u64_from_i32_negative() {
        assert_c_interop(
            SRC,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            uint64_t from_i32(int32_t x);
            uint64_t join(uint32_t hi, uint32_t lo);
            uint64_t from_f64(double x);
            int main(void) { printf("%llu\n", (unsigned long long)from_i32(-1)); return 0; }
            "#,
            "18446744073709551615",
        );
    }

    #[test]
    fn to_u64_needs_a_number() {
        assert_typecheck_error("export func f(a: i32x4) -> u64 { return to_u64(a) }", "numeric");
    }
}
