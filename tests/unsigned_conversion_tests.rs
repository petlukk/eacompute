#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Conversions keep the signedness of non-variable sources ===
    // Regression: to_i16/to_i32/to_i64/to_f32/to_f64 decided zero- vs
    // sign-extension (and uitofp vs sitofp) only for plain variables, so an
    // unsigned expression (`a | b`, a call returning u32, movemask) with its
    // top bit set was silently sign-extended.

    #[test]
    fn test_to_i64_of_unsigned_binary_zero_extends() {
        assert_c_interop(
            r#"
            export func f(a: u32, b: u32) -> i64 {
                return to_i64(a | b)
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int64_t f(uint32_t, uint32_t);
            int main() {
                printf("%lld\n", (long long) f(0x80000000u, 1u));
                return 0;
            }
        "#,
            "2147483649",
        );
    }

    #[test]
    fn test_to_i64_of_call_returning_u32_zero_extends() {
        assert_c_interop(
            r#"
            func id(x: u32) -> u32 {
                return x
            }
            export func f(a: u32) -> i64 {
                return to_i64(id(a)) << 1
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int64_t f(uint32_t);
            int main() {
                printf("%lld\n", (long long) f(0xFFFFFFFFu));
                return 0;
            }
        "#,
            "8589934590",
        );
    }

    #[test]
    fn test_to_i32_of_unsigned_u16_expression_zero_extends() {
        assert_c_interop(
            r#"
            export func f(a: u16, b: u16) -> i32 {
                return to_i32(a + b)
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t f(uint16_t, uint16_t);
            int main() {
                printf("%d\n", f(0x8000, 0x10));
                return 0;
            }
        "#,
            "32784",
        );
    }

    #[test]
    fn test_to_f64_of_unsigned_expression_is_unsigned() {
        assert_c_interop(
            r#"
            export func f(a: u32, b: u32) -> f64 {
                return to_f64(a | b)
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern double f(uint32_t, uint32_t);
            int main() {
                printf("%.1f\n", f(0x80000000u, 0u));
                return 0;
            }
        "#,
            "2147483648.0",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_to_i64_of_movemask_zero_extends() {
        assert_c_interop(
            r#"
            export func f(p: *u8) -> i64 {
                let v: u8x32 = load(p, 0)
                return to_i64(movemask_u8x32(v))
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int64_t f(const uint8_t*);
            int main() {
                uint8_t b[32];
                for (int i = 0; i < 32; i++) b[i] = 0x80;
                printf("%lld\n", (long long) f(b));
                return 0;
            }
        "#,
            "4294967295",
        );
    }
}
