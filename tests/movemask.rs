// Tests for the movemask intrinsic (u8x16/u8x32 → i32 bitmask).

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // --- u8x16 basic: specific bytes match ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_movemask_u8x16_basic() {
        assert_c_interop(
            r#"
            export func test(data: *u8, out: *mut i32) {
                let v: u8x16 = load(data, 0)
                let target: u8x16 = splat(0xFF)
                let mask: i32 = movemask(v .== target)
                out[0] = mask
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, int32_t*);
            int main() {
                uint8_t data[16] = {0xFF,0,0xFF,0, 0,0,0,0, 0xFF,0,0,0, 0,0,0,0xFF};
                int32_t out = 0;
                test(data, &out);
                // bits set: 0,2,8,15 => 0x8105 = 33029
                printf("%d\n", out);
                return 0;
            }"#,
            "33029",
        );
    }

    // --- u8x32 all match: every byte equals target ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_movemask_u8x32_all_match() {
        assert_c_interop(
            r#"
            export func test(data: *u8, out: *mut i32) {
                let v: u8x32 = load(data, 0)
                let target: u8x32 = splat(42)
                let mask: i32 = movemask(v .== target)
                out[0] = mask
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, int32_t*);
            int main() {
                uint8_t data[32];
                for (int i = 0; i < 32; i++) data[i] = 42;
                int32_t out = 0;
                test(data, &out);
                // all 32 bits set = 0xFFFFFFFF = -1 (signed i32)
                printf("%d\n", out);
                return 0;
            }"#,
            "-1",
        );
    }

    // --- u8x32 none match: no bytes equal target ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_movemask_u8x32_none_match() {
        assert_c_interop(
            r#"
            export func test(data: *u8, out: *mut i32) {
                let v: u8x32 = load(data, 0)
                let target: u8x32 = splat(99)
                let mask: i32 = movemask(v .== target)
                out[0] = mask
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, int32_t*);
            int main() {
                uint8_t data[32];
                for (int i = 0; i < 32; i++) data[i] = (uint8_t)i;
                int32_t out = -1;
                test(data, &out);
                printf("%d\n", out);
                return 0;
            }"#,
            "0",
        );
    }

    // --- CSV comma detection: movemask finds comma positions, C counts bits ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_movemask_find_commas() {
        assert_c_interop(
            r#"
            export func find_commas(text: *u8, out: *mut i32) {
                let chunk: u8x32 = load(text, 0)
                let comma: u8x32 = splat(44)
                let mask: i32 = movemask(chunk .== comma)
                out[0] = mask
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void find_commas(const uint8_t*, int32_t*);
            int main() {
                // 32 bytes with commas at known positions
                const char* text = "hello,world,foo,bar,baz,qux,abc,";
                int32_t mask = 0;
                find_commas((const uint8_t*)text, &mask);
                // Count bits set in mask (= number of commas found)
                int count = __builtin_popcount((unsigned)mask);
                printf("%d\n", count);
                return 0;
            }"#,
            "7",
        );
    }

    // --- IR inspection: verify pmovmskb appears in LLVM IR ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_movemask_ir() {
        let ir = compile_to_ir(
            r#"
            export func test(data: *u8, out: *mut i32) {
                let v: u8x16 = load(data, 0)
                let zero: u8x16 = splat(0)
                let mask: i32 = movemask(v .== zero)
                out[0] = mask
            }
        "#,
        );
        assert!(
            ir.contains("pmovmskb"),
            "IR must contain pmovmskb intrinsic call:\n{ir}"
        );
    }
}
