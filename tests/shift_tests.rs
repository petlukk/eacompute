#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Vector Shift Operations (.<<, .>>) ===

    #[test]
    fn test_shift_left_u8x16() {
        assert_c_interop(
            r#"
export func test_shl(a: *u8, out: *mut u8, len: i32) {
    let shift: u8x16 = splat(2)
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        store(out, i, va .<< shift)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_shl(const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 255, 3, 7, 15, 31, 63, 127};
    uint8_t out[16] = {0};
    test_shl(a, out, 16);
    printf("%d %d %d %d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return 0;
}
"#,
            "4 8 16 32 64 128 0 0",
        );
    }
}
