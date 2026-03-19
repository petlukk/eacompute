#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Vector Shift Operations (.<<, .>>) ===

    // --- Shift-left ---

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

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_shift_left_i32x8() {
        assert_c_interop(
            r#"
export func test_shl32(a: *i32, out: *mut i32, len: i32) {
    let shift: i32x8 = splat(3)
    let mut i: i32 = 0
    while i + 8 <= len {
        let va: i32x8 = load(a, i)
        store(out, i, va .<< shift)
        i = i + 8
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_shl32(const int32_t*, int32_t*, int);

int main() {
    int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t out[8] = {0};
    test_shl32(a, out, 8);
    printf("%d %d %d %d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return 0;
}
"#,
            "8 16 24 32 40 48 56 64",
        );
    }

    // --- Shift-right ---

    #[test]
    fn test_shift_right_u8x16() {
        assert_c_interop(
            r#"
export func test_shr(a: *u8, out: *mut u8, len: i32) {
    let shift: u8x16 = splat(2)
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        store(out, i, va .>> shift)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_shr(const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {4, 8, 16, 32, 64, 128, 255, 1, 0,0,0,0,0,0,0,0};
    uint8_t out[16] = {0};
    test_shr(a, out, 16);
    printf("%d %d %d %d %d %d %d %d\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return 0;
}
"#,
            "1 2 4 8 16 32 63 0",
        );
    }

    #[test]
    fn test_shift_right_i32x4_arithmetic() {
        assert_c_interop(
            r#"
export func test_ashr(a: *i32, out: *mut i32, len: i32) {
    let shift: i32x4 = splat(2)
    let mut i: i32 = 0
    while i + 4 <= len {
        let va: i32x4 = load(a, i)
        store(out, i, va .>> shift)
        i = i + 4
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_ashr(const int32_t*, int32_t*, int);

int main() {
    int32_t a[4] = {64, -64, 128, -128};
    int32_t out[4] = {0};
    test_ashr(a, out, 4);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "16 -16 32 -32",
        );
    }

    // --- Type rejection ---

    #[test]
    fn test_shift_rejects_float_vectors() {
        let source = r#"
export func bad(a: *f32, out: *mut f32, n: i32) {
    let va: f32x4 = load(a, 0)
    let shift: f32x4 = splat(2.0)
    store(out, 0, va .<< shift)
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("integer"),
            "expected integer error, got: {msg}"
        );
    }
}
