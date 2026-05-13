#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase B: i16x8, i16x16, maddubs_i16, maddubs_i32 ===

    #[test]
    fn test_i16x8_literal_element() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [100, 200, 300, 400, 500, 600, 700, 800]i16x8
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[7]
                println(y)
            }
            "#,
            "100\n800",
        );
    }

    #[test]
    fn test_i16x8_add_dot() {
        assert_output(
            r#"
            func main() {
                let a: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let b: i16x8 = [10, 20, 30, 40, 50, 60, 70, 80]i16x8
                let c: i16x8 = a .+ b
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[7]
                println(y)
            }
            "#,
            "11\n88",
        );
    }

    #[test]
    fn test_i16x8_splat() {
        assert_output(
            r#"
            func main() {
                let s: i16 = 1000
                let v: i16x8 = splat(s)
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[7]
                println(y)
            }
            "#,
            "1000\n1000",
        );
    }

    #[test]
    fn test_i16x8_reduce_add() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let s: i16 = reduce_add(v)
                println(s)
            }
            "#,
            "36",
        );
    }

    #[test]
    fn test_i16x8_reduce_max() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [10, 200, -30, 150, 5, 1800, 20, 100]i16x8
                let m: i16 = reduce_max(v)
                println(m)
            }
            "#,
            "1800",
        );
    }

    #[test]
    fn test_i16x8_reduce_min() {
        assert_output(
            r#"
            func main() {
                let v: i16x8 = [10, 200, -30, 150, 5, 1800, 20, 100]i16x8
                let m: i16 = reduce_min(v)
                println(m)
            }
            "#,
            "-30",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i16x16_splat_and_element() {
        assert_output(
            r#"
            func main() {
                let s: i16 = 42
                let v: i16x16 = splat(s)
                let x: i16 = v[0]
                println(x)
                let y: i16 = v[15]
                println(y)
            }
            "#,
            "42\n42",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i16x16_literal_add() {
        assert_output(
            r#"
            func main() {
                let a: i16x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i16x16
                let b: i16x16 = splat(100)
                let c: i16x16 = a .+ b
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[15]
                println(y)
            }
            "#,
            "101\n116",
        );
    }

    // === maddubs_i16: u8x16 × i8x16 → i16x8 (fast, wrapping — x86 SSSE3 only) ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_basic() {
        // maddubs_i16([2,3,...], [4,5,...]):
        // pair (2*4)+(3*5) = 8+15 = 23, (2*4)+(3*5)=23, etc.
        // [2,3, 2,3, 2,3, 2,3, 2,3, 2,3, 2,3, 2,3] u8x16
        // [4,5, 4,5, 4,5, 4,5, 4,5, 4,5, 4,5, 4,5] i8x16
        // each i16 lane = 2*4 + 3*5 = 23
        assert_output(
            r#"
            func main() {
                let a: u8x16 = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]u8x16
                let b: i8x16 = [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]i8x16
                let c: i16x8 = maddubs_i16(a, b)
                let x: i16 = c[0]
                println(x)
                let y: i16 = c[7]
                println(y)
            }
            "#,
            "23\n23",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_dot_product() {
        // Dot product of 16 u8 values × 16 i8 weights, accumulate as i32.
        // [1,1,...] × [1,1,...] = 8 pairs each = 2, reduce_add(i16x8) = 16
        assert_output(
            r#"
            func main() {
                let a: u8x16 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]u8x16
                let b: i8x16 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]i8x16
                let c: i16x8 = maddubs_i16(a, b)
                let s: i16 = reduce_add(c)
                println(s)
            }
            "#,
            "16",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_c_interop() {
        assert_c_interop(
            r#"
            export func dot_u8i8_16(act: *u8, wt: *i8, n: i32) -> i16 {
                let mut acc: i16x8 = splat(0)
                let mut i: i32 = 0
                while i < n {
                    let a: u8x16 = load(act, i)
                    let b: i8x16 = load(wt, i)
                    acc = acc .+ maddubs_i16(a, b)
                    i = i + 16
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int16_t dot_u8i8_16(const uint8_t *act, const int8_t *wt, int n);
            int main() {
                uint8_t act[32];
                int8_t  wt[32];
                for (int i = 0; i < 32; i++) { act[i] = 1; wt[i] = 2; }
                int result = (int)dot_u8i8_16(act, wt, 32);
                printf("%d\n", result);
                return 0;
            }
            "#,
            "64",
        );
    }

    #[test]
    fn test_i16x8_load_store_c_interop() {
        assert_c_interop(
            r#"
            export func add_constant_i16x8(src: *i16, dst: *mut i16, n: i32, val: i16) {
                let mut i: i32 = 0
                let splat_val: i16x8 = splat(val)
                while i < n {
                    let chunk: i16x8 = load(src, i)
                    let result: i16x8 = chunk .+ splat_val
                    store(dst, i, result)
                    i = i + 8
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void add_constant_i16x8(const int16_t *src, int16_t *dst, int n, int16_t val);
            int main() {
                int16_t src[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
                int16_t dst[16] = {0};
                add_constant_i16x8(src, dst, 16, 100);
                printf("%d %d %d %d\n", (int)dst[0], (int)dst[7], (int)dst[8], (int)dst[15]);
                return 0;
            }
            "#,
            "101 108 109 116",
        );
    }
}
