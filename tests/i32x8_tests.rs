#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i32x8_splat_reduce_add() {
        assert_c_interop(
            r#"
            export func test(out: *mut i32) {
                let v: i32x8 = splat(7)
                let s: i32 = reduce_add(v)
                out[0] = s
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(int32_t*);
            int main() {
                int32_t out;
                test(&out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "56",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i32x8_load_store_add() {
        assert_c_interop(
            r#"
            export func test(a: *i32, b: *i32, out: *mut i32) {
                let va: i32x8 = load(a, 0)
                let vb: i32x8 = load(b, 0)
                let result: i32x8 = va .+ vb
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int32_t*, const int32_t*, int32_t*);
            int main() {
                int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                int32_t b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
                int32_t out[8];
                test(a, b, out);
                printf("%d %d %d %d %d %d %d %d\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "11 22 33 44 55 66 77 88",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i32x8_select_gt() {
        assert_c_interop(
            r#"
            export func test(input: *i32, out: *mut i32) {
                let v: i32x8 = load(input, 0)
                let thresh: i32x8 = splat(5)
                let one: i32x8 = splat(1)
                let zero: i32x8 = splat(0)
                let result: i32x8 = select(v .> thresh, one, zero)
                let count: i32 = reduce_add(result)
                out[0] = count
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int32_t*, int32_t*);
            int main() {
                int32_t input[8] = {3, 7, 2, 9, 5, 6, 1, 8};
                int32_t out;
                test(input, &out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "4",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_i32x8_mul() {
        assert_c_interop(
            r#"
            export func test(a: *i32, b: *i32, out: *mut i32) {
                let va: i32x8 = load(a, 0)
                let vb: i32x8 = load(b, 0)
                let result: i32x8 = va .* vb
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int32_t*, const int32_t*, int32_t*);
            int main() {
                int32_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                int32_t b[8] = {2, 3, 4, 5, 6, 7, 8, 9};
                int32_t out[8];
                test(a, b, out);
                printf("%d %d %d %d %d %d %d %d\n",
                    out[0], out[1], out[2], out[3],
                    out[4], out[5], out[6], out[7]);
                return 0;
            }
            "#,
            "2 6 12 20 30 42 56 72",
        );
    }
}
