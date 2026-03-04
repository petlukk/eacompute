#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_for_basic() {
        assert_output(
            r#"
            export func main() {
                let mut sum: i32 = 0
                for i in 0..5 {
                    sum = sum + i
                }
                println(sum)
            }
            "#,
            "10",
        );
    }

    #[test]
    fn test_for_step() {
        assert_output(
            r#"
            export func main() {
                let mut sum: i32 = 0
                for i in 0..10 step 2 {
                    sum = sum + i
                }
                println(sum)
            }
            "#,
            "20",
        );
    }

    #[test]
    fn test_for_with_array() {
        assert_c_interop(
            r#"
            export func sum_array(data: *i32, n: i32, out: *mut i32) {
                let mut total: i32 = 0
                for i in 0..n {
                    total = total + data[i]
                }
                out[0] = total
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void sum_array(const int32_t*, int32_t, int32_t*);
            int main() {
                int32_t data[] = {1,2,3,4,5};
                int32_t out;
                sum_array(data, 5, &out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "15",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_for_step_simd() {
        assert_c_interop(
            r#"
            export func double_array(data: *f32, n: i32, out: *mut f32) {
                let two: f32x8 = splat(2.0)
                for i in 0..n step 8 {
                    let v: f32x8 = load(data, i)
                    store(out, i, v .* two)
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void double_array(const float*, int, float*);
            int main() {
                float data[8] = {1,2,3,4,5,6,7,8};
                float out[8];
                double_array(data, 8, out);
                printf("%.0f %.0f %.0f %.0f\n", out[0], out[1], out[6], out[7]);
                return 0;
            }
            "#,
            "2 4 14 16",
        );
    }

    #[test]
    fn test_for_nonzero_start() {
        assert_output(
            r#"
            export func main() {
                let mut sum: i32 = 0
                for i in 3..7 {
                    sum = sum + i
                }
                println(sum)
            }
            "#,
            "18",
        );
    }
}
