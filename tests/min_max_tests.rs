#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_scalar_min_i32() {
        assert_output(
            r#"
            export func main() {
                let a: i32 = 10
                let b: i32 = 3
                println(min(a, b))
            }
            "#,
            "3",
        );
    }

    #[test]
    fn test_scalar_max_i32() {
        assert_output(
            r#"
            export func main() {
                let a: i32 = 10
                let b: i32 = 3
                println(max(a, b))
            }
            "#,
            "10",
        );
    }

    #[test]
    fn test_scalar_min_f32() {
        assert_output(
            r#"
            export func main() {
                let a: f32 = 3.5
                let b: f32 = 1.2
                println(min(a, b))
            }
            "#,
            "1.2",
        );
    }

    #[test]
    fn test_scalar_max_f32() {
        assert_output(
            r#"
            export func main() {
                let a: f32 = 3.5
                let b: f32 = 1.2
                println(max(a, b))
            }
            "#,
            "3.5",
        );
    }

    #[test]
    fn test_scalar_min_negative() {
        assert_output(
            r#"
            export func main() {
                let a: i32 = -5
                let b: i32 = 3
                println(min(a, b))
            }
            "#,
            "-5",
        );
    }

    #[test]
    fn test_min_max_in_loop() {
        assert_c_interop(
            r#"
            export func find_min_max(data: *i32, n: i32, out: *mut i32) {
                let mut lo: i32 = data[0]
                let mut hi: i32 = data[0]
                let mut i: i32 = 1
                while i < n {
                    lo = min(lo, data[i])
                    hi = max(hi, data[i])
                    i = i + 1
                }
                out[0] = lo
                out[1] = hi
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void find_min_max(const int32_t*, int32_t, int32_t*);
            int main() {
                int32_t data[] = {5, -3, 8, 1, -7, 4};
                int32_t out[2];
                find_min_max(data, 6, out);
                printf("%d %d\n", out[0], out[1]);
                return 0;
            }
            "#,
            "-7 8",
        );
    }

    #[test]
    fn test_min_max_ir_uses_llvm_intrinsic() {
        let ir = compile_to_ir(
            r#"
            export func test(a: i32, b: i32, out: *mut i32) {
                out[0] = min(a, b)
                out[1] = max(a, b)
            }
            "#,
        );
        assert!(
            ir.contains("llvm.smin") || ir.contains("@llvm.smin"),
            "IR should use llvm.smin: {ir}"
        );
        assert!(
            ir.contains("llvm.smax") || ir.contains("@llvm.smax"),
            "IR should use llvm.smax: {ir}"
        );
    }
}
