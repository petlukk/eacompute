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

    #[test]
    fn test_scalar_min_f64() {
        assert_output(
            r#"
            export func main() {
                let a: f64 = 3.14159
                let b: f64 = 2.71828
                println(min(a, b))
            }
            "#,
            "2.71828",
        );
    }

    #[test]
    fn test_scalar_max_f64() {
        assert_output(
            r#"
            export func main() {
                let a: f64 = 3.14159
                let b: f64 = 2.71828
                println(max(a, b))
            }
            "#,
            "3.14159",
        );
    }

    #[test]
    fn test_reduce_min_i32x4() {
        assert_c_interop(
            r#"
            export func test(data: *i32, out: *mut i32) {
                let v: i32x4 = load(data, 0)
                out[0] = reduce_min(v)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int32_t*, int32_t*);
            int main() {
                int32_t data[] = {7, -3, 15, 2};
                int32_t out;
                test(data, &out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "-3",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vector_min_f32x8() {
        assert_c_interop(
            r#"
            export func test(a: *f32, b: *f32, out: *mut f32) {
                let va: f32x8 = load(a, 0)
                let vb: f32x8 = load(b, 0)
                let vr: f32x8 = min(va, vb)
                store(out, 0, vr)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const float*, float*);
            int main() {
                float a[] = {1,5,3,7,2,6,4,8};
                float b[] = {8,4,6,2,7,3,5,1};
                float out[8];
                test(a, b, out);
                for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "1 4 3 2 2 3 4 1",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vector_max_f32x8() {
        assert_c_interop(
            r#"
            export func test(a: *f32, b: *f32, out: *mut f32) {
                let va: f32x8 = load(a, 0)
                let vb: f32x8 = load(b, 0)
                let vr: f32x8 = max(va, vb)
                store(out, 0, vr)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const float*, const float*, float*);
            int main() {
                float a[] = {1,5,3,7,2,6,4,8};
                float b[] = {8,4,6,2,7,3,5,1};
                float out[8];
                test(a, b, out);
                for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
                printf("\n");
                return 0;
            }
            "#,
            "8 5 6 7 7 6 5 8",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_vector_min_max_ir_uses_llvm_intrinsic() {
        let ir = compile_to_ir(
            r#"
            export func test(a: *f32, b: *f32, out: *mut f32) {
                let va: f32x8 = load(a, 0)
                let vb: f32x8 = load(b, 0)
                let vmin: f32x8 = min(va, vb)
                let vmax: f32x8 = max(va, vb)
                store(out, 0, vmin)
                store(out, 8, vmax)
            }
            "#,
        );
        assert!(
            ir.contains("llvm.minnum.v8f32"),
            "IR should use llvm.minnum.v8f32: {ir}"
        );
        assert!(
            ir.contains("llvm.maxnum.v8f32"),
            "IR should use llvm.maxnum.v8f32: {ir}"
        );
    }
}
