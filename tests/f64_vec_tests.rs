#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f64x4_splat_reduce() {
        assert_c_interop(
            r#"
            export func test(out: *mut f64) {
                let v: f64x4 = splat(2.5)
                let s: f64 = reduce_add(v)
                out[0] = s
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(double*);
            int main() {
                double out;
                test(&out);
                printf("%.1f\n", out);
                return 0;
            }
            "#,
            "10.0",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f64x4_load_store() {
        assert_c_interop(
            r#"
            export func test(input: *f64, out: *mut f64) {
                let v: f64x4 = load(input, 0)
                let two: f64x4 = splat(2.0)
                let result: f64x4 = v .* two
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const double*, double*);
            int main() {
                double input[4] = {1.0, 2.0, 3.0, 4.0};
                double out[4];
                test(input, out);
                printf("%.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "2.0 4.0 6.0 8.0",
        );
    }

    #[test]
    fn test_f64x2_basic() {
        assert_c_interop(
            r#"
            export func test(input: *f64, out: *mut f64) {
                let v: f64x2 = load(input, 0)
                let s: f64 = reduce_add(v)
                out[0] = s
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const double*, double*);
            int main() {
                double input[2] = {3.14, 2.72};
                double out;
                test(input, &out);
                printf("%.2f\n", out);
                return 0;
            }
            "#,
            "5.86",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f64x4_fma() {
        assert_c_interop(
            r#"
            export func test(a: *f64, b: *f64, c: *f64, out: *mut f64) {
                let va: f64x4 = load(a, 0)
                let vb: f64x4 = load(b, 0)
                let vc: f64x4 = load(c, 0)
                let result: f64x4 = fma(va, vb, vc)
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const double*, const double*, const double*, double*);
            int main() {
                double a[4] = {1.0, 2.0, 3.0, 4.0};
                double b[4] = {2.0, 3.0, 4.0, 5.0};
                double c[4] = {0.5, 0.5, 0.5, 0.5};
                double out[4];
                test(a, b, c, out);
                printf("%.1f %.1f %.1f %.1f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "2.5 6.5 12.5 20.5",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f64x4_select() {
        assert_c_interop(
            r#"
            export func test(input: *f64, out: *mut f64) {
                let v: f64x4 = load(input, 0)
                let thresh: f64x4 = splat(5.0)
                let one: f64x4 = splat(1.0)
                let zero: f64x4 = splat(0.0)
                let result: f64x4 = select(v .> thresh, one, zero)
                store(out, 0, result)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(const double*, double*);
            int main() {
                double input[4] = {3.0, 7.0, 2.0, 9.0};
                double out[4];
                test(input, out);
                printf("%.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "0 1 0 1",
        );
    }
}
