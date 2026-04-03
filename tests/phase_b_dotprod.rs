#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === madd_i16: i16x8 × i16x8 → i32x4 (SSE2 pmaddwd) ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_basic() {
        // madd_i16([2,3,2,3,2,3,2,3], [4,5,4,5,4,5,4,5]):
        // pmaddwd: lane[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]
        // = 2*4 + 3*5 = 23 per i32 lane (4 lanes)
        // reduce_add = 4 * 23 = 92
        assert_output(
            r#"
            func main() {
                let a: i16x8 = [2, 3, 2, 3, 2, 3, 2, 3]i16x8
                let b: i16x8 = [4, 5, 4, 5, 4, 5, 4, 5]i16x8
                let c: i32x4 = madd_i16(a, b)
                let x: i32 = c[0]
                println(x)
                let s: i32 = reduce_add(c)
                println(s)
            }
            "#,
            "23\n92",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_with_maddubs_chain() {
        // Explicit two-step chain replacing the old maddubs_i32:
        // maddubs_i16(u8, i8) → i16x8, then madd_i16(result, ones) → i32x4.
        // act=splat(1), wt=splat(2), 32 elements processed in 2 iterations.
        // Per iter: maddubs_i16 → each i16 = 1*2+1*2 = 4, madd_i16(ones) → 4+4 = 8
        // 2 iters × reduce_add(i32x4 of 8s) = 2 × 32 = 64
        assert_c_interop(
            r#"
            export func dot_u8i8_i32(act: *u8, wt: *i8, n: i32) -> i32 {
                let mut acc: i32x4 = splat(0)
                let ones: i16x8 = splat(1)
                let mut i: i32 = 0
                while i < n {
                    let a: u8x16 = load(act, i)
                    let b: i8x16 = load(wt, i)
                    let t: i16x8 = maddubs_i16(a, b)
                    acc = acc .+ madd_i16(t, ones)
                    i = i + 16
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t dot_u8i8_i32(const uint8_t *act, const int8_t *wt, int n);
            int main() {
                uint8_t act[32];
                int8_t  wt[32];
                for (int i = 0; i < 32; i++) { act[i] = 1; wt[i] = 2; }
                int32_t result = dot_u8i8_i32(act, wt, 32);
                printf("%d\n", result);
                return 0;
            }
            "#,
            "64",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_arbitrary_operands() {
        // Verify madd_i16 works with arbitrary i16 operands (not just ones).
        // [100, 200, 300, 400, 500, 600, 700, 800] × [1, 2, 3, 4, 5, 6, 7, 8]
        // lane0 = 100*1 + 200*2 = 500
        // lane1 = 300*3 + 400*4 = 2500
        // lane2 = 500*5 + 600*6 = 6100
        // lane3 = 700*7 + 800*8 = 11300
        assert_output(
            r#"
            func main() {
                let a: i16x8 = [100, 200, 300, 400, 500, 600, 700, 800]i16x8
                let b: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let c: i32x4 = madd_i16(a, b)
                let x: i32 = c[0]
                println(x)
                let y: i32 = c[1]
                println(y)
                let z: i32 = c[2]
                println(z)
                let w: i32 = c[3]
                println(w)
            }
            "#,
            "500\n2500\n6100\n11300",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_ir_check() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func madd_test(src: *i16, n: i32) -> i32 {
                let mut acc: i32x4 = splat(0)
                let ones: i16x8 = splat(1)
                let mut i: i32 = 0
                while i < n {
                    let a: i16x8 = load(src, i)
                    acc = acc .+ madd_i16(a, ones)
                    i = i + 8
                }
                return reduce_add(acc)
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("madd_i16.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("madd_i16 IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("pmadd.wd"),
            "expected pmadd.wd in IR, got:\n{ir}"
        );
    }

    // === to_i16: scalar narrowing cast ===

    #[test]
    fn test_to_i16_from_i32() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 42
                let y: i16 = to_i16(x)
                println(y)
            }
            "#,
            "42",
        );
    }

    #[test]
    fn test_to_i16_from_i32_negative() {
        assert_output(
            r#"
            func main() {
                let x: i32 = -100
                let y: i16 = to_i16(x)
                println(y)
            }
            "#,
            "-100",
        );
    }

    #[test]
    fn test_to_i16_from_f32() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.7
                let y: i16 = to_i16(x)
                println(y)
            }
            "#,
            "3",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_to_i16_splat_for_madd() {
        // The motivating use case: narrow an i32 scale to i16 for splat into madd_i16.
        assert_c_interop(
            r#"
            export func scaled_dot(src: *i16, n: i32, scale: i32) -> i32 {
                let mut acc: i32x4 = splat(0)
                let s: i16 = to_i16(scale)
                let sv: i16x8 = splat(s)
                let mut i: i32 = 0
                while i < n {
                    let a: i16x8 = load(src, i)
                    acc = acc .+ madd_i16(a, sv)
                    i = i + 8
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t scaled_dot(const int16_t *src, int n, int scale);
            int main() {
                int16_t src[16];
                for (int i = 0; i < 16; i++) src[i] = 1;
                // scale=3: madd_i16([1,1,...], [3,3,...]) = (1*3+1*3)=6 per lane
                // 4 lanes × 6 × 2 iters = 48
                printf("%d\n", scaled_dot(src, 16, 3));
                return 0;
            }
            "#,
            "48",
        );
    }

    // === hadd_i16: horizontal add adjacent i16 pairs (SSSE3 phaddw) ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hadd_i16_basic() {
        // hadd_i16([1,2,3,4,5,6,7,8], [10,20,30,40,50,60,70,80]):
        // result = [1+2, 3+4, 5+6, 7+8, 10+20, 30+40, 50+60, 70+80]
        //        = [3, 7, 11, 15, 30, 70, 110, 150]
        assert_output(
            r#"
            func main() {
                let a: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]i16x8
                let b: i16x8 = [10, 20, 30, 40, 50, 60, 70, 80]i16x8
                let c: i16x8 = hadd_i16(a, b)
                let x0: i16 = c[0]
                let x1: i16 = c[1]
                let x4: i16 = c[4]
                let x7: i16 = c[7]
                println(x0)
                println(x1)
                println(x4)
                println(x7)
            }
            "#,
            "3\n7\n30\n150",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hadd_i16_ir_check() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func f(a: i16x8, b: i16x8) -> i16x8 {
                return hadd_i16(a, b)
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("hadd_i16.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("hadd_i16 IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("phadd.w.128"),
            "expected phadd.w.128 in IR, got:\n{ir}"
        );
    }
}
