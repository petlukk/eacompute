#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === maddubs_i32 AVX2: u8x32 × i8x32 → i32x8 ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_i32_avx2_basic() {
        // splat(2) × splat(4): each byte pair = 2*4 = 8.
        // pmaddubsw: adjacent u8×i8 pairs → i16: 8+8 = 16 (16 lanes).
        // pmaddwd(ones): adjacent i16 pairs → i32: 16+16 = 32 (8 lanes).
        // reduce_add(i32x8) = 8 × 32 = 256. Lane[0] = 32.
        assert_output(
            r#"
            func main() {
                let a: u8x32 = splat(2)
                let b: i8x32 = splat(4)
                let c: i32x8 = maddubs_i32(a, b)
                let s: i32 = reduce_add(c)
                println(s)
                let x: i32 = c[0]
                println(x)
            }
            "#,
            "256\n32",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_i32_avx2_c_interop() {
        assert_c_interop(
            r#"
            export func dot_avx2(
                act: *u8, wt: *i8, n: i32
            ) -> i32 {
                let mut acc: i32x8 = splat(0)
                let mut i: i32 = 0
                while i < n {
                    let a: u8x32 = load(act, i)
                    let b: i8x32 = load(wt, i)
                    acc = acc .+ maddubs_i32(a, b)
                    i = i + 32
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t dot_avx2(
                const uint8_t *act, const int8_t *wt, int n
            );
            int main() {
                uint8_t act[64];
                int8_t  wt[64];
                for (int i = 0; i < 64; i++) {
                    act[i] = 1; wt[i] = 2;
                }
                printf("%d\n", dot_avx2(act, wt, 64));
                return 0;
            }
            "#,
            "128",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_i32_avx2_overflow() {
        // 200 iters × splat(10) × splat(5):
        // per iter per lane: (10*5+10*5)+(10*5+10*5) = 200
        // 8 lanes × 200/iter × 200 iters = 320,000
        assert_c_interop(
            r#"
            export func accum_avx2(
                act: *u8, wt: *i8, iters: i32
            ) -> i32 {
                let mut acc: i32x8 = splat(0)
                let mut it: i32 = 0
                while it < iters {
                    let a: u8x32 = load(act, 0)
                    let b: i8x32 = load(wt, 0)
                    acc = acc .+ maddubs_i32(a, b)
                    it = it + 1
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t accum_avx2(
                const uint8_t *act, const int8_t *wt, int iters
            );
            int main() {
                uint8_t act[32];
                int8_t  wt[32];
                for (int i = 0; i < 32; i++) {
                    act[i] = 10; wt[i] = 5;
                }
                printf("%d\n", accum_avx2(act, wt, 200));
                return 0;
            }
            "#,
            "320000",
        );
    }

    #[test]
    fn test_cvt_f16_f32_wrong_type() {
        let source = r#"
            func main() {
                let a: f32x4 = splat(1.0)
                let b: f32x4 = cvt_f16_f32(a)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("i16x4") || msg.contains("i16x8"), "got: {msg}");
    }

    #[test]
    fn test_cvt_f32_f16_wrong_type() {
        let source = r#"
            func main() {
                let a: i32x4 = splat(1)
                let b: i16x4 = cvt_f32_f16(a)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("f32x4") || msg.contains("f32x8"), "got: {msg}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_maddubs_i32_avx2_ir_check() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func dot_avx2(
                act: *u8, wt: *i8, n: i32
            ) -> i32 {
                let mut acc: i32x8 = splat(0)
                let mut i: i32 = 0
                while i < n {
                    let a: u8x32 = load(act, i)
                    let b: i8x32 = load(wt, i)
                    acc = acc .+ maddubs_i32(a, b)
                    i = i + 32
                }
                return reduce_add(acc)
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("dot_avx2.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("maddubs_i32 AVX2 IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("avx2.pmadd.ub.sw"),
            "expected avx2.pmadd.ub.sw in IR:\n{ir}"
        );
        assert!(
            ir.contains("avx2.pmadd.wd"),
            "expected avx2.pmadd.wd in IR:\n{ir}"
        );
    }

    // === cvt_f16_f32 / cvt_f32_f16 ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cvt_f16_f32_roundtrip() {
        assert_c_interop(
            r#"
            export func test(out: *mut f32, out16: *mut i16) {
                let one_f16: i16 = 15360
                let v: i16x8 = splat(one_f16)
                let f: f32x8 = cvt_f16_f32(v)
                store(out, 0, f)
                let back: i16x8 = cvt_f32_f16(f)
                store(out16, 0, back)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(float*, int16_t*);
            int main() {
                float f[8];
                int16_t h[8];
                test(f, h);
                printf("%.1f\n%d\n", f[0], h[0]);
                return 0;
            }
            "#,
            "1.0\n15360",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cvt_f16_f32_i16x8() {
        assert_c_interop(
            r#"
            export func test(out: *mut f32) {
                let one_f16: i16 = 15360
                let v: i16x8 = splat(one_f16)
                let f: f32x8 = cvt_f16_f32(v)
                store(out, 0, f)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(float*);
            int main() {
                float f[8];
                test(f);
                printf("%.1f %.1f %.1f %.1f\n", f[0], f[1], f[6], f[7]);
                return 0;
            }
            "#,
            "1.0 1.0 1.0 1.0",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cvt_f16_f32_emits_fpext() {
        let ir = compile_to_ir(
            r#"
            export func convert(inp: *i16, out: *mut f32) {
                let v: i16x8 = load(inp, 0)
                let f: f32x8 = cvt_f16_f32(v)
                store(out, 0, f)
            }
            "#,
        );
        assert!(ir.contains("fpext"), "expected fpext in IR:\n{ir}");
        assert!(ir.contains("half"), "expected half type in IR:\n{ir}");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cvt_f16_f32_zero() {
        assert_c_interop(
            r#"
            export func test(out: *mut f32) {
                let zero: i16 = 0
                let v: i16x8 = splat(zero)
                let f: f32x8 = cvt_f16_f32(v)
                store(out, 0, f)
            }
            "#,
            r#"
            #include <stdio.h>
            extern void test(float*);
            int main() {
                float f[8];
                test(f);
                printf("%.1f\n", f[0]);
                return 0;
            }
            "#,
            "0.0",
        );
    }
}
