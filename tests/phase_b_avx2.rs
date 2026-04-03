#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === madd_i16 AVX2: i16x16 × i16x16 → i32x8 ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_avx2_basic() {
        // splat(3) × splat(7): each pair = 3*7 = 21.
        // pmaddwd: adjacent i16 pairs → i32: 21+21 = 42 (8 lanes).
        // reduce_add(i32x8) = 8 × 42 = 336. Lane[0] = 42.
        assert_output(
            r#"
            func main() {
                let a: i16x16 = splat(3)
                let b: i16x16 = splat(7)
                let c: i32x8 = madd_i16(a, b)
                let s: i32 = reduce_add(c)
                println(s)
                let x: i32 = c[0]
                println(x)
            }
            "#,
            "336\n42",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_madd_i16_avx2_c_interop() {
        // Explicit maddubs_i16 + madd_i16 chain at AVX2 width.
        // act=splat(1), wt=splat(2), 64 elements in 2 iters of 32.
        // maddubs_i16 on u8x32/i8x32 is not available (it's 128-bit only),
        // so we test madd_i16 at AVX2 width with i16x16 directly.
        assert_c_interop(
            r#"
            export func dot_avx2(src: *i16, n: i32) -> i32 {
                let mut acc: i32x8 = splat(0)
                let scale: i16x16 = splat(3)
                let mut i: i32 = 0
                while i < n {
                    let a: i16x16 = load(src, i)
                    acc = acc .+ madd_i16(a, scale)
                    i = i + 16
                }
                return reduce_add(acc)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t dot_avx2(const int16_t *src, int n);
            int main() {
                int16_t src[32];
                for (int i = 0; i < 32; i++) src[i] = 5;
                printf("%d\n", dot_avx2(src, 32));
                return 0;
            }
            "#,
            "480",
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
    fn test_madd_i16_avx2_ir_check() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func madd_avx2(src: *i16, n: i32) -> i32 {
                let mut acc: i32x8 = splat(0)
                let ones: i16x16 = splat(1)
                let mut i: i32 = 0
                while i < n {
                    let a: i16x16 = load(src, i)
                    acc = acc .+ madd_i16(a, ones)
                    i = i + 16
                }
                return reduce_add(acc)
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("madd_avx2.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("madd_i16 AVX2 IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
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

    // === widen_u8_u16: u8x16 lower 8 → u16x8 zero-extend ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_widen_u8_u16_basic() {
        assert_c_interop(
            r#"
            export func test(out: *mut u16) {
                let v: u8x16 = splat(200)
                let w: u16x8 = widen_u8_u16(v)
                store(out, 0, w)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(uint16_t*);
            int main() {
                uint16_t out[8];
                test(out);
                printf("%u %u\n", out[0], out[7]);
                return 0;
            }
            "#,
            "200 200",
        );
    }

    #[test]
    fn test_widen_u8_u16_arm_compiles() {
        let opts = ea_compiler::CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..ea_compiler::CompileOptions::default()
        };
        let dir = tempfile::TempDir::new().unwrap();
        let obj = dir.path().join("test.o");
        let result = ea_compiler::compile_with_options(
            r#"
            export func convert(inp: *u8, out: *mut u16) {
                let v: u8x16 = load(inp, 0)
                let w: u16x8 = widen_u8_u16(v)
                store(out, 0, w)
            }
            "#,
            &obj,
            ea_compiler::OutputMode::ObjectFile,
            &opts,
        );
        assert!(result.is_ok(), "ARM compile failed: {result:?}");
    }
}
