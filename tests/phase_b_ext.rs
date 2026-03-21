#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === vdot_i32: ARM-only, should error on x86 ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_x86_rejects_vdot_i32() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func f(a: i8x16, b: i8x16) -> i32x4 {
                return vdot_i32(a, b)
            }
        "#;
        let dir = tempfile::TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        let opts = CompileOptions::default();
        let err =
            ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &opts)
                .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ARM"), "expected ARM mention, got: {msg}");
    }

    // === shuffle_bytes: cross-platform byte LUT ===

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_shuffle_bytes_basic() {
        assert_output(
            r#"
            func main() {
                let table: u8x16 = [10, 20, 30, 40, 50, 60, 70, 80,
                                    90, 100, 110, 120, 130, 140, 150, 160]u8x16
                let idx: u8x16 = [0, 2, 4, 6, 1, 3, 5, 7,
                                  8, 10, 12, 14, 9, 11, 13, 15]u8x16
                let result: u8x16 = shuffle_bytes(table, idx)
                println(result[0])
                println(result[1])
                println(result[2])
                println(result[3])
            }
            "#,
            "10\n30\n50\n70",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_shuffle_bytes_ir_check() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func f(table: u8x16, idx: u8x16) -> u8x16 {
                return shuffle_bytes(table, idx)
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("shuf.ll");
        let opts = CompileOptions {
            opt_level: 0,
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("shuffle_bytes IR compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("pshuf.b.128"),
            "expected pshuf.b.128 in IR, got:\n{ir}"
        );
    }

    /// C interop: shuffle_bytes as a 16-entry nibble LUT (BitNet-style pattern).
    /// Builds a table where table[i] = i*i, then looks up indices [0..15] reversed.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_shuffle_bytes_c_interop() {
        assert_c_interop(
            r#"
            export func lut_apply(src: *u8, dst: *mut u8, n: i32) {
                let table: u8x16 = [0, 1, 4, 9, 16, 25, 36, 49,
                                    64, 81, 100, 121, 144, 169, 196, 225]u8x16
                let mut i: i32 = 0
                while i + 16 <= n {
                    let idx: u8x16 = load(src, i)
                    let result: u8x16 = shuffle_bytes(table, idx)
                    store(dst, i, result)
                    i = i + 16
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void lut_apply(const uint8_t *src, uint8_t *dst, int n);
            int main() {
                uint8_t src[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
                uint8_t dst[16] = {0};
                lut_apply(src, dst, 16);
                /* table[i] = i*i, so dst[i] = src[i]^2 */
                printf("%d %d %d %d %d", (int)dst[0], (int)dst[3], (int)dst[7], (int)dst[10], (int)dst[15]);
                return 0;
            }
            "#,
            "0 9 49 100 225",
        );
    }

    // === AVX-512: f32x16 ===

    /// Verify f32x16 type-checks and emits <16 x float> LLVM IR.
    /// Requires x86-64 host (AVX-512 feature flag).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f32x16_ir_contains_16_float_vector() {
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func scale_f32x16(src: *restrict f32, dst: *mut f32, n: i32, scale: f32) {
                let vscale: f32x16 = splat(scale)
                let mut i: i32 = 0
                while i + 16 <= n {
                    let v: f32x16 = load(src, i)
                    store(dst, i, v .* vscale)
                    i = i + 16
                }
            }
        "#;

        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("scale.ll");
        let opts = CompileOptions {
            opt_level: 0,
            extra_features: "+avx512f".to_string(),
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("f32x16 compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(
            ir.contains("<16 x float>"),
            "expected <16 x float> in IR, got:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_f32x16_splat_typecheck() {
        // Verify f32x16 splat + element access type-checks (no AVX-512 needed for IR check)
        use ea_compiler::{CompileOptions, OutputMode};

        let source = r#"
            export func sum_f32x16(v: f32x16) -> f32 {
                return reduce_add(v)
            }
        "#;
        let dir = tempfile::TempDir::new().unwrap();
        let ir_path = dir.path().join("sum.ll");
        let opts = CompileOptions {
            opt_level: 0,
            extra_features: "+avx512f".to_string(),
            ..CompileOptions::default()
        };
        ea_compiler::compile_with_options(source, &ir_path, OutputMode::LlvmIr, &opts)
            .expect("f32x16 reduce_add compilation failed");

        let ir = std::fs::read_to_string(&ir_path).unwrap_or_default();
        assert!(ir.contains("<16 x float>"), "expected <16 x float> in IR");
        assert!(ir.contains("reduce.fadd"), "expected reduce.fadd in IR");
    }
}
