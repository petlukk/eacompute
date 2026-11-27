//! Float conversions for f32 kernels (v1.21.0): floor_f32x4/x8 (x86
//! vroundps, ARM frintm) and bitcast_f32x4/x8 (no instruction). Integer to
//! float is the existing to_f32 on i32xN (vcvtdq2ps, ARM scvtf), pinned
//! here because Tiro's colour kernel relies on it.
//!
//! Tests that compile for the x86 triple are gated to x86 hosts.

#[cfg(feature = "llvm")]
#[allow(dead_code)]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    fn asm(src: &str, triple: &str, cpu: &str) -> String {
        let dir = TempDir::new().unwrap();
        let out = dir.path().join("t.s");
        let opts = CompileOptions { target_triple: Some(triple.to_string()), target_cpu: Some(cpu.to_string()), ..CompileOptions::default() };
        ea_compiler::compile_with_options(src, &out, OutputMode::Asm, &opts).expect("compile to asm");
        std::fs::read_to_string(out).unwrap()
    }

    const FLOOR8: &str = "export func f(p: *f32, out: *mut f32) {\n    let v: f32x8 = load(p, 0)\n    store(out, 0, floor_f32x8(v))\n}\n";
    const FLOOR4: &str = "export func f(p: *f32, out: *mut f32) {\n    let v: f32x4 = load(p, 0)\n    store(out, 0, floor_f32x4(v))\n}\n";
    const CVT8: &str = "export func f(p: *i32, out: *mut f32) {\n    let v: i32x8 = load(p, 0)\n    store(out, 0, to_f32(v))\n}\n";
    const CVT4: &str = "export func f(p: *i32, out: *mut f32) {\n    let v: i32x4 = load(p, 0)\n    store(out, 0, to_f32(v))\n}\n";
    const BITS8: &str = "export func f(p: *i32, out: *mut f32) {\n    let v: i32x8 = load(p, 0)\n    store(out, 0, bitcast_f32x8(v))\n}\n";

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn floor_negative_and_halves() {
        assert_c_interop(
            FLOOR8,
            r#"
            #include <stdio.h>
            void f(const float *p, float *out);
            int main(void) {
                float p[8] = {-1.5f, -0.0f, 2.999f, -2.0f, 1e30f, -1e-30f, 0.5f, 3.0f};
                float o[8];
                f(p, o);
                for (int k = 0; k < 8; k++) printf("%g ", o[k]);
                printf("\n");
                return 0;
            }
            "#,
            "-2 -0 2 -2 1e+30 -1 0 3",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cvt_i32_f32_round_to_nearest() {
        assert_c_interop(
            CVT8,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const int32_t *p, float *out);
            int main(void) {
                int32_t p[8] = {16777217, -2147483648, 0, -1, 7, 16777219, 3, -16777217};
                float o[8];
                f(p, o);
                for (int k = 0; k < 8; k++) printf("%.1f ", o[k]);
                printf("\n");
                return 0;
            }
            "#,
            "16777216.0 -2147483648.0 0.0 -1.0 7.0 16777220.0 3.0 -16777216.0",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn bitcast_f32_roundtrip() {
        assert_c_interop(
            BITS8,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            void f(const int32_t *p, float *out);
            int main(void) {
                int32_t p[8] = {0x3f800000, 0x40490fdb, 0, (int32_t)0x80000000, 0x7f800000, 1, -1, 0x3f000000};
                float o[8];
                f(p, o);
                printf("%g %.5f %g %g %g %d\n", o[0], o[1], o[2], o[3], o[4], o[7] == 0.5f);
                return 0;
            }
            "#,
            "1 3.14159 0 -0 inf 1",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_instructions() {
        let s = asm(FLOOR8, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!(s.contains("vroundps") && !s.contains("call"), "{s}");
        let s = asm(CVT8, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!(s.contains("vcvtdq2ps"), "{s}");
        let s = asm(BITS8, "x86_64-unknown-linux-gnu", "x86-64-v3");
        assert!(!s.contains("cvt") && !s.contains("call"), "{s}");
    }

    #[test]
    fn arm_instructions() {
        let s = asm(FLOOR4, "aarch64-unknown-linux-gnu", "generic");
        assert!(s.contains("frintm") && !s.contains("\tbl\t"), "{s}");
        let s = asm(CVT4, "aarch64-unknown-linux-gnu", "generic");
        assert!(s.contains("scvtf"), "{s}");
    }

    #[test]
    fn floor_checks_its_type() {
        assert_typecheck_error("export func f(a: f32x4) -> f32x8 { return floor_f32x8(a) }", "f32x8");
        assert_typecheck_error("export func f(a: i32x4) -> f32x8 { return bitcast_f32x8(a) }", "bit widths");
    }
}
