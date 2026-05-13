//! v1.12.0 monomorphic-rename batch: `sat_add` / `sat_sub` / `abs_diff`.
//!
//! Each polymorphic name gets a typed sibling per supported (elem, width):
//!   sat_add_i8x16, sat_add_u8x16, sat_add_i16x8, sat_add_u16x8        (4)
//!   sat_sub_i8x16, sat_sub_u8x16, sat_sub_i16x8, sat_sub_u16x8        (4)
//!   abs_diff_{i,u}{8x16, 16x8}, abs_diff_{i,u}32x4 (ARM-only)         (6)
//!
//! The old polymorphic names continue to compile but record a deprecation
//! warning via DEPRECATED_INTRINSICS — first real exercise of the
//! deprecation infrastructure added in v1.12.0-dev (PR #6).

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;
    use ea_compiler::typeck::TypeChecker;
    use ea_compiler::{CompileOptions, OutputMode};
    use tempfile::TempDir;

    // --- sat_add: numeric correctness, monomorphic spellings ---

    #[test]
    fn sat_add_i8x16_saturates_positive() {
        // i8 saturates at 127. 100 + 50 = 150 → clamps to 127.
        assert_c_interop(
            r#"
            export func test(a: *i8, b: *i8, out: *mut i8) {
                let va: i8x16 = load(a, 0)
                let vb: i8x16 = load(b, 0)
                let r: i8x16 = sat_add_i8x16(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int8_t*, const int8_t*, int8_t*);
            int main() {
                int8_t a[16] = {100,100,-100,-100,127,127,-128,-128, 0,0,0,0,0,0,0,0};
                int8_t b[16] = { 50, 27, -50, -28,  1,  0,  -1,  0, 0,0,0,0,0,0,0,0};
                int8_t out[16];
                test(a, b, out);
                printf("%d %d %d %d %d %d %d %d\n",
                    out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
                return 0;
            }
            "#,
            // 100+50 → 127 (sat),  100+27 → 127 (sat),
            // -100-50 → -128 (sat), -100-28 → -128 (sat),
            // 127+1 → 127 (sat), 127+0 → 127, -128-1 → -128 (sat), -128+0 → -128
            "127 127 -128 -128 127 127 -128 -128",
        );
    }

    #[test]
    fn sat_add_u8x16_saturates_at_255() {
        assert_c_interop(
            r#"
            export func test(a: *u8, b: *u8, out: *mut u8) {
                let va: u8x16 = load(a, 0)
                let vb: u8x16 = load(b, 0)
                let r: u8x16 = sat_add_u8x16(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, const uint8_t*, uint8_t*);
            int main() {
                uint8_t a[16] = {200,200,255,255, 0,0,0,0,0,0,0,0,0,0,0,0};
                uint8_t b[16] = { 50,100,  1,  0, 0,0,0,0,0,0,0,0,0,0,0,0};
                uint8_t out[16];
                test(a, b, out);
                printf("%u %u %u %u\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "250 255 255 255",
        );
    }

    #[test]
    fn sat_add_i16x8_saturates() {
        assert_c_interop(
            r#"
            export func test(a: *i16, b: *i16, out: *mut i16) {
                let va: i16x8 = load(a, 0)
                let vb: i16x8 = load(b, 0)
                let r: i16x8 = sat_add_i16x8(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int16_t*, const int16_t*, int16_t*);
            int main() {
                int16_t a[8] = {30000, -30000, 32767, -32768, 100, 0, 0, 0};
                int16_t b[8] = {10000, -10000,     1,      0, 200, 0, 0, 0};
                int16_t out[8];
                test(a, b, out);
                printf("%d %d %d %d %d\n",
                    out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
            "#,
            "32767 -32768 32767 -32768 300",
        );
    }

    #[test]
    fn sat_add_u16x8_saturates_at_65535() {
        assert_c_interop(
            r#"
            export func test(a: *u16, b: *u16, out: *mut u16) {
                let va: u16x8 = load(a, 0)
                let vb: u16x8 = load(b, 0)
                let r: u16x8 = sat_add_u16x8(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint16_t*, const uint16_t*, uint16_t*);
            int main() {
                uint16_t a[8] = {60000, 65535, 100, 0, 0,0,0,0};
                uint16_t b[8] = { 6000,     1, 200, 0, 0,0,0,0};
                uint16_t out[8];
                test(a, b, out);
                printf("%u %u %u\n", out[0], out[1], out[2]);
                return 0;
            }
            "#,
            "65535 65535 300",
        );
    }

    // --- sat_sub: numeric correctness ---

    #[test]
    fn sat_sub_i8x16_saturates_negative() {
        assert_c_interop(
            r#"
            export func test(a: *i8, b: *i8, out: *mut i8) {
                let va: i8x16 = load(a, 0)
                let vb: i8x16 = load(b, 0)
                let r: i8x16 = sat_sub_i8x16(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int8_t*, const int8_t*, int8_t*);
            int main() {
                int8_t a[16] = {-100, 100, -128, 127, 10, 0,0,0,0,0,0,0,0,0,0,0};
                int8_t b[16] = {  50, -50,    1,  -1,  3, 0,0,0,0,0,0,0,0,0,0,0};
                int8_t out[16];
                test(a, b, out);
                printf("%d %d %d %d %d\n",
                    out[0], out[1], out[2], out[3], out[4]);
                return 0;
            }
            "#,
            "-128 127 -128 127 7",
        );
    }

    #[test]
    fn sat_sub_u8x16_saturates_at_zero() {
        assert_c_interop(
            r#"
            export func test(a: *u8, b: *u8, out: *mut u8) {
                let va: u8x16 = load(a, 0)
                let vb: u8x16 = load(b, 0)
                let r: u8x16 = sat_sub_u8x16(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint8_t*, const uint8_t*, uint8_t*);
            int main() {
                uint8_t a[16] = {10, 5, 255, 0, 0,0,0,0,0,0,0,0,0,0,0,0};
                uint8_t b[16] = {20, 5,   1, 0, 0,0,0,0,0,0,0,0,0,0,0,0};
                uint8_t out[16];
                test(a, b, out);
                printf("%u %u %u %u\n", out[0], out[1], out[2], out[3]);
                return 0;
            }
            "#,
            "0 0 254 0",
        );
    }

    #[test]
    fn sat_sub_i16x8_works() {
        assert_c_interop(
            r#"
            export func test(a: *i16, b: *i16, out: *mut i16) {
                let va: i16x8 = load(a, 0)
                let vb: i16x8 = load(b, 0)
                let r: i16x8 = sat_sub_i16x8(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const int16_t*, const int16_t*, int16_t*);
            int main() {
                int16_t a[8] = {-30000, 30000, 100, 0,0,0,0,0};
                int16_t b[8] = { 10000, -10000,  20, 0,0,0,0,0};
                int16_t out[8];
                test(a, b, out);
                printf("%d %d %d\n", out[0], out[1], out[2]);
                return 0;
            }
            "#,
            "-32768 32767 80",
        );
    }

    #[test]
    fn sat_sub_u16x8_works() {
        assert_c_interop(
            r#"
            export func test(a: *u16, b: *u16, out: *mut u16) {
                let va: u16x8 = load(a, 0)
                let vb: u16x8 = load(b, 0)
                let r: u16x8 = sat_sub_u16x8(va, vb)
                store(out, 0, r)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void test(const uint16_t*, const uint16_t*, uint16_t*);
            int main() {
                uint16_t a[8] = {  100, 65535, 0, 0,0,0,0,0};
                uint16_t b[8] = {  200,     1, 5, 0,0,0,0,0};
                uint16_t out[8];
                test(a, b, out);
                printf("%u %u %u\n", out[0], out[1], out[2]);
                return 0;
            }
            "#,
            "0 65534 0",
        );
    }

    // --- abs_diff: ARM-only — compile-time check only ---
    // (Numeric correctness is already covered by tests/phase14_arm_neon.rs.)

    fn arm_opts() -> CompileOptions {
        CompileOptions {
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
            ..CompileOptions::default()
        }
    }

    fn try_compile_arm(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile_with_options(source, &obj_path, OutputMode::ObjectFile, &arm_opts())
    }

    fn try_compile_x86(source: &str) -> Result<(), ea_compiler::error::CompileError> {
        let dir = TempDir::new().unwrap();
        let obj_path = dir.path().join("test.o");
        ea_compiler::compile(source, &obj_path, OutputMode::ObjectFile)
    }

    #[test]
    fn abs_diff_i8x16_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: i8x16, b: i8x16) -> i8x16 { return abs_diff_i8x16(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    fn abs_diff_u8x16_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff_u8x16(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    fn abs_diff_i16x8_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: i16x8, b: i16x8) -> i16x8 { return abs_diff_i16x8(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    fn abs_diff_u16x8_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: u16x8, b: u16x8) -> u16x8 { return abs_diff_u16x8(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    fn abs_diff_i32x4_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: i32x4, b: i32x4) -> i32x4 { return abs_diff_i32x4(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    fn abs_diff_u32x4_compiles_on_arm() {
        try_compile_arm(
            "export func f(a: u32x4, b: u32x4) -> u32x4 { return abs_diff_u32x4(a, b) }\n",
        )
        .expect("must compile on ARM");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn abs_diff_typed_still_arm_only_on_x86() {
        // try_compile_x86 uses the host default target; on the ubuntu-24.04-arm
        // CI runner the "x86" name is a lie. Gate to actual x86 hosts.
        let err = try_compile_x86(
            "export func f(a: i8x16, b: i8x16) -> i8x16 { return abs_diff_i8x16(a, b) }\n",
        )
        .expect_err("abs_diff_i8x16 must fail on x86 (ARM-only op)");
        assert!(
            format!("{err}").contains("ARM-only"),
            "expected ARM-only error, got: {err}"
        );
    }

    // --- Type-mismatch rejections ---

    #[test]
    fn sat_add_i8x16_rejects_u8x16() {
        let err = try_compile_x86(
            "export func f(a: u8x16, b: u8x16) -> u8x16 { return sat_add_i8x16(a, b) }\n",
        )
        .expect_err("u8x16 must fail sat_add_i8x16 type check");
        assert!(format!("{err}").contains("sat_add_i8x16 expects i8x16"));
    }

    #[test]
    fn sat_sub_u16x8_rejects_i16x8() {
        let err = try_compile_x86(
            "export func f(a: i16x8, b: i16x8) -> i16x8 { return sat_sub_u16x8(a, b) }\n",
        )
        .expect_err("i16x8 must fail sat_sub_u16x8 type check");
        assert!(format!("{err}").contains("sat_sub_u16x8 expects u16x8"));
    }

    #[test]
    fn abs_diff_u32x4_rejects_i32x4() {
        let err = try_compile_arm(
            "export func f(a: i32x4, b: i32x4) -> i32x4 { return abs_diff_u32x4(a, b) }\n",
        )
        .expect_err("i32x4 must fail abs_diff_u32x4 type check");
        assert!(format!("{err}").contains("abs_diff_u32x4 expects u32x4"));
    }

    // --- Deprecation: old polymorphic name still compiles, emits warning ---

    fn warnings_for(source: &str) -> Vec<ea_compiler::DeprecationWarning> {
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let mut tc = TypeChecker::new();
        tc.check_program(&stmts).unwrap();
        tc.warnings()
    }

    #[test]
    fn deprecated_sat_add_still_compiles_and_warns() {
        // The polymorphic name remains functional.
        try_compile_x86("export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }\n")
            .expect("polymorphic sat_add must still compile");

        let warnings =
            warnings_for("export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add(a, b) }\n");
        assert_eq!(warnings.len(), 1);
        let w = &warnings[0];
        assert_eq!(w.name, "sat_add");
        assert_eq!(w.since, "1.12.0");
        assert!(w.advice.contains("sat_add_i8x16"));
    }

    #[test]
    fn deprecated_sat_sub_still_compiles_and_warns() {
        try_compile_x86("export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_sub(a, b) }\n")
            .expect("polymorphic sat_sub must still compile");

        let warnings =
            warnings_for("export func f(a: u16x8, b: u16x8) -> u16x8 { return sat_sub(a, b) }\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].name, "sat_sub");
        assert!(warnings[0].advice.contains("sat_sub_u16x8"));
    }

    #[test]
    fn deprecated_abs_diff_still_compiles_and_warns_on_arm() {
        try_compile_arm("export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff(a, b) }\n")
            .expect("polymorphic abs_diff must still compile on ARM");

        let warnings =
            warnings_for("export func f(a: u8x16, b: u8x16) -> u8x16 { return abs_diff(a, b) }\n");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].name, "abs_diff");
        assert!(warnings[0].advice.contains("abs_diff_u8x16"));
    }

    #[test]
    fn typed_spelling_does_not_warn() {
        // Sanity: calling the new typed name must record no warning.
        let warnings = warnings_for(
            "export func f(a: i8x16, b: i8x16) -> i8x16 { return sat_add_i8x16(a, b) }\n",
        );
        assert!(
            warnings.is_empty(),
            "typed spelling should not warn, got {warnings:?}"
        );
    }
}
