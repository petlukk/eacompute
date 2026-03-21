#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cfg_x86_included() {
        assert_output(
            r#"
            #[cfg(x86_64)]
            export func main() {
                println(42)
            }
            "#,
            "42",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_cfg_aarch64_included() {
        assert_output(
            r#"
            #[cfg(aarch64)]
            export func main() {
                println(42)
            }
            "#,
            "42",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cfg_aarch64_excluded_on_x86() {
        let ir = compile_to_ir(
            r#"
            #[cfg(aarch64)]
            export func arm_only(out: *mut i32) {
                out[0] = 1
            }

            #[cfg(x86_64)]
            export func x86_only(out: *mut i32) {
                out[0] = 2
            }
            "#,
        );
        assert!(
            !ir.contains("arm_only"),
            "arm_only should be excluded on x86"
        );
        assert!(
            ir.contains("x86_only"),
            "x86_only should be included on x86"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cfg_both_platforms_same_name() {
        assert_c_interop(
            r#"
            #[cfg(x86_64)]
            export func platform_func(out: *mut i32) {
                out[0] = 86
            }

            #[cfg(aarch64)]
            export func platform_func(out: *mut i32) {
                out[0] = 64
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void platform_func(int32_t*);
            int main() {
                int32_t out;
                platform_func(&out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "86",
        );
    }

    #[test]
    fn test_cfg_rejects_invalid_target() {
        let source = r#"
            #[cfg(arm64)]
            export func f() -> i32 {
                return 1
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let err = ea_compiler::parse(tokens).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown cfg target") && msg.contains("arm64"),
            "expected 'unknown cfg target' error for 'arm64', got: {msg}"
        );
    }

    #[test]
    fn test_cfg_no_attribute_always_included() {
        assert_output(
            r#"
            export func main() {
                println(99)
            }
            "#,
            "99",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_cfg_mixed_attributed_and_plain() {
        assert_c_interop(
            r#"
            func helper() -> i32 {
                return 10
            }

            #[cfg(x86_64)]
            export func compute(out: *mut i32) {
                out[0] = helper()
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void compute(int32_t*);
            int main() {
                int32_t out;
                compute(&out);
                printf("%d\n", out);
                return 0;
            }
            "#,
            "10",
        );
    }
}
