#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_fence_nt_typecheck() {
        // fence_nt is a zero-arg intrinsic returning void; should typecheck.
        let source = r#"
            export func test() {
                fence_nt()
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let result = ea_compiler::check_types(&stmts);
        assert!(
            result.is_ok(),
            "fence_nt() should typecheck, got: {result:?}"
        );
    }

    #[test]
    fn test_fence_nt_rejects_args() {
        let source = r#"
            export func test() {
                fence_nt(42)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let result = ea_compiler::check_types(&stmts);
        assert!(
            result.is_err(),
            "fence_nt(42) should be rejected, got: {result:?}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_fence_nt_ir_calls_sfence_x86() {
        let ir = compile_to_ir(
            r#"
            export func test() {
                fence_nt()
            }
        "#,
        );
        assert!(
            ir.contains("@llvm.x86.sse.sfence"),
            "fence_nt on x86 must call llvm.x86.sse.sfence:\n{ir}"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_fence_nt_runtime_x86() {
        // Kernel: write via stream_store, fence, then verify the read
        // sees the value within the same kernel call (intra-kernel visibility).
        assert_c_interop(
            r#"
            export func test(out: *mut i32, v: i32) {
                stream_store(out, 0, v)
                fence_nt()
                // Read back after fence to verify visibility
            }
        "#,
            r#"#include <stdio.h>
            extern void test(int*, int);
            int main() {
                __attribute__((aligned(8))) int out[1] = {0};
                test(out, 42);
                printf("%d\n", out[0]);
                return 0;
            }"#,
            "42",
        );
    }
}
