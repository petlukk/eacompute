#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // --- stream_store: non-temporal vector store ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_f32x8() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x8 = load(data, 0)
                stream_store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                __attribute__((aligned(32))) float data[8];
                __attribute__((aligned(32))) float out[8];
                for (int i = 0; i < 8; i++) { data[i] = (float)(i+1); out[i] = 0; }
                test(data, out);
                for (int i = 0; i < 8; i++) printf("%g ", out[i]);
                printf("\n");
                return 0;
            }"#,
            "1 2 3 4 5 6 7 8",
        );
    }

    #[test]
    fn test_stream_store_f32x4() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                stream_store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {10,20,30,40};
                float out[4] = {0};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "10 20 30 40",
        );
    }

    #[test]
    fn test_stream_store_with_offset() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                stream_store(out, 4, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {1,2,3,4};
                float out[8] = {0};
                test(data, out);
                printf("%g %g %g %g %g %g %g %g\n",
                       out[0],out[1],out[2],out[3],
                       out[4],out[5],out[6],out[7]);
                return 0;
            }"#,
            "0 0 0 0 1 2 3 4",
        );
    }

    #[test]
    fn test_stream_store_ir_nontemporal() {
        let ir = compile_to_ir(
            r#"
            export func test(out: *mut f32) {
                let v: f32x4 = splat(1.0)
                stream_store(out, 0, v)
            }
        "#,
        );
        assert!(
            ir.contains("!nontemporal"),
            "stream_store IR must contain !nontemporal metadata:\n{ir}"
        );
    }

    #[test]
    fn test_stream_store_rejects_immutable_pointer() {
        let source = r#"
            export func test(data: *f32) {
                let v: f32x4 = load(data, 0)
                stream_store(data, 0, v)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let result = ea_compiler::check_types(&stmts);
        assert!(
            result.is_err(),
            "stream_store to immutable pointer should fail"
        );
    }

    // --- Offset test (load_masked with non-zero offset) ---

    #[test]
    fn test_load_masked_with_offset() {
        assert_c_interop(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load_masked(data, 4, 2)
                store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(const float*, float*);
            int main() {
                float data[] = {1,2,3,4,50,60,70,80};
                float out[4] = {-1,-1,-1,-1};
                test(data, out);
                printf("%g %g %g %g\n", out[0], out[1], out[2], out[3]);
                return 0;
            }"#,
            "50 60 0 0",
        );
    }
}
