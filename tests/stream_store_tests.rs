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

    // --- stream_store: scalar value support (typeck acceptance) ---

    #[test]
    fn test_stream_store_scalar_i32_typecheck() {
        // Scalar i32 stream_store must typecheck (extension over vector-only).
        // typeck-only path is sufficient; runtime coverage is in test_stream_store_scalar_i32_runtime.
        let source = r#"
            export func test(out: *mut i32, v: i32) {
                stream_store(out, 0, v)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let result = ea_compiler::check_types(&stmts);
        assert!(
            result.is_ok(),
            "scalar i32 stream_store should typecheck, got: {result:?}"
        );
    }

    #[test]
    fn test_stream_store_scalar_i64_typecheck() {
        let source = r#"
            export func test(out: *mut i64, v: i64) {
                stream_store(out, 0, v)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let result = ea_compiler::check_types(&stmts);
        assert!(
            result.is_ok(),
            "scalar i64 stream_store should typecheck, got: {result:?}"
        );
    }

    // --- stream_store: scalar value support (codegen runtime) ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_scalar_i32_runtime() {
        assert_c_interop(
            r#"
            export func test(out: *mut i32, v: i32) {
                stream_store(out, 0, v)
                stream_store(out, 1, v + 100)
            }
        "#,
            r#"#include <stdio.h>
            extern void test(int*, int);
            int main() {
                __attribute__((aligned(8))) int out[2] = {0, 0};
                test(out, 42);
                printf("%d %d\n", out[0], out[1]);
                return 0;
            }"#,
            "42 142",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_scalar_i64_runtime() {
        assert_c_interop(
            r#"
            export func test(out: *mut i64, v: i64) {
                stream_store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void test(int64_t*, int64_t);
            int main() {
                __attribute__((aligned(8))) int64_t out[1] = {0};
                test(out, 0xDEADBEEFCAFEBABELL);
                printf("%llx\n", (unsigned long long)out[0]);
                return 0;
            }"#,
            "deadbeefcafebabe",
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_stream_store_scalar_i64_runtime_aarch64() {
        assert_c_interop(
            r#"
            export func test(out: *mut i64, v: i64) {
                stream_store(out, 0, v)
            }
        "#,
            r#"#include <stdio.h>
            #include <stdint.h>
            extern void test(int64_t*, int64_t);
            int main() {
                __attribute__((aligned(16))) int64_t out[1] = {0};
                test(out, 0x0123456789ABCDEFLL);
                printf("%llx\n", (unsigned long long)out[0]);
                return 0;
            }"#,
            "123456789abcdef",
        );
    }

    // --- x86 objdump assertions: scalar and vector stream_store mnemonics ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_scalar_i32_emits_movnti() {
        assert_intrinsic_in_disassembly(
            r#"
            export func test(out: *mut i32, v: i32) {
                stream_store(out, 0, v)
            }
        "#,
            &["movnti"],
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_scalar_i64_emits_movnti() {
        assert_intrinsic_in_disassembly(
            r#"
            export func test(out: *mut i64, v: i64) {
                stream_store(out, 0, v)
            }
        "#,
            &["movnti"],
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_f32x4_emits_movntps() {
        assert_intrinsic_in_disassembly(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                stream_store(out, 0, v)
            }
        "#,
            &["movntps", "movntdq"],
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_f32x8_emits_vmovntps() {
        assert_intrinsic_in_disassembly(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x8 = load(data, 0)
                stream_store(out, 0, v)
            }
        "#,
            &["vmovntps", "vmovntdq"],
        );
    }

    // --- x86 alignment-failure crash test ---

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_stream_store_misaligned_vector_crashes() {
        // Deliberately misaligned (byte-offset 1 into an aligned array)
        // pointer passed to stream_store of a 256-bit vector should
        // raise SIGSEGV on x86 (vmovntps requires 32-byte alignment).
        //
        // This test documents the alignment contract by demonstrating
        // the consequence of violating it. Uses a child-process pattern
        // so the parent test runner doesn't crash.
        use std::process::{Command, Stdio};

        let dir = tempfile::TempDir::new().unwrap();
        let obj_path = dir.path().join("kernel.o");
        let c_path = dir.path().join("harness.c");
        let bin_path = dir.path().join("test_bin");

        ea_compiler::compile(
            r#"
            export func test(out: *mut f32) {
                let v: f32x8 = splat(1.0)
                stream_store(out, 0, v)
            }
        "#,
            &obj_path,
            ea_compiler::OutputMode::ObjectFile,
        )
        .expect("compile");

        std::fs::write(
            &c_path,
            r#"
            #include <stdint.h>
            extern void test(float*);
            int main() {
                // 32-byte alignment + 1-byte offset = guaranteed misalignment for f32x8
                __attribute__((aligned(32))) static char buf[64];
                float* misaligned = (float*)(buf + 1);
                test(misaligned);
                return 0;
            }
            "#,
        )
        .expect("write C harness");

        Command::new("cc")
            .arg(&c_path)
            .arg(&obj_path)
            .arg("-o")
            .arg(&bin_path)
            .status()
            .expect("link");

        let status = Command::new(&bin_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .expect("run binary");

        // Process should NOT exit cleanly — expect signal-termination.
        assert!(
            !status.success(),
            "misaligned vector stream_store unexpectedly succeeded — \
             alignment contract violated without consequence. Check that \
             codegen still attaches !nontemporal metadata and that LLVM \
             is not silently substituting an aligned movups."
        );
    }

    // --- aarch64 objdump assertions (Pi 5-verified mnemonics, LLVM 18.1.8) ---
    //
    // aarch64 has no scalar non-temporal store instruction. LLVM honors
    // !nontemporal only when it can synthesize stnp (Store Non-temporal Pair) —
    // that is, for 64-bit operands (self-pair to w-pair) and 128-bit vectors
    // (q-register splits to d-pair). For 32-bit and 16-bit scalars, LLVM
    // silently emits plain `str` / `strh` with the NT hint dropped. The four
    // tests below pin the actually-observed LLVM 18 behavior; a future LLVM
    // upgrade that changes any of these emissions will fail loudly rather than
    // silently.

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_stream_store_scalar_i64_emits_stnp_aarch64() {
        // i64 self-pairs to a w-register pair via lsr; emits `stnp w, w`.
        assert_intrinsic_in_disassembly(
            r#"
            export func test(out: *mut i64, v: i64) {
                stream_store(out, 0, v)
            }
        "#,
            &["stnp"],
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_stream_store_f32x4_emits_stnp_aarch64() {
        // 128-bit q-register splits to a d-pair; emits `stnp d, d`.
        assert_intrinsic_in_disassembly(
            r#"
            export func test(data: *f32, out: *mut f32) {
                let v: f32x4 = load(data, 0)
                stream_store(out, 0, v)
            }
        "#,
            &["stnp"],
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_stream_store_scalar_i32_emits_plain_str_aarch64() {
        // PIN: NT hint silently dropped on aarch64 for 32-bit scalar stores.
        // If LLVM ever starts synthesizing stnp for i32 stream_store, this
        // test fails — forcing the docs to be updated to reflect new behavior.
        assert_intrinsic_in_disassembly(
            r#"
            export func test(out: *mut i32, v: i32) {
                stream_store(out, 0, v)
            }
        "#,
            &["str\tw"],
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_stream_store_scalar_i16_emits_plain_strh_aarch64() {
        // PIN: same as i32 — NT hint dropped, plain `strh w, [x]` emitted.
        assert_intrinsic_in_disassembly(
            r#"
            export func test(out: *mut i16, v: i16) {
                stream_store(out, 0, v)
            }
        "#,
            &["strh"],
        );
    }
}
