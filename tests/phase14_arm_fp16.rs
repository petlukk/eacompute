// Lexer/parser recognition tests for f16, f16x4, f16x8 tokens (B4).
// B5: codegen gate — f16 vector types require --fp16 flag.

#[cfg(feature = "llvm")]
mod tests {
    use ea_compiler::lexer::TokenKind;
    use tempfile::TempDir;

    #[test]
    fn test_f16_without_flag_is_rejected() {
        use ea_compiler::{CompileOptions, OutputMode};
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(), // no --fp16
            target_triple: None,
        };
        // Use a pointer parameter to avoid inline cast syntax.
        let src = r#"
            export func k(p: *f16, out: *mut f16) {
                let v: f16x8 = load(p, 0)
                store(out, 0, v)
            }
        "#;
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("f16 without --fp16 should fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("--fp16"),
            "error must mention --fp16, got: {msg}"
        );
    }

    #[test]
    fn test_f16x8_param_without_flag_is_rejected() {
        use ea_compiler::{CompileOptions, OutputMode};
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: String::new(),
            target_triple: None,
        };
        // f16x8 parameter — used to panic in declare_function before this fix
        let src = r#"
            export func k(v: f16x8) { }
        "#;
        let err = ea_compiler::compile_with_options(src, &obj, OutputMode::ObjectFile, &opts)
            .expect_err("f16x8 parameter without --fp16 should fail with error, not panic");
        let msg = format!("{err}");
        assert!(
            msg.contains("--fp16"),
            "error must mention --fp16, got: {msg}"
        );
    }

    #[test]
    fn test_f16_lexer_recognizes_token() {
        let src = "let x: f16 = 0";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        let has_f16 = tokens.iter().any(|t| t.kind == TokenKind::F16);
        assert!(
            has_f16,
            "F16 token missing, got: {:?}",
            tokens
                .iter()
                .map(|t| format!("{:?}", t.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_f16x4_lexer_recognizes_token() {
        let src = "let v: f16x4 = 0";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        let has = tokens.iter().any(|t| t.kind == TokenKind::F16x4);
        assert!(
            has,
            "F16x4 token missing, got: {:?}",
            tokens
                .iter()
                .map(|t| format!("{:?}", t.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_f16x8_lexer_recognizes_token() {
        let src = "let v: f16x8 = 0";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        let has = tokens.iter().any(|t| t.kind == TokenKind::F16x8);
        assert!(
            has,
            "F16x8 token missing, got: {:?}",
            tokens
                .iter()
                .map(|t| format!("{:?}", t.kind))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_f16_parses_as_type() {
        // The parser should accept f16 as a named type in a function signature.
        // We stop at parse (no type-check) to avoid B5+ concerns.
        let src = "func foo(x: f16) {}";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        ea_compiler::parse(tokens).expect("parse failed — f16 not recognized as type");
    }

    #[test]
    fn test_f16x4_parses_as_type() {
        let src = "func foo(x: f16x4) {}";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        ea_compiler::parse(tokens).expect("parse failed — f16x4 not recognized as type");
    }

    #[test]
    fn test_f16x8_parses_as_type() {
        let src = "func foo(x: f16x8) {}";
        let tokens = ea_compiler::tokenize(src).expect("tokenize failed");
        ea_compiler::parse(tokens).expect("parse failed — f16x8 not recognized as type");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_load_store() {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::process::Command;

        let ea = r#"
            export func copy(input: *f16, output: *mut f16) {
                let v: f16x8 = load(input, 0)
                store(output, 0, v)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void copy(const _Float16 *in, _Float16 *out);
            int main(void) {
                _Float16 in_buf[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
                _Float16 out_buf[8] = {0};
                copy(in_buf, out_buf);
                for (int i = 0; i < 8; ++i) printf("%g\n", (double)out_buf[i]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
                "-march=armv8-a+fp16",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), "1\n2\n3\n4\n5\n6\n7\n8");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_elementwise_arithmetic() {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::process::Command;

        let ea = r#"
        export func add_mul(a: *f16, b: *f16, out: *mut f16) {
            let va: f16x8 = load(a, 0)
            let vb: f16x8 = load(b, 0)
            let vsum: f16x8 = va .+ vb
            let vprod: f16x8 = va .* vb
            store(out, 0, vsum)
            store(out, 8, vprod)
        }
    "#;
        let c = r#"
        #include <stdio.h>
        extern void add_mul(const _Float16 *a, const _Float16 *b, _Float16 *out);
        int main(void) {
            _Float16 a[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            _Float16 b[8] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
            _Float16 out[16] = {0};
            add_mul(a, b, out);
            for (int i = 0; i < 16; ++i) printf("%g\n", (double)out[i]);
            return 0;
        }
    "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
                "-march=armv8-a+fp16",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        let lines: Vec<&str> = stdout.trim().split('\n').collect();
        // Sums: 11 22 33 44 55 66 77 88
        // Products: 10 40 90 160 250 360 490 640
        assert_eq!(
            &lines[0..8],
            &["11", "22", "33", "44", "55", "66", "77", "88"]
        );
        assert_eq!(
            &lines[8..16],
            &["10", "40", "90", "160", "250", "360", "490", "640"]
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_fma() {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::process::Command;

        let ea = r#"
            export func fma_test(a: *f16, b: *f16, c: *f16, out: *mut f16) {
                let va: f16x8 = load(a, 0)
                let vb: f16x8 = load(b, 0)
                let vc: f16x8 = load(c, 0)
                let r: f16x8 = fma(va, vb, vc)
                store(out, 0, r)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void fma_test(const _Float16 *a, const _Float16 *b,
                                 const _Float16 *c, _Float16 *out);
            int main(void) {
                _Float16 a[8] = {2,2,2,2,2,2,2,2};
                _Float16 b[8] = {3,3,3,3,3,3,3,3};
                _Float16 c[8] = {1,1,1,1,1,1,1,1};
                _Float16 out[8] = {0};
                fma_test(a, b, c, out);
                for (int i = 0; i < 8; ++i) printf("%g\n", (double)out[i]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
                "-march=armv8-a+fp16",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        // 2*3 + 1 = 7 per lane
        assert_eq!(stdout.trim(), "7\n7\n7\n7\n7\n7\n7\n7");
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_reductions() {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::process::Command;

        let ea = r#"
            export func reductions(a: *f16, out: *mut f16) {
                let v: f16x8 = load(a, 0)
                let s: f16 = reduce_add(v)
                let mx: f16 = reduce_max(v)
                let mn: f16 = reduce_min(v)
                out[0] = s
                out[1] = mx
                out[2] = mn
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void reductions(const _Float16 *a, _Float16 *out);
            int main(void) {
                _Float16 a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
                _Float16 out[3] = {0};
                reductions(a, out);
                printf("%g\n%g\n%g\n", (double)out[0], (double)out[1], (double)out[2]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
                "-march=armv8-a+fp16",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        // sum=36, max=8, min=1
        assert_eq!(stdout.trim(), "36\n8\n1");
    }

    // --- IR-level regression guards (run on any host, no execution required) ---

    #[test]
    fn test_f16_ir_emits_native_half() {
        use ea_compiler::{CompileOptions, OutputMode};
        let src = r#"
            export func add(a: *f16, b: *f16, out: *mut f16) {
                let va: f16x8 = load(a, 0)
                let vb: f16x8 = load(b, 0)
                let r: f16x8 = va .+ vb
                store(out, 0, r)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap();
        assert!(
            ir.contains("<8 x half>"),
            "expected <8 x half> in IR, got:\n{ir}"
        );
        assert!(ir.contains("fadd"), "expected fadd in IR");
    }

    #[test]
    fn test_f16_fma_uses_llvm_intrinsic() {
        use ea_compiler::{CompileOptions, OutputMode};
        let src = r#"
            export func fma_k(a: *f16, b: *f16, c: *f16, out: *mut f16) {
                let va: f16x8 = load(a, 0)
                let vb: f16x8 = load(b, 0)
                let vc: f16x8 = load(c, 0)
                let r: f16x8 = fma(va, vb, vc)
                store(out, 0, r)
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap();
        assert!(
            ir.contains("@llvm.fma.v8f16"),
            "expected llvm.fma.v8f16 intrinsic, got:\n{ir}"
        );
    }

    #[test]
    fn test_f16_reduce_uses_llvm_intrinsics() {
        use ea_compiler::{CompileOptions, OutputMode};
        let src = r#"
            export func red(a: *f16, out: *mut f16) {
                let v: f16x8 = load(a, 0)
                let s: f16 = reduce_add(v)
                let mx: f16 = reduce_max(v)
                let mn: f16 = reduce_min(v)
                out[0] = s
                out[1] = mx
                out[2] = mn
            }
        "#;
        let opts = CompileOptions {
            opt_level: 0,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: Some("aarch64-unknown-linux-gnu".to_string()),
        };
        let dir = TempDir::new().unwrap();
        let ir_path = dir.path().join("k.ll");
        ea_compiler::compile_with_options(src, &ir_path, OutputMode::LlvmIr, &opts).unwrap();
        let ir = std::fs::read_to_string(&ir_path).unwrap();
        assert!(
            ir.contains("@llvm.vector.reduce.fadd.v8f16"),
            "expected fadd reduction intrinsic, got:\n{ir}"
        );
        assert!(
            ir.contains("@llvm.vector.reduce.fmax.v8f16"),
            "expected fmax reduction intrinsic"
        );
        assert!(
            ir.contains("@llvm.vector.reduce.fmin.v8f16"),
            "expected fmin reduction intrinsic"
        );
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_f16_splat() {
        use ea_compiler::{CompileOptions, OutputMode};
        use std::process::Command;

        let ea = r#"
            export func splat_one(out: *mut f16) {
                let v: f16x8 = splat(1.0)
                store(out, 0, v)
            }
        "#;
        let c = r#"
            #include <stdio.h>
            extern void splat_one(_Float16 *out);
            int main(void) {
                _Float16 buf[8] = {0};
                splat_one(buf);
                for (int i = 0; i < 8; ++i) printf("%g\n", (double)buf[i]);
                return 0;
            }
        "#;
        let dir = TempDir::new().unwrap();
        let obj = dir.path().join("k.o");
        let cpath = dir.path().join("h.c");
        let bin = dir.path().join("k_bin");
        let opts = CompileOptions {
            opt_level: 3,
            target_cpu: None,
            extra_features: "+fullfp16".to_string(),
            target_triple: None,
        };
        ea_compiler::compile_with_options(ea, &obj, OutputMode::ObjectFile, &opts)
            .expect("compile failed");
        std::fs::write(&cpath, c).expect("write c");
        let status = Command::new("cc")
            .args([
                cpath.to_str().unwrap(),
                obj.to_str().unwrap(),
                "-o",
                bin.to_str().unwrap(),
                "-lm",
                "-march=armv8-a+fp16",
            ])
            .status()
            .expect("link failed");
        assert!(status.success(), "linker failed");
        let out = Command::new(&bin).output().expect("run failed");
        let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        assert_eq!(stdout.trim(), "1\n1\n1\n1\n1\n1\n1\n1");
    }
}
