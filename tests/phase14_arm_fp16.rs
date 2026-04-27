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
}
