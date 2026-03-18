mod tests {
    fn parse_err(source: &str) -> String {
        let tokens = ea_compiler::tokenize(source).unwrap();
        let err = ea_compiler::parse(tokens).unwrap_err();
        format!("{err}")
    }

    #[test]
    fn test_parse_error_shows_token_text_not_debug() {
        let msg = parse_err("42");
        assert!(!msg.contains("Some("), "error leaks debug format: {msg}");
        assert!(!msg.contains("IntLiteral"), "error leaks token kind: {msg}");
        assert!(
            msg.contains("'42'"),
            "expected quoted token text, got: {msg}"
        );
    }

    #[test]
    fn test_parse_error_expected_type_shows_token() {
        let msg = parse_err("func f(x: +) {}");
        assert!(!msg.contains("Some("), "error leaks debug format: {msg}");
        assert!(msg.contains("'+'"), "expected quoted token, got: {msg}");
    }

    #[test]
    fn test_parse_error_expected_expression_shows_token() {
        let msg = parse_err("export func f() { let x: i32 = }");
        assert!(!msg.contains("Some("), "error leaks debug format: {msg}");
        assert!(msg.contains("'}'"), "expected quoted token, got: {msg}");
    }

    #[test]
    fn test_parse_error_eof_shows_end_of_file() {
        let msg = parse_err("export");
        assert!(
            msg.contains("end of file"),
            "expected 'end of file' for EOF, got: {msg}"
        );
    }

    fn type_err(source: &str) -> String {
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        format!("{err}")
    }

    #[test]
    fn test_const_type_mismatch_shows_value_kind() {
        let msg = type_err("const X: i32 = 1.5\nexport func f() { return }");
        assert!(
            msg.contains("float"),
            "const mismatch should mention the value kind, got: {msg}"
        );
    }
}
