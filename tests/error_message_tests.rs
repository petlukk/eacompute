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

    #[test]
    fn test_store_mismatch_is_descriptive() {
        let msg = type_err(
            "export func f(p: *mut i32, n: i32) {\n    let v: f32x4 = splat(1.0)\n    store(p, 0, v)\n    return\n}",
        );
        assert!(
            msg.contains("pointer element type is"),
            "store mismatch should be descriptive, got: {msg}"
        );
    }

    #[test]
    fn test_immutable_assign_suggests_let_mut() {
        let msg = type_err("export func f() { let x: i32 = 0\n x = 1\n return }");
        assert!(
            msg.contains("let mut"),
            "immutable assign should suggest 'let mut', got: {msg}"
        );
    }

    #[test]
    fn test_store_arg_count_shows_names() {
        let msg = type_err("export func f(p: *mut f32) {\n    store(p, 0)\n    return\n}");
        assert!(
            msg.contains("(ptr, index, vector)"),
            "store arg count error should name the args, got: {msg}"
        );
    }

    #[test]
    fn test_vector_add_suggests_dot_operator() {
        let msg = type_err("export func f(a: f32x4, b: f32x4) -> f32x4 { return a + b }");
        assert!(msg.contains(".+"), "vector + should suggest .+, got: {msg}");
    }

    #[test]
    fn test_vector_multiply_suggests_dot_operator() {
        let msg = type_err("export func f(a: f32x4, b: f32x4) -> f32x4 { return a * b }");
        assert!(msg.contains(".*"), "vector * should suggest .*, got: {msg}");
    }
}
