#[cfg(feature = "llvm")]
mod tests {
    // --- round_f32x8_i32x8 type error tests ---

    #[test]
    fn test_round_wrong_type() {
        let source = r#"export func f(a: i32x8) -> i32x8 { return round_f32x8_i32x8(a) }"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("round_f32x8_i32x8") && msg.contains("f32x8"),
            "expected type error mentioning f32x8, got: {msg}"
        );
    }
}
