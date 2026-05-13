#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    #[test]
    fn lex_wide_int_vec_types() {
        use ea_compiler::tokenize;
        let src = "u8x64 i8x64 i16x32";
        let toks = tokenize(src).unwrap();
        let kinds: Vec<String> = toks.iter().map(|t| format!("{:?}", t.kind)).collect();
        assert!(
            kinds.iter().any(|k| k == "U8x64"),
            "U8x64 token not found in {:?}",
            kinds
        );
        assert!(
            kinds.iter().any(|k| k == "I8x64"),
            "I8x64 token not found in {:?}",
            kinds
        );
        assert!(
            kinds.iter().any(|k| k == "I16x32"),
            "I16x32 token not found in {:?}",
            kinds
        );
    }
}
