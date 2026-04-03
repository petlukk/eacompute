#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i32x4() {
        assert_output(
            r#"
export func main() {
    let v: i32x4 = [10, 20, 30, 40]
    println(v[0])
    println(v[1])
    println(v[2])
    println(v[3])
}
"#,
            "10\n20\n30\n40",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_f32x4() {
        assert_output(
            r#"
export func main() {
    let v: f32x4 = [1.5, 2.5, 3.5, 4.5]
    println(v[0])
    println(v[1])
}
"#,
            "1.5\n2.5",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_u8x16() {
        assert_output(
            r#"
export func main() {
    let v: u8x16 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    println(v[0])
    println(v[15])
}
"#,
            "10\n160",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i8x16() {
        assert_output(
            r#"
export func main() {
    let v: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    println(v[0])
    println(v[15])
}
"#,
            "1\n16",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_from_annotation_i16x8() {
        assert_output(
            r#"
export func main() {
    let v: i16x8 = [1, 2, 3, 4, 5, 6, 7, 8]
    println(v[0])
    println(v[7])
}
"#,
            "1\n8",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_annotation_arithmetic() {
        assert_output(
            r#"
export func main() {
    let a: i32x4 = [1, 2, 3, 4]
    let b: i32x4 = [10, 20, 30, 40]
    let c: i32x4 = a .+ b
    println(c[0])
    println(c[3])
}
"#,
            "11\n44",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_vector_literal_suffix_still_works() {
        assert_output(
            r#"
export func main() {
    let v: i32x4 = [1, 2, 3, 4]i32x4
    println(v[0])
}
"#,
            "1",
        );
    }

    #[test]
    fn test_vector_literal_wrong_count() {
        let source = r#"
export func main() {
    let v: i32x4 = [1, 2, 3]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("4 elements") || format!("{err:?}").contains("expects"),
            "should report element count mismatch, got: {err:?}"
        );
    }

    #[test]
    fn test_vector_literal_wrong_elem_type() {
        let source = r#"
export func main() {
    let v: i32x4 = [1.0, 2.0, 3.0, 4.0]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("expected i32") || format!("{err:?}").contains("element"),
            "should report type mismatch, got: {err:?}"
        );
    }

    #[test]
    fn test_bare_array_literal_still_errors() {
        let source = r#"
export func f(v: i32x4) -> i32x4 {
    return [1, 2, 3, 4]
}
"#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let stmts = ea_compiler::desugar(stmts).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        assert!(
            format!("{err:?}").contains("shuffle"),
            "bare array literal in return should still error, got: {err:?}"
        );
    }
}
