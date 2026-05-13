#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    #[test]
    fn test_bitwise_and() {
        assert_output(
            r#"export func main() {
                let a: i32 = 255
                let b: i32 = 15
                println(a & b)
            }"#,
            "15",
        );
    }

    #[test]
    fn test_bitwise_or() {
        assert_output(
            r#"export func main() {
                let a: i32 = 240
                let b: i32 = 15
                println(a | b)
            }"#,
            "255",
        );
    }

    #[test]
    fn test_bitwise_xor() {
        assert_output(
            r#"export func main() {
                let a: i32 = 255
                let b: i32 = 15
                println(a ^ b)
            }"#,
            "240",
        );
    }

    #[test]
    fn test_shift_left() {
        assert_output(
            r#"export func main() {
                let a: i32 = 1
                println(a << 4)
            }"#,
            "16",
        );
    }

    #[test]
    fn test_shift_right() {
        assert_output(
            r#"export func main() {
                let a: i32 = 256
                println(a >> 4)
            }"#,
            "16",
        );
    }

    #[test]
    fn test_nibble_pack() {
        assert_output(
            r#"export func main() {
                let hi: i32 = 10
                let lo: i32 = 5
                println((hi << 4) | lo)
            }"#,
            "165",
        );
    }

    #[test]
    fn test_nibble_unpack() {
        assert_output_lines(
            r#"
            export func main() {
                let packed: i32 = 165
                let lo: i32 = packed & 15
                let hi: i32 = (packed >> 4) & 15
                println(lo)
                println(hi)
            }
            "#,
            &["5", "10"],
        );
    }

    #[test]
    fn test_bitwise_rejects_float() {
        let tokens = ea_compiler::tokenize(
            r#"export func main() {
                let a: f32 = 1.0
                let b: f32 = 2.0
                let c: f32 = a & b
            }"#,
        )
        .unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("bitwise") || msg.contains("integer"),
            "expected bitwise/integer error, got: {msg}"
        );
    }
}
