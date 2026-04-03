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
}
