#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Integer type hints for pointer-index and u64 operands ===
    // Regression: `p[i]` gave no type hint and u64 was missing from the hint
    // filter, so `p[0] == 1` (p: *u8/*u16/*i64/*u64...) emitted
    // `icmp eq i8 %elem, i32 1` (LLVM verifier failure), and `p[0] > q[0]`
    // on *u8 / `a > b` on u64 silently used signed predicates.

    #[test]
    fn test_index_literal_compare_compiles_for_every_int_type() {
        for ty in ["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"] {
            let ir = compile_to_ir(&format!(
                "export func f(p: *{ty}) -> i32 {{\n    if p[0] == 1 {{ return 1 }}\n    return 0\n}}\n"
            ));
            assert!(ir.contains("icmp eq"), "{ty}: {ir}");
        }
    }

    #[test]
    fn test_u64_scalar_literal_compare_compiles() {
        let ir = compile_to_ir(
            "export func f(x: u64) -> i32 {\n    if x == 1 { return 1 }\n    return 0\n}\n",
        );
        assert!(ir.contains("icmp eq i64"), "{ir}");
    }

    #[test]
    fn test_u8_index_vs_index_compare_is_unsigned() {
        assert_c_interop(
            r#"
            export func gt(p: *u8, q: *u8) -> i32 {
                if p[0] > q[0] { return 1 }
                return 0
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t gt(const uint8_t*, const uint8_t*);
            int main() {
                uint8_t a = 200, b = 100;
                printf("%d %d\n", gt(&a, &b), gt(&b, &a));
                return 0;
            }
        "#,
            "1 0",
        );
    }

    #[test]
    fn test_u64_compare_is_unsigned() {
        assert_c_interop(
            r#"
            export func gt(a: u64, b: u64) -> i32 {
                if a > b { return 1 }
                return 0
            }
        "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern int32_t gt(uint64_t, uint64_t);
            int main() {
                printf("%d %d\n", gt(1ULL << 63, 1), gt(1, 1ULL << 63));
                return 0;
            }
        "#,
            "1 0",
        );
    }
}
