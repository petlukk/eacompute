//! Coverage for the `ptr_as_*` family of pointer-reinterpret intrinsics.
//!
//! `ptr_as_<T>(p)` is a zero-cost reinterpret of one pointer as another typed
//! pointer (same value, different element type). Codegen reuses the underlying
//! pointer value verbatim (`src/codegen/expressions.rs:188`); the only real
//! work happens in `src/typeck/intrinsics.rs::check_ptr_as`, which preserves
//! `mutable` and `restrict` and rewrites the inner element type.
//!
//! Phase 4 of the v1.11.0 audit identified the entire `ptr_as_*` family as
//! untested. Each typed variant gets one round-trip E2E test, plus negative
//! tests for the typeck signature.

#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === E2E round-trip tests, one per typed variant ===
    //
    // Each test reinterprets a buffer of one element type through `ptr_as_<T>`
    // and reads/writes through the cast pointer to confirm the lowering is a
    // genuine no-op pointer reinterpret. We use `u8` buffers as the source
    // representation because C `unsigned char*` is universally aliasable —
    // the C harness allocates raw bytes and the Eä kernel re-types them.

    #[test]
    fn test_ptr_as_i8_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut i32) {
                let p: *i8 = ptr_as_i8(src)
                out[0] = to_i32(p[0])
                out[1] = to_i32(p[1])
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, int32_t*);
            int main() {
                unsigned char buf[2] = {0xFF, 0x7F};
                int32_t out[2];
                k(buf, out);
                printf("%d %d\n", out[0], out[1]);
                return 0;
            }
            "#,
            // 0xFF as i8 = -1; 0x7F as i8 = 127
            "-1 127",
        );
    }

    #[test]
    fn test_ptr_as_u8_roundtrip() {
        // The cast is the load-bearing step: re-typing the pointer changes
        // how indexed bytes are interpreted on write-back. Here we write
        // through a `*mut u8` view, then the C harness verifies the bytes
        // landed unchanged at the source addresses.
        assert_c_interop(
            r#"
            export func k(src: *mut i8) {
                let p: *mut u8 = ptr_as_u8(src)
                p[0] = 255
                p[1] = 127
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(signed char*);
            int main() {
                signed char buf[2] = {0, 0};
                k(buf);
                // After the cast, buf[0] = byte 0xFF, buf[1] = byte 0x7F.
                // Print as unsigned to confirm the raw byte pattern.
                printf("%u %u\n",
                       (unsigned)(unsigned char)buf[0],
                       (unsigned)(unsigned char)buf[1]);
                return 0;
            }
            "#,
            "255 127",
        );
    }

    #[test]
    fn test_ptr_as_i16_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut i32) {
                let p: *i16 = ptr_as_i16(src)
                out[0] = to_i32(p[0])
                out[1] = to_i32(p[1])
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, int32_t*);
            int main() {
                // Little-endian: 0xFFFF=-1, 0x0001=1
                int16_t src[2] = {-1, 1};
                int32_t out[2];
                k((const unsigned char*)src, out);
                printf("%d %d\n", out[0], out[1]);
                return 0;
            }
            "#,
            "-1 1",
        );
    }

    #[test]
    fn test_ptr_as_u16_roundtrip() {
        // Write through a *mut u16 view to verify the cast preserves
        // u16 element semantics (2-byte stride, unsigned writes).
        assert_c_interop(
            r#"
            export func k(src: *mut i8) {
                let p: *mut u16 = ptr_as_u16(src)
                p[0] = 65535
                p[1] = 1
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(signed char*);
            int main() {
                uint16_t buf[2] = {0, 0};
                k((signed char*)buf);
                printf("%u %u\n", (unsigned)buf[0], (unsigned)buf[1]);
                return 0;
            }
            "#,
            "65535 1",
        );
    }

    #[test]
    fn test_ptr_as_i32_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut i32) {
                let p: *i32 = ptr_as_i32(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, int32_t*);
            int main() {
                int32_t src[2] = {-12345, 67890};
                int32_t out[2];
                k((const unsigned char*)src, out);
                printf("%d %d\n", out[0], out[1]);
                return 0;
            }
            "#,
            "-12345 67890",
        );
    }

    #[test]
    fn test_ptr_as_u32_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut u32) {
                let p: *u32 = ptr_as_u32(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, uint32_t*);
            int main() {
                uint32_t src[2] = {4294967295u, 1u};
                uint32_t out[2];
                k((const unsigned char*)src, out);
                printf("%u %u\n", out[0], out[1]);
                return 0;
            }
            "#,
            "4294967295 1",
        );
    }

    #[test]
    fn test_ptr_as_i64_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut i64) {
                let p: *i64 = ptr_as_i64(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, int64_t*);
            int main() {
                int64_t src[2] = {-9000000000LL, 9000000000LL};
                int64_t out[2];
                k((const unsigned char*)src, out);
                printf("%lld %lld\n", (long long)out[0], (long long)out[1]);
                return 0;
            }
            "#,
            "-9000000000 9000000000",
        );
    }

    #[test]
    fn test_ptr_as_u64_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut u64) {
                let p: *u64 = ptr_as_u64(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(const unsigned char*, uint64_t*);
            int main() {
                uint64_t src[2] = {18000000000000000000ULL, 1ULL};
                uint64_t out[2];
                k((const unsigned char*)src, out);
                printf("%llu %llu\n", (unsigned long long)out[0], (unsigned long long)out[1]);
                return 0;
            }
            "#,
            "18000000000000000000 1",
        );
    }

    #[test]
    fn test_ptr_as_f32_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut f32) {
                let p: *f32 = ptr_as_f32(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            extern void k(const unsigned char*, float*);
            int main() {
                float src[2] = {1.5f, -2.25f};
                float out[2];
                k((const unsigned char*)src, out);
                printf("%g %g\n", out[0], out[1]);
                return 0;
            }
            "#,
            "1.5 -2.25",
        );
    }

    #[test]
    fn test_ptr_as_f64_roundtrip() {
        assert_c_interop(
            r#"
            export func k(src: *u8, out: *mut f64) {
                let p: *f64 = ptr_as_f64(src)
                out[0] = p[0]
                out[1] = p[1]
            }
            "#,
            r#"
            #include <stdio.h>
            extern void k(const unsigned char*, double*);
            int main() {
                double src[2] = {3.14, -2.71};
                double out[2];
                k((const unsigned char*)src, out);
                printf("%g %g\n", out[0], out[1]);
                return 0;
            }
            "#,
            "3.14 -2.71",
        );
    }

    // === Mutability and write-through ===
    //
    // `ptr_as_*` must preserve `*mut`: reinterpreting a `*mut u8` as `*mut i32`
    // and writing through the cast pointer must mutate the underlying storage.

    #[test]
    fn test_ptr_as_preserves_mutability_and_writes() {
        assert_c_interop(
            r#"
            export func k(buf: *mut u8) {
                let p: *mut i32 = ptr_as_i32(buf)
                p[0] = 42
                p[1] = -7
            }
            "#,
            r#"
            #include <stdio.h>
            #include <stdint.h>
            extern void k(unsigned char*);
            int main() {
                // 8 bytes is enough for two i32s; aligned for safe access.
                int32_t storage[2] = {0, 0};
                k((unsigned char*)storage);
                printf("%d %d\n", storage[0], storage[1]);
                return 0;
            }
            "#,
            "42 -7",
        );
    }

    // === Typeck rejection ===

    #[test]
    fn test_ptr_as_rejects_non_pointer_argument() {
        // ptr_as_i32 requires its argument to be a pointer; passing a scalar
        // must fail at type-check time.
        let source = r#"
            export func k(x: i64) -> i64 {
                let p: *i32 = ptr_as_i32(x)
                return 0
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("pointer"),
            "expected error mentioning pointer, got: {msg}"
        );
    }

    #[test]
    fn test_ptr_as_rejects_wrong_arity() {
        // ptr_as_f32 takes exactly one argument; two should be rejected.
        let source = r#"
            export func k(a: *u8, b: *u8) {
                let p: *f32 = ptr_as_f32(a, b)
            }
        "#;
        let tokens = ea_compiler::tokenize(source).unwrap();
        let stmts = ea_compiler::parse(tokens).unwrap();
        let err = ea_compiler::check_types(&stmts).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("1 argument") || msg.contains("expects"),
            "expected arity error, got: {msg}"
        );
    }
}
