# Type System

Ea is statically typed with no implicit conversions. Every variable, parameter, and expression has a concrete type known at compile time.

## Scalar Types

| Type | Size | Description |
|------|------|-------------|
| `i8` | 1 byte | Signed 8-bit integer |
| `u8` | 1 byte | Unsigned 8-bit integer |
| `i16` | 2 bytes | Signed 16-bit integer |
| `u16` | 2 bytes | Unsigned 16-bit integer |
| `i32` | 4 bytes | Signed 32-bit integer |
| `u32` | 4 bytes | Unsigned 32-bit integer |
| `i64` | 8 bytes | Signed 64-bit integer |
| `u64` | 8 bytes | Unsigned 64-bit integer |
| `f32` | 4 bytes | 32-bit float (IEEE 754) |
| `f64` | 8 bytes | 64-bit float (IEEE 754) |
| `bool` | 1 byte | Boolean (`true` or `false`) |

Integer literals default to `i32`. Float literals default to `f32`. Use explicit casts (`to_f64(x)`, `to_i64(x)`) to convert between types.

## Vector Types

Vector types hold multiple lanes of the same scalar type. Element-wise operations use dot-operators (`.+`, `.-`, `.*`, `./`, etc.).

### 128-bit Vectors -- SSE (x86) / NEON (ARM)

| Type | Lanes | Element | Size |
|------|-------|---------|------|
| `f32x4` | 4 | `f32` | 16 bytes |
| `f64x2` | 2 | `f64` | 16 bytes |
| `i32x4` | 4 | `i32` | 16 bytes |
| `i16x8` | 8 | `i16` | 16 bytes |
| `i8x16` | 16 | `i8` | 16 bytes |
| `u8x16` | 16 | `u8` | 16 bytes |
| `u16x8` | 8 | `u16` | 16 bytes |

### 256-bit Vectors -- AVX2 (x86 only)

| Type | Lanes | Element | Size |
|------|-------|---------|------|
| `f32x8` | 8 | `f32` | 32 bytes |
| `f64x4` | 4 | `f64` | 32 bytes |
| `i32x8` | 8 | `i32` | 32 bytes |
| `i16x16` | 16 | `i16` | 32 bytes |
| `i8x32` | 32 | `i8` | 32 bytes |
| `u8x32` | 32 | `u8` | 32 bytes |
| `u16x16` | 16 | `u16` | 32 bytes |

These types produce a compile error on ARM targets.

### 512-bit Vectors -- AVX-512 (x86, `--avx512` flag required)

| Type | Lanes | Element | Size |
|------|-------|---------|------|
| `f32x16` | 16 | `f32` | 64 bytes |
| `f64x8` | 8 | `f64` | 64 bytes |
| `i32x16` | 16 | `i32` | 64 bytes |

Using these types without `--avx512` produces a compile error.

## Pointer Types

Pointers represent caller-provided memory. Ea never allocates -- all memory comes from the host language.

| Syntax | Description |
|--------|-------------|
| `*T` | Immutable pointer to `T` |
| `*mut T` | Mutable pointer to `T` |
| `*restrict T` | Immutable pointer, no-alias guarantee |
| `*restrict mut T` | Mutable pointer, no-alias guarantee |

The `restrict` qualifier tells the compiler that the pointer does not alias other pointers, enabling stronger optimizations. Use it when you can guarantee non-overlapping memory.

## Struct Types

User-defined value types declared with `struct`:

```
struct Pixel {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}
```

Structs are passed by value. Access fields with dot syntax: `pixel.r`. Structs can contain scalar types, vector types, and other structs.

## Type Rules

- No implicit conversions between types. Use `to_f32()`, `to_i32()`, etc.
- No generics or polymorphism. Write separate functions for each type.
- Vector dot-operators require both operands to have the same vector type.
- Comparisons on vectors (`.==`, `.<`, `.>`) produce boolean vectors, not scalar bools.
