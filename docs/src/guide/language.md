# Language Basics

This page covers Eä's scalar language features. For SIMD vector types and operations, see [SIMD](simd.md).

## Scalar types

Eä has fixed-size numeric types. No type inference -- you always write the type explicitly.

| Type | Description |
|------|-------------|
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `f32`, `f64` | Floating point |
| `bool` | Boolean (`true` / `false`) |

## Variables

All variables must have an explicit type annotation. Variables are immutable by default.

```
let x: i32 = 5
let y: f32 = 3.14
let flag: bool = true
```

To make a variable mutable, use `mut`:

```
let mut counter: i32 = 0
counter = counter + 1
```

## Constants

Compile-time constants use `const`:

```
const PI: f32 = 3.14159
const BATCH_SIZE: i32 = 256
const EPSILON: f64 = 1e-9
```

Constants can be used in `static_assert` for compile-time checks:

```
const STEP: i32 = 8
static_assert(STEP > 0, "step must be positive")
```

## Arithmetic and comparison

Standard arithmetic operators work on all numeric types:

```
let a: i32 = 10 + 3    // 13
let b: i32 = 10 - 3    // 7
let c: i32 = 10 * 3    // 30
let d: i32 = 10 / 3    // 3 (integer division)
let e: i32 = 10 % 3    // 1 (remainder)
```

Comparison operators return `bool`:

```
let lt: bool = a < b
let gt: bool = a > b
let le: bool = a <= b
let ge: bool = a >= b
let eq: bool = a == b
let ne: bool = a != b
```

## Logical operators

Eä uses words, not symbols, for logical operations:

```
let both: bool = a > 0 and b > 0
let either: bool = a > 0 or b > 0
let neither: bool = not (a > 0 or b > 0)
```

## Control flow

### if / else if / else

```
if x > 0 {
    println(1)
} else if x == 0 {
    println(0)
} else {
    println(-1)
}
```

### while loops

```
let mut i: i32 = 0
while i < n {
    out[i] = data[i] * 2
    i = i + 1
}
```

### for loops

Counted loops with an explicit step:

```
for i in 0..n step 1 {
    out[i] = data[i] * 2
}
```

The step is required. The range `0..n` is half-open: it includes 0 but excludes `n`.

### foreach loops

A simpler counted loop when the step is always 1:

```
foreach (i in 0..n) {
    out[i] = data[i] * 2
}
```

### Loop unrolling

Wrap a loop in `unroll(N)` to unroll it at compile time:

```
unroll(4) {
    foreach (j in 0..4) {
        out[base + j] = data[base + j] * factor
    }
}
```

## Functions

Functions are declared with `func`. All parameter and return types are explicit:

```
func square(x: f32) -> f32 {
    return x * x
}

func add(a: i32, b: i32) -> i32 {
    return a + b
}
```

Functions without a return type return nothing (void):

```
func fill(out: *mut i32, n: i32, val: i32) {
    foreach (i in 0..n) {
        out[i] = val
    }
}
```

### Exported functions

To make a function callable from C/Python/Rust, prefix it with `export`:

```
export func dot_product(a: *f32, b: *f32, n: i32) -> f32 {
    let mut sum: f32 = 0.0
    foreach (i in 0..n) {
        sum = sum + a[i] * b[i]
    }
    return sum
}
```

Only `export func` (and `export kernel`) produce symbols visible from outside. Non-exported functions are internal helpers.

## Pointers

Pointers are how kernels receive data from the host language. There are four pointer variants:

| Syntax | Meaning |
|--------|---------|
| `*T` | Read-only pointer |
| `*mut T` | Mutable pointer (can write through it) |
| `*restrict T` | Read-only, no aliasing (enables optimizations) |
| `*restrict mut T` | Mutable, no aliasing |

### Pointer indexing

Read from a pointer with bracket indexing:

```
let val: f32 = data[i]    // data is *f32
```

Write through a mutable pointer:

```
out[i] = val              // out is *mut f32
```

## Type casts

Explicit casts convert between numeric types:

```
let x: i32 = 42
let f: f32 = to_f32(x)       // 42.0
let d: f64 = to_f64(x)       // 42.0
let back: i32 = to_i32(f)    // 42
let wide: i64 = to_i64(x)    // 42
```

There are no implicit conversions. Mixing types without a cast is a compile error.

## println

`println` is the only output primitive. It exists for debugging:

```
println(42)
println(3.14)
println(true)
println("hello")
```

It accepts integers, floats, bools, and string literals. It does not support format strings.

## What does not exist

Eä is deliberately minimal. The following features do not exist and are not planned:

- No generics or templates
- No traits or interfaces
- No modules or imports
- No heap allocation
- No strings (except literal arguments to `println`)
- No semicolons (statements are newline-separated)
- No closures or lambdas
- No enums or pattern matching
- No exceptions or error handling

One file, one compilation unit. Compose at the C level.
