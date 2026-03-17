# SIMD

SIMD (Single Instruction, Multiple Data) lets you process multiple values in a single CPU instruction. Ea gives you direct control over SIMD vectors -- what you write is what the CPU executes.

## Vector types

A vector type holds a fixed number of elements of the same scalar type. The name encodes both the element type and the lane count:

| Type | Elements | Bits | x86 requirement | ARM requirement |
|------|----------|------|------------------|-----------------|
| `f32x4` | 4 x f32 | 128 | SSE | NEON |
| `i32x4` | 4 x i32 | 128 | SSE | NEON |
| `u8x16` | 16 x u8 | 128 | SSE | NEON |
| `i8x16` | 16 x i8 | 128 | SSE | NEON |
| `i16x8` | 8 x i16 | 128 | SSE | NEON |
| `f64x2` | 2 x f64 | 128 | SSE2 | NEON |
| `f32x8` | 8 x f32 | 256 | AVX2 | not available |
| `i32x8` | 8 x i32 | 256 | AVX2 | not available |
| `u8x32` | 32 x u8 | 256 | AVX2 | not available |
| `f64x4` | 4 x f64 | 256 | AVX2 | not available |
| `f32x16` | 16 x f32 | 512 | AVX-512 | not available |

If you use `f32x8` on a machine without AVX2, or on ARM, the compiler will error. No silent scalar fallback.

To enable 512-bit vectors, compile with `--avx512`:

```bash
ea kernel.ea --lib --avx512
```

## Dot operators

Element-wise vector operations use dot-prefixed operators. This distinguishes them from scalar operations and makes SIMD explicit in the source:

| Operator | Meaning |
|----------|---------|
| `.+` | Element-wise add |
| `.-` | Element-wise subtract |
| `.*` | Element-wise multiply |
| `./` | Element-wise divide |
| `.>` | Element-wise greater than (returns bool vector) |
| `.<` | Element-wise less than |
| `.>=` | Element-wise greater or equal |
| `.<=` | Element-wise less or equal |
| `.==` | Element-wise equal |
| `.!=` | Element-wise not equal |
| `.&` | Element-wise bitwise AND |
| `.\|` | Element-wise bitwise OR |
| `.^` | Element-wise bitwise XOR |

Example:

```
let a: f32x4 = load(data, i)
let b: f32x4 = load(data, i + 4)
let sum: f32x4 = a .+ b
let product: f32x4 = a .* b
let mask: f32x4 = a .> b
```

## Creating vectors

### splat

Broadcast a scalar value to all lanes:

```
let factor: f32 = 2.5
let vf: f32x8 = splat(factor)
// vf = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
```

### Loading from memory

Load a vector from a pointer at an element offset:

```
let v: f32x8 = load(data, i)     // loads data[i..i+8]
```

The offset `i` is in elements, not bytes. `load(data, 8)` loads elements 8 through 15.

There are also typed load intrinsics when you want to be explicit about the element type:

```
let v: f32x8 = load_f32(data, i)
let v: f32x4 = load_f32x4(data, i)
```

### Storing to memory

Write a vector to a mutable pointer:

```
store(out, i, result)     // writes result to out[i..i+8]
```

## Element access

Read individual lanes from a vector by index:

```
let v: f32x4 = load(data, 0)
let first: f32 = v[0]
let second: f32 = v[1]
```

## Conditional selection

`select` picks lanes from two vectors based on a mask:

```
let mask: f32x4 = a .> b
let result: f32x4 = select(mask, a, b)
// where a > b, take a; otherwise take b
```

This compiles to a single blend instruction. There is no branching.

## Scalar vs SIMD comparison

Here is the same operation -- scaling an array -- written both ways.

**Scalar** (processes one element per iteration):

```
export func scale(data: *f32, out: *mut f32, factor: f32, n: i32) {
    foreach (i in 0..n) {
        out[i] = data[i] * factor
    }
}
```

**SIMD** (processes 8 elements per iteration):

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 8
    tail scalar {
        out[i] = data[i] * factor
    }
{
    let vf: f32x8 = splat(factor)
    store(out, i, load(data, i) .* vf)
}
```

The SIMD version does 8 multiplications in a single instruction. On a workload that is compute-bound (multiple operations per element), this translates to a proportional speedup.

## Hardware targeting

By default, Ea targets AVX2 on x86-64 and NEON on AArch64. You can change this:

```bash
# Default (AVX2 on x86)
ea kernel.ea --lib

# Enable AVX-512
ea kernel.ea --lib --avx512

# Target a specific CPU
ea kernel.ea --lib --target=skylake

# Cross-compile for ARM
ea kernel.ea --lib --target-triple=aarch64-unknown-linux-gnu
```

The compiler rejects code that requires features the target does not have. If you use `f32x8` and target ARM, you get a compile error, not slow code.
