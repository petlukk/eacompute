# Common Intrinsics

This page covers the most frequently used intrinsics. For the complete list, see the [Intrinsics Reference](../reference/intrinsics.md).

## splat

Broadcast a scalar to all lanes of a vector:

```
let factor: f32 = 2.5
let vf: f32x8 = splat(factor)
// vf = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
```

Works with all vector types. The return type is inferred from the variable's type annotation.

## load / store

Load a vector from a pointer at an element offset. Store writes a vector back:

```
let v: f32x8 = load(data, i)       // read 8 floats starting at data[i]
store(out, i, v)                    // write 8 floats starting at out[i]
```

Offsets are in elements, not bytes.

### Typed loads

When you want to be explicit about the element type:

```
let v: f32x8 = load_f32(data, i)
let v: f32x4 = load_f32x4(data, i)
```

These are equivalent to plain `load` but make the element type visible in the source.

## stream_store

Non-temporal store that bypasses the CPU cache. Use for write-only output buffers where you will not read the data back soon:

```
let result: f32x8 = a .* b
stream_store(out, i, result)
```

This avoids polluting the cache with output data, leaving more cache space for inputs. Only beneficial for large output arrays that will not be re-read immediately.

## fma

Fused multiply-add: computes `a * b + c` in a single instruction with a single rounding (more accurate than separate multiply and add):

```
// Scalar
let result: f32 = fma(a, b, c)

// Vector
let va: f32x8 = load(a, i)
let vb: f32x8 = load(b, i)
let vc: f32x8 = load(c, i)
let result: f32x8 = fma(va, vb, vc)
```

Works on `f32`, `f64`, and all float vector types. Maps to the hardware FMA instruction.

## reduce_add

Sum all lanes of a vector down to a scalar:

```
let v: f32x8 = load(data, i)
let sum: f32 = reduce_add(v)
```

Works on all integer and float vector types. Useful for dot products, reductions, and histogram accumulation.

## reduce_max / reduce_min

Find the maximum or minimum value across all lanes:

```
let v: f32x4 = load(data, i)
let biggest: f32 = reduce_max(v)
let smallest: f32 = reduce_min(v)
```

Works on integer and float vector types.

## select

Per-lane conditional: where the mask is true, take from `a`; where false, take from `b`:

```
let mask: f32x8 = a .> b
let result: f32x8 = select(mask, a, b)
// element-wise max(a, b)
```

Compiles to a blend instruction. No branching. This is how you write branchless SIMD code.

## sqrt / rsqrt

Square root and reciprocal square root (1/sqrt):

```
let x: f32 = 16.0
let root: f32 = sqrt(x)       // 4.0

let v: f32x8 = load(data, i)
let roots: f32x8 = sqrt(v)
let inv_roots: f32x8 = rsqrt(v)
```

`rsqrt` uses the fast hardware approximation. Works on `f32`, `f64`, and float vector types.

## min / max

Element-wise minimum and maximum. Works on both scalars and vectors:

```
// Scalar
let smaller: f32 = min(a, b)
let larger: f32 = max(a, b)

// Vector
let va: f32x8 = load(data, i)
let vb: f32x8 = load(data, j)
let mins: f32x8 = min(va, vb)
let maxs: f32x8 = max(va, vb)
```

## movemask

Extract comparison results to an integer bitmask. x86 only:

```
let mask: f32x8 = a .> b
let bits: i32 = movemask(mask)
// bit k is 1 if lane k of a > lane k of b
```

Useful for branching on SIMD comparison results or counting matching elements. Each bit in the result corresponds to one vector lane.

## sat_add / sat_sub

Saturating addition and subtraction. Values clamp to the type's min/max instead of wrapping on overflow. Cross-platform (ARM NEON + x86 SSE2):

```
let bright: u8x16 = sat_add(pixels, boost)    // clamps at 255
let dark: u8x16 = sat_sub(pixels, reduce)     // clamps at 0
```

Works with `i8x16`, `u8x16`, `i16x8`, `u16x8`. Signed vs unsigned saturation is determined by the element type. Both arguments must have the same type.

## Masked memory operations

For tail handling, masked loads and stores read/write only the valid lanes:

```
let rem: i32 = n - i
let v: f32x4 = load_masked(data, i, rem)    // load only 'rem' elements
store_masked(out, i, result, rem)            // store only 'rem' elements
```

The `rem` parameter specifies how many elements (starting from lane 0) are valid. Lanes beyond `rem` are zero-filled on load and not written on store.

## Summary table

| Intrinsic | Input | Output | Description |
|-----------|-------|--------|-------------|
| `splat(s)` | scalar | vector | Broadcast to all lanes |
| `load(ptr, i)` | pointer, offset | vector | Load vector from memory |
| `store(ptr, i, v)` | pointer, offset, vector | void | Write vector to memory |
| `stream_store(ptr, i, v)` | pointer, offset, vector | void | Non-temporal write |
| `fma(a, b, c)` | 3 values | same type | `a * b + c` fused |
| `reduce_add(v)` | vector | scalar | Sum all lanes |
| `reduce_max(v)` | vector | scalar | Max across lanes |
| `reduce_min(v)` | vector | scalar | Min across lanes |
| `select(m, a, b)` | mask, 2 vectors | vector | Per-lane conditional |
| `sqrt(x)` | scalar/vector | same type | Square root |
| `rsqrt(x)` | scalar/vector | same type | Reciprocal square root |
| `min(a, b)` | 2 values | same type | Element-wise minimum |
| `max(a, b)` | 2 values | same type | Element-wise maximum |
| `sat_add(a, b)` | 2 int vectors | same type | Saturating addition |
| `sat_sub(a, b)` | 2 int vectors | same type | Saturating subtraction |
| `movemask(m)` | bool vector | i32 | Extract lane mask to bits |
