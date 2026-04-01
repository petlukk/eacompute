# All Intrinsics

Complete reference for every built-in function in Eä.

## Memory

### load

Load a vector from a pointer at a byte offset. The return type is inferred from context.

```
let v: f32x8 = load(ptr, i);
```

### Typed Scalar Loads

Load a single scalar value from a pointer at a byte offset.

| Intrinsic | Return Type |
|-----------|-------------|
| `load_f32(ptr, i)` | `f32` |
| `load_f64(ptr, i)` | `f64` |
| `load_i32(ptr, i)` | `i32` |
| `load_i16(ptr, i)` | `i16` |
| `load_i8(ptr, i)` | `i8` |
| `load_u8(ptr, i)` | `u8` |
| `load_u16(ptr, i)` | `u16` |
| `load_u32(ptr, i)` | `u32` |
| `load_u64(ptr, i)` | `u64` |

```
let x: f32 = load_f32(ptr, i);
```

### Typed Vector Loads

Load a full vector from a pointer at a byte offset.

| Intrinsic | Return Type |
|-----------|-------------|
| `load_f32x4(ptr, i)` | `f32x4` |
| `load_f32x8(ptr, i)` | `f32x8` |
| `load_f32x16(ptr, i)` | `f32x16` |
| `load_i32x4(ptr, i)` | `i32x4` |
| `load_i32x8(ptr, i)` | `i32x8` |
| `load_i16x8(ptr, i)` | `i16x8` |
| `load_i8x16(ptr, i)` | `i8x16` |
| `load_u8x16(ptr, i)` | `u8x16` |
| `load_u8x32(ptr, i)` | `u8x32` |

```
let v: f32x8 = load_f32x8(data, i * 32);
```

### store

Write a vector to a pointer at a byte offset.

```
store(out, i, result);
```

### stream_store

Non-temporal store that bypasses the CPU cache. Use for write-only output that will not be read back soon.

```
stream_store(out, i, result);
```

### load_masked

Masked vector load. Lanes where the mask is false are not loaded.

```
let v: f32x8 = load_masked(ptr, i, mask);
```

### store_masked

Masked vector store. Only lanes where the mask is true are written.

```
store_masked(out, i, value, mask);
```

### gather

Load elements from scattered memory addresses using an index vector. x86 only -- not available on ARM.

```
let v: f32x8 = gather(ptr, indices);
```

### scatter

Store elements to scattered memory addresses using an index vector. AVX-512 only (`--avx512` flag required).

```
scatter(ptr, indices, values);
```

### prefetch

Issue a prefetch hint to bring data into cache.

```
prefetch(ptr, i);
```

## Math

### sqrt

Square root. Works on scalar `f32`/`f64` and all float vector types.

```
let y: f32 = sqrt(x);
let v: f32x8 = sqrt(vec);
```

### rsqrt

Reciprocal square root (approximate). Scalar `f32` and `f32` vector types.

```
let y: f32 = rsqrt(x);
let v: f32x8 = rsqrt(vec);
```

### exp

Exponential function. Float types.

```
let y: f32 = exp(x);
```

### fma

Fused multiply-add: computes `a * b + c` in a single operation with one rounding step. Works on scalar `f32` and all float vector types.

```
let y: f32 = fma(a, b, c);
let v: f32x8 = fma(va, vb, vc);
```

### min

Element-wise minimum. Works on scalar (`i32`, `f32`, `f64`) and vector types.

```
let m: i32 = min(a, b);
let v: f32x8 = min(va, vb);
```

### max

Element-wise maximum. Works on scalar (`i32`, `f32`, `f64`) and vector types.

```
let m: i32 = max(a, b);
let v: f32x8 = max(va, vb);
```

## Reduction

Reduce a vector to a single scalar value.

### reduce_add

Sum all lanes.

```
let sum: f32 = reduce_add(v);   // f32x8 -> f32
let sum: i32 = reduce_add(iv);  // i32x8 -> i32
```

### reduce_max

Maximum across all lanes.

```
let m: f32 = reduce_max(v);
```

### reduce_min

Minimum across all lanes.

```
let m: f32 = reduce_min(v);
```

### reduce_add_fast

Unordered float reduction. Faster than `reduce_add` but does not guarantee summation order, so results may differ slightly due to floating-point rounding. Float vectors only.

```
let sum: f32 = reduce_add_fast(v);
```

## Vector

### splat

Broadcast a scalar value to all lanes of a vector. The vector type is inferred from context.

```
let v: f32x8 = splat(1.0);
```

### shuffle

Reorder lanes of a vector according to an index tuple. The indices are compile-time constants.

```
let reversed: f32x4 = shuffle(v, (3, 2, 1, 0));
```

### select

Per-lane conditional select. Where the mask is true, take from `a`; where false, take from `b`.

```
let result: f32x8 = select(mask, a, b);
```

### movemask

Extract a comparison result bitmask from a boolean vector to a scalar `i32`. Each bit corresponds to the sign bit of one lane. **x86 only** -- not available on ARM.

```
let bits: i32 = movemask(cmp_result);
```

## Conversion

### Scalar Casts

| Intrinsic | Description |
|-----------|-------------|
| `to_f32(x)` | Convert to `f32` |
| `to_f64(x)` | Convert to `f64` |
| `to_i32(x)` | Convert to `i32` |
| `to_i64(x)` | Convert to `i64` |

```
let f: f32 = to_f32(i);
let n: i32 = to_i32(x);
```

### Widening Conversions

Widen narrow integer lanes to wider float or integer lanes. Only the first N lanes of the input are consumed.

| Intrinsic | Input | Output |
|-----------|-------|--------|
| `widen_i8_f32x4(v)` | `i8x16` | `f32x4` |
| `widen_u8_f32x4(v)` | `u8x16` | `f32x4` |
| `widen_i8_f32x8(v)` | `i8x16` | `f32x8` |
| `widen_u8_f32x8(v)` | `u8x16` | `f32x8` |
| `widen_i8_f32x16(v)` | `i8x16` | `f32x16` |
| `widen_u8_f32x16(v)` | `u8x16` | `f32x16` |
| `widen_u8_i32x4(v)` | `u8x16` | `i32x4` |
| `widen_u8_i32x8(v)` | `u8x16` | `i32x8` |
| `widen_u8_i32x16(v)` | `u8x16` | `i32x16` |

```
let pixels: f32x8 = widen_u8_f32x8(raw_bytes);
```

#### Lane-offset variants

The `_4`, `_8`, `_12` suffixes select which 4 bytes of the input to widen, eliminating the need for a shuffle before widening:

| Intrinsic | Input | Output | Bytes used |
|-----------|-------|--------|------------|
| `widen_u8_f32x4_4(v)` | `u8x16` | `f32x4` | 4-7 |
| `widen_u8_f32x4_8(v)` | `u8x16` | `f32x4` | 8-11 |
| `widen_u8_f32x4_12(v)` | `u8x16` | `f32x4` | 12-15 |
| `widen_i8_f32x4_4(v)` | `i8x16` | `f32x4` | 4-7 |
| `widen_i8_f32x4_8(v)` | `i8x16` | `f32x4` | 8-11 |
| `widen_i8_f32x4_12(v)` | `i8x16` | `f32x4` | 12-15 |
| `widen_u8_i32x4_4(v)` | `u8x16` | `i32x4` | 4-7 |
| `widen_u8_i32x4_8(v)` | `u8x16` | `i32x4` | 8-11 |
| `widen_u8_i32x4_12(v)` | `u8x16` | `i32x4` | 12-15 |

Process all 16 bytes of a u8x16 as 4 groups of f32x4 without any shuffles:

```
let f0: f32x4 = widen_u8_f32x4(v)      // bytes 0-3
let f1: f32x4 = widen_u8_f32x4_4(v)    // bytes 4-7
let f2: f32x4 = widen_u8_f32x4_8(v)    // bytes 8-11
let f3: f32x4 = widen_u8_f32x4_12(v)   // bytes 12-15
```

### Narrowing Conversions

Convert wider lanes to narrower lanes, with clamping and rounding.

| Intrinsic | Input | Output |
|-----------|-------|--------|
| `narrow_f32x4_i8(v)` | `f32x4` | `i8` (4 bytes) |

```
let packed = narrow_f32x4_i8(float_pixels);
```

### Multiply-Add Byte Pairs

Multiply unsigned bytes by signed bytes and add adjacent pairs. **x86 only.**

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `maddubs_i16(a, b)` | `(u8x16, i8x16) -> i16x8` | Multiply and add adjacent pairs to 16-bit |
| `maddubs_i32(a, b)` | `(u8x16, i8x16) -> i32x4` | Multiply and add adjacent quads to 32-bit |

```
let products: i16x8 = maddubs_i16(unsigned_bytes, signed_weights);
```

### vdot_i32

Signed integer dot product: multiplies groups of 4 `i8` pairs and sums each group into one `i32` lane. **ARM only** -- requires `--dotprod` flag (ARMv8.2-A dot product extension). Maps to NEON `sdot`.

```
let dot: i32x4 = vdot_i32(activations, weights);
acc = acc .+ vdot_i32(a, b);  // accumulate explicitly
```

| Signature | `(i8x16, i8x16) -> i32x4` |
|-----------|---------------------------|

### I8MM Matrix Multiply

Matrix multiply-accumulate on int8 data. **ARM only** -- requires `--i8mm` flag (ARMv8.6-A I8MM extension). Available on Cortex-A78+, Apple M1+.

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `smmla_i32(acc, a, b)` | `(i32x4, i8x16, i8x16) -> i32x4` | Signed x signed |
| `ummla_i32(acc, a, b)` | `(i32x4, u8x16, u8x16) -> i32x4` | Unsigned x unsigned |
| `usmmla_i32(acc, a, b)` | `(i32x4, u8x16, i8x16) -> i32x4` | Unsigned x signed |

The accumulator is the first argument. Each instruction performs a 2x8 x 8x2 matrix multiply and adds the result to the accumulator. Use `splat(0)` as accumulator for the first iteration.

```
let zero: i32x4 = splat(0);
let result: i32x4 = smmla_i32(zero, activations, weights);
// accumulate over multiple chunks:
acc = smmla_i32(acc, next_a, next_b);
```

### Widening Multiply

Multiply narrow integer lanes and produce wider output. **ARM only** (base NEON). Input types are 64-bit NEON vectors.

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `wmul_i16(a, b)` | `(i8x8, i8x8) -> i16x8` | Signed 8-bit to 16-bit |
| `wmul_u16(a, b)` | `(u8x8, u8x8) -> u16x8` | Unsigned 8-bit to 16-bit |
| `wmul_i32(a, b)` | `(i16x4, i16x4) -> i32x4` | Signed 16-bit to 32-bit |
| `wmul_u32(a, b)` | `(u16x4, u16x4) -> u32x4` | Unsigned 16-bit to 32-bit |

```
let wide: i16x8 = wmul_i16(bytes_a, bytes_b);
```

### Absolute Difference

Element-wise `|a - b|`. **ARM only** (base NEON). Maps to a single instruction (`sabd`/`uabd`).

| Intrinsic | Supported Types |
|-----------|----------------|
| `abs_diff(a, b)` | `i8x16`, `u8x16`, `i16x8`, `u16x8`, `i32x4`, `u32x4` |

```
let diff: u8x16 = abs_diff(frame_a, frame_b);
```

### Saturating Arithmetic

Addition and subtraction that clamp to the type's min/max instead of wrapping on overflow. **Cross-platform** (ARM NEON + x86 SSE2).

| Intrinsic | Supported Types |
|-----------|----------------|
| `sat_add(a, b)` | `i8x16`, `u8x16`, `i16x8`, `u16x8` |
| `sat_sub(a, b)` | `i8x16`, `u8x16`, `i16x8`, `u16x8` |

Signed vs unsigned saturation is determined by the element type. Both arguments must have the same type.

```
let bright: u8x16 = sat_add(pixels, boost);    // clamps at 255, never wraps
let dark: u8x16 = sat_sub(pixels, reduce);     // clamps at 0, never wraps
```

### shuffle_bytes

Byte-level table lookup: each byte in `indices` selects a byte from `table`. Cross-platform. x86: SSSE3 `pshufb`. ARM: NEON `tbl`. Out-of-range indices (>15) zero the lane on both platforms.

```
let result: u8x16 = shuffle_bytes(table, indices);
```

| Signature | `(u8x16, u8x16) -> u8x16` |
|-----------|---------------------------|

### Rounding & Packing

| Intrinsic | Signature | Description | Platform |
|---|---|---|---|
| `round_f32x4_i32x4` | `(f32x4) -> i32x4` | Round-to-nearest-even. x86: `cvtps2dq`. ARM: `fcvtns`. | cross-platform |
| `pack_sat_i32x4` | `(i32x4, i32x4) -> i16x8` | Saturating narrow. x86: `packssdw`. ARM: `sqxtn`. | cross-platform |
| `pack_sat_i16x8` | `(i16x8, i16x8) -> i8x16` | Saturating narrow. x86: `packsswb`. ARM: `sqxtn`. | cross-platform |
| `round_f32x8_i32x8` | `(f32x8) -> i32x8` | Round-to-nearest-even float to integer. x86: `vcvtps2dq` (AVX2). | x86-only |
| `pack_sat_i32x8` | `(i32x8, i32x8) -> i16x16` | Saturating narrow two i32x8 into i16x16. x86: `vpackssdw` (AVX2). | x86-only |
| `pack_sat_i16x16` | `(i16x16, i16x16) -> i8x32` | Saturating narrow two i16x16 into i8x32. x86: `vpacksswb` (AVX2). | x86-only |

> **Lane order (x86):** `pack_sat_i32x8` and `pack_sat_i16x16` pack per 128-bit lane on x86, not sequentially. Output layout: `[a0-a3, b0-b3, a4-a7, b4-b7]`. This matches `maddubs_i32` lane layout -- no fixup shuffle needed in quant pipelines.

## Debug

### println

Print a value to stdout. Accepts scalars (`i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`), string literals, and vector types. Lowers to C `printf`. No format strings.

```
println(42);
println(3.14);
println("hello");
println(my_vector);
```
