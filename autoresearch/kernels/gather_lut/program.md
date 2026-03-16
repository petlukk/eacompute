# Ea Kernel Optimization -- LUT Apply via Gather

You are optimizing an Ea kernel that applies a 256-entry float lookup table to an array of byte indices: `out[i] = lut[data[i]]`. This tests the SIMD `gather` intrinsic for indirect loads.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `lut_apply_gather` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must be an exact float match with the C reference.
3. One change per iteration. State your hypothesis clearly.
4. The `lut_apply_gather` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel uses `gather(lut, indices_i32x8)` to do 8 indirect loads at once. However, **gather is often slower than scalar loops on x86** due to gather microarchitecture limitations (each gather decomposes into serial cache lookups internally on many CPUs).

**Key question: is SIMD gather actually faster than scalar indexing for this workload?**

The LUT is only 1KB (256 floats) and fits entirely in L1 cache. The data array is large (up to 16M elements). This means the bottleneck is reading the byte indices sequentially and writing f32 output -- the LUT lookups themselves are cheap (L1 hits).

**Dimensions to explore:**
- **Scalar fallback**: remove gather entirely, use a scalar loop. On many CPUs this is faster because scalar loads from L1 are 4-5 cycle latency vs gather's 20+ cycles
- **Partial gather**: gather 4 at a time (i32x4) instead of 8, may have lower latency
- **Unrolled scalar**: unroll scalar loop 4x or 8x for better ILP
- **Mixed approach**: load u8 bytes with SIMD, extract indices, do scalar LUT lookups, pack results back into f32x8 for vectorized store
- **Prefetch on data array**: the data array is the streaming access pattern
- **Stream store**: output is write-only, stream_store avoids cache pollution
- **Loop unrolling**: process 16 or 32 elements per iteration with multiple gather ops

**What probably won't help:**
- Prefetching the LUT -- it's 1KB, permanently in L1
- Wider SIMD (f32x16) without AVX-512
- Algorithmic changes -- the operation is fundamentally index-lookup-store

## Available Ea Features

**SIMD types:**

| 128-bit | 256-bit | 512-bit |
|---------|---------|---------|
| f32x4   | f32x8   | f32x16  |
| f64x2   | f64x4   |         |
| i32x4   | i32x8   | i32x16  |
| i16x8   | i16x16  |         |
| i8x16   | i8x32   |         |
| u8x16   | u8x32   |         |

**Intrinsics:**
- Memory: load, load_f32x4, load_f32x8, load_i32x8 (typed variants for all vector types), store, stream_store, gather, scatter, prefetch(ptr, offset)
- Arithmetic: fma, sqrt, rsqrt, exp, min, max
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**gather signature:** `gather(base_ptr, indices_i32x8)` returns f32x8. Indices are i32 element offsets.
**widen_u8_i32x8 signature:** `widen_u8_i32x8(u8x16_vec)` returns i32x8 (lower 8 bytes zero-extended).

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

**Casting:** `to_i32(x)`, `to_f32(x)`, etc. for scalar type conversions.

**Pointer indexing:** `ptr[i]` where `i` is any integer type (u8, i32, etc.). Example: `lut[data[i]]` works when data is `*u8` because u8 is a valid index type.

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
