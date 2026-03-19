# Eä Kernel Optimization — 3x3 Int8 Convolution

You are optimizing an Eä int8 depthwise convolution kernel: 3x3 kernel, NHWC layout, i32 accumulator.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `conv2d_3x3_u8i8_safe` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include both the i16 and i32 variants (the benchmark measures the i32 variant).

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must exactly match the C reference (integer exact).
3. One change per iteration. State your hypothesis clearly.
4. The `conv2d_3x3_u8i8_safe` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Key intrinsics:**
- `maddubs_i32(u8x16, i8x16) -> i32x4`: unsigned×signed multiply-add, 16 bytes → 4 int32s. This is the core compute primitive (pmaddubsw + pmaddwd fused).
- `maddubs_i16(u8x16, i8x16) -> i16x8`: faster but can overflow for large C_in.

**Dimensions to explore:**
- Loop unrolling: process multiple output pixels per iteration (col unroll)
- Channel loop unrolling: process more than 32 channels per inner step
- Weight reuse: same 9×C_in weights used for every output pixel
- Row unrolling: process multiple output rows simultaneously
- Prefetch: input rows or weight data ahead of use
- Accumulator width: i32x4 vs i32x8 (wider SIMD for accumulation)
- For-loop vs while-loop: `for` may unroll differently

## Conv2d-Specific Notes

- **C_in is always a multiple of 32** (benchmark uses C_in=64).
- **9 weight positions** per output pixel, each dot-producing C_in channels.
- **Weight-stationary**: weights are small (9×64 = 576 bytes), fit in L1. Input is large.
- **NHWC layout**: channels are contiguous, spatial positions are strided.
- **The i16 variant** (`conv2d_3x3_u8i8`) must also be included in the file but is not benchmarked.

## Available Eä Features

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
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min (reduce_add_fast is float-only, uses unordered tree reduction — faster than reduce_add but non-deterministic FP order)
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16(u8x16, i8x16)->i16x8, maddubs_i32(u8x16, i8x16)->i32x4

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
