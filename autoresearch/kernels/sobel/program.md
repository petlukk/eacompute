# Eä Kernel Optimization — Sobel Edge Detection

You are optimizing an Eä SIMD kernel for Sobel edge detection: L1 gradient magnitude `|Gx| + |Gy|`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `sobel` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the NumPy reference within rtol=1e-4.
3. One change per iteration. State your hypothesis clearly.
4. The `sobel` function signature must not change: `(input: *restrict f32, out: *mut f32, width: i32, height: i32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32`)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)
- Mixed widths (e.g., f32x8 main loop + f32x4 or scalar tail)

**Other dimensions to explore:**
- Loop unrolling: process multiple output rows per iteration
- Load reuse: overlapping stencil windows share 6 of 9 values between adjacent pixels
- Prefetch tuning: rows ahead of current position
- Store strategy: store vs stream_store for output
- Absolute value: `select(v .< zero, zero .- v, v)` vs rewriting to avoid abs

## Sobel-Specific Notes

- **9 loads per output pixel** from 3 rows. Adjacent pixels share 6 loads — horizontal sliding window can reuse data.
- **No loop-carried dependency** — each output pixel is independent. Pure throughput-bound.
- **Overlapping loads are unaligned** — `load(input, row + x - 1)` and `load(input, row + x + 1)` are offset by 1 element from `load(input, row + x)`.
- **Border handling**: skip row 0, row height-1, col 0, col width-1. The function signature includes width/height.
- **2D access pattern**: `input[y * width + x]`. Row stride = width.

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
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
