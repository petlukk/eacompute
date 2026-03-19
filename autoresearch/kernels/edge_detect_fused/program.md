# Ea Kernel Optimization — Fused Edge Detection (blur + sobel + threshold)

You are optimizing an Ea SIMD kernel that fuses gaussian blur, sobel gradient magnitude, and thresholding into a single pass over a 5x5 stencil window. This avoids the three separate passes of the unfused pipeline.

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `edge_detect_fused` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must produce >= 95% pixel match vs the unfused C reference.
3. One change per iteration. State your hypothesis clearly.
4. The `edge_detect_fused` function signature must not change: `(input: *restrict f32, out: *mut f32, width: i32, height: i32, thresh: f32)`.
5. No dead code. No comments longer than one line.
6. The file contains other exported functions too (gaussian_blur_3x3, sobel_magnitude, threshold_f32x8, dilate_3x3). Keep them all — only optimize `edge_detect_fused`.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**SIMD width is a key dimension:**
- f32x4 (128-bit SSE) — current baseline
- f32x8 (256-bit AVX2) — doubles throughput if register pressure allows
- Mixed widths (e.g., f32x8 main loop + f32x4 or scalar tail)

**Other dimensions to explore:**
- Prefetch distance and pattern (currently +16 elements ahead on all 5 rows)
- Row buffering: precompute partial sums per row to reduce redundant loads
- Loop unrolling: process 2+ output pixels per inner iteration
- Horizontal sliding window: reuse shifted loads between adjacent x positions
- Store strategy: store vs stream_store for output (output is write-only)
- FMA usage: replace multiply-then-add with fma() where applicable
- Separable decomposition: the Gaussian weights [1,4,6,4,1] are separable — could do horizontal then vertical partial sums

## Kernel-Specific Notes

- **5x5 stencil** reading from 5 rows, 5 columns per output pixel. 25 loads per pixel (minus p22 which is unused for gradient).
- **Compute-bound at large sizes** — 24 loads + ~30 arithmetic ops per pixel. Memory bandwidth is not the bottleneck for 2048x2048.
- **Border**: skips 2 pixels on each edge (y in 2..height-2, x in 2..width-2).
- **Binary output**: result is 0.0 or 1.0 after thresholding, so small FP differences in intermediate gradient magnitude can flip pixels near the threshold boundary.
- **Algebraic fusion**: the fused kernel computes blur+sobel as a single convolution with weights derived from convolving the Gaussian kernel [1,2,1; 2,4,2; 1,2,1]/16 with Sobel [-1,0,1; -2,0,2; -1,0,1]. The combined horizontal gradient weights are [1,2,0,-2,-1; 4,8,0,-8,-4; 6,12,0,-12,-6; 4,8,0,-8,-4; 1,2,0,-2,-1]/16.

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

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
