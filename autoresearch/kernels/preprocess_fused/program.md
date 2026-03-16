# Ea Kernel Optimization — MNIST Preprocess Fused (Normalize + Standardize + Clip)

You are optimizing an Ea kernel for MNIST image preprocessing: fused normalize + standardize + clip in a single pass. One f32 input array (pixel values 0-255), one f32 output array. The kernel reads each element, scales it (multiply by 1/255), subtracts the mean, multiplies by inverse std, then clamps to [0.0, 1.0].
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `preprocess_fused` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference with atol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `preprocess_fused` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel is a fused pipeline: scale, subtract, multiply, two-sided clamp. It reads one f32 array and writes one f32 array. Currently uses f32x8 SIMD with a scalar tail.

**This kernel is bandwidth-bound at large sizes.** One load + one store per element, with moderate compute (3 arithmetic ops + 2 selects). At 47M floats (28*28*60000), the input array is ~180MB — far exceeds L3 cache.

**Dimensions to explore:**
- **FMA fusion**: replace `(v * scale - mean)` with `fma(v, scale, -mean)` to fuse multiply-add
- **Unrolling**: process 16 or 32 floats per iteration instead of 8 (dual f32x8 registers)
- **Prefetch distance**: add `prefetch(input, i + offset)` to hint the memory subsystem
- **stream_store vs store**: output is write-only, `stream_store` avoids read-for-ownership
- **Combine scale+inv_std**: precompute `scale * inv_std` and `mean * inv_std` to reduce ops per element
- **Loop structure**: `for` vs `while`, `unroll(N)` hint

**What probably won't help:**
- u8 input widening — the caller pre-converts to f32 before calling
- Algorithmic changes — the operation is fundamentally load-compute-store
- SIMD width increase beyond f32x8 without AVX-512
- Removing the scalar tail — only runs for remainder elements

## Benchmark Details

Scored on the LARGEST size: N=47,040,000 (28*28*60000 = full MNIST dataset). This ensures the benchmark measures memory-bandwidth behavior, not cache-local compute. Smaller sizes (78K, 784K, 7.8M) are also measured for breakdown but do not affect the score.

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
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min (reduce_add_fast is float-only, uses unordered tree reduction — faster than reduce_add but non-deterministic FP order)
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
