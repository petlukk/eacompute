# Eä Kernel Optimization — Video Anomaly Detection (Fused)

You are optimizing an Eä kernel for video frame anomaly detection: fused diff + threshold + count in a single pass. Two input frames, one scalar output (anomaly pixel count).
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `anomaly_count_fused` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within ±1.0.
3. One change per iteration. State your hypothesis clearly.
4. The `anomaly_count_fused` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel is already SIMD-optimized with dual f32x8 accumulators, prefetch, and branchless select. It reads two input arrays and produces a scalar count.

**This kernel is likely bandwidth-bound.** Two loads per element (a[i], b[i]) with minimal compute (sub, abs, compare, accumulate). At 4M floats, both arrays are 16MB each — exceeds L2.

**Dimensions to explore:**
- **stream_store vs store**: not applicable (no output array written)
- **Wider unrolling**: process 32 or 64 floats per iteration instead of 16
- **Prefetch distance**: currently +64, try +128 or +256 for larger arrays
- **Reduce accumulator overhead**: accumulate as integer (count) instead of float
- **Integer pipeline**: convert to i32 comparison — avoid float select overhead
- **Loop structure**: `for` vs `while`, `unroll(N)` hint

**What probably won't help:**
- Algorithmic changes — the operation is fundamentally load-compare-count
- SIMD width increase beyond f32x8 without AVX-512
- Removing the scalar tail — only runs for remainder elements

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
- Reduction: reduce_add, reduce_max, reduce_min
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
