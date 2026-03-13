# Eä Kernel Optimization — Prefix Sum (Inclusive Scan)

You are optimizing an Eä kernel for inclusive prefix sum: `out[i] = sum(data[0..=i])`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `prefix_sum_f32` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-2 (relaxed — long prefix sum accumulates FP error).
3. One change per iteration. State your hypothesis clearly.
4. The `prefix_sum_f32` function signature must not change: `(data: *f32, out: *mut f32, len: i32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Important note:** Prefix sum has an inherent sequential dependency — each output depends on the previous. SIMD approaches must use clever shuffle/add patterns within a vector, then propagate across vectors. Pure scalar with good ILP may be competitive.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32` arithmetic)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)

Wider SIMD is not always faster. The sequential dependency in prefix sum means SIMD requires multi-step in-register scans plus cross-vector propagation. Let the benchmark prove which width wins.

**Scalar approaches:**
- Simple loop (baseline)
- Unrolled scalar (manually unroll 2x, 4x, 8x with carried sum)
- Software pipelining (overlap loads with adds)

**SIMD approaches:**
- Block-wise scan: compute prefix sum within fixed-size blocks using SIMD shuffle/add patterns, then propagate block totals across blocks
- In-register scan: for f32x4, use 3 shuffle+add steps to compute prefix sum within a vector; for f32x8, use a sequence of permute+add steps
- Hybrid: SIMD for partial sums within blocks, scalar propagation of block boundaries

**Prefetch strategies:**
- None (baseline)
- Near prefetch: `prefetch(data, offset)` one cache line ahead
- Far prefetch: prefetch multiple cache lines ahead

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
- Memory: load, store, stream_store, gather, scatter, prefetch(ptr, offset)
- Arithmetic: fma, sqrt, rsqrt, exp, min, max
- Reduction: reduce_add, reduce_max, reduce_min
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
