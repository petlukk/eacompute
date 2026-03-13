# Eä Kernel Optimization — FMA

You are optimizing an Eä SIMD kernel for fused multiply-add: `result[i] = a[i] * b[i] + c[i]`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `fma_kernel_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `fma_kernel_f32x8` function signature must not change.
5. Other variants (f32x4, foreach, foreach+unroll) may be included but are not measured.
6. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32` arithmetic)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)
- f32x16 (512-bit AVX-512, requires `--avx512`)
- Mixed widths (e.g., f32x8 main loop + f32x4 tail)

Wider SIMD is not always faster. Bandwidth-bound kernels may see no benefit. Dependency-bound kernels may prefer scalar with more ILP. Let the benchmark prove which width wins.

**Other dimensions to explore:**
- Loop unrolling (1x, 2x, 4x, 8x)
- Prefetch (none, near, far, multi-level)
- Store strategy (store vs stream_store)
- Accumulator count (for reductions)
- Instruction ordering (interleaved vs batched)

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
