# Ea Kernel Optimization -- Conditional Masked Scatter-Add

You are optimizing an Ea kernel for conditional scatter-add: given arrays of values, indices, and a float mask, scatter-add values to output positions where mask exceeds a threshold. `if mask[i] > threshold: output[indices[i]] += values[i]`

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `scatter_add` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference exactly (same order of operations, exact float equality).
3. One change per iteration. State your hypothesis clearly.
4. The `scatter_add` function signature must not change.
5. No dead code. No comments longer than one line.

## Key Constraint: Sequential Dependencies

This kernel is fundamentally sequential due to potential index conflicts -- two elements may scatter to the same output position. The order of accumulation matters for exact float reproducibility. Any SIMD approach must preserve the sequential semantics.

## Strategy Space

The baseline is a simple scalar conditional scatter. Random access pattern (indices in [0, 10000)) means cache misses dominate at large sizes.

**Dimensions to explore:**
- **Branchless scalar**: always compute `output[idx] += values[i] * (mask[i] > threshold ? 1.0 : 0.0)` -- eliminates branch misprediction but does unnecessary memory ops
- **Block-skip with SIMD mask check**: load 8 mask values as f32x8, compare against threshold. If all-zero (no elements pass), skip the entire block. Saves random-access stores for blocks where no element passes (~50% at threshold=0.5, but blocks of 8 consecutive masked-out elements are rare)
- **Prefetch indices**: prefetch output[indices[i+K]] to warm cache for upcoming scatter writes
- **Gather-modify-scatter (AVX-512 only)**: use gather to load output[indices], add values, scatter back. Requires AVX-512 and careful handling of duplicate indices within a vector
- **Histogram-like binning**: sort or bucket by output index to improve cache locality. Major code complexity.
- **Loop unrolling**: unroll the scalar loop to expose ILP in the branch predictor and load pipeline

**What probably won't help much:**
- Pure SIMD vectorization -- the random scatter pattern is inherently scalar
- Stream stores -- output access is random, not sequential
- Wider SIMD types -- the bottleneck is random memory access, not compute

**Realistic expectations:** This kernel is memory-latency-bound due to random access. Gains over the scalar baseline will likely be modest (5-20%). The best strategies focus on reducing branch overhead and improving cache behavior.

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
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min (reduce_add_fast is float-only, uses unordered tree reduction)
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
