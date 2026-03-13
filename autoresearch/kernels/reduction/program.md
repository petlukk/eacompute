# Eä Kernel Optimization — Horizontal Reduction (Sum)

You are optimizing an Eä SIMD kernel for horizontal sum: `total = sum(data[0..len])`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `sum_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-2 (relaxed due to FP associativity).
3. One change per iteration. State your hypothesis clearly.
4. The `sum_f32x8` function signature must not change.
5. Other variants (f32x4, foreach, unroll, max, min) may be included but are not measured.
6. No dead code. No comments longer than one line.

## The Eä Way (optimization priority)

1. Use the widest SIMD type available (f32x8 > f32x4 > scalar on AVX2)
2. Write explicit vector loops with load/store — don't rely on auto-vectorization
3. Use explicit tail handling for remainder elements
4. Prefer intrinsics (fma, reduce_add, reduce_max, reduce_min) over manual equivalents
5. foreach/unroll are fallbacks, not first choices

## Reduction-Specific Optimization Strategies

- **Multi-accumulator**: Use 2-4 independent accumulator vectors to break the loop-carried dependency chain and expose ILP on superscalar CPUs. Merge accumulators after the main loop.
- **Unrolled loads**: Load multiple vectors per iteration to hide memory latency and increase throughput.
- **Prefetch**: Use `prefetch(ptr, offset)` to bring cache lines ahead of the load loop.
- **Wider SIMD**: f32x8 processes 8 elements per iteration vs 4 for f32x4.
- **Minimize scalar tail**: Handle as few elements as possible in the scalar tail loop.
- **reduce_add for final reduction**: The `reduce_add(vec)` intrinsic efficiently reduces a vector to a scalar sum.

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
