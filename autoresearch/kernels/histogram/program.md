# Ea Kernel Optimization — Histogram

You are optimizing an Ea kernel for histogram: count occurrences of values 0..255 in an i32 array.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `histogram_i32` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the reference exactly (integer comparison, atol=0).
3. One change per iteration. State your hypothesis clearly.
4. The `histogram_i32` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Width: scalar is the natural choice (random access pattern defeats SIMD).**
Histogram is random-access bound — each data[i] indexes into hist at an unpredictable location. SIMD is very difficult because gather/scatter with conflicts is slow. Scalar with cache-aware tricks (multiple histogram copies to reduce conflicts) is likely optimal.

**Dimensions to explore:**

- **Multi-histogram**: Use 2-4 separate histogram arrays (each 256 entries), process different elements into each, merge at end. This reduces store-to-load forwarding stalls when consecutive data values map to the same bin.
- **Loop unrolling**: Process 2-4 elements per iteration to expose instruction-level parallelism in the load pipeline.
- **Prefetch**: Use `prefetch(data, offset)` to bring data cache lines ahead of the current position.
- **Note**: Histogram is random-access bound — each data[i] indexes into hist at an unpredictable location. SIMD is very difficult because gather/scatter with conflicts is slow. Scalar with cache-aware tricks (multiple histogram copies to reduce conflicts) is likely optimal.

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

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
