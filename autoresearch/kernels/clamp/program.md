# Eä Kernel Optimization — Clamp

You are optimizing an Eä SIMD kernel for clamping: `result[i] = clamp(data[i], lo, hi)`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `clamp_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `clamp_f32x8` function signature must not change.
5. Other variants (f32x4, foreach) may be included but are not measured.
6. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32` + if/else clamping)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)
- Mixed widths (e.g., f32x8 main loop + f32x4 tail)

Wider SIMD is not always faster. Clamp is compute-light — loop overhead and prefetch effects may dominate. Let the benchmark prove which width wins.

**Other dimensions to explore:**
- Clamp implementation: min/max vs select-based
- Loop unrolling (1x, 2x, 4x, 8x)
- Prefetch (none, input, output, both)
- Store strategy (store vs stream_store)

## Clamp Optimization Strategies

- **min/max intrinsics**: `min(max(v, lo), hi)` is the canonical clamp — two instructions, no masking needed. Compare to `select`-based approach which generates comparison masks + blend.
- **Unrolled loop**: Process multiple vectors per iteration. Clamp has no loop-carried dependency — each element is independent.
- **stream_store**: For large arrays where output won't be re-read soon, non-temporal stores bypass read-for-ownership.
- **Prefetch**: Use `prefetch(ptr, offset)` to bring input cache lines ahead.
- **Note**: Clamp is a mix of compute and bandwidth. The compute is light (2 ops per vector) but generates mask instructions. The interesting question is whether min/max or select produces faster code.

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
