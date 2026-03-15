# Eä Kernel Optimization — Dot Product

You are optimizing an Eä SIMD kernel for dot product: `result = sum(a[i] * b[i])`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `dot_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-2 (relaxed due to FP associativity).
3. One change per iteration. State your hypothesis clearly.
4. The `dot_f32x8` function signature must not change.
5. Other variants (f32x4, foreach) may be included but are not measured.
6. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32` arithmetic)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)
- Mixed widths (e.g., f32x8 main loop + f32x4 tail)

Wider SIMD is not always faster. With two input arrays, register pressure limits how many accumulators fit. Scalar with many accumulators might beat wide SIMD with few. Let the benchmark prove which width wins.

**Other dimensions to explore:**
- Accumulator count (1, 2, 3, 4 FMA chains)
- Prefetch (none, one array, both arrays, various distances)
- Loop unrolling
- Load/compute ordering

## Dot Product Optimization Strategies

- **Multi-accumulator**: Use 2-4 independent FMA accumulator vectors to break the loop-carried dependency chain. FMA has ~4-cycle latency on modern x86 — multiple accumulators let the OOO engine interleave independent chains.
- **FMA intrinsic**: `fma(a, b, acc)` computes `a * b + acc` in one instruction, fusing multiply and add with single rounding.
- **Unrolled loads**: Load multiple vector pairs per iteration to amortize loop overhead and hide memory latency.
- **Prefetch**: Use `prefetch(ptr, offset)` to bring cache lines ahead of the load loop.
- **reduce_add for final reduction**: The `reduce_add(vec)` intrinsic reduces a vector to a scalar sum using ordered (sequential) addition.
- **reduce_add_fast for faster final reduction**: `reduce_add_fast(vec)` uses unordered tree reduction — log2(width) parallel add levels instead of width sequential adds. Significantly faster on latency-bound reductions. Float-only. Acceptable when rtol=1e-2.
- **Merge accumulators**: After the main loop, add all accumulator vectors together before the final reduce_add.

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

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
