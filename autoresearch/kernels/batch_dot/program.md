# Eä Kernel Optimization — Batch Dot Product

You are optimizing an Eä SIMD kernel for batch dot product over a vector database.
For each of `n_vecs` database vectors, compute `dot(query, db[v])`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `batch_dot` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-3.
3. One change per iteration. State your hypothesis clearly.
4. The `batch_dot` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**The kernel structure:**
- Outer loop over `n_vecs` database vectors
- Inner loop: dual-accumulator FMA dot product over `dim` elements
- Single `reduce_add` at the end of each vector

**Dimensions to explore:**
- Accumulator count (2, 4 — currently 2 FMA chains: acc0, acc1)
- Loop unrolling (process 32 or 48 elements per iteration instead of 16)
- Prefetch strategy (currently prefetch next vector start at base+dim; try multiple points to cover the 1.5-3KB stride jump between vectors)
- Load/compute ordering (batched vs interleaved)
- SIMD width (f32x4 vs f32x8)
- reduce_add_fast for the final horizontal sum

## Batch Dot Optimization Strategies

- **Multi-accumulator**: Independent FMA chains break loop-carried dependencies. FMA has ~4-cycle latency — need 4+ independent chains for full throughput.
- **Prefetch next vector**: At dim=768, each vector is 3KB. A single prefetch only covers 1 cache line (64 bytes). Multiple prefetch points at different offsets into the next vector can hide the stride jump latency.
- **Loop unrolling**: Processing more elements per iteration reduces branch overhead.
- **reduce_add_fast for final reduction**: `reduce_add_fast(vec)` uses unordered tree reduction — log2(width) parallel add levels instead of width sequential adds. Faster than `reduce_add` but non-deterministic FP order. Acceptable at rtol=1e-3.

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

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
