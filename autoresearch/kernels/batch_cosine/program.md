# Eä Kernel Optimization — Batch Cosine Similarity

You are optimizing an Eä SIMD kernel for batch cosine similarity over a vector database.
For each of `n_vecs` database vectors, compute `cosine(query, db[v]) = dot(q,d) / (||q|| * ||d||)`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `batch_cosine` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-3.
3. One change per iteration. State your hypothesis clearly.
4. The `batch_cosine` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**The kernel has two parts:**
1. **Query norm precomputation** (runs once): sum-of-squares over `dim` elements, then sqrt. Minor cost.
2. **Per-vector inner loop** (runs `n_vecs` times): simultaneously computes dot product and database vector norm-squared in a single pass over `dim` elements. This is the hot path.

**Dimensions to explore:**
- Accumulator count (2, 4, 6 — currently 4 accumulators: dot0, dot1, sq0, sq1)
- Loop unrolling (process 32 or 48 elements per iteration instead of 16)
- Prefetch distance (currently prefetch next vector's start at base+dim)
- Load/compute ordering (batched vs interleaved)
- rsqrt instead of sqrt for db_norm (approximate but faster, acceptable at rtol=1e-3)
- Precompute 1/query_norm to replace division with multiplication in the final step
- SIMD width (f32x4 vs f32x8 — wider isn't always better with 4 accumulators)

## Batch Cosine Optimization Strategies

- **Fused dot+norm**: The current kernel computes dot product and db norm in a single pass. This halves memory bandwidth vs two separate passes. Preserve this fusion.
- **Multi-accumulator**: Independent accumulator chains break loop-carried dependencies. FMA has ~4-cycle latency — need 4+ independent chains for full throughput.
- **Prefetch next vector**: At dim=768, each vector is 3KB. Prefetching the next vector's start hides the stride jump latency.
- **rsqrt approximation**: `rsqrt(x)` is ~4x faster than `1/sqrt(x)` and accurate to ~1e-3. Use for db_norm.
- **Reciprocal precomputation**: Compute `inv_query_norm = 1.0 / query_norm` once, then multiply instead of dividing per vector.
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
