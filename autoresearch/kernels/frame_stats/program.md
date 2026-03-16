# Eä Kernel Optimization — Frame Stats (Triple Reduction)

You are optimizing an Eä kernel for single-pass frame statistics: min, max, and sum computed simultaneously over a float array. One input array, three scalar outputs.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `frame_stats` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Min/max must match exactly (within 1e-3). Sum must match within relative tolerance 1e-3.
3. One change per iteration. State your hypothesis clearly.
4. The `frame_stats` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel currently uses dual f32x8 accumulators (2 each for min, max, sum = 6 vector registers) with prefetch. This is a **reduction kernel** with three simultaneous reductions.

**Key bottleneck analysis:**
- Each iteration does: 2 loads, 2 adds, 2 mins, 2 maxs = 8 vector ops on 2 loads
- At small sizes (L2/L3 cache) this is **latency-bound** — more accumulators help
- At large sizes (4096² = 64MB, main memory) this is **bandwidth-bound** — extra register pressure hurts
- **The benchmark scores on the LARGEST size (4096²).** Optimizations that win in cache but lose at real-world scale will be rejected.
- Be very careful with register pressure: 3 reductions × N accumulators must fit in 16 YMM registers

**Dimensions to explore:**
- **More accumulators**: 4 per reduction (12 registers total) — breaks dependency chains further
- **Wider unrolling**: process 32 or 48 floats per iteration
- **Prefetch distance**: currently +64 elements, try +128 or +256
- **reduce_add_fast**: use unordered reduction for the sum (faster horizontal reduction)
- **Register pressure**: with 3 reductions × N accumulators, register spilling is a real risk
- **Remove prefetch**: hardware prefetch may be sufficient for sequential access

**What probably won't help:**
- stream_store (no output array written in the hot loop)
- Algorithmic changes — it's fundamentally load-reduce
- SIMD width beyond f32x8 without AVX-512

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
