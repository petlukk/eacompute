# Eä Kernel Optimization — Particle Life (N-Body)

You are optimizing an Eä kernel for N-body particle simulation: pairwise force computation with sqrt/division, velocity integration, and position wrapping.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `particle_life_step` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-4, atol=1e-4.
3. One change per iteration. State your hypothesis clearly.
4. The `particle_life_step` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**The inner loop is the bottleneck.** For each particle i, the inner loop iterates over all N particles j computing distances, sqrt, division, and force accumulation. This is O(N²) and dominates runtime.

**Dimensions to explore:**

- **Inner loop unrolling**: process 2 or 4 j-particles per iteration to hide latency
- **rsqrt instead of sqrt+div**: `rsqrt(dist2)` gives `1/dist` directly, replacing `sqrt(dist2)` + two divisions (`dx/dist`, `dy/dist`). Approximate but much faster.
- **Strength of reduction**: precompute `force/dist` = `strength * (1 - dist*r_max_inv) / dist` = `strength * (r_max_inv_recip - 1) * rsqrt(dist2)` — fewer divides
- **Prefetch**: prefetch px[j+K], py[j+K] ahead of use in the inner loop
- **Accumulator strategy**: multiple (fx0,fy0), (fx1,fy1) accumulators to break dependency chains
- **Loop splitting**: separate the force computation (read-only on positions) from the velocity/position update
- **Branch elimination**: replace `if dist2 > 0 && dist2 < r_max2` with branchless select/mask

**What probably won't help:**
- SIMD vectorization of the inner loop: the conditional branch (`if dist2 > 0 && dist2 < r_max2`) and gather from `matrix[ti * num_types + types[j]]` make this hard to vectorize. Focus on scalar optimizations first.
- Tiling/blocking: the interaction matrix is tiny (6×6), not worth blocking for cache.

## Particle Life Specifics

- **N ranges from 500 to 2000** in the benchmark. Inner loop is 250K to 4M iterations per step.
- **sqrt + 2 divisions per interaction** is the critical path. Replacing with rsqrt saves ~2 divisions.
- **r_max = 80.0, size = 800.0** — about 1% of pairs interact (within r_max). The branch is mostly not-taken.
- **The outer loop has a loop-carried dependency** (px[i], py[i] update after force). Cannot parallelize across i without changing semantics.
- **Types array access** `types[j]` in inner loop: random i32 load, likely cached since N ≤ 2000.

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

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
