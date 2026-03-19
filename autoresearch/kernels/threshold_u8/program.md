# Ea Kernel Optimization — Pixel Threshold (u8x16)

You are optimizing an Ea kernel for binary pixel thresholding: for each byte, output 255 if > threshold else 0. Operates entirely in uint8 space with no float conversion, using u8x16 SIMD vectors. Two pointer args (src, dst), one scalar threshold, one length.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `threshold_u8x16` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must be an exact byte match vs the C reference.
3. One change per iteration. State your hypothesis clearly.
4. The `threshold_u8x16` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel uses u8x16 compare + select per 16-byte chunk. Each iteration: 1 load, 1 compare, 1 select, 1 store. Very low compute per byte.

**This kernel is bandwidth-bound at large sizes.** One load + one store per element (2 bytes of memory traffic per input byte). At 4096x4096 (16MB input + 16MB output), data far exceeds L2/L3. The memory subsystem is the bottleneck, not ALU.

**Dimensions to explore:**
- **SIMD width**: u8x16 (128-bit) vs u8x32 (256-bit) — doubles throughput per iteration
- **Unrolling**: process 2x or 4x vectors per iteration to hide latency and amortize loop overhead
- **Prefetch**: prefetch(src, offset) to warm cache lines ahead of the read position
- **stream_store**: non-temporal stores bypass cache for write-only output, freeing cache for reads
- **Loop structure**: `for` vs `while`, `unroll(N)` hint

**What probably won't help:**
- Float conversion — the whole point is staying in u8 space
- Complex arithmetic — there is almost none; memory is the bottleneck
- Multiple accumulators — there is no reduction, just load-compare-store
- Algorithmic changes — the operation is fundamentally per-element

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

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
