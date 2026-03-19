# Ea Kernel Optimization — Brightness/Contrast (u8 -> f32 Pipeline)

You are optimizing an Ea kernel that loads u8 pixel data, widens to f32, applies brightness+contrast adjustment with clamping, and stores f32 output. This tests the full widen-compute-store pipeline common in image processing.

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `brightness_contrast` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within ±0.002.
3. One change per iteration. State your hypothesis clearly.
4. The `brightness_contrast` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel widens u8 input to f32 via widen_u8_f32x4, applies scale+bias via FMA, clamps with min/max, and stores f32 output. It processes 8 elements per iteration (two f32x4 groups from one u8x16 load).

**The bottleneck is likely the widen/shuffle overhead and store bandwidth**, not the arithmetic. Each iteration does 1 u8x16 load, 1 shuffle, 2 widens, 2 FMAs, 4 min/max, 2 stores.

**Dimensions to explore:**
- **Wider widen**: use widen_u8_f32x8 to get 8 floats at once, avoiding the shuffle+second widen. Process via f32x8 arithmetic
- **Loop unrolling**: process 16 or 32 u8 per iteration (2-4x unroll) to reduce loop overhead and improve ILP
- **Prefetch**: add prefetch(input, offset) for upcoming cache lines
- **stream_store**: output is write-only, stream_store avoids polluting cache
- **for-loop with step**: `for i in 0..len step N` instead of while loop
- **Combined unroll + wider SIMD**: load u8x16, widen to 2x f32x8, process, store
- **Prefetch distance tuning**: try +64, +128, +256 for different array sizes

**What probably won't help:**
- Algorithmic changes — the operation is load-widen-compute-clamp-store
- Reducing compute — FMA+clamp is already minimal
- Integer arithmetic path — output is f32, must use float pipeline

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
- Construction: splat, select, shuffle
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Key Insight: widen_u8_f32x8

The intrinsic `widen_u8_f32x8(chunk)` widens the low 8 bytes of a u8x16 to an f32x8 vector in one call. This avoids the shuffle+two-widen pattern and enables f32x8 (256-bit) arithmetic. This is likely the single biggest optimization available.

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
