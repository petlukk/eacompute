# Eä Kernel Optimization — SAXPY

You are optimizing an Eä SIMD kernel for SAXPY: `y[i] = a * x[i] + y[i]`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `saxpy_f32x8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `saxpy_f32x8` function signature must not change.
5. Other variants (f32x4, foreach) may be included but are not measured.
6. No dead code. No comments longer than one line.

## The Eä Way (optimization priority)

1. Use the widest SIMD type available (f32x8 > f32x4 > scalar on AVX2)
2. Write explicit vector loops with load/store — don't rely on auto-vectorization
3. Use explicit tail handling for remainder elements
4. Prefer intrinsics (fma, reduce_add) over manual scalar equivalents
5. foreach/unroll are fallbacks, not first choices

## SAXPY Optimization Strategies

- **Unrolled loop**: Process multiple vectors per iteration to amortize loop overhead and expose ILP. Unlike reduction, SAXPY has no loop-carried dependency — each iteration is independent.
- **stream_store**: For large arrays where output won't be re-read soon, non-temporal stores bypass the cache hierarchy and avoid read-for-ownership overhead.
- **Prefetch**: Use `prefetch(ptr, offset)` to bring input cache lines ahead. SAXPY reads two arrays (x, y) and writes one (y), so the memory subsystem juggles 3 streams.
- **splat once**: Broadcast the scalar `a` to a vector outside the loop with `splat(a)`.
- **FMA intrinsic**: `fma(va, vx, vy)` computes `a * x + y` in one instruction.
- **Note**: SAXPY is bandwidth-bound — arithmetic is cheap (one FMA per 8 elements). Optimization focuses on memory access patterns, not compute.

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
