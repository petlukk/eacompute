# Ea Kernel Optimization — Int8 Dot Product (maddubs)

You are optimizing an Ea kernel for quantized int8 dot product: `dot_u8i8` computes the dot product of n uint8 activations and n int8 weights, returning an i16 sum. This is the inner loop of int8 quantized inference (TFLite, ONNX Runtime, XNNPACK).
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `dot_u8i8` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference exactly.
3. One change per iteration. State your hypothesis clearly.
4. The `dot_u8i8` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel uses `maddubs_i16(u8x16, i8x16) -> i16x8` which maps to SSSE3 `pmaddubsw`. Each call processes 16 byte pairs into 8 i16 results (pairwise multiply-add). Currently uses a single accumulator.

**This kernel is compute-bound for small/medium n and bandwidth-bound for large n.** The maddubs instruction has high throughput but the single accumulator creates a dependency chain.

**Dimensions to explore:**
- **Unrolling**: process 32, 48, or 64 bytes per iteration with independent accumulators to break dependency chains
- **Multiple accumulators**: 2x or 4x i16x8 accumulators, reduce_add at the end
- **Prefetch**: `prefetch(act, offset)` and `prefetch(wt, offset)` — two input streams
- **Loop structure**: `while` vs `for`, `unroll(N)` hint
- **Widen to i32 accumulators**: use `maddubs_i32` to avoid i16 overflow risk and enable larger n, but trades throughput

**i16 overflow risk:** Each maddubs output lane can be up to 127*255*2 = 64770, close to i16 max (32767) — actually exceeds it! With accumulation, overflow happens quickly. For large n, widening to i32 accumulators (`maddubs_i32`) may be both safer and enable more aggressive accumulation strategies. The benchmark uses small values (0-3, -2 to 2) to avoid overflow in correctness checks.

**What probably won't help:**
- Changing SIMD width beyond 16 bytes without AVX2 i8x32 maddubs (not available)
- Algorithmic changes — the operation is fundamentally load-multiply-add-reduce
- Removing the second accumulator declaration (acc1 is already unused dead code)

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
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min (reduce_add_fast is float-only, uses unordered tree reduction)
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
