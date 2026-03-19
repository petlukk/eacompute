# Ea Kernel Optimization — Tokenizer Prepass (Fused)

You are optimizing an Ea kernel for tokenizer preprocessing: a fused classify + lowercase + boundary detection kernel operating entirely in u8x16 SIMD space. One input array (text bytes), three output arrays (classes, lowered, boundaries).
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `text_prepass_fused` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. All three output arrays must exactly match the C reference byte-for-byte.
3. One change per iteration. State your hypothesis clearly.
4. The `text_prepass_fused` function signature must not change: `(text: *u8, flags: *mut u8, lower: *mut u8, boundaries: *mut u8, len: i32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel is already SIMD-optimized using u8x16 vectors. It classifies each byte (whitespace=1, letter=2, digit=4, punct=8, nonascii=16), lowercases A-Z, and detects boundaries where class changes between adjacent bytes. The fused version does all three in a single pass but duplicates classification for both current and previous bytes.

**This kernel is compute-bound via fusion.** Each iteration performs classification twice (current + previous), lowercase once, and boundary detection once. All three operations are independent per position. The bottleneck is instruction count, not memory bandwidth.

**Dimensions to explore:**
- **Carry previous flags**: instead of recomputing classification of prev, store last vector of curr_flags and shift/reload for next iteration boundary check. This halves classification work.
- **Unrolling**: process 32 bytes per iteration (two u8x16 loads). Amortizes loop overhead, enables more ILP.
- **Prefetch**: `prefetch(text, offset)` to warm cache lines ahead of the read position.
- **Instruction reordering**: interleave classification and lowercase ops to maximize execution port utilization.
- **Simplified classification**: reduce the number of select/compare ops by reordering range checks or combining conditions.
- **Stream stores**: `stream_store` for the three output arrays since they are write-only and will not be read back soon.

**What probably won't help:**
- Wider SIMD types (u8x32) would help but depend on AVX2; u8x16 is the portable target.
- Algorithmic changes — the three operations are fundamentally per-byte.
- Removing the scalar tail — only runs for remainder elements.
- Removing unfused kernels — they must remain in the file.

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
- Memory: load, load_u8x16, store, stream_store, prefetch(ptr, offset)
- Arithmetic: min, max
- Reduction: reduce_add, reduce_max, reduce_min
- Construction: splat, select
- Conversion: widen_u8_i32x4, widen_u8_i32x8, widen_u8_i32x16, widen_u8_f32x4, widen_u8_f32x8

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.<=`, `.>=`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
