# Ea Kernel Optimization — First-Order IIR / Exponential Moving Average

You are optimizing an Ea kernel for exponential moving average (EMA): `y[i] = alpha * x[i] + (1 - alpha) * y[i-1]`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `ema_filter` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-4.
3. One change per iteration. State your hypothesis clearly.
4. The `ema_filter` function signature must not change: `(input: *restrict f32, output: *mut f32, len: i32, alpha: f32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

**This kernel has an inherent cross-iteration dependency** — `y[i]` depends on `y[i-1]`. This is fundamentally sequential and cannot be naively vectorized. Naive SIMD will produce incorrect results.

**This kernel will likely show 0% gain**, similar to prefix_sum. The serial dependency chain is the bottleneck, not compute throughput or memory bandwidth. A valid result is confirming that the scalar baseline is already optimal.

**Dimensions to explore:**

- **Minimize per-iteration overhead**: use `fma(alpha, input[i], beta * output[i-1])` to let LLVM emit a single FMA instruction instead of separate multiply+add
- **Prefetch**: `prefetch(input, offset)` to hide memory latency on the input stream (output is written sequentially and should be in cache)
- **Scalar unrolling**: unroll 2x or 4x with carried state — compute `y[i]` then immediately `y[i+1]` using `y[i]` as the dependency. This doesn't break the dependency chain but may improve instruction scheduling
- **Block-parallel decomposition**: process independent blocks, then fix up boundaries. Each block starts with y[-1]=0, computes its local EMA, then adjusts: for first-order IIR with coefficient beta, the correction at position k within a block is `correction * beta^k`. This requires computing powers of beta (expensive) and may not be faster than scalar
- **Multi-channel interleaving**: if the data were multi-channel (e.g., RGBA), you could SIMD across channels. With single-channel data, this doesn't apply
- **Approximation via block overlap**: for small alpha (large beta close to 1.0), the impulse response decays as beta^k. After ~log(eps)/log(beta) samples, the contribution is negligible. Process overlapping blocks and discard the overlap region. Only valid for small alpha values
- **Accept scalar is optimal**: the dependency chain latency (FMA = 4-5 cycles on modern x86) times N elements is a hard lower bound. If the scalar loop is already achieving close to one FMA per 4-5 cycles, there is nothing to optimize

**What probably won't help:**
- SIMD vectorization of the main loop — the dependency on y[i-1] prevents it
- `foreach` or `kernel` constructs — these assume independence across iterations
- Wider data types — the bottleneck is latency, not throughput

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

**Loop constructs:** while, foreach, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
