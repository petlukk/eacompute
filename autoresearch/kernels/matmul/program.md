# Ea Kernel Optimization — Matrix Multiply

You are optimizing an Ea SIMD kernel for matrix multiplication: `C = A * B` where all matrices are flat row-major arrays of size `n*n`.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `matmul_f32` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match numpy `A @ B` within rtol=1e-3 (relaxed due to FP associativity).
3. One change per iteration. State your hypothesis clearly.
4. The `matmul_f32` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

Matrix multiply is compute-bound with O(n^3) ops on O(n^2) data. Loop order affects cache behavior dramatically — ikj order is cache-friendly for row-major storage.

**Width is a dimension, not a default:**
- Scalar loops (plain `while` with `f32` arithmetic)
- f32x4 (128-bit SSE)
- f32x8 (256-bit AVX2)

Wider SIMD is not always faster. Matrix multiply has high arithmetic intensity, so register blocking and cache behavior often matter more than raw vector width.

**Other dimensions to explore:**
- Register blocking (tile sizes: 1x4, 4x4, 2x8, etc.)
- Loop ordering (ijk, ikj, jki, etc.)
- Loop unrolling
- Prefetch strategies
- Accumulator count per tile

## Matrix Multiply Optimization Strategies

- **Loop reordering (ikj)**: In row-major storage, the ijk order causes strided access on B. The ikj order makes both A and C accesses sequential, dramatically improving cache behavior.
- **Register blocking**: Compute a small tile (e.g., 4x8) of C per inner loop iteration. Keeps partial sums in registers, reducing memory traffic.
- **SIMD vectorization**: Process multiple j-columns simultaneously using f32x4 or f32x8 vectors. Load contiguous rows of B and accumulate into vector registers.
- **FMA intrinsic**: `fma(a, b, acc)` computes `a * b + acc` in one instruction. Use with `splat()` to broadcast a scalar A element across a vector.
- **Prefetch**: Use `prefetch(ptr, offset)` to bring B rows into cache ahead of the inner loop.
- **Loop unrolling**: Process multiple rows of A per outer iteration to increase instruction-level parallelism.

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
