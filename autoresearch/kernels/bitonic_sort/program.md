# Ea Kernel Optimization — In-Place f32 Sort (Bitonic / Sorting Networks)

You are optimizing an Ea kernel that sorts an f32 array in-place. The baseline uses scalar insertion sort on blocks of 8, then bottom-up merge passes.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `sort_f32` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must be sorted in non-decreasing order and contain the same elements as the input.
3. One change per iteration. State your hypothesis clearly.
4. The `sort_f32` function signature must not change: `export func sort_f32(data: *mut f32, len: i32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel currently uses scalar insertion sort on blocks of 8, then merge passes. This is O(n log n) but with high constant factors.

**Sorting is compute-bound, not bandwidth-bound.** The data is read and written many times during sorting. Reducing comparison/swap count and branch mispredictions is key.

**Dimensions to explore:**
- **Larger sorted blocks**: Sort blocks of 16 or 32 using an unrolled sorting network before merging. Fewer merge passes needed.
- **SIMD min/max for compare-and-swap**: `min(a, b)` and `max(a, b)` give you a branchless compare-and-swap pair. Load two f32x8 vectors, apply min/max to get sorted pairs.
- **Branchless scalar sort**: Replace `if` comparisons with min/max on scalar floats to avoid branch misprediction.
- **Better merge strategy**: The current in-place merge uses rotation (O(n) per element). A merge that uses temporary storage or a smarter in-place technique would help.
- **Selection sort for small blocks**: May have fewer data movements than insertion sort.
- **Unrolled sorting networks**: For fixed-size blocks (8, 16, 32), a hardcoded network of min/max pairs is optimal and branchless.

**Known Ea limitations for sorting:**
- No shuffle/permute intrinsics for rearranging elements within a SIMD vector
- No insert/extract element intrinsics for SIMD vectors
- No indirect indexing into SIMD registers
- SIMD min/max works on corresponding lanes of two vectors — useful for "compare-and-swap" of two f32x8 blocks (sorted merge network between registers) but NOT for sorting within a single register
- The main SIMD opportunity: load two sorted f32x8 blocks, use min/max to produce the low 8 and high 8 of a bitonic merge step

**What probably won't help:**
- Trying to sort within a single f32x8 register (no cross-lane shuffle)
- Radix sort (needs bit manipulation Ea may not support well)
- Quicksort (branch-heavy, poor cache behavior on random data)

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

**Scalar min/max:** `min(a, b)` and `max(a, b)` work on scalar f32 values too — branchless compare-and-swap.

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
