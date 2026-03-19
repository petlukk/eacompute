# Ea Kernel Optimization — Fused Parse + Hash Table Aggregate (1BRC)

You are optimizing an Ea kernel for the 1 Billion Row Challenge: fused text parsing + hash table aggregation in a single pass. For each line in a text buffer, the kernel parses a station name and temperature, hashes the name, and inserts/updates an open-addressing hash table (1024 slots) tracking min/max/sum/count per station.

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit `parse_aggregate` to improve performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all constants and the function from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Hash table output must match the C reference exactly (min, max, sum, count per station).
3. One change per iteration. State your hypothesis clearly.
4. The `parse_aggregate` function signature must not change.
5. No dead code. No comments longer than one line.

## Strategy Space

The kernel is a sequential loop over lines with data-dependent control flow (semicolon search, hash probe). It uses SIMD u8x16 for key comparison in the hash table.

**This kernel is compute-bound.** The inner loop does string parsing, hashing, temperature parsing, and hash table probing — all with data-dependent branches. The text buffer is read sequentially but the hash table has random access patterns.

**Dimensions to explore:**
- **Hash function quality**: reduce collisions to minimize linear probing. Try different hash constants or hash function structure.
- **SIMD key comparison width**: currently u8x16 covers names up to 16 bytes. Could use u8x32 for wider coverage but most names are short.
- **Prefetch on hash table arrays**: prefetch ht_keys[slot*64], ht_key_len[slot], ht_min[slot] etc. before probing — hash table access is random and likely cache-missing at scale.
- **Loop structure**: the semicolon backward scan, hash computation, and temperature parse are sequential. Reordering or fusing these may help.
- **Temperature parsing**: the current approach uses a while-loop for integer digits. Since temperatures are always X.Y or XX.Y or -X.Y or -XX.Y format, a fixed-pattern parse could eliminate the loop.
- **Branch reduction**: replace `if/else` chains with arithmetic where possible (e.g., branchless negative handling).
- **Key copy optimization**: the byte-by-byte key copy on insert could use wider stores.

**What probably won't help:**
- Changing TABLE_SIZE — the benchmark has ~50 stations, 1024 slots is already sparse
- Full SIMD vectorization of the outer loop — each line has variable length, data-dependent parsing
- Reordering hash table layout — the parallel array structure is required by the function signature

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
