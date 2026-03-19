# Ea Kernel Optimization — ChaCha20 Counter-Mode Encryption

You are optimizing an Ea kernel that implements ChaCha20 (RFC 7539) counter-mode encryption using i32x4 SIMD vectors. The kernel processes plaintext in 64-byte blocks: each block generates a keystream via 20 rounds of quarter-round operations, then XORs the keystream with plaintext.

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit the kernel to improve ChaCha20 throughput (GB/s). You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence. Include all functions from the original file.

## Rules

1. Only valid Ea syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference byte-for-byte. This is a cipher — there is zero tolerance.
3. One change per iteration. State your hypothesis clearly.
4. The `chacha20_encrypt` function signature must not change (same 8 parameters: key, nonce, counter, plaintext, ciphertext, len, ks_i32, ks_u8).
5. The `chacha20_block` signature can change (it is internal; the benchmark tests via `chacha20_encrypt` only).
6. No dead code. No comments longer than one line.

## Current Architecture

- **State**: 4 x i32x4 row vectors (row0..row3), 16 x i32 = 64 bytes
- **Column rounds**: Quarter round directly on rows (a=r0, b=r1, c=r2, d=r3)
- **Diagonal rounds**: shuffle rows to align diagonals, quarter round, shuffle back
- **Rotation**: `(v .<< splat(n)) .| ((v .>> splat(32-n)) .& mask)` — the `.& mask` is needed because `.>>` on i32x4 is arithmetic (sign-extending)
- **Keystream XOR**: dual-pointer trick — `ks_i32` for i32x4 stores from block function, `ks_u8` for u8x16 loads during XOR
- **Tail handling**: `load_masked`/`store_masked` for final partial block
- **Inner loop**: 10 iterations of double-rounds, each iteration loads/stores state through `out` buffer

## Strategy Space

**Wider vectors — i32x8 (AVX2) multi-block**:
Use i32x8 to hold state from 2 independent blocks in parallel. Each i32x8 lane pair processes a different counter value. This is the standard multi-block SIMD ChaCha20 approach and probably the single biggest win. Doubles throughput per iteration with no extra rounds. Requires restructuring the block function to accept 2 counters and produce 128 bytes of keystream.

**Register pressure management**:
The current kernel loads/stores working state to the `out` buffer every round iteration. Keeping all 4 rows + 4 initial rows in registers (8 x i32x4) would eliminate 8 loads + 4 stores per double-round iteration = 120 memory ops per block. This is likely significant.

**Loop unrolling**:
Unroll the 10-iteration double-round loop (full or partial, e.g., 2x or 5x). Reduces loop overhead and branch mispredictions, helps instruction scheduling.

**Keystream buffer elimination**:
Instead of store-then-reload through the dual-pointer (i32 store, u8 load), XOR plaintext directly as i32x4. Load plaintext via a `*i32` parameter (reinterpret), XOR with keystream i32x4, store result via `*mut i32`. This avoids the u8/i32 conversion entirely for full 64-byte blocks. Only use the u8 path for the tail.

**Eliminating the rotation mask**:
The `.& mask` in each rotation costs 1 vector instruction. There are 4 rotations per quarter round, 8 quarter rounds per double-round, 10 double-rounds = 320 mask ops per block. If there's a way to clear high bits without the mask (or prove they don't matter), that saves significant instruction count.

**Prefetch**:
Add `prefetch(plaintext, offset + N)` before XOR phase to bring upcoming plaintext into cache.

**Stream stores**:
Use `stream_store` for ciphertext output (write-only, bypass cache).

**What probably won't help:**
- Algorithmic changes (ChaCha20 is a standard, can't change the algorithm)
- Reducing round count (must be 20 rounds / 10 double-rounds)
- AES-NI or crypto-specific intrinsics (Ea doesn't have them)
- f32/f64 SIMD (ChaCha20 is entirely integer)

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
- Memory: load, store, stream_store, gather, scatter, prefetch(ptr, offset), load_masked(ptr, offset, count), store_masked(ptr, offset, vec, count)
- Arithmetic: fma, sqrt, rsqrt, exp, min, max
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min
- Construction: splat, select, shuffle
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, for, unroll(N)

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`

**Other:** const, static_assert, *restrict, *restrict mut, array literals like `[1, 2, 3, 0]i32x4`

**Important Ea notes:**
- `.>>` on i32x4 is arithmetic shift (sign-extending). To get logical right shift behavior, mask off the sign-extended bits with `.&`.
- `shuffle(v, [indices])` reorders lanes within a single vector.
- `splat(value)` broadcasts a scalar to all lanes.
- Typed load/store: `load(ptr, offset)` where offset is in elements, not bytes.
- `store(ptr, offset, value)` similarly element-indexed.

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
