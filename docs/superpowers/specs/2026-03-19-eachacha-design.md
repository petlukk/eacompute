# eachacha: ChaCha20 Crypto Demo — Design Spec

## Goal

Build a standalone demo project (`eachacha`) that implements RFC 7539 ChaCha20 encryption in Ea SIMD kernels, with a fused encrypt+statistics kernel, benchmarked against NumPy, generic C, and OpenSSL. Brutally honest measurement. If we lose, we lose.

## Why This Demo

Traditional encryption is a "tax" on performance. The pitch: with Ea's SIMD kernels and `_parallel` threading, encryption runs faster than I/O can deliver data. Encryption cost becomes zero in practice.

The deeper argument: Ea can fuse encryption with downstream logic (statistics, filtering, validation) in a single memory pass. OpenSSL is a black box — you send data in, get encrypted data out, and then do a second pass for your actual work. Ea eliminates that second pass.

## Architecture

Standalone repo at `/root/dev/eachacha` (same pattern as eastat, easobel). Depends on `ea-compiler` pip package, not on eacompute source.

Two Ea kernels, one C reference implementation, one Python benchmark suite.

## Kernels

### `chacha20.ea` — Full RFC 7539 ChaCha20

**Quarter round** (the core primitive):
```
a = a .+ b; d = d .^ a; d = (d .<< splat(16)) .| (d .>> splat(16))
c = c .+ d; b = b .^ c; b = (b .<< splat(12)) .| (b .>> splat(20))
a = a .+ b; d = d .^ a; d = (d .<< splat(8))  .| (d .>> splat(24))
c = c .+ d; b = b .^ c; b = (b .<< splat(7))  .| (b .>> splat(25))
```

Operations: addition (`.+`), XOR (`.^`), rotation via shift (`.<<`, `.>>`) and OR (`.|`). All on i32x4.

**Exports:**
- `chacha20_block(state: *i32, out: *mut i32)` — Single 64-byte block. 10 double rounds (20 quarter rounds total). Adds original state back at the end per RFC.
- `chacha20_encrypt(key: *i32, nonce: *i32, counter: i32, plaintext: *u8, ciphertext: *mut u8, len: i32)` — Counter mode encryption. Generates keystream blocks, XORs with plaintext. Increments counter per block. Handles partial final block.

**State constants** (RFC 7539 "expand 32-byte k"):
```
0x61707865, 0x3320646e, 0x79622d32, 0x6b206574
```

**State layout** (RFC 7539 Section 2.3):
```
cccccccc  cccccccc  cccccccc  cccccccc    (constants above)
kkkkkkkk  kkkkkkkk  kkkkkkkk  kkkkkkkk    (key words 0-3)
kkkkkkkk  kkkkkkkk  kkkkkkkk  kkkkkkkk    (key words 4-7)
bbbbbbbb  nnnnnnnn  nnnnnnnn  nnnnnnnn    (counter + nonce)
```

**Endianness:** Little-endian throughout (matching x86-64 and standard AArch64). Key, nonce, and counter bytes are loaded as little-endian i32 words per RFC 7539 Section 2.1. On little-endian platforms, raw pointer loads produce correct byte order.

**Quarter round assignment:**
- Column rounds: QR(0,4,8,12), QR(1,5,9,13), QR(2,6,10,14), QR(3,7,11,15)
- Diagonal rounds: QR(0,5,10,15), QR(1,6,11,12), QR(2,7,8,13), QR(3,4,9,14)

**SIMD strategy:** The state is stored as 4 row vectors (i32x4 each):
- `row0` = constants, `row1` = key[0:3], `row2` = key[4:7], `row3` = counter+nonce

Column rounds operate directly on these 4 vectors (each QR hits one lane per row).

For diagonal rounds, rows must be rotated to realign lanes:
- `row1`: rotate left by 1 → `shuffle(row1, [1, 2, 3, 0])`
- `row2`: rotate left by 2 → `shuffle(row2, [2, 3, 0, 1])`
- `row3`: rotate left by 3 → `shuffle(row3, [3, 0, 1, 2])`

After the diagonal QRs, reverse the rotations:
- `row1`: rotate right by 1 → `shuffle(row1, [3, 0, 1, 2])`
- `row2`: rotate right by 2 → `shuffle(row2, [2, 3, 0, 1])`
- `row3`: rotate right by 3 → `shuffle(row3, [1, 2, 3, 0])`

This is the standard SIMD ChaCha20 approach.

**Byte-level XOR strategy:** After `chacha20_block` produces 16 x i32 keystream words, store them to a temporary i32 buffer. Then load the same memory as u8x16 vectors (pointer reinterpretation via cast to `*u8`), and XOR with plaintext u8x16 vectors. This works because the keystream buffer is caller-provided contiguous memory; reading it through a `*u8` pointer gives the little-endian byte representation of each i32 word, which is correct per RFC 7539.

**`chacha20_block` is internal** (not exported). Only `chacha20_encrypt` is exported. The block function is inlined or called internally within the encrypt function. Since Ea has single-file compilation, the block logic can be a helper function in the same file.

**Partial final block:** Generate a full 64-byte keystream block into a temporary buffer. Use a scalar tail loop to XOR only the remaining `len % 64` bytes. No masked SIMD needed — the tail is small enough that scalar is fine.

### `chacha20_fused.ea` — Encrypt + Statistics in One Pass

**Export:**
- `chacha20_encrypt_stats(key: *i32, nonce: *i32, counter: i32, plaintext: *u8, ciphertext: *mut u8, len: i32, out_sum: *mut i64, out_count: *mut i32, out_min: *mut u8, out_max: *mut u8)` — Same encryption as above, but also computes sum, count, min, max of plaintext bytes in the same loop. One memory read instead of two.

The statistics are computed on plaintext before XOR, so the fusion adds only a few SIMD operations per block (widen u8 to i32 for sum accumulation, compare for min/max). The memory access pattern is identical to the non-fused kernel.

### `chacha20_ref.c` — Generic C Reference

Standard ChaCha20 in plain C. No intrinsics, no SIMD pragmas. Compiled with `cc -O3`. Same algorithm, same RFC compliance. This is the "fair fight" — what a competent C programmer would write without SIMD knowledge.

## Benchmark Design

### `bench.py` — All Comparisons

| Competitor | What | How |
|---|---|---|
| NumPy XOR | Naive baseline | `np.bitwise_xor(data, key_repeated)` — not real crypto, just XOR throughput |
| Generic C | Fair fight | `chacha20_ref.c` compiled with `-O3`, called via ctypes |
| OpenSSL | Industry standard | `cryptography.hazmat.primitives.ciphers` ChaCha20 |
| Ea (single core) | Our kernel, 1 thread | `chacha20_encrypt()` via auto-generated binding |
| Ea parallel | Our kernel, all cores | Manual ThreadPoolExecutor in bench.py: partition data into chunks, assign counter ranges, call `chacha20_encrypt` per thread. Auto-generated `_parallel` won't work here because each chunk needs a different starting counter value. |
| Ea fused (single core) | Encrypt + stats, 1 thread | `chacha20_encrypt_stats()` |
| Separate encrypt + stats | Two passes | Encrypt first, then compute stats separately — shows fusion savings |

**Parameters:**
- Data size: 64 MB (bandwidth-relevant, fast to run)
- Warmup: 3 runs (discarded)
- Timed: 10 runs
- Report: median GB/s + stddev
- Key/nonce: fixed test values (benchmark, not security test)

**Honest reporting rules:**
- All numbers in one table, no cherry-picking
- If OpenSSL wins, say so. Include the ratio.
- Report single-core AND parallel separately (no hiding behind thread count)
- Note the hardware: CPU model, core count, cache sizes
- Note compiler: ea version, LLVM version, cc version, optimization flags

### `test_vectors.py` — RFC 7539 Verification

**Before any benchmarking runs:**
1. Quarter round test vector (RFC 7539 Section 2.1.1)
2. Full block test vector (RFC 7539 Section 2.3.2)
3. Encryption test vector (RFC 7539 Section 2.4.2)
4. Cross-verification: encrypt with Ea, decrypt with OpenSSL (and vice versa)
5. Fused kernel produces identical ciphertext as non-fused kernel
6. Fused statistics match `np.sum()`, `np.min()`, `np.max()` on plaintext

If any test fails: print error, abort. No benchmarks on incorrect code.

## Project Structure

```
/root/dev/eachacha/
├── chacha20.ea              # RFC 7539 ChaCha20 kernel
├── chacha20_fused.ea        # Encrypt + stats fused kernel
├── chacha20_ref.c           # Generic C reference (no intrinsics)
├── build.sh                 # ea --lib + ea bind + cc -O3
├── bench.py                 # Full benchmark suite
├── test_vectors.py          # RFC 7539 correctness tests
└── README.md                # Results, honest analysis
```

## Build

```bash
#!/bin/bash
set -e
EA="ea"  # or path to ea binary

# Compile Ea kernels
$EA chacha20.ea --lib
$EA bind chacha20.ea --python
$EA chacha20_fused.ea --lib
$EA bind chacha20_fused.ea --python

# Compile C reference
cc -O3 -shared -fPIC -o libchacha20_ref.so chacha20_ref.c
```

Requires: `ea-compiler` pip package (or ea binary on PATH), `cc`, Python 3 with numpy and cryptography.

## What We Do NOT Build

- No Poly1305 MAC (separate algorithm, not part of this demo)
- No CLI tool or "transparent tunnel" UI
- No streaming file I/O (benchmarks run in-memory for clean measurement)
- No FastAPI server or web interface
- No auto-parallelism in the kernel (threading at binding level per Ea philosophy)

## Future: Autoresearch Optimization (Phase 2)

After baseline benchmarks are established, run eacompute's autoresearch system on the ChaCha20 kernels to explore:
- Prefetch strategies
- Loop unrolling (process multiple blocks per iteration)
- Wider vectors (i32x8 on AVX2)
- Dual accumulators for the fused kernel
- Stream stores for output

This is phase 2. Phase 1 is correct, clean, honestly benchmarked.

## Success Criteria

1. RFC 7539 test vectors pass
2. Cross-verification with OpenSSL passes
3. Fused kernel produces identical ciphertext
4. All benchmark numbers reported honestly
5. README tells the full story — wins AND losses
