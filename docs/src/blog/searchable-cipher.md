# Searching Encrypted Logs Without Decrypting Them (Much)

*Or: what happens when you let a SIMD compiler loose on a crypto problem.*

---

## The Problem That Shouldn't Exist

You have 100 GB of encrypted log files. Something is on fire. You need to find every line containing "ERROR".

Here's your options:

**Option A: The "correct" way.** Decrypt the entire file to `/tmp`. Grep it. Delete the decrypted copy. Hope nobody read it while it was sitting there. Also hope your 50 GB `/tmp` partition can handle it. Also hope you remembered to `shred` instead of `rm`.

**Option B: The galaxy brain way.** Use Fully Homomorphic Encryption to search directly on ciphertext. Wait approximately until the heat death of the universe. Receive your answer. Realize the fire already consumed the building.

**Option C: What we actually built.** Decrypt 4 KB at a time into a tiny buffer. Search it. Zero the buffer. Move on. The plaintext exists for about as long as a TikTok attention span.

We went with Option C.

## The "Fusion" Trick

Here's the thing about ChaCha20 — it's a stream cipher. You XOR a keystream with your plaintext, byte by byte. Decryption is the same operation: XOR the keystream with ciphertext, get plaintext back.

So why not XOR *and* search at the same time?

```
Standard pipeline:
  decrypt(ciphertext) → plaintext_file → grep(plaintext_file) → results
  Memory: 100 GB plaintext on disk. Two passes over data.

Fused pipeline:
  for each 4 KB chunk:
    decrypt → search → extract lines → zero buffer
  Memory: 4 KB. One pass. Plaintext never on disk.
```

This is called **loop fusion** — cramming multiple operations into a single loop so you only read the data once. It's the same trick that makes `a * b + c` with FMA faster than doing `a * b` then `+ c` separately. Except here we're fusing cryptographic decryption with string search.

And because we're writing in [Eä](https://github.com/petlukk/eacompute) — a language where you control the SIMD instructions directly — the "search" part isn't some scalar byte-by-byte comparison. It's the same algorithm glibc uses in `memmem`:

```
// For each 16-byte chunk of decrypted plaintext:
let bits: i32 = movemask(chunk .== splat(first_byte))
if bits == 0 {
    skip  // no match possible in these 16 bytes
}
// else: verify the candidate positions
```

`vpcmpeqb` + `vpmovmskb`. Compare 16 bytes at once, get a bitmask of hits. If the bitmask is zero, skip. That's it. Most chunks contain zero instances of the letter 'E', so most chunks are skipped entirely.

## Multi-Needle: The Actually Useful Part

Searching for one string is cute. Searching for `["ERROR", "FATAL", "PANIC"]` without decrypting three times — that's useful.

The v2 kernel takes multiple needles and OR:s their bitmasks:

```
bits = 0
for each unique first-byte:
    bits = bits + movemask(chunk .== splat(first_byte))
if bits == 0:
    skip  // none of the needles start with any byte in this chunk
```

One decryption pass. Three patterns. All the matched lines extracted with `\n`-boundary detection using the same SIMD trick.

The result:

```python
from eachacha import encrypt, search

key = bytes(range(32))
nonce = bytes(12)
ct = encrypt(b"INFO ok\nERROR disk full\nFATAL crash\n", key, nonce)

result = search(ct, [b"ERROR", b"FATAL"], key, nonce)
for i in range(result.match_count):
    print(f"[{result.needle_ids[i]}] {result.lines[i]}")

# [0] b'ERROR disk full'
# [1] b'FATAL crash'
```

Three lines of Python. The `search()` function does the rest — picks the right kernel, allocates buffers, packs needles, calls the SIMD kernel, returns structured results.

## The Numbers

AMD EPYC 9354P, 2 vCPUs, 64 MB test data:

**Single needle:**

| Implementation | GB/s |
|---|---:|
| **Eä fused decrypt+search** | **1.28** |
| Decrypt then C memmem (two passes) | 0.96 |
| C memmem on plaintext (no decrypt) | 2.22 |

**Three needles:**

| Implementation | GB/s |
|---|---:|
| **Eä v2 multi-needle (one pass)** | **0.52** |
| Eä v1 called 3 times (three passes) | 0.41 |
| C memmem x3 on plaintext | 0.78 |

The fused approach is **1.34× faster** than decrypt-then-search for single needle, and **1.28× faster** than three separate searches for multi-needle. Not because our search is faster — it's because we only decrypt once.

And yes, 0.52 GB/s on *encrypted* data is 67% of what C memmem achieves on *plaintext*. We're paying the ChaCha20 tax, but the fusion rebate is generous.

## What We're NOT Claiming

Let's be honest about the security model, because someone on Hacker News will definitely yell at me if I'm not:

| What we guarantee | What we don't claim |
|---|---|
| Plaintext never on disk | "Zero RAM exposure" |
| 4 KB buffer, zeroed per window | Equivalent to FHE |
| Only match offsets + lines leave the kernel | Side-channel resistance |
| 400-million-fold reduction in exposure surface | That this is a good idea for nuclear launch codes |

This is a **practical middle ground** between "decrypt everything to a temp file" (what everyone does today) and "compute on ciphertext" (what nobody can afford to do today). It's not cryptographically novel. It's an engineering choice: minimize exposure, maximize throughput, accept the 4 KB window.

Also, no special hardware required. No SGX enclaves. No trust assumptions about chip manufacturers. Just software and a CPU that can XOR things fast.

## The Autoresearch Detour

We tried to make it faster. The Eä project has an automated optimization loop ([autoresearch](https://github.com/petlukk/eacompute/tree/main/autoresearch)) that iteratively modifies kernels and benchmarks them:

```
Iteration 1: Remove buffer zeroing → 0% improvement (zeroing isn't the bottleneck)
Iteration 2: unroll(10) on round loops → parse error (oops)
Iteration 3: 2-block ILP instead of 4-block → 24% SLOWER
Iteration 4: Inline rotation functions → 5% slower (LLVM already inlines them)
Iteration 5: Timeout (the kernel is 866 lines, agent couldn't finish)
```

Total improvement from 5 iterations: **0%**.

The kernel is compute-bound. ChaCha20's 20 rounds of quarter-round operations dominate. The search and line extraction are fast — the bottleneck is the math. Which is honestly a good place to be: it means we're not leaving performance on the table.

## 2,098 Lines

That's the total across all four kernels:

| Kernel | Lines | What it does |
|---|---:|---|
| `chacha20.ea` | 272 | Encrypt (1.78 GB/s) |
| `chacha20_fused.ea` | 384 | Encrypt + stats in one pass |
| `chacha20_search.ea` | 576 | Single-needle encrypted search |
| `chacha20_search_v2.ea` | 866 | Multi-needle + context lines |

For comparison, OpenSSL's ChaCha20 implementation is somewhere north of 100,000 lines of C and assembly. Ours passes the same RFC 7539 test vectors and cross-verifies byte-for-byte with OpenSSL.

I'm not saying we're better than OpenSSL. OpenSSL in native C with AVX-512 would destroy us. But we're 272 lines that a human can read in 10 minutes, and we still hit 1.78 GB/s. That's the trade-off.

## Try It

```bash
pip install eachacha
```

```python
from eachacha import encrypt, search

key = bytes(range(32))
nonce = bytes(12)
ct = encrypt(b"your secret logs here", key, nonce)
result = search(ct, b"secret", key, nonce)
print(result.offsets)  # [5]
```

Or build from source for native CPU optimization:

```bash
pip install ea-compiler
git clone https://github.com/petlukk/eachacha
cd eachacha && ./build.sh && pip install -e .
```

---

*Benchmarks measured on 2 vCPUs. Your numbers will be different, probably better. The ratios hold. [Code, benchmarks, and 109 tests on GitHub.](https://github.com/petlukk/eachacha)*

*Eä is open-source. The compiler is ~12,000 lines of Rust. [Docs.](https://petlukk.github.io/eacompute/) [GitHub.](https://github.com/petlukk/eacompute)*
