# NEON idiom: runtime SIMD permute

`permute_runtime` is **x86-only**. NEON has no equivalent for arbitrary
runtime-indexed lookup into an f32x8 / i32x8 table:

- `tbl.16b` / `tbx.16b` are byte-level with a 16-byte table cap — useless
  for f32/i32 LUTs.
- SVE2 `tbl` with vector indices is the closest match, but Cortex-A76
  (Pi 5) does not expose SVE2.

For ARM, the canonical Eä idiom for a small runtime LUT (≤ 8 entries) is
a **compare-and-select chain**, one `select` per LUT entry:

```ea
// Lookup `table[indices[k]]` for indices_v: i32xN, where table has
// num_types <= 8 entries (here num_types = 6).
let t0: f32xN = splat(table[0])
let t1: f32xN = splat(table[1])
let t2: f32xN = splat(table[2])
let t3: f32xN = splat(table[3])
let t4: f32xN = splat(table[4])
let t5: f32xN = splat(table[5])

let s0: f32xN = select(indices_v .== splat(0), t0, splat(0.0))
let s1: f32xN = select(indices_v .== splat(1), t1, s0)
let s2: f32xN = select(indices_v .== splat(2), t2, s1)
let s3: f32xN = select(indices_v .== splat(3), t3, s2)
let s4: f32xN = select(indices_v .== splat(4), t4, s3)
let result: f32xN = select(indices_v .== splat(5), t5, s4)
```

**Hoist the broadcasts above the inner loop.** If the compiler keeps the
`splat(table[k])` calls inside the inner loop, it may re-load `table[k]`
per iteration. Bind them to named values in the surrounding scope.

## When this beats `gather` (even on x86)

Measured on AVX-512 (Zen 4) with the `particle_life` kernel at N=2000:

| Variant                    | Speedup vs scalar |
|----------------------------|-------------------|
| `gather`                   | 1.34×             |
| 6-element select-chain     | 2.93×             |

Crossover where `gather` wins back is around `num_types ≥ 12-16`. Below
that, the select chain dominates because hardware `vgather` is
microcoded-heavy.

## When to use `tbl.16b` instead

If the table is **bytes** with at most 16 entries (e.g., S-box, character
class table), use `shuffle_bytes(u8x16 table, u8x16 indices)`. It lowers
to `tbl.16b` on NEON and `vpshufb` on x86 — a single instruction on
both platforms.

`permute_runtime` and `shuffle_bytes` solve different problems; the
byte-domain version is fine on ARM, the f32/i32-domain version is not.

## See also

- Reference: `docs/src/reference/intrinsics.md` (`permute_runtime`)
- Cookbook: `docs/src/cookbook/neon-runtime-permute-workaround.md`
- Experimental kernel: `autoresearch/kernels/particle_life/kernel_v113.ea`
