# NEON Workaround: Runtime SIMD Permute

`permute_runtime` is one of the x86-only intrinsics in Eä. This page
documents the canonical NEON workaround.

`permute_runtime(tbl: f32x8, idx: i32x8) -> f32x8` lowers to AVX2's
`vpermps` on x86 — one instruction, eight runtime-indexed lane picks,
one result vector. On AArch64 NEON there is no such instruction. Pi 5's
Cortex-A76 has no SVE2, so there's no `tbl z0.s, {z1.s}, z2.s` fallback.
`tbl.16b` / `tbx.16b` are byte-level with a 16-byte table cap — useless
for f32/i32 LUTs. A real runtime f32-lane permute is hardware-impossible
on this target.

Pre-v1.14.0, Eä had no `permute_runtime` at all. v1.14.0 ships the
intrinsic with x86-only codegen and a hard ARM error that names this
document and the select-chain idiom. The pattern that would otherwise
be "figure it out yourself" is now a documented, benchmarked idiom with
a known crossover point.

## The `particle_life` motivation

Olorin's `particle_life` kernel computes per-particle force based on a
type-pair lookup table. Each particle pair `(i, j)` has integer types
`type_i` and `type_j`; the combined index `type_i * num_types + type_j`
addresses a small LUT of f32 force multipliers. With `num_types = 6`
that's a 36-entry table, but any given vectorized step only touches 6
distinct row entries.

On x86 the kernel uses one `permute_runtime` per vector of 8 particles:

```ea
// AVX2 permute: works on x86 but errors on ARM with a pointer to
// docs/idioms/neon-runtime-permute.md.
export func force_lut_x86(row: f32x8, idx: i32x8, out: *mut f32) {
    let v: f32x8 = permute_runtime(row, idx)
    store(out, 0, v)
}
```

That kernel hard-errors on AArch64 at codegen time with:

> permute_runtime has no NEON equivalent on ARM. Use a compare-and-select
> chain keyed on each LUT index. See
> docs/idioms/neon-runtime-permute.md for canonical patterns.

## The compare-and-select pattern

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

Build with `--target-triple=aarch64-unknown-linux-gnu`. LLVM lowers the
`select` + `.==` chain to `fcmeq` / `bsl` pairs on NEON — one compare
and one bitselect per LUT entry, no memory traffic after the initial
broadcast.

## When this beats `gather` (even on x86)

The same select-chain pattern outperforms `gather` on x86 for small
tables. Measured on AVX-512 (Zen 4) with the `particle_life` kernel at
N=2000:

| Variant                    | Speedup vs scalar |
|----------------------------|-------------------|
| `gather`                   | 1.34×             |
| 6-element select-chain     | 2.93×             |

Crossover where `gather` wins back is around `num_types ≥ 12-16`. Below
that, the select chain dominates because hardware `vgather` is
microcoded-heavy. The portable select-chain pattern gives you the better
number on ARM *and* the better number on x86 for the small-table case.

## When to use `tbl.16b` instead

If the table is **bytes** with at most 16 entries (e.g., S-box, character
class table), use `shuffle_bytes(u8x16 table, u8x16 indices)`. It lowers
to `tbl.16b` on NEON and `vpshufb` on x86 — a single instruction on
both platforms.

`permute_runtime` and `shuffle_bytes` solve different problems; the
byte-domain version is fine on ARM, the f32/i32-domain version is not.

## Why no silent fallback?

Eä's design rule is "the programmer sees the cost." A silent scalar
fallback for `permute_runtime()` on ARM would hide a potential
performance cliff behind syntax that looks like a single SIMD
instruction. The compare-and-select pattern keeps the cost explicit:
**six comparisons and six blends appear as six compares and six selects
in the source**. A profiler shows the `fcmeq` / `bsl` chains, not a
deceptively single-line `permute_runtime` that hides them. Reviewer
reading the diff sees the operations. Future you reading the source in
six months sees the operations. There is no hidden performance cliff.

This is the same philosophy that gates `gather` behind an ARM hard error
and errors on f16x8 arithmetic without `--fp16`: explicit cost beats
convenient syntax that masks slowdown.

## When SVE2 lands

If your ARM target is Apple M4, Graviton 3+, or Snapdragon X, SVE2 *does*
have a real vector-indexed table lookup (`tbl z0.s, {z1.s}, z2.s`) that
covers the f32-lane-permute case. Eä's SVE2 codegen is deferred to a
future release. Until then, the compare-and-select pattern is the
universal AArch64 workaround: it works on Pi 5 today, and it stays
correct on M4 / Graviton even after a hardware-permute lowering lands
(it just gets superseded by the better path on that target).

## See also

- Terse reference idiom: [`docs/idioms/neon-runtime-permute.md`](../../idioms/neon-runtime-permute.md)
- Reference: `docs/src/reference/intrinsics.md` (`permute_runtime`)
- Experimental kernel: `autoresearch/kernels/particle_life/kernel_v113.ea`
- Related workaround: [`docs/src/cookbook/neon-gather-workaround.md`](neon-gather-workaround.md)
