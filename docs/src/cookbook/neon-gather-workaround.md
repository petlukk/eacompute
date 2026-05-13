# NEON Gather Workaround

`gather(p: *f32, idx: i32xN) -> f32xN` lowers to AVX2's `vgatherdps`
on x86 — one instruction, four (or eight) indexed loads, one result
vector. On AArch64 NEON there is no such instruction. Pi 5's Cortex-A76
has no SVE / SVE2 either, so there's no `ld1w {z0.s}, p0/z, [x1, z1.s,
sxtw]` fallback. A real gather is hardware-impossible on this target.

Pre-v1.11.0, Eä errored on `gather()` for ARM with a message that said
"use a scalar loop on ARM" — true but unhelpful, and the de-facto
workaround everyone arrived at (a stack-buffer round-trip through
memory) was strictly worse than the canonical scalar-compose pattern
already used by llama.cpp's IQ kernels.

v1.11.0 ships two new vector-construction intrinsics —
`f32x4_from_scalars` and `f32x8_from_scalars` — and rewrites the ARM
`gather()` error to point at them by name plus
[`docs/idioms/neon-gather.md`](../../idioms/neon-gather.md). The pattern
that used to be "figure it out yourself" is now a one-line compose with
a documented idiom.

## The IQ3 motivation

Olorin's IQ3_S / IQ3_XXS dequant kernels need exactly this. Each block
holds packed 3-bit indices into a 256-entry lookup table of dequantized
f32 values. The hot loop reads four indices, fetches four LUT entries,
and writes them out as an f32x4 vector.

On x86 the kernel is one line:

```ea
// AVX2 gather: works on x86 but errors on ARM with a pointer to f32x{4,8}_from_scalars.
export func lut_dequant_x86(lut: *f32, idx: i32x4, out: *mut f32) {
    let v: f32x4 = gather(lut, idx)
    store(out, 0, v)
}
```

That kernel hard-errors on AArch64 at codegen time with:

> gather has no NEON equivalent on ARM. Use scalar load_u32 +
> f32x4_from_scalars (or f32x8_from_scalars) to compose the result
> explicitly. See docs/idioms/neon-gather.md for the canonical pattern.

## The compose pattern

The workaround is to read the four (or eight) values one at a time and
build the vector explicitly. The new intrinsic does the building:

```ea
// IQ3 LUT dequant pattern: gather four entries from a lookup table.
export func iq3_lut_dequant_lane(lut: *f32, indices: *i32, out: *mut f32) {
    let i0: i32 = indices[0]
    let i1: i32 = indices[1]
    let i2: i32 = indices[2]
    let i3: i32 = indices[3]
    let v0: f32 = lut[i0]
    let v1: f32 = lut[i1]
    let v2: f32 = lut[i2]
    let v3: f32 = lut[i3]
    let v: f32x4 = f32x4_from_scalars(v0, v1, v2, v3)
    store(out, 0, v)
}
```

The same shape works for an 8-wide gather over an i32x8 index vector
via `f32x8_from_scalars(...)` — same idea, eight scalar loads instead
of four.

For the real per-row loop, the four scalar reads sit visibly inside an
ordinary while loop:

```ea
// IQ3 LUT dequant row: walk an index array, fetch 4 LUT entries at a time,
// store into the output stream. The four scalar loads sit visibly in the
// source — the programmer sees the cost.
export func iq3_dequant_row(lut: *f32, indices: *i32, out: *mut f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        let v0: f32 = lut[indices[i]]
        let v1: f32 = lut[indices[i + 1]]
        let v2: f32 = lut[indices[i + 2]]
        let v3: f32 = lut[indices[i + 3]]
        let v: f32x4 = f32x4_from_scalars(v0, v1, v2, v3)
        store(out, i, v)
        i = i + 4
    }
}
```

Build with `--target-triple=aarch64-unknown-linux-gnu`. LLVM's NEON
lowering folds the `insertelement` chain inside `f32x4_from_scalars`
into a sequence of `ins v.s[i]` instructions — the same code GCC and
Clang produce for `vsetq_lane_f32` chains. No stack buffer, no
load-and-shuffle dance.

## Why no silent fallback?

Eä's design rule is "the programmer sees the cost." A silent scalar
fallback for `gather()` on ARM would hide a 4-8× slowdown behind syntax
that looks like SIMD. The compose pattern keeps the cost explicit:
**four scalar reads appear as four scalar reads in the source**. A
profiler shows four `ldr s0, [x1, x2, lsl #2]` instructions, not a
deceptively single-line `gather` that hides them. Reviewer reading the
diff sees the loads. Future you reading the source in six months sees
the loads. There is no hidden performance cliff.

This is the same philosophy that gates `--fp16` behind a flag and
errors on f16x8 arithmetic without it: explicit cost beats convenient
syntax that masks slowdown.

## When SVE2 lands

If your ARM target is Apple M4, Graviton 3+, or Snapdragon X, SVE2 *does*
have a real gather (`ld1w` with a scatter-gather addressing mode). Eä's
SVE2 codegen is deferred to a future release. Until then, the compose
pattern is the universal AArch64 workaround: it works on Pi 5 today, and
it stays correct on M4 / Graviton even after a hardware-gather lowering
lands (it just gets superseded by the better path on that target).

## See also

- Terse reference idiom: [`docs/idioms/neon-gather.md`](../../idioms/neon-gather.md)
- Test coverage: `tests/phase14_arm_neon.rs`
  (`test_f32x4_from_scalars`, `test_f32x8_from_scalars`,
  `test_gather_on_arm_points_to_compose`)
