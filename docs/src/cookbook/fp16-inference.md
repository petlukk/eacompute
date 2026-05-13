# Native f16 Inference

LLM weights ship as f16. KV cache rows are f16. Attention reads f16 every
step. Until v1.11.0, every one of those reads paid for a `cvt_f16_f32`
into an f32 register before any compute could happen — the storage was
narrow, but the SIMD math was wide. On Pi 5 (Cortex-A76 with FEAT_FP16
silicon) that round-trip is now optional. Compile with `--fp16` and
`f16x8` arithmetic lowers straight to NEON's `fadd v.8h` / `fmul v.8h` /
`fmla v.8h` — half the register pressure, no cvt, no widening.

This is opt-in. Existing code that uses `cvt_f16_f32` / `cvt_f32_f16` keeps
working unchanged — they're stable on any AArch64. `--fp16` is the
"compute stays in half-precision the whole way" mode for hot paths that
load f16, multiply f16, and write f16 back.

## Where the round-trip hurts

In an LLM attention kernel, every token's Q ⋅ Kᵀ inner product loads a
row of cached keys. Storage is f16 — that's the whole point of a half
KV cache. Compute is f32. The original portable shape looks like this:

```ea
// Before: f32 round-trip on every f16 load.
// Storage is f16, compute is f32, every load pays the cvt cost.
export func rmsnorm_via_f32(x: *i16, scale: f32, out: *mut i16) {
    let v_h: i16x4 = load(x, 0)
    let v: f32x4 = cvt_f16_f32(v_h)
    let sq: f32x4 = v .* v
    let sum: f32 = reduce_add(sq)
    let n: f32 = 4.0
    let eps: f32 = 0.000001
    let denom: f32 = sqrt(sum / n + eps)
    let inv: f32 = 1.0 / denom
    let inv_v: f32x4 = splat(inv * scale)
    let r: f32x4 = v .* inv_v
    let r_h: i16x4 = cvt_f32_f16(r)
    store(out, 0, r_h)
}
```

(Half-precision values are addressed as `i16` here because the
cvt-based path predates the `f16` type — `cvt_f16_f32(i16x4)` is
the documented bit-level entry point.)

That kernel works on any AArch64. But every `cvt_f16_f32` is real work
— a `fcvtl v.4s, v.4h` instruction, plus the data spends the whole
compute pass in 128-bit f32 registers, halving how many lanes fit and
doubling register pressure on FMA chains.

## What `--fp16` changes

With the new `f16` scalar / `f16x4` / `f16x8` types and `--fp16` enabled,
the same shape stays in half-precision end to end:

```ea
// Native f16 — no f32 round-trip on the per-element path.
// Storage is f16; compute is f16 except for the single scalar sqrt
// (no useful f16 scalar sqrt on Cortex-A76; the round-trip is one
// operation, not N).
export func rmsnorm_f16(x: *f16, scale: f16, out: *mut f16) {
    let v: f16x8 = load(x, 0)
    let sq: f16x8 = v .* v
    let sum: f16 = reduce_add(sq)

    // Scalar branch: sum/N + eps, sqrt, reciprocal. Done in f32 because
    // there is no single-lane f16 sqrt instruction on Cortex-A76.
    let n: f16 = to_f16(8.0)
    let mean_h: f16 = sum / n
    let mean_f: f32 = to_f32(mean_h)
    let eps: f32 = 0.000001
    let inv_f: f32 = 1.0 / sqrt(mean_f + eps)
    let inv_h: f16 = to_f16(inv_f)

    // Back to f16x8 for the per-element multiply.
    let inv_v: f16x8 = splat(inv_h)
    let scale_v: f16x8 = splat(scale)
    let r: f16x8 = v .* inv_v .* scale_v
    store(out, 0, r)
}
```

Build with:

```bash
ea rmsnorm.ea --lib --fp16 --target-triple=aarch64-unknown-linux-gnu
```

The vector multiplies and the reduction now lower to `fmul v.8h` and
`faddv h, v.8h` directly — the LLVM IR has `<8 x half>` everywhere on
the per-lane path, and `fpext` only appears around the scalar sqrt. Eight
lanes fit in a single Q register instead of eight lanes spread across two
f32 registers, so chained FMAs hit twice the per-cycle throughput on the
A76's pipelines.

## Attention dot-product

The same idea applies to the inner product over the KV row. The pre-`--fp16`
version converted on every load; the native form fuses without the cvt:

```ea
// Attention dot product over an f16 KV cache row.
// Before --fp16: each f16x8 load became cvt_f16_f32 to an f32x8 register
// before any compute. With --fp16, the cvt is gone — fma runs directly
// on <8 x half> registers.
export func attn_dot_f16(q: *f16, k: *f16, n: i32) -> f16 {
    let mut acc: f16x8 = splat(to_f16(0.0))
    let mut i: i32 = 0
    while i < n {
        let qv: f16x8 = load(q, i)
        let kv: f16x8 = load(k, i)
        acc = fma(qv, kv, acc)
        i = i + 8
    }
    return reduce_add(acc)
}
```

Each loop iteration is one `ld1 v.8h` × 2, one `fmla v.8h`, no cvt. The
load-to-FMA chain stays in 128-bit Q registers the whole way.

## When to reach for it

Use native f16 when:

- Storage is f16 and compute is dominated by element-wise multiply-add
  (KV cache reads in attention, weight-times-activation in MLP rows,
  RMSNorm scaling, RoPE rotation).
- The target hardware actually has FEAT_FP16 (Cortex-A76 / A78 / X1+,
  Apple M-series, Neoverse V1+). On hardware without it, `--fp16`
  errors before codegen — there's no silent fallback.
- You measure first. The win comes from register pressure and SIMD
  width, not from the cvt instruction itself; bandwidth-bound shapes
  that already saturate memory won't speed up.

Keep the f32 round-trip when:

- A single scalar sqrt / divide / reciprocal lives on the critical path.
  The cvt is one instruction; f32's better-conditioned scalar math is
  worth more than the cvt costs. The `rmsnorm_f16` example above does
  exactly this for the per-batch sqrt.
- The kernel needs to ship to non-FEAT_FP16 hardware. The
  `cvt_f16_f32` / `cvt_f32_f16` path is the portable form and does not
  go away in v1.11.0 — it works with or without `--fp16`.

## Performance

Quantitative numbers ship with Phase 6 of the v1.11.0 audit
(post-merge). The architectural story is straightforward: the cvt is
gone from the load-to-compute path and the lane count per register
doubles from 4 to 8, so attention / RMSNorm / RoPE hot paths fall to
roughly the same per-instruction shape as a hand-written
`vfmlal_low_f16` chain. Olorin's gemma4 inference uses this path under
`--fp16`. See the spec for the design rationale.

## See also

- End-to-end test fixture: `tests/data/rmsnorm_f16.ea`
- ARM FP16 test suite: `tests/phase14_arm_fp16.rs`
