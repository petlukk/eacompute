# Fast Transcendentals: `exp_poly_f32`

Eä's `exp()` intrinsic calls libm via `llvm.exp.v*f32`. LLVM has no
hardware vector `exp` on any current ISA — not on AVX2, not on AVX-512,
not on NEON, not on SVE. So `exp(f32x8)` always lowers to a loop of
eight sequential `expf` calls. The SIMD pattern in the source is
cosmetic: throughput stays at one element per scalar libm call.

In a softmax loop or a tanh-GELU activation, that scalarization is the
entire kernel. `exp_poly_f32(f32xN) -> f32xN` (new in v1.11.0) replaces
it with a polynomial that stays in SIMD registers — seven to eight FMAs
per lane, no libm call, no scalarization.

The throughput win depends on the baseline:

- **Modern x86 with a fast `expf` in glibc** (AMD Zen 4 / glibc 2.42 on
  our reference benchmark): **2.93× isolated**, 2.60× inside softmax.
- **Pi 5 (ARM Cortex-A76, glibc `expf`, no `libmvec`)** measured inside
  a real GELU kernel in Olorin: **2.23×** end-to-end on `gemma4_gelu`,
  consistent across 64-12288-lane shapes.
- **Older libm or scalar-only environments without a vectorized `expf`**:
  the gap widens — the spec's original "~10×" headline holds against
  the slowest baselines but does not match every modern glibc.

See [`docs/release/v1.11.0/perf-results.md`](../../release/v1.11.0/perf-results.md)
for the methodology, full numbers, and a discussion of why glibc 2.42's
`expf` is faster than the spec assumed. The win is real on every baseline
we measured; the magnitude is environment-sensitive.

## The contract

`exp_poly_f32` is **not** a drop-in replacement for `exp()`. It trades
two things for the throughput:

1. **Bounded input range.** It is defined on `[-50, 50]` per lane.
   Outside that range the polynomial diverges — no NaN or Inf
   guarantees, no clamping. The caller clamps if their inputs may
   exceed.
2. **Bounded accuracy.** Relative error is ≤ 2⁻¹⁸ (~3.8e-6) inside the
   safe range. That's enough for softmax (normalize-by-sum absorbs the
   error) and for tanh-GELU activations. It is **not** enough for
   anything that needs full f32 precision — keep `exp()` for those.

The name encodes the trade. Reading `exp_poly_f32` at a call site, you
know it's a polynomial on f32 lanes — same explicitness style as
`pack_sat_i32x8`, `widen_u8_i32x4`, `cvt_f16_f32`.

## Softmax

The canonical use case is numerically-stable softmax over a small fixed
window. The maximum subtract keeps inputs in a tight range, exp is the
dominant cost, and the final normalize absorbs the polynomial's relative
error.

```ea
// Stable softmax over 8 lanes using exp_poly_f32.
// reduce_max keeps inputs in a tight range so we never approach the
// [-50, 50] contract edge — the test fixture uses values 1..8 and the
// shifted range is [-7, 0].
export func softmax(x: *f32, out: *mut f32) {
    let v: f32x8 = load(x, 0)
    let mx: f32 = reduce_max(v)
    let mxv: f32x8 = splat(mx)
    let shifted: f32x8 = v .- mxv
    let ev: f32x8 = exp_poly_f32(shifted)
    let s: f32 = reduce_add(ev)
    let inv: f32 = 1.0 / s
    let invv: f32x8 = splat(inv)
    let r: f32x8 = ev .* invv
    store(out, 0, r)
}
```

The full integration test for this shape (`test_exp_poly_f32_softmax_integration`
in `tests/phase14_exp_poly.rs`) compares against `expf`-based reference
softmax with relative tolerance 1e-3 (~2⁻¹⁰) — the polynomial's 2⁻¹⁸
error compounds through reduce_add, but the normalize-by-sum still
collapses it to a tolerance that's well below softmax's typical
numerical envelope.

## tanh-GELU via algebraic identity

Eä has no `tanh_poly_f32` (yet — adding one would be a separate
minimax fit). Instead, you express tanh in terms of exp:

```
tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
```

The GELU activation `gelu(x) ≈ 0.5 · x · (1 + tanh(c · (x + 0.044715 · x³)))`
with `c = sqrt(2/π)` then becomes a single `exp_poly_f32` plus arithmetic:

```ea
// tanh-GELU via algebraic identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
export func gelu_tanh(input: *f32, output: *mut f32) {
    let x: f32x8 = load(input, 0)

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    let c0: f32x8 = splat(0.7978845608)
    let c1: f32x8 = splat(0.044715)
    let xsq: f32x8 = x .* x
    let xcube: f32x8 = xsq .* x
    let inner: f32x8 = c0 .* fma(c1, xcube, x)

    // Clamp inner to [-50, 50] so exp_poly_f32's bounded contract holds.
    let hi: f32x8 = splat(50.0)
    let lo: f32x8 = splat(-50.0)
    let clamped: f32x8 = min(max(inner, lo), hi)

    // tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
    let two: f32x8 = splat(2.0)
    let one: f32x8 = splat(1.0)
    let e2x: f32x8 = exp_poly_f32(clamped .* two)
    let num: f32x8 = e2x .- one
    let den: f32x8 = e2x .+ one
    let t: f32x8 = num ./ den

    // gelu = 0.5 * x * (1 + t)
    let half: f32x8 = splat(0.5)
    let result: f32x8 = half .* x .* (one .+ t)
    store(output, 0, result)
}
```

Two things to notice in the kernel:

- **The `min(max(...))` clamp.** GELU inputs in practice live in
  ~[-3, 3], far from the contract edge. The clamp costs two
  instructions and removes the failure mode if anything outside that
  expected range slips in. It's a cheap and explicit safety belt.
- **The single `exp_poly_f32` call.** The identity `tanh(x) = (e^{2x} -
  1) / (e^{2x} + 1)` lets one polynomial evaluation cover both halves
  of the tanh — no separate sinh/cosh, no second exp call.

## Safely clamping in-range

The contract is bounded on `[-50, 50]` per lane, so the safest defensive
pattern is the min-then-max chain shown above:

```ea
let hi: f32x8 = splat(50.0)
let lo: f32x8 = splat(-50.0)
let safe: f32x8 = min(max(unbounded, lo), hi)
let e: f32x8 = exp_poly_f32(safe)
```

Both `min` and `max` lower to single NEON `fminnm` / `fmaxnm` (or x86
`vminps` / `vmaxps`) instructions, so the clamp is two FMAs of overhead
on a polynomial that's already 7-8 FMAs per lane.

If you can prove statically that your input range is bounded — softmax
after a `reduce_max` subtract, for instance — skip the clamp. The
guarantee is mathematical, not enforced; the polynomial does not check.

## When to keep `exp()`

`exp_poly_f32` is the right tool for **SIMD hot paths where 2⁻¹⁸ relative
error is acceptable**. Examples: softmax, GELU, attention scoring,
exponential decay in beam search.

Keep `exp()` (which calls libm) for:

- Scalar code, or vector code where the SIMD scalarization isn't
  visible (rare loops, one-shot computations).
- Code that needs full f32 precision — physics, statistics, log-sum-exp
  inside an accuracy-critical reduction.
- Inputs that may exceed `[-50, 50]` and a libm-correct NaN/Inf is
  required.

## See also

- v1.11.0 intrinsic catalog: [`docs/release/v1.11.0/intrinsic-catalog.md`](../../release/v1.11.0/intrinsic-catalog.md)
- Spec (algorithm, error analysis, minimax coefficients):
  `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`
- Test fixtures (accuracy, range, softmax integration):
  `tests/phase14_exp_poly.rs`
