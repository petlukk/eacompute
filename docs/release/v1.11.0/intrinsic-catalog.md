# Eä Intrinsic Catalog (v1.11.0)

This is the canonical user-facing reference for every intrinsic available
in Eä v1.11.0. Each entry lists the signature(s), the flag and target it
requires (if any), and the source module it lowers through. Signatures
are taken verbatim from the v1.11.0 inventory and are audit-corrected.

Conventions:

- `T` means a generic scalar / vector type that the intrinsic accepts;
  the polymorphic surface is enumerated in the row.
- `imm` means an integer compile-time constant.
- A row with no "Required flags" column is cross-platform with no extra
  flag — it lowers via portable LLVM IR or has explicit x86/ARM dispatch
  in codegen.
- "(NEW v1.11.0)" marks intrinsics added or substantially extended by
  this release.

## Memory ops

| Name | Signature | Notes |
|---|---|---|
| `load` | `(p: *T, i: i32) -> TxN` | Aligned vector load. f16 widths require `--fp16` (NEW v1.11.0). |
| `store` | `(p: *mut T, i: i32, v: TxN)` | Aligned vector store. f16 widths require `--fp16` (NEW v1.11.0). |
| `stream_store` | `(p: *mut T, i: i32, v: TxN)` | Non-temporal store (`movntdq` / `stnp`). |
| `gather` | `(p: *f32, idx: i32xN) -> f32xN` | **x86-only.** ARM users compose via `f32x{4,8}_from_scalars`. See [`docs/idioms/neon-gather.md`](../../idioms/neon-gather.md). |
| `scatter` | `(p: *mut f32, idx: i32xN, v: f32xN)` | x86-only, requires `--avx512`. |
| `load_masked` | `(p: *T, i: i32, mask) -> TxN` | Per-lane mask load. |
| `store_masked` | `(p: *mut T, i: i32, v: TxN, mask)` | Per-lane mask store. |
| `prefetch` | `(p: *T, i: i32)` | Hardware prefetch hint. |

## Arithmetic

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `fma` | `(a: TxN, b: TxN, c: TxN) -> TxN` | none (f16 widths require `--fp16`) | a*b + c, single instruction. |
| `abs` | `(T) -> T` for scalar / vector float / int | none | NEW v1.11.0 generalization. |
| `sqrt` | `(TxN) -> TxN` for f32 / f64 | none | Hardware fsqrt. |
| `rsqrt` | `(TxN) -> TxN` for f32 / f64 | none | Reciprocal sqrt (rsqrte + Newton refine). |
| `exp` | `(TxN) -> TxN` for f32 / f64 | none | Calls libm via `llvm.exp.v*` — **scalarizes**. Use `exp_poly_f32` for SIMD throughput. |
| `exp_poly_f32` (NEW v1.11.0) | `(f32xN) -> f32xN` for N ∈ {4, 8} | none | Degree-5 minimax polynomial. Defined on `[-50, 50]`; relative error ≤ 2⁻¹⁸. Throughput vs scalar `exp()` is baseline-dependent: 2.93× isolated on modern x86 (glibc 2.42 `expf`), 2.23× on Pi 5 inside a real GELU kernel; higher against older or scalar-only libm. See [`docs/src/cookbook/fast-transcendentals.md`](../../src/cookbook/fast-transcendentals.md) and [`docs/release/v1.11.0/perf-results.md`](perf-results.md). |

## Lane ops

### Construction and broadcast

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `splat` | `(scalar T) -> TxN` (type inferred from context) | f16 needs `--fp16` | |
| `f32x4_from_scalars` (NEW v1.11.0) | `(f32, f32, f32, f32) -> f32x4` | none | Canonical NEON gather workaround. |
| `f32x8_from_scalars` (NEW v1.11.0) | `(f32, f32, f32, f32, f32, f32, f32, f32) -> f32x8` | none | |

### Concat / extract

| Name | Signature | Notes |
|---|---|---|
| `concat_i8x16` (NEW v1.11.0) | `(i8x16, i8x16) -> i8x32` | |
| `concat_u8x16` (NEW v1.11.0) | `(u8x16, u8x16) -> u8x32` | |
| `concat_i8x32` (NEW v1.11.0) | `(i8x32, i8x32) -> i8x64` | |
| `concat_u8x32` (NEW v1.11.0) | `(u8x32, u8x32) -> u8x64` | |
| `concat_i32x8` (NEW v1.11.0) | `(i32x8, i32x8) -> i32x16` | |
| `concat_f32x8` (NEW v1.11.0) | `(f32x8, f32x8) -> f32x16` | |
| `lo128_i8x32` / `hi128_i8x32` (NEW v1.11.0) | `(i8x32) -> i8x16` | Low / high 128-bit half. |
| `lo128_u8x32` / `hi128_u8x32` (NEW v1.11.0) | `(u8x32) -> u8x16` | |
| `lo256_i8x64` / `hi256_i8x64` (NEW v1.11.0) | `(i8x64) -> i8x32` | |
| `lo256_u8x64` / `hi256_u8x64` (NEW v1.11.0) | `(u8x64) -> u8x32` | |
| `lo256_i32x16` / `hi256_i32x16` (NEW v1.11.0) | `(i32x16) -> i32x8` | |
| `lo256_f32x16` / `hi256_f32x16` (NEW v1.11.0) | `(f32x16) -> f32x8` | |

### Shuffle / blend / broadcast pairs

| Name | Signature | Notes |
|---|---|---|
| `shuffle` | `(v: TxN, mask: i32xN) -> TxN` | Per-lane permute (existing). |
| `shuffle_i32x8` (NEW v1.11.0) | `(i32x8, mask: i32x8) -> i32x8` | Constant mask. |
| `shuffle_i32x16` (NEW v1.11.0) | `(i32x16, mask: i32x16) -> i32x16` | Constant mask. |
| `blend_i32` (NEW v1.11.0) | `(a, b, mask: imm) -> i32xN` | Bit-mask blend. |
| `bcast_even_pairs_i32x8` (NEW v1.11.0) | `(i32x8) -> i32x8` | Broadcast even-indexed pairs. |
| `bcast_odd_pairs_i32x8` (NEW v1.11.0) | `(i32x8) -> i32x8` | |
| `bcast_even_pairs_i32x16` (NEW v1.11.0) | `(i32x16) -> i32x16` | |
| `bcast_odd_pairs_i32x16` (NEW v1.11.0) | `(i32x16) -> i32x16` | |
| `shuffle_bytes` | `(v, mask: i8xN) -> i8xN` for widths 16, 32 | Per-byte permute; width 32 is x86-only. |
| `movemask` | `(TxN) -> i32` | Pack lane sign bits to scalar mask. |

## Reductions

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `reduce_add` | `(TxN) -> T` | f16 needs `--fp16` | Ordered for float, unordered for int. |
| `reduce_max` | `(TxN) -> T` | f16 needs `--fp16` | |
| `reduce_min` | `(TxN) -> T` | f16 needs `--fp16` | |
| `reduce_add_fast` | `(TxN) -> T` for float | f16 needs `--fp16` | Reassociative fast-math reduce. |
| `addp_i16` (NEW v1.11.0) | `(i16x8, i16x8) -> i16x8` | ARM-only | NEON `addp` pairwise add. |
| `addp_i32` (NEW v1.11.0) | `(i32x4, i32x4) -> i32x4` | ARM-only | |
| `hadd_i16` | `(i16xN, i16xN) -> i16xN` for N ∈ {8, 16} | **x86-only** | SSSE3/AVX2 `phaddw`. ARM uses `addp_i16` (identical semantics). |

## Conversions

### Scalar

| Name | Signature | Notes |
|---|---|---|
| `to_f32` | `(numeric scalar) -> f32` | Pre-existing. |
| `to_f64` | `(numeric scalar) -> f64` | Pre-existing. |
| `to_f16` (NEW v1.11.0) | `(numeric scalar) -> f16` | |
| `to_i16` (NEW v1.11.0) | `(numeric scalar) -> i16` | |
| `to_i32` | `(numeric scalar) -> i32` | Pre-existing. |
| `to_i64` | `(numeric scalar) -> i64` | Pre-existing. |

### Vector conversion / cvt

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `cvt_f16_f32` (NEW v1.11.0) | `(i16xN) -> f32xN` for N ∈ {4, 8, 16} | none (4-wide); x86-only (8-wide, 16-wide) | NEON `fcvtl` / F16C `vcvtph2ps`. |
| `cvt_f32_f16` (NEW v1.11.0) | `(f32xN) -> i16xN` for N ∈ {4, 8} | none (4-wide); x86-only (8-wide) | NEON `fcvtn` / F16C `vcvtps2ph`. Asymmetric: 16-wide round-trip not yet available (deferred to v1.12.0). |

### Widening / narrowing / packing

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `widen_u8_f32x4`, `widen_i8_f32x4` | `(u8x16 \| i8x16) -> f32x4` (low 4 lanes) | none | Pre-existing. |
| `widen_u8_f32x8`, `widen_i8_f32x8` | `(u8x16 \| i8x16) -> f32x8` | none | Pre-existing. |
| `widen_u8_f32x16`, `widen_i8_f32x16` | `(u8x16 \| i8x16) -> f32x16` | `--avx512` | Pre-existing. |
| `widen_u8_i32x4` | `(u8x16) -> i32x4` | none | Pre-existing. |
| `widen_u8_i32x8` | `(u8x16) -> i32x8` | none | Pre-existing. |
| `widen_u8_i32x16` | `(u8x16) -> i32x16` | `--avx512` | Pre-existing. |
| `widen_i8_f32x4_4`, `widen_i8_f32x4_8`, `widen_i8_f32x4_12` (NEW v1.11.0) | `(i8x16) -> f32x4` | none | Lane-offset variants (4 lanes starting at byte 4 / 8 / 12). |
| `widen_u8_f32x4_4`, `widen_u8_f32x4_8`, `widen_u8_f32x4_12` (NEW v1.11.0) | `(u8x16) -> f32x4` | none | |
| `widen_u8_i32x4_4`, `widen_u8_i32x4_8`, `widen_u8_i32x4_12` (NEW v1.11.0) | `(u8x16) -> i32x4` | none | |
| `widen_u8_u16` (NEW v1.11.0) | `(u8x16) -> u16x8` | none | Zero-extend the **low 8 lanes**; upper 8 lanes of the input are discarded. |
| `narrow_f32x4_i8` | `(f32x4) -> i8x16` | none | Pre-existing. |
| `pack_sat_i16x8` (NEW v1.11.0) | `(i16x8, i16x8) -> i8x16` | none | Signed saturate to i8. |
| `pack_sat_i32x4` (NEW v1.11.0) | `(i32x4, i32x4) -> i16x8` | none | Signed saturate to i16. |
| `pack_sat_i16x16` (NEW v1.11.0) | `(i16x16, i16x16) -> i8x32` | **x86-only** | AVX2 `vpacksswb` (ARM uses `pack_sat_i16x8` twice). |
| `pack_sat_i32x8` (NEW v1.11.0) | `(i32x8, i32x8) -> i16x16` | **x86-only** | AVX2 `vpackssdw`. |
| `pack_usat_i16x8` (NEW v1.11.0) | `(i16x8, i16x8) -> u8x16` | none | Unsigned saturate to u8. |
| `pack_usat_i32x4` (NEW v1.11.0) | `(i32x4, i32x4) -> u16x8` | none | |
| `pack_usat_i16x16` (NEW v1.11.0) | `(i16x16, i16x16) -> u8x32` | **x86-only** | AVX2 `vpackuswb`. |
| `pack_usat_i32x8` (NEW v1.11.0) | `(i32x8, i32x8) -> u16x16` | **x86-only** | AVX2 `vpackusdw`. |
| `round_f32x4_i32x4` (NEW v1.11.0) | `(f32x4) -> i32x4` | none | Round-to-nearest. |
| `round_f32x8_i32x8` (NEW v1.11.0) | `(f32x8) -> i32x8` | none | |

## Comparison + selection + saturating arithmetic

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `select` | `(mask, a: TxN, b: TxN) -> TxN` | none | Per-lane blend. |
| `min` | `(TxN, TxN) -> TxN` for int / float | none | |
| `max` | `(TxN, TxN) -> TxN` for int / float | none | |
| `sat_add` (NEW v1.11.0) | `(T, T) -> T` for `i8x16 / u8x16 / i16x8 / u16x8` | none | Saturating add. |
| `sat_sub` (NEW v1.11.0) | `(T, T) -> T` for same families | none | Saturating sub. |
| `abs_diff` (NEW v1.11.0) | `(T, T) -> T` for `i8x16 / u8x16 / i16x8 / u16x8 / i32x4 / u32x4` | ARM-only | NEON `sabd` / `uabd`. |

## Bit ops

### Vector bitcasts and byte-shifts

| Name | Signature | Notes |
|---|---|---|
| `bitcast_i8x16` (NEW v1.11.0) | `(any 128-bit vec) -> i8x16` | Zero-cost LLVM `bitcast`. |
| `bitcast_i8x32` (NEW v1.11.0) | `(any 256-bit vec) -> i8x32` | |
| `bitcast_i32x4` (NEW v1.11.0) | `(any 128-bit vec) -> i32x4` | |
| `bitcast_i32x8` (NEW v1.11.0) | `(any 256-bit vec) -> i32x8` | |
| `bsrli_i8x16` (NEW v1.11.0) | `(i8x16, imm: i32) -> i8x16` | Byte-shift right logical. |
| `bsrli_i8x32` (NEW v1.11.0) | `(i8x32, imm: i32) -> i8x32` | x86 SSSE3+; on ARM the wide form errors and points at `bsrli_i8x16`. |
| `bslli_i8x16` (NEW v1.11.0) | `(i8x16, imm: i32) -> i8x16` | Byte-shift left logical. |
| `bslli_i8x32` (NEW v1.11.0) | `(i8x32, imm: i32) -> i8x32` | Same arch story as `bsrli_i8x32`. |

### Scalar bitwise (NEW v1.11.0 — language addition)

`&`, `|`, `^`, `<<`, `>>` are now valid on integer scalars (`i32`, `i64`,
`u32`, etc.), mirroring the existing dot-prefixed vector ops `.&`, `.|`,
`.^`, `.<<`, `.>>`.

### Pointer casts (NEW v1.11.0)

| Name | Signature | Notes |
|---|---|---|
| `ptr_as_i8` / `ptr_as_u8` | `(*T) -> *i8` / `*u8` | Zero-cost typed pointer cast. |
| `ptr_as_i16` / `ptr_as_u16` | `(*T) -> *i16` / `*u16` | |
| `ptr_as_i32` / `ptr_as_u32` | `(*T) -> *i32` / `*u32` | |
| `ptr_as_i64` / `ptr_as_u64` | `(*T) -> *i64` / `*u64` | |
| `ptr_as_f32` / `ptr_as_f64` | `(*T) -> *f32` / `*f64` | |

## Dot products

### Cross-platform

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `maddubs_i16` | `(u8x16, i8x16) -> i16x8` / `(u8x32, i8x32) -> i16x16` | x86-only; ARM uses `usmmla_i32` if `--i8mm`, else a manual zero-/sign-extend + multiply + `addp_i16` chain | SSSE3/AVX2 `pmaddubsw`. **`wmul_i16` alone is signed×signed and produces wrong results for `u8 >= 128`** — the ARM error spells out the safe portable recipe. |

### x86-only

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `madd_i16` | `(i16xN, i16xN) -> i32x(N/2)` for N ∈ {8, 16, 32} | 32-wide requires `--avx512` | SSE2/AVX2/AVX-512 `pmaddwd`. ARM error suggests `wmul_i32(lo, lo) + wmul_i32(hi, hi) + addp_i32`. |

### ARM-only

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `vdot_i32` | `(acc: i32x4, a: i8x16, b: i8x16) -> i32x4` | `--dotprod` | NEON `sdot` accumulating dot product. |
| `vdot_lane_i32` (NEW v1.11.0) | `(acc: i32x4, a: i8x16, b: i8x16, lane: imm) -> i32x4` | `--dotprod` | NEON `sdot`-by-lane. |
| `smmla_i32` | `(acc: i32x4, a: i8x16, b: i8x16) -> i32x4` | `--i8mm` | I8MM 8×8→32 signed matmul. |
| `ummla_i32` (NEW v1.11.0) | `(acc: i32x4, a: u8x16, b: u8x16) -> i32x4` | `--i8mm` | I8MM unsigned matmul. |
| `usmmla_i32` (NEW v1.11.0) | `(acc: i32x4, a: u8x16, b: i8x16) -> i32x4` | `--i8mm` | I8MM mixed-sign matmul — the canonical ARM replacement for `maddubs_i16`. |

## ARM-only widening multiply

| Name | Signature | Required flags | Notes |
|---|---|---|---|
| `wmul_i16` (NEW v1.11.0) | `(i8x8, i8x8) -> i16x8` | ARM-only | NEON `smull`. Signed only — see warning under `maddubs_i16`. |
| `wmul_u16` (NEW v1.11.0) | `(u8x8, u8x8) -> u16x8` | ARM-only | NEON `umull`. |
| `wmul_i32` (NEW v1.11.0) | `(i16x4, i16x4) -> i32x4` | ARM-only | NEON `smull`. |
| `wmul_u32` (NEW v1.11.0) | `(u16x4, u16x4) -> u32x4` | ARM-only | NEON `umull`. |

## Native FP16 (NEW v1.11.0 — ARM-only, requires `--fp16`)

`f16x4` and `f16x8` flow through every existing float intrinsic plus the
arithmetic operators. Each lowers to a native `<N x half>` NEON
instruction (`fadd v.8h`, `fmul v.8h`, `fmla v.8h`, `fcvtl`, etc.) on
Cortex-A76 and later. Without `--fp16` the compiler errors at the
library-API entry rather than falling back to a silent f32 round-trip.

| Op | Eä syntax | Result |
|---|---|---|
| Element-wise arithmetic | `a .+ b` / `a .- b` / `a .* b` / `a ./ b` on `f16xN` | `<N x half>` arithmetic |
| `splat` | `splat(scalar_f16) -> f16xN` | `dup v.8h, w` |
| `load` / `store` | `load(p: *f16, i) -> f16xN`, `store(p: *mut f16, i, v)` | aligned `<N x half>` load/store |
| `fma` | `fma(a, b, c)` on `f16xN` | `fmla v.8h` |
| `reduce_add`, `reduce_add_fast` | `reduce_add(v: f16xN) -> f16` | `llvm.vector.reduce.fadd.v{N}f16` → `faddv h` |
| `reduce_min`, `reduce_max` | same shape | `fminnmv`, `fmaxnmv` |

Scalar `f16` arithmetic (single-lane) follows the same rule: needs
`--fp16`. No literal syntax — get a scalar `f16` via `to_f16(...)`,
reduction, splat, or memory load.

## CLI flags affecting intrinsics

| Flag | Effect | Arch |
|---|---|---|
| `--avx512` | Enables AVX-512 width forms (`f32x16`, `i32x16`, `madd_i16(i16x32)`, `scatter`). | x86 only |
| `--fp16` (NEW v1.11.0) | Enables `f16x4` / `f16x8` compute. | ARM only — errors with `--fp16 is incompatible with non-ARM target` on x86 |
| `--dotprod` | Enables `vdot_i32`, `vdot_lane_i32`. | ARM only |
| `--i8mm` (NEW v1.11.0) | Enables `smmla_i32`, `ummla_i32`, `usmmla_i32`. | ARM only — errors with `error: --i8mm is only valid for AArch64 targets` on x86 |

## Breaking change — migrating from `maddubs_i32`

v1.11.0 removes the polymorphic `maddubs_i32(u8x16, i8x16) -> i32x4`. It
hid a two-instruction chain (`pmaddubsw + pmaddwd`) behind a single name,
which broke Eä's "programmer sees the cost" contract. Rewrite:

```ea
// Before (v1.10):
let r: i32x4 = maddubs_i32(a_u8, b_i8)

// After (v1.11.0):
let t: i16x8 = maddubs_i16(a_u8, b_i8)
let ones: i16x8 = splat(1i16)
let r: i32x4 = madd_i16(t, ones)
```

`maddubs_i16` keeps its asymmetric `(u8, i8)` shape. On ARM (no x86
`pmaddubsw` / `pmaddwd`), the canonical replacement is `usmmla_i32` with
`--i8mm`; without `--i8mm`, the error message at codegen names the safe
portable recipe (zero-extend the u8 operand, sign-extend the i8 operand,
multiply, `addp_i16` — `wmul_i16` alone is signed-only and **silently
wrong** for `u8 >= 128`).

## See also

- v1.11.0 inventory: `docs/release/v1.11.0/inventory.md`
- v1.11.0 changelog: `docs/release/v1.11.0/CHANGELOG.md`
- API consistency audit: `docs/release/v1.11.0/audit-findings.md`
- Test coverage matrix: `docs/release/v1.11.0/test-coverage.md`
- Cookbook entries: `docs/src/cookbook/fp16-inference.md`,
  `docs/src/cookbook/fast-transcendentals.md`,
  `docs/src/cookbook/neon-gather-workaround.md`
- Specs: `docs/superpowers/specs/2026-04-27-pi5-neon-enablement-design.md`,
  `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`
