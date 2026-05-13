# v1.11.0 Inventory

Branch: `feat/i8mm-intrinsics` → `main` (audit before merge)
Diff base: `origin/main`
HEAD: `7c6a9c0` (fix(codegen): SharedLib linker dispatches on opts.target_triple)
Commits in branch: 99
Files changed: 76, +11,766 / −1,189

The branch combines three logical layers:

1. **Pre-existing i8mm work** (≈73 commits): `vdot_lane_i32`, `bitcast_*`,
   sign-agnostic `shuffle_bytes`, `bsrli/bslli` (4 variants), `pack_usat`
   (4 variants) + `pack_sat` (4 variants), `shuffle_i32x{8,16}`, `blend_i32`,
   `smmla/ummla/usmmla`, AVX-512 lane intrinsics, AVX-512 int dot, vector
   literal annotation form, scalar bitwise operators, `widen_u8_u16`,
   multi-width widen variants, parallel bindings, inspect improvements.
2. **Pi 5 NEON Enablement** (19 commits): gather compose primitives
   (`f32x{4,8}_from_scalars`) + native FP16 compute (`--fp16`, `f16xN`
   types). Spec: `docs/superpowers/specs/2026-04-27-pi5-neon-enablement-design.md`.
3. **`exp_poly_f32`** (6 commits): polynomial vector exp. Spec:
   `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`.

Plus the head commit `7c6a9c0` (Windows cross-compile linker fix).

## New CLI Flags

| Flag | Effect | Arch |
|---|---|---|
| `--fp16` | Appends `+fullfp16` to LLVM target features; enables native f16 SIMD codegen (splat/load/store/fma/reductions) | ARM only — rejected with `--fp16 is incompatible with non-ARM target` on x86 |
| `--i8mm` | Appends `+i8mm` to LLVM target features; required for `smmla_i32` / `ummla_i32` / `usmmla_i32` | ARM only — rejected with `error: --i8mm is only valid for AArch64 targets` on x86 |

`--avx512` and `--dotprod` already existed in `origin/main`.

## New Types

| Type | Description | Arch |
|---|---|---|
| `f16` | New scalar floating-point type (16-bit, IEEE 754 half-precision). Recognized by lexer, parser, typeck (`Type::F16`), `is_float()` returns true, `size_bits()` returns 16. Arithmetic requires `--fp16` (ARM-only). | any (decl); ARM+`--fp16` for compute |
| `f16x4`, `f16x8` | Vector types over `f16` | ARM+`--fp16` for compute |
| `i8x8`, `u8x8`, `i16x4`, `u16x4`, `u16x8`, `u16x16`, `i32x2`, `u32x4`, `i8x64`, `u8x64`, `i16x32` | New SIMD vector type tokens added to the lexer for completeness; the wide x64/x32 types match AVX-512 BW width | varies by intrinsic consumer |

A scalar `to_f16` cast and a `ptr_as_f16`-adjacent `to_f16` conversion arm
were added alongside `to_i16` (the int conversion was already implied via
`Type::I16` but no `to_i16` arm existed).

## New Intrinsics

The classification rule: an intrinsic whose codegen uses `if self.is_arm`
to dispatch between an x86 LLVM intrinsic and an ARM LLVM intrinsic (or
that uses pure portable LLVM IR like `shufflevector` / `insertelement`)
is **Cross-platform**. An intrinsic that hard-errors with `is x86-only`
or `is ARM-only` is single-arch.

### Cross-platform

| Name | Signature | Source | Tests |
|---|---|---|---|
| `abs` | `(T) -> T` for scalar/vector float (or int via fabs/abs intrinsics) | `src/codegen/simd_arithmetic.rs` | `tests/abs_tests.rs` |
| `bitcast_i8x16` | `(any 128-bit vec) -> i8x16` | `src/codegen/simd_pack.rs` | `tests/phase_b_ext.rs` |
| `bitcast_i8x32` | `(any 256-bit vec) -> i8x32` | `src/codegen/simd_pack.rs` | `tests/phase_b_ext.rs` |
| `bitcast_i32x4` | `(any 128-bit vec) -> i32x4` | `src/codegen/simd_pack.rs` | `tests/phase_b_ext.rs` |
| `bitcast_i32x8` | `(any 256-bit vec) -> i32x8` | `src/codegen/simd_pack.rs` | `tests/phase_b_ext.rs` |
| `concat_i8x16` | `(i8x16, i8x16) -> i8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `concat_u8x16` | `(u8x16, u8x16) -> u8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `concat_i8x32` | `(i8x32, i8x32) -> i8x64` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `concat_u8x32` | `(u8x32, u8x32) -> u8x64` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `concat_i32x8` | `(i32x8, i32x8) -> i32x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `concat_f32x8` | `(f32x8, f32x8) -> f32x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `lo128_i8x32` | `(i8x32) -> i8x16` (low 128 lanes) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `hi128_i8x32` | `(i8x32) -> i8x16` (high 128 lanes) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `lo128_u8x32` | `(u8x32) -> u8x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `hi128_u8x32` | `(u8x32) -> u8x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `lo256_i8x64` | `(i8x64) -> i8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `hi256_i8x64` | `(i8x64) -> i8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `lo256_u8x64` | `(u8x64) -> u8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `hi256_u8x64` | `(u8x64) -> u8x32` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `lo256_i32x16` | `(i32x16) -> i32x8` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `hi256_i32x16` | `(i32x16) -> i32x8` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `lo256_f32x16` | `(f32x16) -> f32x8` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `hi256_f32x16` | `(f32x16) -> f32x8` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `shuffle_i32x8` | `(i32x8, mask: i32x8) -> i32x8` (constant mask) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `shuffle_i32x16` | `(i32x16, mask: i32x16) -> i32x16` (constant mask) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `blend_i32` | `(a, b, mask: imm) -> i32xN` (bit-mask blend) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `bcast_even_pairs_i32x8` | `(i32x8) -> i32x8` (broadcast even-indexed pairs) | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `bcast_odd_pairs_i32x8` | `(i32x8) -> i32x8` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `bcast_even_pairs_i32x16` | `(i32x16) -> i32x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs`, `tests/phase_b_avx512_arm_safety.rs` |
| `bcast_odd_pairs_i32x16` | `(i32x16) -> i32x16` | `src/codegen/simd_lane.rs` | `tests/phase_b_avx512_lane.rs` |
| `f32x4_from_scalars` | `(f32, f32, f32, f32) -> f32x4` | `src/codegen/simd_lane.rs` | `tests/phase14_arm_neon.rs` |
| `f32x8_from_scalars` | `(f32, f32, f32, f32, f32, f32, f32, f32) -> f32x8` | `src/codegen/simd_lane.rs` | `tests/phase14_arm_neon.rs` |
| `bslli_i8x16` | `(i8x16, imm: i32) -> i8x16` (byte-shift left logical) | `src/codegen/simd_byteshift.rs` (ARM branch + x86 branch) | `tests/phase14_byteshift.rs` |
| `bslli_i8x32` | `(i8x32, imm: i32) -> i8x32` | `src/codegen/simd_byteshift.rs` | `tests/phase14_byteshift.rs` |
| `bsrli_i8x16` | `(i8x16, imm: i32) -> i8x16` (byte-shift right logical) | `src/codegen/simd_byteshift.rs` | `tests/phase14_byteshift.rs` |
| `bsrli_i8x32` | `(i8x32, imm: i32) -> i8x32` | `src/codegen/simd_byteshift.rs` | `tests/phase14_byteshift.rs` |
| `cvt_f16_f32` | `(i16xN) -> f32xN` (i16x4→f32x4 cross-platform; i16x8→f32x8 x86 only) | `src/codegen/simd_pack.rs` | `tests/phase14_arm_ext.rs`, `tests/phase_b_avx2.rs`, `tests/phase_b_avx512_dotprod.rs` |
| `cvt_f32_f16` | `(f32xN) -> i16xN` (f32x4→i16x4 cross-platform; f32x8→i16x8 x86 only) | `src/codegen/simd_pack.rs` | `tests/phase14_arm_ext.rs`, `tests/phase_b_avx2.rs` |
| `round_f32x4_i32x4` | `(f32x4) -> i32x4` (round-to-nearest) | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `round_f32x8_i32x8` | `(f32x8) -> i32x8` | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `pack_sat_i16x8` | `(i16x8, i16x8) -> i8x16` (signed saturate to i8) | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `pack_sat_i32x4` | `(i32x4, i32x4) -> i16x8` | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `pack_usat_i16x8` | `(i16x8, i16x8) -> u8x16` (unsigned saturate to u8) | `src/codegen/simd_pack_unsigned.rs` | `tests/phase14_pack_unsigned.rs` |
| `pack_usat_i32x4` | `(i32x4, i32x4) -> u16x8` | `src/codegen/simd_pack_unsigned.rs` | `tests/phase14_pack_unsigned.rs` |
| `sat_add` | `(T, T) -> T` (saturating add for int vecs) | `src/codegen/simd_saturating.rs` (ARM branch + x86 generic intrinsic) | `tests/phase14_sat.rs` |
| `sat_sub` | `(T, T) -> T` (saturating sub) | `src/codegen/simd_saturating.rs` | `tests/phase14_sat.rs` |
| `exp_poly_f32` | `(f32xN) -> f32xN` (degree-5 minimax polynomial, defined on `[-50, 50]`) | `src/codegen/simd_exp_poly.rs` | `tests/phase14_exp_poly.rs` |
| `to_f16` | `(numeric scalar) -> f16` | `src/typeck/intrinsics.rs` + scalar conversion path | `tests/phase14_arm_fp16.rs`, `tests/data/rmsnorm_f16.ea` |
| `to_i16` | `(numeric scalar) -> i16` | `src/typeck/intrinsics.rs` + scalar conversion path | `tests/phase_b_dotprod.rs` |
| `ptr_as_i8` | `(*T) -> *i8` (zero-cost pointer cast) | `src/typeck/intrinsics.rs` `check_ptr_as` | none (Phase 4 gap) |
| `ptr_as_u8` | `(*T) -> *u8` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_i16` | `(*T) -> *i16` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_u16` | `(*T) -> *u16` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_i32` | `(*T) -> *i32` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_u32` | `(*T) -> *u32` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_i64` | `(*T) -> *i64` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_u64` | `(*T) -> *u64` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_f32` | `(*T) -> *f32` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `ptr_as_f64` | `(*T) -> *f64` | `src/typeck/intrinsics.rs` | none (Phase 4 gap) |
| `widen_u8_u16` | `(u8x16) -> u16x16` (zero-extend) | `src/codegen/simd_arithmetic.rs` | `tests/phase_b_avx2.rs` |
| `widen_i8_f32x4_4`, `widen_i8_f32x4_8`, `widen_i8_f32x4_12` | `(i8x16) -> f32x4` (lane-offset variants) | `src/codegen/simd.rs` (existing `compile_widen_i8_f32` reused) | `tests/phase14_widen.rs` |
| `widen_u8_f32x4_4`, `widen_u8_f32x4_8`, `widen_u8_f32x4_12` | `(u8x16) -> f32x4` | `src/codegen/simd.rs` | `tests/phase14_widen.rs` |
| `widen_u8_i32x4_4`, `widen_u8_i32x4_8`, `widen_u8_i32x4_12` | `(u8x16) -> i32x4` | `src/codegen/simd.rs` | `tests/phase14_widen.rs` |

### x86-only

| Name | Signature | Required flags | Source | Tests |
|---|---|---|---|---|
| `madd_i16` | `(i16xN, i16xN) -> i32x(N/2)` for N ∈ {8, 16, 32} (SSE2/AVX2/AVX-512 pmaddwd) | AVX-512 width requires `--avx512` at codegen-feature level | `src/codegen/simd_x86_dotprod.rs` | `tests/phase14_arm.rs` (rejection), `tests/phase_b_avx2.rs`, `tests/phase_b_avx512_dotprod.rs`, `tests/phase_b_dotprod.rs` |
| `hadd_i16` | `(i16x8, i16x8) -> i16x8` / `(i16x16, i16x16) -> i16x16` (SSSE3/AVX2 phaddw) | none extra | `src/codegen/simd_x86_dotprod.rs` | `tests/phase14_arm.rs` (rejection), `tests/phase_b_dotprod.rs` |
| `pack_sat_i16x16` | `(i16x16, i16x16) -> i8x32` (AVX2) | none | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `pack_sat_i32x8` | `(i32x8, i32x8) -> i16x16` (AVX2) | none | `src/codegen/simd_pack.rs` | `tests/phase14_pack.rs` |
| `pack_usat_i16x16` | `(i16x16, i16x16) -> u8x32` (AVX2 packuswb) | none | `src/codegen/simd_pack_unsigned.rs` | `tests/phase14_pack_unsigned.rs` |
| `pack_usat_i32x8` | `(i32x8, i32x8) -> u16x16` (AVX2 packusdw) | none | `src/codegen/simd_pack_unsigned.rs` | `tests/phase14_pack_unsigned.rs` |

Note: the wider `cvt_f16_f32(i16x8)` and `cvt_f32_f16(f32x8)` forms are
x86-only because they require 256-bit half register support; the i16x4 /
f32x4 forms are cross-platform.

### ARM-only

| Name | Signature | Required flags | Source | Tests |
|---|---|---|---|---|
| `abs_diff` | `(T, T) -> T` for `i8x16 / u8x16 / i16x8 / u16x8 / i32x4 / u32x4` (NEON sabd/uabd) | none | `src/codegen/simd_saturating.rs` | `tests/phase14_arm_neon.rs` |
| `addp_i16` | `(i16x8, i16x8) -> i16x8` (NEON addp pairwise add) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `addp_i32` | `(i32x4, i32x4) -> i32x4` (NEON addp) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `wmul_i16` | `(i8x8, i8x8) -> i16x8` (NEON smull widening multiply) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `wmul_u16` | `(u8x8, u8x8) -> u16x8` (NEON umull) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `wmul_i32` | `(i16x4, i16x4) -> i32x4` (NEON smull) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `wmul_u32` | `(u16x4, u16x4) -> u32x4` (NEON umull) | none | `src/codegen/simd_wmul.rs` | `tests/phase14_arm_neon.rs` |
| `vdot_lane_i32` | `(acc: i32x4, a: i8x16, b: i8x16, lane: imm) -> i32x4` (NEON sdot-by-lane) | `--dotprod` | `src/codegen/simd_dotprod.rs` | `tests/phase14_arm.rs`, `tests/phase_b_ext.rs` |
| `smmla_i32` | `(acc: i32x4, a: i8x16, b: i8x16) -> i32x4` (signed 8x8 matmul) | `--i8mm` | `src/codegen/simd_dotprod.rs` | `tests/phase14_arm.rs` |
| `ummla_i32` | `(acc: i32x4, a: u8x16, b: u8x16) -> i32x4` (unsigned 8x8 matmul) | `--i8mm` | `src/codegen/simd_dotprod.rs` | `tests/phase14_arm_i8mm.rs` |
| `usmmla_i32` | `(acc: i32x4, a: u8x16, b: i8x16) -> i32x4` (mixed-sign 8x8 matmul) | `--i8mm` | `src/codegen/simd_dotprod.rs` | `tests/phase14_arm_i8mm.rs` |
| Native f16 vector arithmetic (`+`, `-`, `*`, `/` on `f16xN`) | binary op desugar | `--fp16` | `src/codegen/simd_arithmetic.rs` (existing path; f16 vec flows through `is_float_type()`) | `tests/phase14_arm_fp16.rs` |
| `splat`, `load`, `store`, `fma`, `reduce_add`, `reduce_add_fast`, `reduce_min`, `reduce_max` on `f16xN` | extends existing intrinsics to f16 | `--fp16` | `src/codegen/simd_fp16.rs` | `tests/phase14_arm_fp16.rs` |

## New Source Files

| File | Lines | Responsibility |
|---|---|---|
| `src/bind_handler.rs` | 115 | C/Rust/Python/PyTorch/CMake binding-emit handler (extracted from `main.rs`) |
| `src/codegen/simd_byteshift.rs` | 133 | `bslli_i8x{16,32}` / `bsrli_i8x{16,32}` codegen |
| `src/codegen/simd_conv.rs` | 228 | Conversion-intrinsic codegen helpers |
| `src/codegen/simd_dotprod.rs` | 333 | `vdot_i32`, `vdot_lane_i32`, `smmla_i32`, `ummla_i32`, `usmmla_i32` (extracted from `simd.rs`) |
| `src/codegen/simd_exp_poly.rs` | 196 | `exp_poly_f32` polynomial codegen |
| `src/codegen/simd_fp16.rs` | 195 | Native f16 splat/load/store/fma/reductions, gated on `--fp16` |
| `src/codegen/simd_lane.rs` | 220 | AVX-512 lane intrinsics: `concat_*`, `lo*_*`, `hi*_*`, `shuffle_i32x{8,16}`, `blend_i32`, `bcast_*_pairs_*`, `f32x{4,8}_from_scalars` |
| `src/codegen/simd_pack.rs` | 436 | `pack_sat_*`, `round_f32x{4,8}_i32x{4,8}`, `bitcast_*`, `cvt_f16_f32`, `cvt_f32_f16` |
| `src/codegen/simd_pack_unsigned.rs` | 221 | `pack_usat_*` |
| `src/codegen/simd_saturating.rs` | 153 | `sat_add`, `sat_sub`, `abs_diff` |
| `src/codegen/simd_util.rs` | 328 | Shared codegen utilities for SIMD (extracted from `simd.rs`) |
| `src/codegen/simd_wmul.rs` | 210 | `addp_i{16,32}`, `wmul_{i,u}{16,32}` (NEON widening multiply + pairwise add) |
| `src/codegen/simd_x86_dotprod.rs` | 234 | `madd_i16`, `hadd_i16` (SSE2/SSSE3/AVX2/AVX-512 pmaddwd / phaddw) |
| `src/typeck/intrinsics_byteshift.rs` | 110 | typeck for byte-shift intrinsics |
| `src/typeck/intrinsics_dotprod.rs` | 390 | typeck for dot-product / matmul intrinsics |
| `src/typeck/intrinsics_f16.rs` | 71 | typeck for `cvt_f16_f32` / `cvt_f32_f16` |
| `src/typeck/intrinsics_lane.rs` | 236 | typeck for AVX-512 lane intrinsics (`concat`, `lo/hi_extract`, `shuffle_i32`, `blend_i32`, `bcast_pairs`) |
| `src/typeck/intrinsics_neon.rs` | 240 | typeck for ARM NEON intrinsics (`abs_diff`, `addp_i{16,32}`, `wmul_{i,u}{16,32}`) |
| `src/typeck/intrinsics_pack.rs` | 339 | typeck for pack/round intrinsics |
| `src/typeck/intrinsics_simd.rs` | 273 (added 33) | New entries `check_f32_from_scalars`, `check_exp_poly_f32`, `check_widen_u8_u16` |

All new source files are ≤ 500 lines (per the hard rule). The largest is
`src/codegen/simd_pack.rs` at 436 lines.

## New Test Files

| File | Lines | Coverage |
|---|---|---|
| `tests/abs_tests.rs` | 184 | Scalar / vector `abs` intrinsic |
| `tests/common/mod.rs` | 2 | Shared test helpers (very small stub) |
| `tests/data/rmsnorm_f16.ea` | 29 | f16 RMSNorm end-to-end fixture |
| `tests/lex_wide_vec_types.rs` | 28 | New SIMD vector type tokens (i8x64, u8x64, etc.) |
| `tests/phase14_arm_ext.rs` | 141 | `cvt_f16_f32` / `cvt_f32_f16` on ARM |
| `tests/phase14_arm_fp16.rs` | 642 | Native f16 NEON: arithmetic, splat, load/store, fma, reductions |
| `tests/phase14_arm_i8mm.rs` | 192 | `ummla_i32`, `usmmla_i32` |
| `tests/phase14_arm_neon.rs` | 355 | NEON gather compose (`f32x{4,8}_from_scalars`), `abs_diff`, `addp_i{16,32}`, `wmul_*`, gather error message |
| `tests/phase14_bitwise.rs` | 112 | Scalar bitwise operators (`&`, `\|`, `^`, `<<`, `>>`) |
| `tests/phase14_byteshift.rs` | 161 | `bsrli_i8x{16,32}`, `bslli_i8x{16,32}` |
| `tests/phase14_exp_poly.rs` | 527 | `exp_poly_f32`: accuracy, range, IR regression guard, softmax integration, negative type tests |
| `tests/phase14_pack.rs` | 197 | `pack_sat_*`, `round_f32x{4,8}_i32x{4,8}` |
| `tests/phase14_pack_unsigned.rs` | 154 | `pack_usat_*` |
| `tests/phase14_sat.rs` | 309 | `sat_add`, `sat_sub` |
| `tests/phase14_widen.rs` | 170 | `widen_*_*` lane-offset variants, `widen_u8_u16` |
| `tests/phase_b_avx512_arm_safety.rs` | 179 | Verify AVX-512 lane intrinsics emit arch-safe errors on ARM |
| `tests/phase_b_avx512_dotprod.rs` | 208 | AVX-512 `madd_i16` (32-wide) |
| `tests/phase_b_avx512_lane.rs` | 589 | AVX-512 lane intrinsics: `concat_*`, `lo/hi_*`, `shuffle_i32x{8,16}`, `blend_i32`, `bcast_*_pairs_*` |
| `tests/phase_b_dotprod.rs` | 272 | AVX2 `madd_i16`, `hadd_i16`, `to_i16` |
| `tests/phase_b_ext.rs` | 125 | `bitcast_*` extension tests, `vdot_lane_i32` |
| `tests/vector_literal_tests.rs` | 181 | Vector literal annotation form (`let v: i32x4 = [1,2,3,4]`) |

`tests/phase_b.rs` itself was retained (still present at HEAD) but
heavily refactored alongside the `phase_b_*` family expansion. The new
`phase_b_*` test files above are net-additions covering AVX2 / AVX-512 /
ARM-safety / dotprod / ext surfaces split out as the suite grew.

Two test files exceed 500 lines (`phase14_arm_fp16.rs` at 642 and
`phase_b_avx512_lane.rs` at 589). The 500-line rule applies to source
files; these are tests but worth a Phase 2 split if the project policy
extends to tests.

## Notable Bug Fixes

- **`7c6a9c0` — Windows cross-compile linker fix.** `SharedLib` linker
  dispatch previously gated on `#[cfg(target_os = "windows")]` (host
  triple), so Linux→Windows cross-compilation (`--target-triple=x86_64-pc-windows-gnu`)
  fell through to `cc -shared` and produced ELF `.so` files dressed up
  as `.dll`. New dispatch:
  - target windows + host windows → `lld-link.exe` (unchanged)
  - target windows + host non-windows → `x86_64-w64-mingw32-gcc -shared
    -static-libgcc` (override with `WINDOWS_CC` env)
  - else → `cc -shared -lm` (unchanged)

  Validated by Olorin's windows-port: `ea` now cross-compiles 39 SIMD
  kernels into PE DLLs that load via `libloading::Library::new` on
  Windows. (`src/lib.rs`, +39/−39.)

- **`maddubs_i32` → `madd_i16` replacement** (commit `89130cb`,
  breaking change). The old single intrinsic hid a 2-instruction chain
  (`pmaddubsw + pmaddwd`) behind one name, violating the "programmer
  sees the cost" philosophy. The chain is now explicit: callers write
  `let t: i16x8 = maddubs_i16(a, b); let r: i32x4 = madd_i16(t, ones)`.
  This is the only breaking change in v1.11.0.

- **NEON `gather()` error rewrite.** `src/codegen/simd_masked.rs:211`
  now points the user at `f32x{4,8}_from_scalars` + `docs/idioms/neon-gather.md`
  instead of saying "use a scalar loop on ARM". Covered by
  `tests/phase14_arm_neon.rs` negative tests.

## Phase 4 Gaps Noted Here

These are tests that should exist but do not; flagged for Phase 4, not
fixed in this audit:

- All ten `ptr_as_*` intrinsics: no tests under `tests/`. The intrinsic
  is implemented in `src/typeck/intrinsics.rs` (`check_ptr_as`) but no
  test file exercises it. (`tests/ptr_ptr_tests.rs` is an unrelated
  pre-existing file.)

## Concerns / Notes

- **Scalar `to_f16` vs intrinsic `to_f16`.** The branch adds `to_f16`
  to the conversion arm in `intrinsics.rs` (alongside `to_f32`, `to_f64`,
  `to_i16`, `to_i32`, `to_i64`). The user-facing surface is the scalar
  conversion `to_f16(x)` — same shape as `to_f32`. No separate intrinsic
  file exists; the new `intrinsics_f16.rs` covers only the vector
  bit-level `cvt_*` pair.
- **Vector literal annotation** (`let v: i32x4 = [1,2,3,4]`) is not a
  named intrinsic but a parser feature — counted as a language addition,
  not an intrinsic.
- **Scalar bitwise operators** (`&`, `|`, `^`, `<<`, `>>` on integer
  scalars) are also language additions, not intrinsics. Coverage:
  `tests/phase14_bitwise.rs`.
