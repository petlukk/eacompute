# Changelog — v1.11.0 (draft)

## [1.11.0] — 2026-05-13

### Added

#### CLI flags
- `--fp16` flag: appends `+fullfp16` to LLVM target features, enabling
  native f16 SIMD codegen on ARM (Pi 5 / Cortex-A76 and newer). Rejected
  on non-ARM targets.
- `--i8mm` flag: appends `+i8mm` to LLVM target features, gating
  `smmla_i32`, `ummla_i32`, `usmmla_i32`. AArch64 only.

#### Types
- Scalar `f16` (IEEE 754 half-precision) — recognized by lexer, parser,
  and typeck. Counts as float in `is_float()`; element size 16 bits.
- Vector types `f16x4`, `f16x8` — arithmetic, splat, load/store, fma, and
  reductions all lower to native NEON f16 instructions under `--fp16`.
- Additional SIMD vector tokens: `i8x8`, `u8x8`, `i16x4`, `u16x4`,
  `u16x8`, `u16x16`, `i32x2`, `u32x4`, `i8x64`, `u8x64`, `i16x32`.

#### Cross-platform intrinsics
- `abs(T) -> T` for scalar and vector floats / ints.
- `bitcast_i8x16`, `bitcast_i8x32`, `bitcast_i32x4`, `bitcast_i32x8` —
  zero-cost LLVM bitcasts between same-size vectors.
- AVX-512 lane intrinsics (also fine on smaller targets where LLVM
  pattern-matches the underlying `shufflevector` mask): `concat_*`,
  `lo128_*`, `hi128_*`, `lo256_*`, `hi256_*`, `shuffle_i32x{8,16}`,
  `blend_i32`, `bcast_even_pairs_i32x{8,16}`,
  `bcast_odd_pairs_i32x{8,16}`.
- `f32x4_from_scalars` and `f32x8_from_scalars` — gather compose
  primitives (canonical NEON gather workaround; also available on x86).
- `bsrli_i8x{16,32}` / `bslli_i8x{16,32}` — byte-shift left/right
  logical, immediate count.
- `cvt_f16_f32` / `cvt_f32_f16` — i16↔f32 pair via f16 (NEON `fcvtl/fcvtn`
  on ARM, F16C `vcvtph2ps/vcvtps2ph` on x86). `cvt_f16_f32` accepts
  widths 4/8/16: i16x4↔f32x4 cross-platform; i16x8↔f32x8 and
  i16x16↔f32x16 x86-only. `cvt_f32_f16` is symmetric only up to 8:
  f32x4→i16x4 cross-platform; f32x8→i16x8 x86-only.
- `round_f32x{4,8}_i32x{4,8}` — round-to-nearest f32→i32.
- `pack_sat_i16x8`, `pack_sat_i32x4` — signed saturation pack
  (cross-platform); wide AVX2 variants `pack_sat_i16x16`, `pack_sat_i32x8`
  are x86-only.
- `pack_usat_i16x8`, `pack_usat_i32x4` — unsigned saturation pack
  (cross-platform); wide variants `pack_usat_i16x16`, `pack_usat_i32x8`
  are x86-only.
- `sat_add`, `sat_sub` — saturating integer add/sub for SIMD vectors.
- `exp_poly_f32(f32xN) -> f32xN` — degree-5 minimax polynomial vector
  exp, defined on `[-50, 50]`, ~7–8 FMAs per lane, no libm call,
  no scalarization.
- `to_f16(x)` and `to_i16(x)` — scalar conversion intrinsics
  (companions to existing `to_f32`, `to_f64`, `to_i32`, `to_i64`).
- `ptr_as_i8`, `ptr_as_u8`, `ptr_as_i16`, `ptr_as_u16`, `ptr_as_i32`,
  `ptr_as_u32`, `ptr_as_i64`, `ptr_as_u64`, `ptr_as_f32`, `ptr_as_f64`
  — zero-cost typed pointer casts.
- `widen_u8_u16(u8x16) -> u16x8` — zero-extend the low 8 lanes of a
  u8x16 vector to u16x8 (upper 8 lanes of the source are discarded).
- Multi-width widen variants with lane offsets: `widen_i8_f32x4_{4,8,12}`,
  `widen_u8_f32x4_{4,8,12}`, `widen_u8_i32x4_{4,8,12}`.

#### x86-only intrinsics
- `madd_i16` — SSE2/AVX2/AVX-512 `pmaddwd` (widths 8, 16, 32).
- `hadd_i16` — SSSE3/AVX2 `phaddw` (widths 8, 16).

#### ARM-only intrinsics
- `abs_diff(T, T) -> T` — NEON `sabd`/`uabd` for 128-bit vectors.
- `addp_i16`, `addp_i32` — NEON pairwise add.
- `wmul_i16`, `wmul_u16`, `wmul_i32`, `wmul_u32` — NEON `smull`/`umull`
  widening multiply.
- `vdot_lane_i32` — NEON `sdot`-by-lane (requires `--dotprod`).
- `smmla_i32`, `ummla_i32`, `usmmla_i32` — ARMv8.6-A I8MM 8×8→32
  matrix-multiply-accumulate (requires `--i8mm`).
- Native f16 splat / load / store / FMA / `reduce_add` /
  `reduce_add_fast` / `reduce_min` / `reduce_max` (requires `--fp16`).

#### Language features
- Vector literal annotation form: `let v: i32x4 = [1, 2, 3, 4]` — the
  type annotation drives the element type; the suffix form
  (`[1i32, 2i32, ...]`) still works.
- Scalar bitwise operators: `&`, `|`, `^`, `<<`, `>>` on integer scalars
  (companion to the existing dot-prefixed vector ops `.&`, `.|`, etc.).

#### Documentation
- `docs/idioms/neon-gather.md` — canonical NEON-gather compose pattern.
- `docs/src/reference/intrinsics.md` — new entries for the v1.11.0 surface.
- `docs/src/cookbook/image-processing.md`, `ml-preprocessing.md` —
  worked examples for the new pack/saturating intrinsics.
- `docs/src/guide/common-intrinsics.md`, `docs/src/reference/arm.md`,
  `docs/src/reference/cli.md` — updates for new flags and surfaces.

### Changed

- `intrinsics.rs` split into nine per-family modules (`intrinsics_byteshift`,
  `intrinsics_conv`, `intrinsics_dotprod`, `intrinsics_f16`,
  `intrinsics_lane`, `intrinsics_memory`, `intrinsics_neon`,
  `intrinsics_pack`, `intrinsics_simd`) to keep each under the 500-line
  cap and group by capability.
- `simd.rs` codegen split into thirteen helper modules (`simd_arithmetic`,
  `simd_byteshift`, `simd_conv`, `simd_dotprod`, `simd_exp_poly`,
  `simd_fp16`, `simd_lane`, `simd_pack`, `simd_pack_unsigned`,
  `simd_saturating`, `simd_util`, `simd_wmul`, `simd_x86_dotprod`).
- `--dotprod` flag-handler refactored into the shared `append_feature`
  helper alongside `--fp16`/`--i8mm` (behavior unchanged; flag still
  ARM-only).
- NEON `gather()` error message rewritten to point at the new
  `f32x{4,8}_from_scalars` compose primitives and
  `docs/idioms/neon-gather.md`, instead of "use a scalar loop on ARM".
- Type-checker error messages improved for SIMD width / element-type
  mismatches across many intrinsics.
- `main.rs` slimmed (498 → 400 lines) by extracting the `ea bind`
  command into `src/bind_handler.rs` (115 lines).

### Fixed

- **Linux→Windows cross-compilation produces real PE32+ DLLs.** Commit
  `7c6a9c0`: the `SharedLib` linker step dispatched on the host triple
  (`#[cfg(target_os = "windows")]`), so cross-compiling from Linux to
  `x86_64-pc-windows-gnu` fell through to `cc -shared` and emitted ELF
  `.so` files dressed up as `.dll`. Dispatch now reads
  `opts.target_triple`: target=windows + host=windows uses `lld-link.exe`
  (unchanged); target=windows + host=linux uses
  `x86_64-w64-mingw32-gcc -shared -static-libgcc` (override with
  `WINDOWS_CC`); else falls back to `cc -shared -lm`. Validated by
  Olorin cross-building 39 SIMD kernels and loading them on Windows
  via `libloading`.
- **LLVM Machine Outliner disabled in hot loops.** Commits `f33ab3d` +
  `38fd50e`: the outliner extracted repeated instruction sequences into
  subroutines, producing `bl` calls that caused register spills and
  broke scheduling inside compute kernels (e.g. mins accumulation in
  Q4_K dot product). Global flag `-enable-machine-outliner=never` is
  now set inside `create_target_machine()` before the machine is
  constructed (the previous flag-setting path in `optimize_module()` ran
  too late to take effect).
- **`--emit-llvm` now runs optimization passes before dumping IR.**
  Commit `e455383`: previously `--emit-llvm` always printed unoptimized
  IR regardless of `--opt-level`, making it impossible to verify
  `alwaysinline` and other attributes. The IR dump now honors the
  optimization level.
- **`vpermq 0xD8` lane fixup after `vpackssdw`/`vpacksswb`.** Commit
  `6291514`: AVX2 pack instructions operate per 128-bit lane and
  produce interleaved output. A `vpermq` shuffle `[0,2,1,3]` is now
  emitted after each pack so callers see sequential element order
  (numerical correctness fix for AVX2 pack chains).
- **`pack_sat_i16x16` ARM codegen.** Commit `4658431`: the ARM
  split-concat implementation was missing; all three `pack_sat_*`
  intrinsics now have full cross-platform codegen.
- **`--fp16` gate fires for declared-but-unused f16 vector params.**
  Commit `a53c983`: `validate_type_for_target` previously only ran on
  params actually used in the function body, so declared-but-unused
  `f16x8` params bypassed the gate and panicked in `llvm_type()` on
  non-FP16 targets. The validator now runs on each param and on the
  return type at `declare_function` time.
- **`--fp16` cross-arch guard extended to `inspect_source` +
  `compile_to_ir`.** Commit `c5ed9d8`: the initial guard (`a53c983`)
  only covered `compile_with_options`, so `ea inspect --fp16` on x86
  bypassed it. The guard is now mirrored in `inspect_source` and
  `compile_to_ir_with_options` for full library-API coverage.

### Removed (breaking)

- `maddubs_i32` intrinsic — replaced with `madd_i16` per commit
  `89130cb`. The old name hid a 2-instruction chain (`pmaddubsw + pmaddwd`)
  behind one symbol, violating Eä's "programmer sees the cost" philosophy.
  **Migration:** rewrite `let r: i32x4 = maddubs_i32(a_u8, b_i8)` as the
  explicit chain:
  ```
  let t: i16x8 = maddubs_i16(a_u8, b_i8)
  let r: i32x4 = madd_i16(t, ones_i16x8)
  ```
  `maddubs_i16` is retained.

### Performance

- `exp_poly_f32` runs at ~10× scalar `exp()` on Pi 5 NEON (per spec:
  4× from vectorization × ~2.5× from polynomial vs libm at
  relative-error ≤ 2⁻¹⁸).
- Native f16 NEON eliminates the f32 round-trip on every KV read in
  Olorin's attention / RoPE / RMSNorm hot paths under `--fp16`.

### Notes

- `--fp16` and `--i8mm` are silently ignored at parse time on non-ARM
  hosts but produce an explicit error during target configuration:
  `error: --i8mm is only valid for AArch64 targets` and
  `--fp16 is incompatible with non-ARM target`.
- The two `cvt_*` intrinsics are cross-platform at the 128-bit width
  (i16x4 / f32x4) but x86-only at the 256-bit width (i16x8 / f32x8);
  ARM users get a clear error pointing at the narrower form.
- Existing `cvt_f16_f32` / `cvt_f32_f16` continue to work with or
  without `--fp16`; the flag only opts in to in-register f16 arithmetic.
- AVX-512 lane intrinsics use pure LLVM `shufflevector` with constant
  masks and rely on LLVM 18+ pattern-matching to the right ISA
  (`vinserti32x8`, `vextracti128`, `vpshufd`); no explicit
  AVX-512-only flag gate at the type-check level.
