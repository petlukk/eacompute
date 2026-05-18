# Changelog

## v1.14.0 — UNRELEASED

### Added

- `permute_runtime(f32x8, i32x8) -> f32x8` and `permute_runtime(i32x8, i32x8) -> i32x8` intrinsics. Lowers to `vpermps` / `vpermd` on x86 (AVX2). ARM is rejected with a codegen error pointing at the [NEON runtime-permute workaround](docs/src/cookbook/neon-runtime-permute-workaround.md). Motivating consumer: `autoresearch/kernels/particle_life/kernel_v113.ea` (measured 2.93× over scalar on Zen 4). See `docs/src/reference/intrinsics.md` for the full spec.
- `prefetch_write(ptr, offset)` — write-intent prefetch hint. Lowers to `prefetchw` on x86 (requires `PRFCHW` CPUID; fallback `prefetcht0` on older targets) / `prfm pstl1keep` on aarch64. Reuses the existing `prefetch` typeck path and `llvm.prefetch.p0` codegen with `(rw=1, locality=3)`. Motivating consumer: chacha20 ciphertext stores; eakv dequantize destinations.
- `prefetch_nta(ptr, offset)` — non-temporal prefetch hint, bypasses L2/L3 caching. Lowers to `prefetchnta` on x86 / `prfm pldl1strm` on aarch64. Same typeck/codegen paths with `(rw=0, locality=0)`. Motivating consumer: streaming-read kernels (Q4 dequantize input, large one-pass scans).
- `shuffle(a, b, [indices])` — two-source compile-time shuffle. Indices in `[0, width)` select from `a`; indices in `[width, 2 * width)` select from `b`. Lowers via LLVM `shufflevector`; backends pick instruction (x86 `vunpcklps` / `vblendps`, aarch64 `zip1` / `tbl`). Existing single-source `shuffle(v, [indices])` is unchanged. Motivating consumers: interleave / zip primitives, lane-by-lane blends, ChaCha-style concatenate-permute across two blocks.

## v1.13.0 — 2026-05-15 — ea bench + first aarch64 baselines + Specification umbrella

Standing benchmark suite for the v1.11.0 audit kernels. Converts performance regression detection from vigilance-dependent to mechanical: `ea bench <manifest.toml>` builds an `.ea` kernel + C harness, runs the harness pinned to one core (`taskset` on Linux), captures JSONL measurements, wraps them with environment metadata, and diffs against a committed baseline JSON. Day-one manifests cover `exp_poly_f32` (x86_64 only — kernel uses `f32x8`), `fp16_kv` (aarch64), and `gather_compose` (x86 + ARM variants), with baselines captured on the maintainer's x86_64 dev host and the Raspberry Pi 5 (Cortex-A76). Warn-only regression gate at 10% in v1.13.0 — regressions print `WARNING:` but the process still exits 0 so we collect runner-variance signal for one release before deciding the threshold. Companion docs: a new Specification umbrella at `docs/src/reference/index.md` gives the language reference a single canonical entry point, and `RELEASING.md` documents the maintainer pre-tag ritual.

### Added

#### Tooling
- **`ea bench <manifest.toml>` subcommand.** Builds an `.ea` kernel + C harness, runs the harness pinned to one core (`taskset`-gated on Linux), captures JSONL measurements on stdout, relays harness stderr with a `[harness] ` prefix, and emits a single result JSON to stdout (or `--out PATH`). Flags: `--target=`, `--avx512`, `--fp16`, `--i8mm`, `--dotprod`, `--opt-level=`, `--update-baseline`, `--no-diff`, `--out PATH`. See `docs/src/reference/bench.md` for the manifest schema and harness contract.
- **JSONL harness contract.** The v1.11.0 audit harnesses (`exp_poly_f32_harness.c`, `fp16_kv_harness.c`, `gather_compose_harness.c`) now emit one `{"kernel":"...","median_ns":N,...}` line per measurement on stdout, with banners and verify messages on stderr. The existing methodology (deterministic LCG fill, warmup, median of N runs of M inner calls, volatile sink) is unchanged.
- **Committed baselines for x86_64.** `benchmarks/v1.11.0/exp_poly_f32.baseline.json` and `gather_compose_x86.baseline.json` captured on the maintainer's dev host. Future runs diff against these.
- **Committed baselines for aarch64 (Cortex-A76).** `benchmarks/v1.11.0/fp16_kv.baseline.json` and `gather_compose_arm.baseline.json` captured on Raspberry Pi 5 (`taskset`-pinned, `--fp16` / `--dotprod` enabled, eacompute `88da108`). Both surface design-spec gaps the bench was built to make visible — see Notes below.
- **CI smoke step.** `.github/workflows/ci.yml` runs `ea bench benchmarks/v1.11.0/exp_poly_f32.bench.toml` on the x86 Linux job and uploads the result JSON as a build artifact. Warn-only this release (`|| true` guard).
- **Reference doc.** `docs/src/reference/bench.md` documents the manifest schema, harness contract, output schema, diff/baseline semantics, and a "how to add a new benchmark" recipe.

#### Documentation
- **Specification umbrella.** New `docs/src/reference/index.md` names the seven existing reference files (`types`, `intrinsics`, `cli`, `arm`, `bindings`, `python-api`, `bench`) as the normative specification, with stability/deprecation notes and a "spec wins" pointer over informative Guide and Cookbook content. Linked from the README as the single canonical "this is the spec" entry point. Closes the v1.13.0 roadmap item asking for a single doc to link as "the spec"; takes the umbrella shape rather than re-attempting a monolithic file (the previous `Specification.md` was actively removed in commit `48fa313` and stayed gitignored).
- **`RELEASING.md`.** Top-level maintainer release checklist: refresh aarch64 baselines on the Pi 5 first (intermittent access), refresh x86_64 baselines on the dev host, finalize CHANGELOG date + version bump + migration docs + `docs/public-api.txt` snapshot, run `cargo test`, tag and push. CI handles the rest via `.github/workflows/release.yml`. Replaces the v1.13.0 roadmap's "Pi 5 self-hosted CI runner" item, which is not viable given intermittent Pi access.

### Fixed

- **`exp_poly_f32.bench.toml` arch declaration.** Manifest previously declared `arch = ["x86_64", "aarch64"]`, but the kernel uses `f32x8` throughout — strictly AVX2, rejected by the type checker on ARM (`f32x8 requires AVX2; use f32x4 on ARM`). Narrowed to `arch = ["x86_64"]`. A polynomial-exp variant for ARM would need an `f32x4`-width kernel and would ship as a separate `exp_poly_f32_arm.bench.toml`; no consumer has asked for it yet.

### Notes

- **aarch64 baselines surface microarch ceilings, not regressions or codegen bugs.** Disassembly investigation root-caused both gaps to Cortex-A76 hardware limits rather than codegen misses. `fp16_kv`'s `kv_native` runs at **0.85×** of `kv_roundtrip` (4114 vs 3501 ns); codegen is native `fmla.8h` / `fmul.8h`, but A76 issues f16 and f32 SIMD ops at the same per-cycle rate — the lane-doubling benefit only materializes on Neoverse-V1/V2 (and later A-series A78+). `gather_compose_arm`'s `compose_x4` runs at **0.96×** of `scalar_loop` (98743 vs 94713 ns); the disassembly is 4 dependent scalar loads feeding 1 vector store, which is the structural ceiling for unbounded-LUT gather on NEON (no real gather instruction; SVE2's `LD1W` is unavailable on A76). Both committed as honest A76 measurements, not regressions versus a prior release. Tracked as follow-up issues for spec-target wording refinement.
- **Pi 5 self-hosted runner deferred.** Maintainer access to the Pi (`peter@10.46.0.27`) is intermittent — a flaky self-hosted runner trains everyone to ignore CI signal. This release adopts a "run `ea bench` at release-cut" ritual instead of per-PR CI. See [`RELEASING.md`](RELEASING.md).
- Autoresearch archive integration and pre-v1.11.0 benchmark migration (`fma_kernel`, `horizontal_reduction`) are out of scope for v1.13.0. Both consume `ea bench` once it exists.
- Committed baselines are host-specific; CI runner deltas (often 20%+ on libm-backed kernels) are expected and warn-only for v1.13.0.

## v1.12.0 — 2026-05-13 — deprecation infrastructure + u64 widening multiply + typed sat_add/sat_sub/abs_diff

Seven PRs since v1.11.0. The intrinsic surface now has typed spellings for `sat_add` / `sat_sub` / `abs_diff` (polymorphic forms deprecated, removal v2.0.0), a cross-platform u32×u32→u64 widening multiply (`wmul_u64_lo`/`hi`, unblocks Poly1305), and additions that close API symmetries (`widen_u8_u16_hi`, `cvt_f32_f16` width-16, i16/u16 lane extractors). New `u64x{2,4,8}` vector types underpin the widening-multiply work. A deprecation-warning runtime + `docs/migrations/` directory + `cargo public-api` CI gate land alongside, exercised by the rename batch. Two latent codegen bugs from v1.11.0 caught and fixed by the new end-to-end test discipline. 847 tests on 3-platform CI (x86 Linux, Linux ARM64, Windows).

### Added

#### Cross-platform intrinsics
- `wmul_u64_lo(u32x4, u32x4) -> u64x2` and `wmul_u64_hi(u32x4, u32x4) -> u64x2` — unsigned widening multiply, u32×u32 → u64. `_lo` widens logical lanes 0,1; `_hi` widens logical lanes 2,3. Pair them to widen all four u32 lanes. **First cross-platform `wmul_*` family member** — every other `wmul_*` is ARM-only.
  - **ARM**: `umull v.2d, v.2s, v.2s` (lo) / `umull2 v.2d, v.4s, v.4s` (hi) — single instruction each via `llvm.aarch64.neon.umull.v2i64` with the appropriate shufflevector pattern (LLVM 18 pattern-matches the high-half extract+umull to `umull2`).
  - **x86**: lowers via the canonical IR pattern `mul(zext, zext)`; backend emits `vpmuludq` after `vpmovzxdq` (lo) or `vpshufd` (hi). No reliance on the deprecated `llvm.x86.sse2.pmulu.dq` intrinsic (removed in LLVM 7+).
  - **Motivating use case**: Poly1305 5-limb radix-2^26 accumulator (25 widening multiplies per block).
- `widen_u8_u16_hi(u8x16) -> u16x8` — zero-extend the **high** 8 lanes of a `u8x16` to a `u16x8`. Sibling of `widen_u8_u16` (low half). ARM: `ushll2 v.8h, v.16b, #0` (1 insn). x86: `vpxor + vpunpckhbw` (2 insn).
- `lo128_i16x16` / `hi128_i16x16` / `lo128_u16x16` / `hi128_u16x16` / `lo256_i16x32` / `hi256_i16x32` — lane extractors filling the i16/u16 gap in the existing `lo*`/`hi*` family. Pure dispatch wiring; reuses the generic `compile_lo_extract` / `compile_hi_extract`. ARM emits `ushll`/`ushll2` (where 128-bit input applies); x86 emits `vextractf128 $1` for `hi128_*` and free truncation for `lo128_*`.
- `cvt_f32_f16(f32x16) -> i16x16` — AVX-512 width-16 form. Closes the asymmetry: `cvt_f16_f32` already accepted widths `{4, 8, 16}`; `cvt_f32_f16` was capped at `{4, 8}`. Lowers via existing `fptrunc <16 x float> to <16 x half>` → single `vcvtps2ph $4, %zmm0, %ymm0`.

#### Typed (monomorphic) spellings for sat_add / sat_sub / abs_diff
- 14 new intrinsic names; the polymorphic parents continue to compile but emit a deprecation warning. See **Deprecated** section.
  - `sat_add_i8x16`, `sat_add_u8x16`, `sat_add_i16x8`, `sat_add_u16x8` (cross-platform).
  - `sat_sub_i8x16`, `sat_sub_u8x16`, `sat_sub_i16x8`, `sat_sub_u16x8` (cross-platform).
  - `abs_diff_i8x16`, `abs_diff_u8x16`, `abs_diff_i16x8`, `abs_diff_u16x8`, `abs_diff_i32x4`, `abs_diff_u32x4` (ARM-only).
- Codegen for all typed spellings forwards to the existing `compile_sat_add` / `compile_sat_sub` / `compile_abs_diff` — same lowering, no behavior change. First real exercise of the deprecation-warning runtime.

#### Types
- `u64x2` (128-bit, both platforms), `u64x4` (256-bit, x86 AVX2), `u64x8` (512-bit, x86 AVX-512). Vector types over `u64`; mirror the existing `f64x{2,4}` plumbing. ARM rejects `u64x4` / `u64x8` at the type-validation site with a narrowing hint ("u64xN requires AVX-512/AVX2; use u64x2 on ARM").
- New lexer tokens: `TokenKind::U64x2`, `U64x4`, `U64x8`. Additive — no rename of existing tokens.

#### Deprecation-warning infrastructure
- `src/typeck/deprecations.rs`: `DeprecationInfo`, `DeprecationWarning`, and a `DEPRECATED_INTRINSICS` table. Calling an intrinsic listed in the table records a warning on the active `TypeChecker` (the intrinsic still compiles normally).
- `TypeChecker::with_deprecations(table)` for tests; `TypeChecker::warnings()` accessor; `ea_compiler::check_types_with_warnings(stmts)` library entry point.
- The compiler driver (`ea` CLI, `compile_with_options`, `compile_to_ir_with_options`, `inspect_source`) prints any collected warnings to stderr after a successful type check.
- Production `DEPRECATED_INTRINSICS` now has three entries: the polymorphic `sat_add`, `sat_sub`, `abs_diff` (first real consumers).

#### Migration documentation
- New `docs/migrations/` directory with one file per breaking release. `docs/migrations/README.md` documents the deprecation-cycle policy: minor-release warning → at least one full minor cycle → removal in the next major release.
- `docs/migrations/v1.11.0.md` retroactively documents the `maddubs_i32` → `maddubs_i16` + `madd_i16` migration.
- `docs/migrations/v1.12.0.md` covers the sat_add / sat_sub / abs_diff typed-spelling migration, plus a summary of the additive changes that don't need migration.

#### CI
- New `public-api-check` job in `.github/workflows/ci.yml`: runs `cargo public-api --simplified` and diffs against `docs/public-api.txt`. Fails CI on unintended public-API drift; intentional changes require updating the snapshot in the same PR.
- Initial snapshot committed at `docs/public-api.txt`.

### Deprecated

- **Polymorphic `sat_add` / `sat_sub` / `abs_diff` — use the typed spellings.** The polymorphic forms continue to compile but emit a deprecation warning at each call site, pointing at the typed replacement. Removal scheduled for v2.0.0. See `docs/migrations/v1.12.0.md` for the migration recipes.

### Fixed

- **x86 codegen for `sat_add` / `sat_sub` no longer produces an unresolved symbol at link time.** The previous codegen called `llvm.x86.sse2.padds.b` / `paddus.b` / `psubs.w` etc. directly, but LLVM 7+ removed those target-specific intrinsics in favor of the canonical target-independent forms (`llvm.sadd.sat.vNiM`, `llvm.uadd.sat.vNiM`, etc.). The old names link-failed with `undefined reference to llvm.x86.sse2.padds.b` whenever a program was actually built into an executable. Existing tests only compiled to object file, so the bug went unnoticed in v1.11.0. Codegen now uses the canonical intrinsic names, which the backend lowers to `padds`/`paddus` on x86 and `sqadd`/`uqadd` on ARM. Removes the x86/ARM branching in `src/codegen/simd_saturating.rs`.
- **`wmul_u64` codegen on x86** — caught during development of the new intrinsic: the analogous `llvm.x86.sse2.pmulu.dq` was also removed in LLVM 7+; new codegen uses canonical `mul(zext, zext)` IR. Same lesson, separate fix.

## v1.11.0 — 2026-05-13 — ARM I8MM + FP16 intrinsics, exp_poly_f32, pack/saturate surface

### Added

#### CLI flags
- `--fp16` flag: appends `+fullfp16` to LLVM target features, enabling native f16 SIMD codegen on ARM (Pi 5 / Cortex-A76 and newer). Rejected on non-ARM targets.
- `--i8mm` flag: appends `+i8mm` to LLVM target features, gating `smmla_i32`, `ummla_i32`, `usmmla_i32`. AArch64 only.

#### Types
- Scalar `f16` (IEEE 754 half-precision) — recognized by lexer, parser, and typeck. Counts as float in `is_float()`; element size 16 bits.
- Vector types `f16x4`, `f16x8` — arithmetic, splat, load/store, fma, and reductions all lower to native NEON f16 instructions under `--fp16`.
- Additional SIMD vector tokens: `i8x8`, `u8x8`, `i16x4`, `u16x4`, `u16x8`, `u16x16`, `i32x2`, `u32x4`, `i8x64`, `u8x64`, `i16x32`.

#### Cross-platform intrinsics
- `abs(T) -> T` for scalar and vector floats / ints.
- `bitcast_i8x16`, `bitcast_i8x32`, `bitcast_i32x4`, `bitcast_i32x8` — zero-cost LLVM bitcasts between same-size vectors.
- AVX-512 lane intrinsics (also fine on smaller targets where LLVM pattern-matches the underlying `shufflevector` mask): `concat_*`, `lo128_*`, `hi128_*`, `lo256_*`, `hi256_*`, `shuffle_i32x{8,16}`, `blend_i32`, `bcast_even_pairs_i32x{8,16}`, `bcast_odd_pairs_i32x{8,16}`.
- `f32x4_from_scalars` and `f32x8_from_scalars` — gather compose primitives (canonical NEON gather workaround; also available on x86).
- `bsrli_i8x{16,32}` / `bslli_i8x{16,32}` — byte-shift left/right logical, immediate count.
- `cvt_f16_f32` / `cvt_f32_f16` — i16↔f32 pair via f16 (NEON `fcvtl/fcvtn` on ARM, F16C `vcvtph2ps/vcvtps2ph` on x86). `cvt_f16_f32` accepts widths 4/8/16: i16x4↔f32x4 cross-platform; i16x8↔f32x8 and i16x16↔f32x16 x86-only. `cvt_f32_f16` is symmetric only up to 8: f32x4→i16x4 cross-platform; f32x8→i16x8 x86-only.
- `round_f32x{4,8}_i32x{4,8}` — round-to-nearest f32→i32.
- `pack_sat_i16x8`, `pack_sat_i32x4` — signed saturation pack (cross-platform); wide AVX2 variants `pack_sat_i16x16`, `pack_sat_i32x8` are x86-only.
- `pack_usat_i16x8`, `pack_usat_i32x4` — unsigned saturation pack (cross-platform); wide variants `pack_usat_i16x16`, `pack_usat_i32x8` are x86-only.
- `sat_add`, `sat_sub` — saturating integer add/sub for SIMD vectors.
- `exp_poly_f32(f32xN) -> f32xN` — degree-5 minimax polynomial vector exp, defined on `[-50, 50]`, ~7–8 FMAs per lane, no libm call, no scalarization.
- `to_f16(x)` and `to_i16(x)` — scalar conversion intrinsics (companions to existing `to_f32`, `to_f64`, `to_i32`, `to_i64`).
- `ptr_as_i8`, `ptr_as_u8`, `ptr_as_i16`, `ptr_as_u16`, `ptr_as_i32`, `ptr_as_u32`, `ptr_as_i64`, `ptr_as_u64`, `ptr_as_f32`, `ptr_as_f64` — zero-cost typed pointer casts.
- `widen_u8_u16(u8x16) -> u16x8` — zero-extend the low 8 lanes of a u8x16 vector to u16x8 (upper 8 lanes of the source are discarded).
- Multi-width widen variants with lane offsets: `widen_i8_f32x4_{4,8,12}`, `widen_u8_f32x4_{4,8,12}`, `widen_u8_i32x4_{4,8,12}`.

#### x86-only intrinsics
- `madd_i16` — SSE2/AVX2/AVX-512 `pmaddwd` (widths 8, 16, 32).
- `hadd_i16` — SSSE3/AVX2 `phaddw` (widths 8, 16).

#### ARM-only intrinsics
- `abs_diff(T, T) -> T` — NEON `sabd`/`uabd` for 128-bit vectors.
- `addp_i16`, `addp_i32` — NEON pairwise add.
- `wmul_i16`, `wmul_u16`, `wmul_i32`, `wmul_u32` — NEON `smull`/`umull` widening multiply.
- `vdot_lane_i32` — NEON `sdot`-by-lane (requires `--dotprod`).
- `smmla_i32`, `ummla_i32`, `usmmla_i32` — ARMv8.6-A I8MM 8×8→32 matrix-multiply-accumulate (requires `--i8mm`).
- Native f16 splat / load / store / FMA / `reduce_add` / `reduce_add_fast` / `reduce_min` / `reduce_max` (requires `--fp16`).

#### Language features
- Vector literal annotation form: `let v: i32x4 = [1, 2, 3, 4]` — the type annotation drives the element type; the suffix form (`[1i32, 2i32, ...]`) still works.
- Scalar bitwise operators: `&`, `|`, `^`, `<<`, `>>` on integer scalars (companion to the existing dot-prefixed vector ops `.&`, `.|`, etc.).

#### Documentation
- `docs/idioms/neon-gather.md` — canonical NEON-gather compose pattern.
- `docs/src/reference/intrinsics.md` — new entries for the v1.11.0 surface.
- `docs/src/cookbook/image-processing.md`, `ml-preprocessing.md` — worked examples for the new pack/saturating intrinsics.
- `docs/src/guide/common-intrinsics.md`, `docs/src/reference/arm.md`, `docs/src/reference/cli.md` — updates for new flags and surfaces.

### Changed

- `intrinsics.rs` split into nine per-family modules (`intrinsics_byteshift`, `intrinsics_conv`, `intrinsics_dotprod`, `intrinsics_f16`, `intrinsics_lane`, `intrinsics_memory`, `intrinsics_neon`, `intrinsics_pack`, `intrinsics_simd`) to keep each under the 500-line cap and group by capability.
- `simd.rs` codegen split into thirteen helper modules (`simd_arithmetic`, `simd_byteshift`, `simd_conv`, `simd_dotprod`, `simd_exp_poly`, `simd_fp16`, `simd_lane`, `simd_pack`, `simd_pack_unsigned`, `simd_saturating`, `simd_util`, `simd_wmul`, `simd_x86_dotprod`).
- `--dotprod` flag-handler refactored into the shared `append_feature` helper alongside `--fp16`/`--i8mm` (behavior unchanged; flag still ARM-only).
- NEON `gather()` error message rewritten to point at the new `f32x{4,8}_from_scalars` compose primitives and `docs/idioms/neon-gather.md`, instead of "use a scalar loop on ARM".
- Type-checker error messages improved for SIMD width / element-type mismatches across many intrinsics.
- `main.rs` slimmed (498 → 400 lines) by extracting the `ea bind` command into `src/bind_handler.rs` (115 lines).

### Fixed

- **Linux→Windows cross-compilation produces real PE32+ DLLs.** Commit `7c6a9c0`: the `SharedLib` linker step dispatched on the host triple (`#[cfg(target_os = "windows")]`), so cross-compiling from Linux to `x86_64-pc-windows-gnu` fell through to `cc -shared` and emitted ELF `.so` files dressed up as `.dll`. Dispatch now reads `opts.target_triple`: target=windows + host=windows uses `lld-link.exe` (unchanged); target=windows + host=linux uses `x86_64-w64-mingw32-gcc -shared -static-libgcc` (override with `WINDOWS_CC`); else falls back to `cc -shared -lm`. Validated by Olorin cross-building 39 SIMD kernels and loading them on Windows via `libloading`.
- **LLVM Machine Outliner disabled in hot loops.** Commits `f33ab3d` + `38fd50e`: the outliner extracted repeated instruction sequences into subroutines, producing `bl` calls that caused register spills and broke scheduling inside compute kernels (e.g. mins accumulation in Q4_K dot product). Global flag `-enable-machine-outliner=never` is now set inside `create_target_machine()` before the machine is constructed (the previous flag-setting path in `optimize_module()` ran too late to take effect).
- **`--emit-llvm` now runs optimization passes before dumping IR.** Commit `e455383`: previously `--emit-llvm` always printed unoptimized IR regardless of `--opt-level`, making it impossible to verify `alwaysinline` and other attributes. The IR dump now honors the optimization level.
- **`vpermq 0xD8` lane fixup after `vpackssdw`/`vpacksswb`.** Commit `6291514`: AVX2 pack instructions operate per 128-bit lane and produce interleaved output. A `vpermq` shuffle `[0,2,1,3]` is now emitted after each pack so callers see sequential element order (numerical correctness fix for AVX2 pack chains).
- **`pack_sat_i16x16` ARM codegen.** Commit `4658431`: the ARM split-concat implementation was missing; all three `pack_sat_*` intrinsics now have full cross-platform codegen.
- **`--fp16` gate fires for declared-but-unused f16 vector params.** Commit `a53c983`: `validate_type_for_target` previously only ran on params actually used in the function body, so declared-but-unused `f16x8` params bypassed the gate and panicked in `llvm_type()` on non-FP16 targets. The validator now runs on each param and on the return type at `declare_function` time.
- **`--fp16` cross-arch guard extended to `inspect_source` + `compile_to_ir`.** Commit `c5ed9d8`: the initial guard (`a53c983`) only covered `compile_with_options`, so `ea inspect --fp16` on x86 bypassed it. The guard is now mirrored in `inspect_source` and `compile_to_ir_with_options` for full library-API coverage.

### Removed (breaking)

- `maddubs_i32` intrinsic — replaced with `madd_i16` per commit `89130cb`. The old name hid a 2-instruction chain (`pmaddubsw + pmaddwd`) behind one symbol, violating Eä's "programmer sees the cost" philosophy. **Migration:** rewrite `let r: i32x4 = maddubs_i32(a_u8, b_i8)` as the explicit chain:
  ```
  let t: i16x8 = maddubs_i16(a_u8, b_i8)
  let r: i32x4 = madd_i16(t, ones_i16x8)
  ```
  `maddubs_i16` is retained.

### Performance

- `exp_poly_f32` throughput vs scalar `exp()` is baseline-dependent. Measured on the Phase 6 benchmarks:
  - **2.93× isolated** on x86 AMD Zen 4 + glibc 2.42 (modern scalar `expf`).
  - **2.60×** inside the bundled softmax kernel on the same host.
  - **2.23×** end-to-end on Pi 5 inside Olorin's `gemma4_gelu` kernel (Cortex-A76, glibc `expf`, no `libmvec`).
  - The spec's original "~10×" headline is an upper bound against older or scalar-only libm without a vectorized `expf`; against modern glibc the win is the 2-3× range shown above.
- Native f16 NEON eliminates the f32 round-trip on every KV read in Olorin's attention / RoPE / RMSNorm hot paths under `--fp16`.

### Notes

- `--fp16` and `--i8mm` are silently ignored at parse time on non-ARM hosts but produce an explicit error during target configuration: `error: --i8mm is only valid for AArch64 targets` and `--fp16 is incompatible with non-ARM target`.
- The two `cvt_*` intrinsics are cross-platform at the 128-bit width (i16x4 / f32x4) but x86-only at the 256-bit width (i16x8 / f32x8); ARM users get a clear error pointing at the narrower form.
- Existing `cvt_f16_f32` / `cvt_f32_f16` continue to work with or without `--fp16`; the flag only opts in to in-register f16 arithmetic.
- AVX-512 lane intrinsics use pure LLVM `shufflevector` with constant masks and rely on LLVM 18+ pattern-matching to the right ISA (`vinserti32x8`, `vextracti128`, `vpshufd`); no explicit AVX-512-only flag gate at the type-check level.

## v1.7.0 — exp intrinsic, multi-width widen, i32x16

**`exp()` intrinsic** — scalar f32/f64 plus all float vector widths (f32x4, f32x8, f32x16, f64x2, f64x4). Maps to `llvm.exp.*`. Motivated by softmax/attention kernels (eakv GQA) that need exponentiation in the hot path.

**Multi-width widen intrinsics** — `widen_u8_f32x8`, `widen_i8_f32x8`, `widen_u8_f32x16`, `widen_i8_f32x16`, `widen_u8_i32x4`, `widen_u8_i32x8`, `widen_u8_i32x16`. Extends the existing `widen_u8_f32x4`/`widen_i8_f32x4` pair to all practical SIMD widths. Enables wider quantized-to-float conversion pipelines without manual lane splitting.

**`i32x16` vector type** — 512-bit integer vector. Requires `--avx512`. Completes the i32 vector width progression (i32x4 → i32x8 → i32x16).

420 tests, all files ≤500 lines, clippy/fmt clean.

## v1.6.0 — Profile-driven consolidation

Every feature motivated by `perf stat` analysis across 6 demos. Profile first, feature second, two proofs before merge.

**Scalar `min` / `max`** — branchless min/max for i32, f32, f64. Lowers to `llvm.smin`/`llvm.smax` (CMOVcc) and `llvm.minnum`/`llvm.maxnum` (MINSS/CSEL). Attacks the 25.6% branch mispredict tax identified in 1BRC microarchitectural analysis.

**`f64x4` / `f64x2` vector types** — double-precision SIMD for statistical accumulators. f64x4 on x86 AVX2, f64x2 on ARM NEON. All existing intrinsics (load, store, splat, reduce_add, fma, select, dot operators) work on f64 vectors.

**`for` loop syntax** — `for i in 0..n step 8 { }`. Desugared to while loop in the same pass as kernel→func. Half-open range `[start, end)`, optional step clause.

**`#[cfg(x86_64)]` / `#[cfg(aarch64)]` conditional compilation** — merge platform-specific kernels into one file. Evaluated at desugar time. Eliminates the scan.ea/scan_arm.ea file duplication pattern.

**Pointer-to-pointer `**T`** — recursive pointer parsing for batch operations. `frames: **f32` gives clean signatures for multi-buffer workloads like astro_stack batch accumulation.

**`i32x8` verification** — token and parser support already existed. Full operation coverage confirmed: load, store, splat, reduce_add, select, arithmetic, multiply.

420 tests (45 new), all files ≤500 lines, clippy/fmt clean.

## v1.5.4 — Freestanding kernel correctness

**Disable LLVM loop-idiom libcalls** — LLVM's LoopIdiomRecognize pass silently replaces store-loops with `memset`/`memcpy` calls. For freestanding kernels with no C runtime, this causes linker failures on Windows (`/NODEFAULTLIB`) and hides what the programmer actually wrote on all platforms. The optimizer now sets `-disable-loop-idiom-memset` and `-disable-loop-idiom-memcpy` via `LLVMParseCommandLineOptions` before running passes. Kernels contain exactly the code the programmer wrote — no synthesized libcalls, on any platform.

**ARM build fix** — `c_char` is `u8` on ARM Linux but `i8` on x86. The loop-idiom flag setup used `*const i8` for the LLVM CLI argument pointers, which failed to compile on aarch64. Fixed by using inferred pointer types.

## v1.5 — Multi-kernel files, `static_assert`, `ea inspect`

**Multi-kernel files** — multiple structs, constants, helper functions, and exported kernels in a single `.ea` file. The full pipeline (parser, desugarer, type checker, codegen, metadata, header, all five binding generators) handles everything seamlessly. No special syntax needed — just write multiple exports.

```
struct Vec2 { x: f32, y: f32 }
const SCALE: f32 = 2.0

export kernel add(a: *f32, b: *f32, out: *mut f32) over i in n step 8 { ... }
export kernel mul(a: *f32, b: *f32, out: *mut f32) over i in n step 8 { ... }
export func dot(a: *f32, b: *f32, n: i32) -> f32 { ... }
```

**`static_assert`** — compile-time assertions evaluated during type checking. No code emitted.

```
const STEP: i32 = 8
static_assert(STEP % 4 == 0, "STEP must be SIMD-aligned")
static_assert(STEP > 0 && STEP <= 16, "STEP must be in range 1..16")
```

Supports arithmetic (`+`, `-`, `*`, `/`, `%`), comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`), and boolean logic (`&&`, `||`, `!`) on compile-time constants. Non-constant references produce clear errors.

**`ea inspect`** — analyze post-optimization instruction mix, loops, vector width, and register usage.

```bash
ea inspect kernel.ea                  # all exports, native target
ea inspect kernel.ea --avx512         # with AVX-512
ea inspect kernel.ea --target=skylake # specific CPU
```

```
=== vscale (exported) ===
  vector instructions:  12
  scalar instructions:   4
  vector width:         256-bit (f32x8)
  loops:                2 (1 main, 1 tail)
  vector registers:     ymm0, ymm1, ymm2, ymm3 (4 used)
```

## v1.4 — Output annotations

Mark `*mut` pointer parameters as outputs with buffer sizing hints. Binding generators auto-allocate and return buffers, eliminating the allocate-call-unpack pattern from host code.

```
export func transform(data: *f32, out result: *mut f32 [cap: n], n: i32) {
    let mut i: i32 = 0
    while i < n {
        result[i] = data[i] * 2.0
        i = i + 1
    }
}
```

Three forms:

| Syntax | Behavior |
|--------|----------|
| `out result: *mut f32 [cap: n]` | Auto-allocated by binding, returned to caller |
| `out result: *mut f32 [cap: n, count: actual]` | Auto-allocated with separate actual-length path |
| `out result: *mut f32` | Caller provides buffer (stays in signature) |

Generated bindings handle allocation per target:

| Target | Auto-allocation | Return type |
|--------|----------------|-------------|
| Python | `np.empty(n, dtype=np.float32)` | `np.ndarray` |
| Rust | `vec![Default::default(); n]` | `Vec<f32>` |
| C++ | `std::vector<float>(n)` | `std::vector<float>` |
| PyTorch | `torch.empty(n, dtype=torch.float32)` | `Tensor` |

Type checker validates: `out` requires `*mut` pointer type, cap identifiers must reference preceding input params or constants. Metadata JSON emits `direction`, `cap`, and `count` fields per arg. Backward-compatible: old JSON without these fields works unchanged.

## v1.3 — Kernel construct, compile-time constants, tail strategies

**`kernel`** — declarative loop construct for data-parallel operations:

```
export kernel double_it(data: *i32, out: *mut i32)
    over i in n step 1
{
    out[i] = data[i] * 2
}
```

Desugars to a function with a generated loop. The range bound (`n`) becomes the last parameter automatically. SIMD kernels use `step 4`/`step 8` for explicit vectorization.

**`tail`** — handle remainder elements when array length isn't a multiple of step:

```
export kernel vscale(data: *f32, out: *mut f32, factor: f32)
    over i in n step 8
    tail scalar { out[i] = data[i] * factor }
{
    store(out, i, load(data, i) .* splat(factor))
}
```

Three strategies: `tail scalar { ... }` (element-wise loop), `tail mask { ... }` (single masked iteration), `tail pad` (skip remainder).

**`const`** — compile-time constants inlined at every use site:

```
const BLOCK_SIZE: i32 = 64
const PI: f64 = 3.14159265358979
```

Supports integer and float types. Constants are validated at type-check time and referenced in kernel bodies, cap expressions, and function parameters.

## v1.2 — `ea bind` multi-language bindings

**`ea bind`** now generates native bindings for five targets from a single kernel:

| Flag | Output | What you get |
|------|--------|--------------|
| `--python` | `kernel.py` | NumPy ctypes module with dtype checks, length collapsing |
| `--rust` | `kernel.rs` | `extern "C"` FFI + safe wrappers with `&[T]`/`&mut [T]` |
| `--pytorch` | `kernel_torch.py` | `torch.autograd.Function` per export, tensor contiguity/device checks |
| `--cpp` | `kernel.hpp` | `namespace ea`, `extern "C"` declarations, `std::span` overloads |
| `--cmake` | `CMakeLists.txt` + `EaCompiler.cmake` | Ready-to-build CMake project skeleton |

All generators share a common JSON parser (`bind_common.rs`) and the same length-collapsing heuristic: parameters named `n`/`len`/`length`/`count`/`size`/`num` after a pointer arg are auto-filled from the slice/array/tensor size.

## v1.1 — ARM/NEON support, integration examples, CI

**ARM/AArch64 cross-compilation** — compile kernels for ARM targets with NEON (128-bit) SIMD:

```bash
ea kernel.ea --lib --target=aarch64   # produces kernel.so for ARM
```

The compiler validates vector widths at the type-check level: 128-bit types (`f32x4`, `i32x4`, `u8x16`, `i16x8`) work on ARM; 256-bit+ types (`f32x8`, `i32x8`) and x86-specific intrinsics (`maddubs`, `gather`, `scatter`) produce clear error messages with alternatives.

**Integration examples** — manual integration patterns for embedding Eä kernels into host projects. Most are now superseded by `ea bind`; see [FFmpeg filter](integrations/ffmpeg-filter/) for the remaining manual example.

**CI** — build and test on Linux x86_64, Linux ARM (aarch64), and Windows on every push.

## v1.0 — error diagnostics, masked ops, scatter/gather

**`foreach`** — auto-vectorized element-wise loops with phi-node codegen:

```
export func scale(data: *f32, out: *mut f32, n: i32, factor: f32) {
    foreach (i in 0..n) {
        out[i] = data[i] * factor
    }
}
```

`foreach` generates a scalar loop with phi nodes. LLVM may auto-vectorize at `-O2+`.
For guaranteed SIMD width, use explicit `load`/`store` with `f32x4`/`f32x8`.

**`unroll(N)`** — hint to unroll the following loop:

```
unroll(4) foreach (i in 0..n) { out[i] = data[i] * factor }
unroll(4) while i < n { ... }
```

Relies on LLVM unrolling heuristics. Not a hard guarantee.

**`prefetch(ptr, offset)`** — software prefetch hint for large-array streaming:

```
prefetch(data, i + 16)
```

**`--header`** — generate a C header alongside the object file:

```bash
ea kernel.ea --header    # produces kernel.o + kernel.h
```

```c
// kernel.h (generated)
#ifndef KERNEL_H
#define KERNEL_H
#include <stdint.h>
void scale(const float* data, float* out, int32_t n, float factor);
#endif
```

**`--emit-asm`** — emit assembly for inspection:

```bash
ea kernel.ea --emit-asm  # produces kernel.s
```

## v0.4.0

### Breaking Changes
- `maddubs(u8x16, i8x16) -> i16x8` renamed to `maddubs_i16` — update all kernels

### New Intrinsics
- `maddubs_i32(u8x16, i8x16) -> i32x4` — safe accumulation via pmaddubsw+pmaddwd chain.
  Programmer explicitly chooses the overflow model by choosing the instruction.
  No silent widening.

### Demos
- `demo/conv2d_3x3/conv_safe.ea` — i32x4 accumulator variant, immune to accumulator overflow
