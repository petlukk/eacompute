# v1.11.0 API Consistency Audit Findings

Compiled: 2026-05-13. Branch: `feat/i8mm-intrinsics`. Reviewer: subagent
(Phase 3 of the pre-merge audit plan).

Inputs read:
- `docs/release/v1.11.0/inventory.md` (the 63-intrinsic catalog)
- `docs/superpowers/specs/2026-04-27-pi5-neon-enablement-design.md`
- `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`
- `src/codegen/simd.rs` dispatch (`compile_simd_call`)
- `src/typeck/intrinsics.rs` dispatch (`check_intrinsic_call`)
- All 9 new `src/codegen/simd_*.rs` files
- All 6 new `src/typeck/intrinsics_*.rs` files
- All NEW `CompileError::codegen_error` / `type_error` sites in the diff
- `src/lib.rs` arch-validation entry points
- `src/main.rs` CLI flag wiring (lines 100–185, 320–370)

## Summary

| Category | Findings | Critical | Important | Minor |
|---|---|---|---|---|
| Naming convention | 4 | 0 | 0 | 4 |
| Error messages | 3 | 0 | 3 | 0 |
| Dispatch pattern | 2 | 0 | 0 | 2 |
| Cross-arch behavior | 1 | 0 | 0 | 1 |
| **Total** | **10** | **0** | **3** | **7** |

All three Important findings were error-message rewrites in the same file
(`src/codegen/simd_x86_dotprod.rs`) and have been fixed in this phase.
No Critical findings were uncovered.

## Findings

### F-01. `sat_add`, `sat_sub` are polymorphic with the type left out of the name

- **Severity:** Minor
- **Category:** Naming
- **Location:** `src/codegen/simd_saturating.rs`, `src/typeck/intrinsics_conv.rs:96` (`check_sat_add`)
- **Issue:** `sat_add(a, b)` accepts i8x16, u8x16, i16x8, u16x8 (four element
  types) without the type appearing in the name. The project's documented
  pattern (`CLAUDE.md` "Design Philosophy" → "Concrete, not generic. No
  generics, no traits, no polymorphism. Write separate kernels for each
  type.") favors monomorphic intrinsic names. The adjacent `pack_sat_i16x8`,
  `pack_sat_i32x4` family makes the same word `sat` already type-explicit,
  so the inconsistency is local to the name `sat`. Same applies to `sat_sub`.
  Note: pre-existing polymorphic intrinsics (`min`/`max`/`abs`/`sqrt`/
  `splat`/`load`/`store`/`fma`/`reduce_*`/`select`/`shuffle`) are
  exceptions per the documented `exp()` precedent and are out of scope.
- **Recommended fix:** rename to monomorphic forms in v1.12.0:
  `sat_add_i8x16` / `sat_add_u8x16` / `sat_add_i16x8` / `sat_add_u16x8`
  and the same for sub. This is a breaking change so it should land with
  a major-version bump alongside `exp()` and the other documented exceptions
  (per eabrain's "deferred to v1.12.0" note).
- **Disposition:** Deferred to v1.12.0 (breaking change; queued behind the
  `exp()` monomorphization).

### F-02. `abs_diff` is polymorphic across 6 element-type combinations

- **Severity:** Minor
- **Category:** Naming
- **Location:** `src/typeck/intrinsics_neon.rs:13` (`check_abs_diff`),
  `src/codegen/simd_saturating.rs`
- **Issue:** `abs_diff` accepts (i8x16, u8x16, i16x8, u16x8, i32x4, u32x4) —
  6 separate types — without naming them. The adjacent NEON family is
  monomorphic (`addp_i32`, `wmul_i16`, `wmul_u16`, …). Inconsistent within
  the ARM-only block.
- **Recommended fix:** rename to `abs_diff_i8x16` / `abs_diff_u8x16` /
  `abs_diff_i16x8` / … in v1.12.0. Same rationale as F-01.
- **Disposition:** Deferred to v1.12.0.

### F-03. `widen_u8_u16` lane-discard behavior is not in the name

- **Severity:** Minor
- **Category:** Naming
- **Location:** `src/typeck/intrinsics_conv.rs:168`, table row at
  `docs/release/v1.11.0/inventory.md:135`
- **Issue:** `widen_u8_u16(u8x16) -> u16x8` keeps only the low 8 lanes and
  silently discards the upper 8. The sibling family
  `widen_u8_f32x4_{0,4,8,12}` *does* encode the lane offset in the name.
  By the project's "explicit over implicit" rule, the default `widen_u8_u16`
  could be named `widen_u8_u16_lo` (or `_0`) and a `widen_u8_u16_hi` variant
  added, matching the lane-offset family. Currently the name presents as a
  symmetric widen but the codegen is asymmetric.
- **Recommended fix:** add an explicit `_lo` / `_hi` suffix family
  (`widen_u8_u16_lo` keeps current behavior; add `widen_u8_u16_hi` for the
  upper half) in v1.12.0. Or document the "low-8" default in the doc
  comment on `check_widen_u8_u16` so callers see it during IDE hover.
- **Disposition:** Deferred to Phase 7 (doc-comment improvement) + v1.12.0
  (additive sibling intrinsic). The doc-comment fix is non-breaking and
  cheap; the `_hi` variant is a follow-on feature.

### F-04. `cvt_f16_f32` accepts 3 widths but `cvt_f32_f16` accepts only 2

- **Severity:** Minor
- **Category:** Naming / API symmetry
- **Location:** `src/typeck/intrinsics_f16.rs:12–70`,
  `src/codegen/simd_pack.rs:373–435`
- **Issue:** `cvt_f16_f32` supports widths {4, 8, 16}, but `cvt_f32_f16`
  supports only {4, 8}. The names are well-formed (type-explicit, follow
  the pattern), but the API surface is asymmetric: a user widening f16→f32
  to 16-wide can't round-trip back. The error message at typeck
  (`"cvt_f32_f16 expects f32x4 or f32x8"`) is correct but doesn't explain
  why 16 is missing.
- **Recommended fix:** add a `cvt_f32_f16` width-16 form using
  `llvm.x86.avx512.mask.vcvtps2ph.512` (x86-only, gated on `--avx512`) so
  the round-trip is symmetric. Alternatively, document the asymmetry in
  both intrinsic doc comments so callers see it during IDE hover.
- **Disposition:** Deferred to v1.12.0 (additive feature; round-tripping
  16-wide is a real need for `cvt_f16_f32(i16x16)` callers).

### F-05. `maddubs_i16` ARM rejection didn't suggest an alternative

- **Severity:** Important (FIXED in this phase)
- **Category:** Errors
- **Location:** `src/codegen/simd_x86_dotprod.rs:18` (was: "no NEON
  equivalent")
- **Issue:** The canonical helpful-error pattern from the Pi 5 NEON spec
  is "what you did wrong + what to do instead + pointer to canonical
  doc/idiom." `maddubs_i16` on ARM was failing with just
  `"maddubs_i16 is x86-only (SSSE3/AVX2 pmaddubsw); no NEON equivalent"`,
  which tells the user the failure but offers no fix path. ARM has at
  least two reasonable substitutions:
    - `usmmla_i32` (when `--i8mm` is available — exact mixed-sign 8-bit
      semantics rolled into one matmul, the inventory's documented
      replacement chain)
    - `wmul_i16` (split halves) + `addp_i16` (fuse pairs) for the slow
      portable path
- **Recommended fix:** rewrite the error to name both alternatives and
  call out the `--i8mm` gate for `usmmla_i32`.
- **Disposition:** Fixed in this phase. New text:

  > `maddubs_i16 is x86-only (SSSE3/AVX2 pmaddubsw); on ARM, use
  > wmul_i16(i8x8 lo, i8x8 lo) + wmul_i16(i8x8 hi, i8x8 hi) + addp_i16
  > to fuse the adjacent pairs, or use usmmla_i32 for the mixed-sign 8-bit
  > dot product (--i8mm required)`

  Existing rejection test (`tests/phase14_arm.rs:92`) only asserts
  `.contains("x86")` so the new text is compatible.

### F-06. `madd_i16` ARM rejection didn't suggest an alternative

- **Severity:** Important (FIXED in this phase)
- **Category:** Errors
- **Location:** `src/codegen/simd_x86_dotprod.rs:106`
- **Issue:** Same pattern as F-05. Error was `"madd_i16 is x86-only
  (SSE2/AVX2 pmaddwd); no NEON equivalent"`. The ARM equivalent
  decomposition is `wmul_i32 + addp_i32`.
- **Recommended fix:** name the decomposition explicitly.
- **Disposition:** Fixed in this phase. New text:

  > `madd_i16 is x86-only (SSE2/AVX2 pmaddwd); on ARM, use
  > wmul_i32(lo i16x4, lo i16x4) + wmul_i32(hi i16x4, hi i16x4) +
  > addp_i32 to fuse adjacent products into the i32 result`

  Test `tests/phase14_arm.rs:107` (`.contains("x86")`) unaffected.

### F-07. `hadd_i16` ARM rejection didn't suggest an alternative

- **Severity:** Important (FIXED in this phase)
- **Category:** Errors
- **Location:** `src/codegen/simd_x86_dotprod.rs:189`
- **Issue:** Same pattern. Error was `"hadd_i16 is x86-only (SSSE3/AVX2
  phaddw); no NEON equivalent"`. But the ARM-side intrinsic `addp_i16`
  added by this same branch has identical semantics (pairwise add of
  adjacent lanes across both operands). The user has a perfect fit, just
  with a different name, and the previous message hid that.
- **Recommended fix:** point at `addp_i16`.
- **Disposition:** Fixed in this phase. New text:

  > `hadd_i16 is x86-only (SSSE3/AVX2 phaddw); on ARM, use addp_i16(a, b)
  > which has identical semantics (pairwise add of adjacent lanes across
  > both operands)`

  Test `tests/phase14_arm.rs:122` (`.contains("x86")`) unaffected.

### F-08. `compile_simd_call` dispatch uses both `compile_*` and `emit_*` prefixes

- **Severity:** Minor
- **Category:** Dispatch
- **Location:** `src/codegen/simd.rs:171–348` (the main `match name { ... }`)
- **Issue:** Most arms call helpers named `compile_<name>` (e.g.
  `compile_sat_add`, `compile_wmul_i16`, `compile_pack_sat_i32x4`). The
  new lane-intrinsics family uses `emit_<name>` instead (`emit_concat`,
  `emit_lo_extract`, `emit_hi_extract`, `emit_shuffle_i32`,
  `emit_blend_i32`, `emit_bcast_pairs`, `emit_f32_from_scalars`). The
  helpers all live in `src/codegen/simd_lane.rs`. Functionally identical;
  cosmetic inconsistency.
- **Recommended fix:** rename `emit_*` to `compile_*` in `simd_lane.rs`
  (and the corresponding `simd.rs` call sites). Touches ~7 internal
  identifiers; no public surface affected.
- **Disposition:** Deferred to Phase 7.

### F-09. `compile_simd_call` dispatch arms not strictly alphabetical

- **Severity:** Minor
- **Category:** Dispatch
- **Location:** `src/codegen/simd.rs:183–339`
- **Issue:** Arms are family-grouped (widen family together, conversion
  family together, lane intrinsics together, etc.) which is sensible and
  readable. But within families the order is approximate (e.g.
  `round_f32x8_i32x8` lands between `wmul_u32` and `pack_sat_i32x8`
  rather than next to its `round_f32x4_i32x4` sibling). Not worth
  reflowing aggressively, but a one-pass cleanup would help future
  maintainers find arms.
- **Recommended fix:** within each family block, sort arms alphabetically;
  keep family groupings separated by a blank line. Mirror the layout in
  `src/typeck/intrinsics.rs::check_intrinsic_call` to match.
- **Disposition:** Deferred to Phase 7.

### F-10. `--avx512` and `--i8mm` arch validation only at CLI, not at library API

- **Severity:** Minor
- **Category:** Cross-arch
- **Location:** `src/main.rs:174–182` vs `src/lib.rs:106–113`
- **Issue:** `--fp16` validation is mirrored into the library API
  (`validate_fp16_compatibility` runs in `compile_with_options`,
  `compile_to_ir_with_options`, and `inspect_source` per commits
  `a53c983` + `c5ed9d8`). The `--avx512`-on-ARM and `--i8mm`-on-x86 checks
  only live in `src/main.rs`. A library API consumer who sets
  `CompileOptions { extra_features: "+i8mm".into(), target_triple:
  Some("x86_64-...") }` will *not* hit a flag-level rejection — but every
  i8mm-gated intrinsic (`smmla_i32`, `ummla_i32`, `usmmla_i32`) has a
  per-intrinsic `if !self.is_arm` guard, so the user gets a clean
  per-call error rather than a silent fallback. Same for `--avx512`
  (every avx512-only intrinsic is guarded). The flag itself is benignly
  set but never engages.

  The inventory explicitly notes "`--avx512` is unchanged in this branch"
  and commit `33ab1f8` notes "`--i8mm` … mirroring the --avx512/--dotprod
  pattern" — so the CLI-only gate is intentional. `--fp16` was the
  exception because f16-typed arithmetic uses plain operators (`.+`,
  `.-`, etc.) with no per-intrinsic guard to catch it.
- **Recommended fix:** for full consistency, add
  `validate_avx512_compatibility` and `validate_i8mm_compatibility`
  helpers in `src/lib.rs` and call them from the same three entry points
  as `validate_fp16_compatibility`. Cheap, ~20 lines. Or leave as-is and
  document the design rationale ("per-intrinsic guards are sufficient
  for explicit-name intrinsics; only type-driven gates need
  library-layer mirroring") in the inventory's "Concerns / Notes"
  section.
- **Disposition:** Deferred to Phase 7. The behavior is not broken (no
  silent fallback path exists) — this is purely about symmetry.

## Items NOT flagged (sample of patterns checked and accepted)

- **`exp_poly_f32`**: type-explicit, algorithm-explicit, justified
  deviation from polymorphic `exp()`. Spec-documented at
  `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`.
- **`f32x4_from_scalars` / `f32x8_from_scalars`**: type+width in the
  name, no surprises.
- **`widen_i8_f32x4_4`, `_8`, `_12`** and the `widen_u8_i32x4_*` family:
  type+width+offset in the name. Pattern is consistent.
- **`madd_i16` returning `i32x(N/2)`**: the multiply-add idiom returning
  a wider type is standard across SSE/NEON/AVX, and the i16 part
  describes the input element type which is the load-bearing fact.
- **`bitcast_i8x16` accepting any 128-bit vector**: the *output* type is
  in the name, the input is intentionally polymorphic — same as
  `widen_u8_u16` would conceptually be. Consistent with the project
  pattern of "name the destination."
- **`vdot_lane_i32`, `smmla_i32`, `ummla_i32`, `usmmla_i32`**:
  type-explicit; arch-specific names with `_i32` accumulator type
  visible. Each has a clear `--i8mm` / `--dotprod` gate-and-suggest
  error pair.
- **NEON `gather()` error rewrite** (commit predates this audit):
  exemplary canonical pattern. Used as the reference style for F-05
  through F-07 fixes.
- **`--fp16` library-layer gate**: triple-mirror in
  `compile_with_options`, `compile_to_ir_with_options`,
  `inspect_source`. Verified in `src/lib.rs:126,274,295`.
- **Per-intrinsic arch guards**: every x86-only or ARM-only intrinsic
  emits `CompileError::codegen_error` before any LLVM IR is generated.
  No silent scalar fallback was found in the new surface.
- **Module declarations in `src/codegen/mod.rs:1–46`**: alphabetical and
  clean.
- **Helper extraction**: `compile_smmla_i32` / `compile_ummla_i32` /
  `compile_usmmla_i32` share the same gate-then-emit shape — readable;
  no further deduplication needed.

## Cross-arch verification table

For each new arch-specific or feature-flag-gated intrinsic, the
wrong-target behavior was verified by reading the codegen file directly.
"Verified?" = Y means a `CompileError::codegen_error` is emitted before
any LLVM IR is generated, with a message that names the alternative
(canonical pattern) or names the missing flag.

| Intrinsic | Required flag | Wrong-target behavior | Verified? |
|---|---|---|---|
| `smmla_i32` | `--i8mm` | x86: hard-error; ARM no `--i8mm`: hard-error suggesting `--i8mm` | Y (`simd_dotprod.rs:131,136`) |
| `ummla_i32` | `--i8mm` | same | Y (`simd_dotprod.rs:176,181`) |
| `usmmla_i32` | `--i8mm` | same | Y (`simd_dotprod.rs:221,226`) |
| `vdot_i32` | `--dotprod` | x86: hard-error "no x86 equivalent"; ARM no `--dotprod`: suggests `--dotprod` | Y (`simd_dotprod.rs:18,28`) |
| `vdot_lane_i32` | `--dotprod` | same | Y (`simd_dotprod.rs:68,78`) |
| `madd_i16` (any width) | none for 8/16; AVX-512 for 32 | x86 ok; ARM: hard-error with substitution | Y (F-06 fix) |
| `maddubs_i16` (any width) | none for 16/32; AVX-512 for 64 | same | Y (F-05 fix) |
| `hadd_i16` (8/16) | none | x86 ok; ARM: hard-error pointing at `addp_i16` | Y (F-07 fix) |
| `bsrli_i8x32`, `bslli_i8x32` | none | x86 ok; ARM: hard-error pointing at `_i8x16` variant | Y (`simd_byteshift.rs:43,107`) |
| `shuffle_bytes` width 32 | none | ARM 32-wide: hard-error pointing at 16-wide | Y (`simd_dotprod.rs:300`) |
| `round_f32x8_i32x8` | none | ARM: hard-error pointing at `round_f32x4_i32x4` | Y (`simd_pack.rs:55`) |
| `pack_sat_i32x8`, `pack_sat_i16x16` | none | ARM: hard-error pointing at 4/8 variants | Y (`simd_pack.rs:89,126`) |
| `pack_usat_i32x8`, `pack_usat_i16x16` | none | same | Y (`simd_pack_unsigned.rs:18,54`) |
| `cvt_f16_f32` i16x8 / i16x16 | x86-only widths | ARM: hard-error suggesting i16x4 | Y (`simd_pack.rs:381`) |
| `cvt_f32_f16` f32x8 | x86-only width | ARM: hard-error suggesting f32x4 | Y (`simd_pack.rs:415`) |
| `wmul_*`, `addp_*`, `abs_diff` | none | x86: hard-error "is ARM-only (NEON); no x86 equivalent" | Y (`simd_wmul.rs`, `simd_saturating.rs`) |
| f16 vector arithmetic (`+`/`-`/`*`/`/`) | `--fp16` | non-ARM: hard-error at lib-API entry; ARM no `--fp16`: hard-error suggesting `cvt_f16_f32`+f32 path | Y (`lib.rs:106–113`, `codegen/mod.rs:392–395`) |
| f16 splat/load/store/fma/reductions | `--fp16` | same | Y (`simd_fp16.rs`, dispatched via `call_uses_f16` in `simd.rs:178`) |
| `gather` | none, but x86-only intrinsic | ARM: canonical error pointing at `f32x{4,8}_from_scalars` + `docs/idioms/neon-gather.md` | Y (`simd_masked.rs:211`) |
| `scatter` | `--avx512` | x86 no `--avx512`: hard-error suggesting `--avx512`; ARM has no path (falls through scatter typeck unchanged) | Y (`simd_masked.rs:291`) |

## Conclusion

The branch is **ready for Phase 4** (test-coverage audit).

- 0 Critical findings.
- 3 Important findings, all error-message rewrites in
  `src/codegen/simd_x86_dotprod.rs` — fixed in this phase. The new
  messages follow the canonical "what failed + what to do instead"
  pattern set by the NEON gather error.
- 7 Minor findings, all deferred to Phase 7 (cleanup) or v1.12.0
  (breaking changes). None block the v1.11.0 → main merge.

The arch-safety surface is solid: every new arch-only or
feature-gated intrinsic emits a per-call compile-time error on the
wrong target, with no silent scalar fallback paths. The `--fp16`
library-layer gate is correctly mirrored across all three entry
points. The `--avx512` and `--i8mm` CLI-only gates are intentional
per the commit history and are defended by per-intrinsic guards
(F-10).

The naming inconsistencies (F-01, F-02) and the asymmetric
`widen_u8_u16` / `cvt_f32_f16` surfaces (F-03, F-04) are real but
small and best handled in v1.12.0 alongside the already-deferred
`exp()` monomorphization, since they involve breaking renames or
additive intrinsics rather than fixes to existing behavior.
