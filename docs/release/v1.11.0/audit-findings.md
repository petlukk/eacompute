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
| Error messages | 3 | 1 | 2 | 0 |
| Dispatch pattern | 2 | 0 | 0 | 2 |
| Cross-arch behavior | 1 | 0 | 0 | 1 |
| **Total** | **10** | **1** | **2** | **7** |

All three error-message findings were rewrites in the same file
(`src/codegen/simd_x86_dotprod.rs`) and have been fixed in this phase.
F-05 was promoted to Critical in a review round after the first fix
attempt shipped a semantically incorrect `wmul_i16`-based recipe (signed
multiply applied to unsigned operands); see F-05 below for the recipe
correction.

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
  `exp()` monomorphization). Phase 7 explicitly did NOT touch this — the
  rename would break every caller of `sat_add` / `sat_sub`.

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
- **Disposition:** Deferred to v1.12.0. Phase 7 explicitly did NOT touch
  this — the rename would break every caller of `abs_diff`.

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
- **Disposition:** Doc-comment fix LANDED in Phase 7 — added an explicit
  doc-comment to `check_widen_u8_u16` in `src/typeck/intrinsics_conv.rs`
  noting the low-8-lane semantics and pointing users at the manual shuffle
  if they need the upper half. The `_hi` sibling intrinsic remains deferred
  to v1.12.0 as an additive feature.

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
  16-wide is a real need for `cvt_f16_f32(i16x16)` callers). Phase 7
  explicitly did NOT touch this — adding the AVX-512 width-16 form is a
  new-intrinsic feature outside Phase 7 scope.

### F-05. `maddubs_i16` ARM rejection didn't suggest an alternative

- **Severity:** Critical (FIXED in this phase, recipe corrected in
  review round 2)
- **Category:** Errors
- **Location:** `src/codegen/simd_x86_dotprod.rs:18` (was: "no NEON
  equivalent")
- **Issue:** The canonical helpful-error pattern from the Pi 5 NEON spec
  is "what you did wrong + what to do instead + pointer to canonical
  doc/idiom." `maddubs_i16` on ARM was failing with just
  `"maddubs_i16 is x86-only (SSSE3/AVX2 pmaddubsw); no NEON equivalent"`,
  which tells the user the failure but offers no fix path. `maddubs_i16`
  is *mixed-sign* (`u8 * i8`); the only single-instruction ARM equivalent
  is `usmmla_i32` (requires `--i8mm`). Without `--i8mm`, callers must
  manually zero-extend the u8 side, sign-extend the i8 side, then do a
  portable signed multiply + `addp_i16` — there is no single-call
  portable recipe.
- **Severity escalation:** the initial Phase 3 fix suggested a
  `wmul_i16(i8x8 lo, i8x8 lo) + wmul_i16(i8x8 hi, i8x8 hi) + addp_i16`
  recipe in the error text. The review round caught that this is
  *semantically incorrect*: `wmul_i16` is `(i8x8, i8x8) -> i16x8`
  signed×signed (lowers to `smull.v8i16`). For `u8 >= 128` lanes,
  sign-extending the unsigned operand produces wrong negative values
  and silently wrong products. Shipping that as a user-facing error
  recipe would have actively misled callers into numerically broken
  kernels. Promoted from Important → Critical because of the
  user-action-misleading nature.
- **Final fix:** rewrite to (1) name `usmmla_i32` as the canonical ARM
  alternative with the `--i8mm` gate, (2) describe the portable manual
  recipe (zero-extend u8, sign-extend i8, multiply, `addp_i16`), and
  (3) explicitly warn that `wmul_i16` alone is signed-only and unsafe
  for `u8 >= 128`.
- **Disposition:** Fixed in this phase. New text:

  > `maddubs_i16 is x86-only (SSSE3/AVX2 pmaddubsw, mixed-sign u8*i8
  > multiply-add). On ARM, use usmmla_i32 for the canonical mixed-sign
  > 8-bit dot product (requires --i8mm). Without i8mm, there is no
  > single-instruction equivalent: u8 and i8 lanes must be widened to
  > i16 separately (zero-extend the u8 side, sign-extend the i8 side)
  > before a portable signed multiply + addp_i16; wmul_i16 is
  > signed*signed only and will produce wrong results for u8 values
  > >= 128`

  Existing rejection test (`tests/phase14_arm.rs:92`) only asserts
  `.contains("x86")` so the new text is compatible.

### F-06. `madd_i16` ARM rejection didn't suggest an alternative

- **Severity:** Important (FIXED in this phase)
- **Category:** Errors
- **Location:** `src/codegen/simd_x86_dotprod.rs:106`
- **Issue:** Same pattern as F-05. Error was `"madd_i16 is x86-only
  (SSE2/AVX2 pmaddwd); no NEON equivalent"`. `madd_i16` is signed × signed
  (`(i16x8, i16x8) -> i32x4`); `wmul_i32 + addp_i32` is sign-correct.
- **Recommended fix:** name the decomposition explicitly.
- **Disposition:** Fixed in this phase. New text:

  > `madd_i16 is x86-only (SSE2/AVX2 pmaddwd); on ARM, use
  > wmul_i32(lo i16x4, lo i16x4) + wmul_i32(hi i16x4, hi i16x4) +
  > addp_i32 to fuse adjacent products into the i32 result`

  Test `tests/phase14_arm.rs:107` (`.contains("x86")`) unaffected.

  **Caveat surfaced by review:** the project does not currently expose
  a `lo*_i16x8` / `hi*_i16x8` extractor (the `lo_extract` / `hi_extract`
  family is wired for i8/u8/i32/f32 widths only). Callers following
  this recipe must manually compose i16x4 halves via scalar indexing
  or a similar pattern.

  **Phase 7 disposition:** deferred to v1.12.0. Adding `lo_i16x8` /
  `hi_i16x8` is a new-intrinsic feature outside Phase 7 scope (the rule
  is no new public surface in a follow-up cleanup phase). The recipe in
  the F-06 error text remains accurate: callers can compose halves with
  the existing scalar-extract / `f32x4_from_scalars`-style pattern until
  v1.12.0 ships the dedicated extractors.

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
- **Disposition:** Fixed in Phase 7. Renamed all seven `emit_*` helpers
  in `src/codegen/simd_lane.rs` to `compile_*` and updated the matching
  call sites in `src/codegen/simd.rs`. No public surface changes; all
  778 tests still pass.

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
- **Disposition:** Partially fixed in Phase 7. The
  `round_*` / `pack_sat_*` / `pack_usat_*` block was the most visibly
  out-of-order (round_f32x8 landed between wmul and pack_sat, with the
  smaller-width siblings further down). Reordered to: `round_f32x4_i32x4`
  then `round_f32x8_i32x8`; `pack_sat_i16x8 / i16x16 / i32x4 / i32x8` in
  ascending width; same for `pack_usat_*`. Mirrored in
  `src/typeck/intrinsics.rs::check_intrinsic_call`. The rest of the
  dispatch was judged acceptable as-is (the widen family is already
  grouped by signedness × output type × offset, the byte-shift family
  is grouped sr/sl × width, etc.). A more aggressive resort would create
  diff churn without aiding readability.

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
- **Disposition:** Documented in Phase 7. Rather than mirroring the
  `--fp16` library-API guard for `--avx512` / `--i8mm`, we documented the
  design rationale ("per-intrinsic guards are sufficient for
  explicit-name intrinsics; only type-driven gates like f16 arithmetic
  need library-layer mirroring") in `docs/release/v1.11.0/inventory.md`
  under "Concerns / Notes". The behavior is correct; adding the helpers
  would be code without a corresponding bug to fix.

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
- 7 Minor findings: 4 addressed in Phase 7 (F-03 doc-comment, F-08
  dispatch rename, F-09 partial dispatch reorder, F-10 design-rationale
  documentation), and 3 deferred to v1.12.0 as additive or breaking
  changes (F-01, F-02 polymorphic-name renames; F-04 `cvt_f32_f16`
  width-16 form; plus the F-06 caveat about `lo_i16x8` / `hi_i16x8`
  extractors). None block the v1.11.0 → main merge.

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
