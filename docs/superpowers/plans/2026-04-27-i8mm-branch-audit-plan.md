# `feat/i8mm-intrinsics` Branch Audit & Release Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute the audit phases. Each phase below is a discrete task with checkbox steps.

**Goal:** Pre-merge audit, documentation update, follow-up cleanup, version bump, and merge of the `feat/i8mm-intrinsics` branch into `main`.

**Branch state:** 98 commits ahead of `main`. 76 files changed, +11,727 / −1,150 lines. Current version `1.10.0`.

**Not an implementation task.** This plan executes verification, documentation, and integration work — no new feature code.

---

## What's in the branch (3 logical layers)

1. **Pre-existing i8mm work** (~73 commits): vdot_lane_i32, bitcast_*, shuffle_bytes (sign-agnostic), bsrli/bslli (4 variants), pack_usat (4 variants), shuffle_i32x{8,16}, blend_i32, smmla/ummla/usmmla, AVX-512 lane intrinsics, AVX-512 int dot, vector literal annotation form, scalar bitwise, widen_u8_u16, multi-width widen variants, error-message improvements, parallel bindings, inspect improvements, and more.

2. **Pi 5 NEON Enablement** (19 commits, this session — pushed): Part A gather compose primitives + Part B native FP16 compute. Spec: `docs/superpowers/specs/2026-04-27-pi5-neon-enablement-design.md`.

3. **`exp_poly_f32`** (6 commits, this session — pushed): polynomial vector exp without scalarization. Spec: `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`.

The pre-existing i8mm layer was never separately audited and reviewed against project standards as a unit. This plan does that audit, plus integrates Pi 5 + exp_poly cleanly.

---

## Hard Rules (enforced at every step)

1. No file exceeds 500 lines.
2. Every feature proven by end-to-end test.
3. No `// TODO`, `// HACK`, `// placeholder`, `// for now`.
4. No premature features.
5. `cargo fmt && cargo clippy --all-targets --all-features -- -D warnings` clean before every commit.
6. C interop is the product — every intrinsic must produce callable C ABI functions.

---

## Phase 1: Inventory

**Goal:** Produce a complete catalog of what's new in the branch. Inputs to all later phases.

### Files

- Create: `docs/release/v1.11.0/inventory.md` (the catalog)
- Create: `docs/release/v1.11.0/CHANGELOG.md` (release notes derived from inventory)

### Steps

- [ ] **Step 1: Catalog all new intrinsics**

```bash
git diff main..HEAD -- src/typeck/intrinsics.rs | grep '^+\s*"' | sort -u > /tmp/new_intrinsic_names.txt
git diff main..HEAD -- src/codegen/simd.rs | grep -E '^\+\s*"\w+"' | sort -u >> /tmp/new_intrinsic_names.txt
sort -u /tmp/new_intrinsic_names.txt
```

For each name in the output, look up the typeck signature (`grep -A 5 'check_<name>' src/typeck/`) and the codegen entry (`grep -A 3 '<name>' src/codegen/simd.rs`). Record:
- Name
- Type signature (input → output)
- Target/arch availability (any, x86, ARM, requires `--avx512`, requires `--fp16`, etc.)
- Source file for codegen
- Test file(s) covering it

- [ ] **Step 2: Catalog all new CLI flags**

```bash
git diff main..HEAD -- src/main.rs | grep '^+\s*"--' | sort -u
```

Expected: at minimum `--fp16`, `--dotprod`, `--i8mm`, possibly `--avx512` if it was already there. For each flag, record name + effect + cross-arch behavior.

- [ ] **Step 3: Catalog all new types**

```bash
git diff main..HEAD -- src/typeck/types.rs | grep -E '^\+\s+\w+,$' | head
```

Expected: `F16`, `U16` (added by pack_usat work), maybe a few others. Plus all the new vector types from `parse_vector_suffix` updates.

- [ ] **Step 4: Catalog all new test files**

```bash
git diff main..HEAD --stat -- tests/ | grep -E '\| +\d+ \++$'
```

Lists net-new test files (lines added with no removed).

- [ ] **Step 5: Catalog all new source files**

```bash
git diff main..HEAD --stat -- src/ | awk -F'|' '$1 ~ /\.rs$/ && $2 !~ /-/ {print $1}'
```

Plus removed files (rare — should flag if anything was removed without justification).

- [ ] **Step 6: Write `inventory.md`**

Structure:
```markdown
# v1.11.0 Inventory

## New CLI Flags
| Flag | Effect | Arch |
|---|---|---|

## New Types
| Type | Description | Arch |
|---|---|---|

## New Intrinsics
### Cross-platform
| Name | Signature | Source | Tests |
|---|---|---|---|

### x86-only
| Name | Signature | Required flags | Source | Tests |
|---|---|---|---|---|

### ARM-only
| Name | Signature | Required flags | Source | Tests |
|---|---|---|---|---|

## New Source Files
| File | Lines | Responsibility |
|---|---|---|

## New Test Files
| File | Lines | Coverage |
|---|---|---|
```

Force-add (`docs/` is gitignored): `git add -f docs/release/v1.11.0/inventory.md`. Commit message: `docs: v1.11.0 inventory catalog`.

---

## Phase 2: Code quality audit

**Goal:** Verify the cumulative diff respects the project's hard rules.

### Steps

- [ ] **Step 1: Line count audit on every modified `src/*.rs` file**

```bash
git diff main..HEAD --stat -- src/ | awk '{print $1}' | grep '\.rs$' | xargs wc -l 2>/dev/null | sort -n
```

Any file over 500 lines is a hard-rule violation. Already known: `src/codegen/statements.rs` at 503. Goal: 0 files over 500.

For each over-500 file, plan and execute a targeted extraction (one helper function or one logical section into a new file). Note the existing pattern: `simd_math.rs` was split during Pi 5 NEON work; mirror that approach.

- [ ] **Step 2: Clippy + fmt (already known clean from per-task reviews; re-verify)**

```bash
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings
```

If any warnings, fix.

- [ ] **Step 3: Stub / placeholder scan**

```bash
git diff main..HEAD | grep -E '^\+.*(//\s*(TODO|HACK|FIXME|XXX|placeholder|for now|hardcoded|temporary))' | head -50
```

Any matches in NEW lines must be addressed (either resolve, or convert to a proper deferred-work entry in eabrain). Existing pre-branch instances are acceptable to defer.

- [ ] **Step 4: Stale `#[allow(dead_code)]` scan**

```bash
git diff main..HEAD | grep -B 1 '#\[allow(dead_code)\]' | head
```

For each new `#[allow(dead_code)]`, verify the field/function is genuinely unused. If used, remove the attribute. If unused, decide: actually remove the dead code, or keep allow if a follow-up task will use it (in which case eabrain note the deferred consumer).

- [ ] **Step 5: Commit message audit (squash candidates)**

```bash
git log --oneline main..HEAD | head -100
```

Read the messages. Any "fix" commits that should have been squashed into the feature commit they're fixing? In this project's history: yes — there are `chore:` cleanups after several `feat:` commits. Decision point: keep history as-is (transparent timeline) or squash before merge for cleaner main log. Default: keep as-is unless the user asks otherwise. Note the choice in the audit output.

- [ ] **Step 6: Commit `chore: post-audit cleanup`** for any fixes from steps 1-4. If nothing needed fixing, no commit.

---

## Phase 3: API consistency audit

**Goal:** New intrinsic surface should follow the project's existing naming + signature conventions. Inconsistencies are debt.

### Steps

- [ ] **Step 1: Naming convention audit**

Reference patterns from the spec philosophy:
- Type-explicit: `widen_u8_i32x4`, `pack_sat_i32x8`, `cvt_f16_f32`, `f32x4_from_scalars` ✓
- Operation + variant + type: `compile_*_f16` family ✓
- Operation + algorithm + type: `exp_poly_f32` ✓ (justified deviation from polymorphic `exp()`)

For each new intrinsic from Phase 1's catalog, classify against these patterns. Flag any names that don't fit:
- Polymorphic intrinsics that should be monomorphic (e.g., does any new intrinsic accept multiple element types in typeck?)
- Type-implicit names that hide what they do (e.g., a `bigint` or `simd_thing` would be a flag)
- Width-implicit names where width isn't obvious from operand type

Likely flags: most pre-existing i8mm-era intrinsics already follow conventions; the audit is a sanity check, not a refactor.

- [ ] **Step 2: Error message style audit**

Reference: spec philosophy says "explicit cost, helpful error messages". The Pi 5 NEON gather error rewrite is the canonical example:
> `"gather has no NEON equivalent on ARM. Use scalar load + f32x4_from_scalars (or f32x8_from_scalars) to compose the result explicitly. See docs/idioms/neon-gather.md for the canonical pattern."`

```bash
grep -rn 'CompileError::codegen_error\|CompileError::type_error' src/ | grep -v test | head -50
```

For each error message touched in this branch (`git log -p main..HEAD --grep='codegen_error\|type_error' -- src/`), verify it tells the user what to do. Flag any that just say "expected X, got Y" without a path forward where one exists.

- [ ] **Step 3: Dispatch pattern audit**

Walk `src/codegen/simd.rs::compile_simd_call`. For each new arm, verify:
- Arm is in a sensible group (alphabetical or by family)
- Helper functions in `simd_*.rs` follow the existing `compile_*` or `emit_*` naming
- f16-specific paths route through the gate (already verified in Pi 5 NEON review, re-confirm at the dispatch level)

Walk `src/typeck/intrinsics.rs::check_intrinsic_call`. Same audit on the typeck side.

- [ ] **Step 4: Cross-arch behavior audit**

For each intrinsic that's arch-specific (or feature-flag-gated), verify it produces a clear compile-time error on the wrong target — never a silent fallback. Reference: `--avx512`-on-ARM, `--fp16`-on-x86, `--i8mm`-on-x86 all error explicitly.

For each intrinsic newly added in this branch, check the cross-arch code path:
```bash
grep -rn 'is_arm\|avx512\|fp16\|i8mm' src/codegen/ | head -50
```

Look for `if !self.is_arm { ... }` or similar arch checks. Each should error if the intrinsic isn't supported on the current target.

- [ ] **Step 5: Write `audit-findings.md`** in `docs/release/v1.11.0/`

For each issue found in steps 1-4, record: severity (Critical/Important/Minor), location (file:line), what's wrong, recommended fix. If no issues found, write that.

- [ ] **Step 6: Commit findings + apply Critical/Important fixes**

If Critical or Important issues exist, fix them in this phase. Minor issues go to Phase 7 (follow-ups) for batching.

---

## Phase 4: Test coverage audit

**Goal:** Every new feature has end-to-end + IR-level coverage. No silent regressions possible.

### Steps

- [ ] **Step 1: Coverage matrix**

For each intrinsic from Phase 1's catalog, fill in:
| Intrinsic | E2E test | IR-level guard | Negative test (typeck reject) | Cross-arch test |
|---|---|---|---|---|

Use:
```bash
git grep -l "<intrinsic_name>" tests/
```

For each row, mark:
- ✓ if test exists
- ✗ if missing
- N/A if not applicable (e.g., IR guards aren't needed for arithmetic that just adds an intrinsic dispatch)

- [ ] **Step 2: Identify gaps**

Any `✗` in the E2E column for a non-trivial intrinsic is a gap. Any `✗` in the IR-level column for a perf-critical intrinsic (like `exp_poly_f32` or the f16 lowering) is a gap.

Likely findings:
- Most pre-existing i8mm intrinsics have E2E tests (the project's TDD discipline).
- IR-level guards may be missing for some pre-existing intrinsics that depend on specific LLVM lowering (e.g., `bsrli_i8x16` should emit `vpsrldq` — is there a guard?).

- [ ] **Step 3: Fill the gaps**

For each `✗`, write the missing test. Follow patterns from `tests/phase14_exp_poly.rs` (IR guards) and `tests/phase14_arm_fp16.rs` (E2E with C harness).

Don't backfill EVERY pre-existing intrinsic — only the ones touched in this branch's commits or that depend on specific LLVM behavior. Pre-existing well-tested intrinsics from older branches don't need re-validation here.

- [ ] **Step 4: Run the full suite, confirm 0 failures**

```bash
cargo test --tests --features=llvm 2>&1 | grep "FAILED\|^test result" | tail -10
```

- [ ] **Step 5: Commit any new tests**

```
git commit -m "test: backfill coverage gaps for v1.11.0 intrinsics"
```

---

## Phase 5: Documentation audit

**Goal:** All user-facing docs current. New users should be able to discover and use every new intrinsic + flag.

### Files

- Likely modified: `CLAUDE.md`, `README.md`, `docs/idioms/` (if any new patterns), `docs/src/SUMMARY.md` (cookbook table of contents)
- New: `docs/release/v1.11.0/intrinsic-catalog.md` (single-page reference for users)
- New cookbook entries: `docs/src/cookbook/fp16-inference.md`, `docs/src/cookbook/fast-transcendentals.md`, `docs/src/cookbook/neon-gather-workaround.md`

### Steps

- [ ] **Step 1: CLAUDE.md audit**

Already partially done at the end of Pi 5 NEON work. Re-verify:
- CLI Reference table includes `--fp16`, `--dotprod`, `--i8mm`, `--avx512` (and any others surfaced by Phase 1)
- Source layout table includes all new `src/codegen/simd_*.rs` files (`simd_conv.rs`, `simd_fp16.rs`, `simd_exp_poly.rs`, plus any from the i8mm era that may have been added without CLAUDE.md tracking)
- "Files Near 500-Line Limit" table reflects current state (after Phase 2 fixes)

If anything is stale, update CLAUDE.md.

- [ ] **Step 2: README.md audit**

```bash
git log --oneline main..HEAD -- README.md
```

If README hasn't been touched, verify it doesn't claim things that are now wrong. Check:
- Project tagline / one-liner
- Quickstart examples (do they still work? do they showcase the new SIMD surface?)
- "Targets" section if it exists (must mention ARM FP16 if added)

If updates needed, commit.

- [ ] **Step 3: Write `intrinsic-catalog.md`**

`docs/release/v1.11.0/intrinsic-catalog.md` — single-page user reference covering ALL intrinsics in v1.11.0 (not just new). Structure:

```markdown
# Eä Intrinsic Catalog (v1.11.0)

## Memory ops
- `load`, `store`, `stream_store`, `gather`, `scatter`, `load_masked`, `store_masked`

## Arithmetic
- `fma`, `abs`, `sqrt`, `rsqrt`, `exp`, `exp_poly_f32` (NEW v1.11.0)

## Lane ops
- `splat`, `concat_*`, `lo128_*`, `hi128_*`, `shuffle_i32x{8,16}`, `blend_i32`
- `f32x4_from_scalars`, `f32x8_from_scalars` (NEW v1.11.0)

## Reductions
- `reduce_add`, `reduce_max`, `reduce_min`, `reduce_add_fast`

## Conversions
- `to_f32`, `to_f64`, `to_f16` (NEW v1.11.0), `to_i16`, `to_i32`, `to_i64`
- `cvt_f16_f32`, `cvt_f32_f16`
- `widen_u8_*`, `widen_i8_*`, `narrow_f32x4_i8`
- `pack_sat_*`, `pack_usat_*`, `round_f32x*_i32x*`

## Comparison + selection
- `select`, `min`, `max`, `sat_add`, `sat_sub`

## Bit ops
- `bsrli_i8x*`, `bslli_i8x*`, `bitcast_i8x*`, `bitcast_i32x*`

## Dot products
- `madd_i16`, `maddubs_i16`, `hadd_i16`, `vdot_i32`, `vdot_lane_i32`
- `smmla_i32`, `ummla_i32`, `usmmla_i32`

## Misc
- `prefetch`, `movemask`, `addp_i*`, `wmul_*`, `shuffle_bytes`
```

For each, list type signature(s) and required flags. Pull from Phase 1's inventory. Mark NEW v1.11.0 entries clearly.

Force-add (docs/ gitignored): `git add -f docs/release/v1.11.0/intrinsic-catalog.md`.

- [ ] **Step 4: Cookbook entries**

The existing `docs/src/cookbook/` has 4 pages (image-processing, ml-preprocessing, numpy-comparison, text-processing). Add 3 new entries showcasing the v1.11.0 surface:

**`docs/src/cookbook/fp16-inference.md`** — Native f16 KV cache + RMSNorm + attention. Real-world example: how Olorin's gemma4 inference loop uses `f16x8` load → `.* `→ `reduce_add` → scalar sqrt → splat-and-multiply for RMSNorm. Show before-and-after: the cvt_f16_f32 round-trip path (worked but wasted a load through f32) vs the native f16 path (`--fp16` flag, ~1.5-2x speedup on the KV path on Pi 5). Code blocks should compile against `tests/phase14_arm_fp16.rs` patterns.

**`docs/src/cookbook/fast-transcendentals.md`** — `exp_poly_f32` for softmax + GELU. Show why the LLVM `@llvm.exp.v*f32` scalarizes (no hardware exp on any arch), what the polynomial gets you (~10x in tight loops at GELU/softmax-acceptable error), and the algebraic-identity tanh trick (`tanh(x) = (exp(2x)-1)/(exp(2x)+1)`). Two complete example kernels: softmax-8-wide and tanh-GELU. Reference the spec for the bounded-range contract (`[-50, 50]`).

**`docs/src/cookbook/neon-gather-workaround.md`** — IQ3 LUT dequant on Pi 5. Show the AVX2 `gather()` pattern alongside the NEON workaround using `f32x4_from_scalars`. Compare against the existing `docs/idioms/neon-gather.md` (the idioms file is terse + reference-y; the cookbook entry is a longer narrative with the IQ3 motivation). Forward-reference: when SVE2 hardware is the target, the SVE-gather codegen path (deferred) lifts this restriction.

For each new cookbook page:
- Match the writing style of `docs/src/cookbook/numpy-comparison.md` (prose-first, code-supports-prose, real-world motivation)
- Code blocks must be compilable Eä — verify by extracting and running them through `ea --emit-llvm` to catch syntax errors before commit
- Cross-link: each page links to the relevant spec in `docs/superpowers/specs/` and the relevant test file in `tests/`

Update `docs/src/SUMMARY.md` (the bookdown TOC) to include the 3 new pages under the existing Cookbook section.

- [ ] **Step 5: Commit docs updates**

```
git commit -m "docs: v1.11.0 intrinsic catalog + 3 cookbook entries (FP16, transcendentals, NEON gather) + CLAUDE.md refresh"
```

---

## Phase 6: Performance audit (perf-critical features)

**Goal:** Confirm the perf claims in the specs hold on real hardware.

### Files

- Create: `benchmarks/v1.11.0/exp_poly_f32_bench.ea`
- Create: `benchmarks/v1.11.0/fp16_kv_bench.ea`
- Create: `benchmarks/v1.11.0/gather_compose_bench.ea`
- Create: `docs/release/v1.11.0/perf-results.md`

### Steps

- [ ] **Step 1: `exp_poly_f32` benchmark**

Goal: confirm ~10x speedup vs. `exp()` (libm-scalarized) in a softmax loop. Write a benchmark kernel:

```ea
// benchmarks/v1.11.0/exp_poly_f32_bench.ea
export func softmax_libm(x: *f32, out: *mut f32, n: i32) {
    // Uses exp() — scalarizes to libm
    ...
}

export func softmax_poly(x: *f32, out: *mut f32, n: i32) {
    // Uses exp_poly_f32
    ...
}
```

C harness measures wall-clock for N iterations on a vector of length 8192. Report tok/s or μs/call.

Expected: ~10x speedup on x86 (verified via `cargo bench` or C harness with `clock_gettime`).

- [ ] **Step 2: FP16 KV path benchmark**

Olorin's actual KV-load + RMSNorm pattern, two variants:
- f32 path (load f16 bits as i16, `cvt_f16_f32`, compute in f32, write back via `cvt_f32_f16`)
- f16 path (native `*f16` load, native f16 compute, native store)

Expected: ~1.5-2x speedup on Pi 5 (the gain is from removing the f32 round-trip; f16 compute itself isn't faster per element).

- [ ] **Step 3: Gather compose benchmark**

IQ3-LUT-style pattern, two variants:
- x86: native `gather()` (AVX2)
- ARM: scalar load + `f32x4_from_scalars` compose

Expected: x86 gather faster than ARM compose (as expected — that's the whole reason gather is x86-only). Confirm the ARM compose isn't slower than the original "use a scalar loop" pattern would have been (it should be at least as fast).

- [ ] **Step 4: Write `perf-results.md`**

```markdown
# v1.11.0 Performance Results

## exp_poly_f32 (x86 AVX2)
| Workload | exp() | exp_poly_f32 | Speedup |
|---|---|---|---|
| softmax(8192 f32) | XX μs | YY μs | Z.Zx |

## FP16 KV path (Pi 5 Cortex-A76 + FEAT_FP16)
| Workload | f32 round-trip | native f16 | Speedup |

## Gather compose (Pi 5 Cortex-A76)
| Workload | scalar loop (old) | from_scalars | Notes |
```

Force-add. Commit: `docs: v1.11.0 perf results`.

- [ ] **Step 5: If results miss the spec target**

The spec said "~10x speedup" for exp_poly_f32 and "removes f32 round-trip" for FP16. If actuals are significantly worse:
- Investigate whether opt_level=3 is enabled
- Check whether the bench loop is being optimized away (use volatile sink)
- If genuinely slower than expected, this is a Critical finding — block merge until resolved

If results are within margin (e.g., 7-10x for exp_poly_f32 instead of exactly 10x), record actuals and proceed.

---

## Phase 7: Follow-up cleanup

**Goal:** Address all queued follow-ups from prior reviews.

### Steps (one commit per item, or batched as logically appropriate)

- [ ] **Step 1: `src/codegen/statements.rs` 500-line violation**

File is at 503 lines (3 over). Pre-existing before this branch — needs a small extract. Look for the largest function or a coherent ~10-20 line block that could go into a new helper file (e.g., `src/codegen/statements_loops.rs` if a loop-statement section is large).

Verify with `wc -l`:
```bash
wc -l src/codegen/statements.rs  # should be ≤ 500 after fix
```

- [ ] **Step 2: `tests/phase14_arm_ext.rs:115` test assertion broadening**

The B5 follow-up note: assertion was widened to accept `"AVX2"` along with `"x86-only"` and `"256-bit"`. Final reviewer flagged this as overly defensive. Investigate: with HEAD's current code, does the test still fail with just the original `"x86-only" || "256-bit"` assertion?

If the original passes: revert the broadening.
If the original fails: keep the broadening, add a comment explaining WHY (the validate_type_for_target gate ordering produces an AVX2-mentioning error path).

Either way, document the choice.

- [ ] **Step 3: Investigate any other findings from Phases 2-4**

Critical items were already fixed in their phase. Important items batched here. Minor items deferred to v1.12.0 if they're not blocking.

- [ ] **Step 4: Single commit per fix, descriptive messages**

```
git commit -m "refactor: extract <X> from statements.rs to honor 500-line cap"
git commit -m "test: revert B5 assertion broadening (original passes after gate refactor)"
```

(Or: `chore: keep B5 assertion broadening; document AVX2 reason` if revert is wrong.)

- [ ] **Step 5: Verify nothing broke**

```bash
cargo test --tests --features=llvm 2>&1 | grep "FAILED\|^test result" | tail -10
cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings
```

---

## Phase 8: Release prep

**Goal:** Ready the branch for a clean v1.11.0 release.

### Files

- Modify: `Cargo.toml` (version bump)
- Modify: `Cargo.lock` (auto-regenerated)
- Create or update: `CHANGELOG.md` at repo root (if it exists; otherwise create)
- Modify: `README.md` if version is mentioned anywhere

### Steps

- [ ] **Step 1: Bump `Cargo.toml`**

Current: `version = "1.10.0"`. Target: `version = "1.11.0"` (minor bump — lots of new features, no breaking changes per spec).

```bash
sed -i 's/^version = "1\.10\.0"/version = "1.11.0"/' Cargo.toml
```

Verify: `grep '^version' Cargo.toml`.

Run `cargo build` to regenerate `Cargo.lock`.

- [ ] **Step 2: Generate CHANGELOG entry**

Pull from `docs/release/v1.11.0/inventory.md` (Phase 1 output). Format as Keep-a-Changelog style:

```markdown
## [1.11.0] — 2026-04-27

### Added
- `--fp16` CLI flag: enable native FEAT_FP16 compute on ARM (Pi 5 Cortex-A76+)
- `--dotprod` CLI flag: enable ARM dot-product instructions (sdot/udot/vdot)
- `--i8mm` CLI flag: enable ARM Int8 Matrix Multiply (smmla/ummla/usmmla)
- New SIMD types: `f16x4`, `f16x8`, `u16x16`
- New scalar type: `f16`
- New intrinsics:
  - Compose: `f32x4_from_scalars`, `f32x8_from_scalars`
  - Polynomial: `exp_poly_f32` (~10x faster than scalarized `exp()`)
  - f16 family: native splat, load, store, fma, reductions, element-wise arithmetic
  - Conversion: `to_f16`
  - Pre-existing layer: `vdot_lane_i32`, `bitcast_i8x{16,32}`, `bitcast_i32x{4,8}`,
    `shuffle_bytes` (sign-agnostic), `bsrli_i8x{16,32}`, `bslli_i8x{16,32}`,
    `pack_usat_i{16,32}x*`, `shuffle_i32x{8,16}`, `blend_i32`, `smmla/ummla/usmmla_i32`,
    `widen_u8_u16`, multi-width widen variants
- New docs: `docs/idioms/neon-gather.md`

### Changed
- `gather()` on ARM now points users at the canonical `f32x{4,8}_from_scalars`
  workaround pattern (was: "use a scalar loop on ARM")
- `--fp16` cross-arch check moved from main.rs to library layer
  (`compile_with_options`, `compile_to_ir_with_options`, `inspect_source` all gate)
- `to_f32(f16)` now correctly emits `fpext half to float` (was: type-punned passthrough)
- `simd_math.rs::compile_conversion` extracted to new `simd_conv.rs` to honor 500-line cap

### Performance
- exp_poly_f32: ~10x faster than scalarized libm `expf` in softmax/GELU loops
- FP16 KV path: removes f32 round-trip on every KV access on Pi 5

### Notes
- `exp()` (libm-backed) unchanged — for full-domain correctness, keep using it
- `cvt_f16_f32` / `cvt_f32_f16` (bits-as-i16) unchanged — for code that doesn't have FEAT_FP16
- f16 / f32x4_from_scalars are forward-compat with future `exp_poly_f16`,
  `tanh_approx`, `log_approx` extensions (not in v1.11.0)
```

If `CHANGELOG.md` doesn't exist at repo root, create it. If it does, prepend the v1.11.0 entry.

- [ ] **Step 3: Smoke test the release build**

```bash
cargo build --release
./target/release/ea --help 2>&1 | head -20  # confirm --fp16, etc., listed
```

- [ ] **Step 4: Final test sweep on release build**

```bash
cargo test --tests --features=llvm --release 2>&1 | grep "FAILED\|^test result" | tail -10
```

Expected: 0 failures.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit -m "chore: bump version to 1.11.0 + CHANGELOG"
```

---

## Phase 9: Merge

**Goal:** Land the branch on `main` cleanly.

### Steps

- [ ] **Step 1: Re-sync from origin**

```bash
git fetch origin
git log HEAD..origin/main --oneline   # any commits we don't have?
```

If `origin/main` has new commits since this branch diverged, decide: merge `origin/main` into the feature branch, or rebase. Prefer merge (preserves the linear development history).

```bash
git merge origin/main  # if needed
cargo test --tests --features=llvm  # verify still green after merge
```

- [ ] **Step 2: Push final state**

```bash
git push origin feat/i8mm-intrinsics
```

CI runs: x86 Linux, x86 Windows, ARM Linux. All should pass green.

- [ ] **Step 3: Open PR**

```bash
gh pr create --base main --head feat/i8mm-intrinsics \
  --title "v1.11.0: i8mm + Pi 5 NEON enablement + exp_poly_f32" \
  --body "$(cat <<'EOF'
## Summary

98 commits delivering three layers of SIMD compiler work:

1. **Pre-existing i8mm** (~73 commits): vdot_lane_i32, bitcast_*, shuffle_bytes,
   bsrli/bslli, pack_usat, shuffle_i32x*, blend_i32, smmla/ummla/usmmla,
   AVX-512 lane intrinsics, AVX-512 int dot, more.
2. **Pi 5 NEON Enablement** (19 commits): NEON gather compose primitives
   (f32x{4,8}_from_scalars) + native FP16 compute (--fp16 flag, f16 types,
   splat/load/store/fma/reductions, to_f16 conversion).
3. **exp_poly_f32** (6 commits): polynomial vector exp without scalarization
   (~10x speedup of softmax/GELU).

See `docs/release/v1.11.0/CHANGELOG.md` for the full inventory.

## Specs

- `docs/superpowers/specs/2026-04-27-pi5-neon-enablement-design.md`
- `docs/superpowers/specs/2026-04-27-exp-poly-f32-design.md`

## Test plan

- [x] `cargo fmt --check` clean
- [x] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [x] Full test suite passes (0 failures) on x86 + ARM CI runners
- [x] Performance benchmarks meet spec targets (see `docs/release/v1.11.0/perf-results.md`)
- [x] All source files ≤ 500 lines
- [x] No `// TODO` / `// HACK` / dead `#[allow(dead_code)]`

## Migration

**One breaking change:** `maddubs_i32` was removed and replaced by `madd_i16`
(commit `89130cb`). The old intrinsic hid a 2-instruction chain (`pmaddubsw +
pmaddwd`) behind one symbol, violating the "programmer sees the cost" rule.
Callers must rewrite as the explicit chain:
```
let t: i16x8 = maddubs_i16(a_u8, b_i8)
let r: i32x4 = madd_i16(t, ones_i16x8)
```

All other additions are net-new intrinsics, types, or flags.
Existing `exp()` and `cvt_f16_f32` / `cvt_f32_f16` semantics unchanged.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Wait for CI green, then merge**

```bash
gh pr checks <PR_URL>  # confirm green
gh pr merge <PR_URL> --merge  # NOT --squash, NOT --rebase (preserve commit history)
```

- [ ] **Step 5: Tag the release**

```bash
git checkout main
git pull
git tag -a v1.11.0 -m "v1.11.0: i8mm + Pi 5 NEON + exp_poly_f32"
git push origin v1.11.0
```

The release workflow on tag push will build binaries.

- [ ] **Step 6: Delete the feature branch**

```bash
git branch -d feat/i8mm-intrinsics  # local
git push origin --delete feat/i8mm-intrinsics  # remote
```

---

## Follow-up list (deferred to post-v1.11.0)

These are explicit non-goals for this audit + release. Logged in eabrain for future cycles.

| Item | Why deferred |
|---|---|
| Olorin-side IQ3_S/IQ3_XXS port | Downstream consumer, not eacompute. Owned by Olorin team / next session. Unblocked by Pi 5 NEON Part A landing. |
| `tanh_approx_f32`, `log_approx_f32`, `sin/cos_approx_f32` | YAGNI for v1.11. Each is its own minimax fit + spec. Add when consumers exist. tanh-GELU expressible via algebraic identity over `exp_poly_f32`. |
| `exp_poly_f64` | No current consumer. f64 polynomial needs different coefficients. |
| `exp_poly_f16` | f16-native softmax is a future Olorin optimization. Add when a kernel needs it. |
| `exp_poly_hi_f32` (higher precision tier) | YAGNI. Add only when a kernel needs > 2⁻¹⁸ relative error from a polynomial path. |
| Scalar `exp_poly_f32` | Transcendentals run in hot SIMD loops; add scalar form when a kernel needs it. |
| SVE / SVE2 gather codegen | Pi 5 has no SVE. Add when M4/Graviton/Snapdragon X is a target. |
| x86 AVX-512-FP16 path | Sapphire Rapids / Granite Rapids. Separate spec. |
| `compile_to_ir_with_options` test coverage for `--fp16` rejection | No CLI exposure today; test added if a consumer surfaces. |
| Polymorphic `exp()` typeck cleanup | Pre-existing minor "Concrete, not generic" violation. Not in scope of this audit. |

These all get eabrain `note` entries with `--type note` after merge so the next session has the context.

---

## Self-Review

**Spec coverage:** This is an audit plan, not a feature plan. Coverage isn't measured against a spec — it's measured against "does this catch real risks before merge".

Risks covered:
- ✓ File-size violations (Phase 2)
- ✓ Stub / placeholder code (Phase 2)
- ✓ Stale `#[allow]` attributes (Phase 2)
- ✓ Inconsistent naming (Phase 3)
- ✓ Unhelpful error messages (Phase 3)
- ✓ Missing test coverage (Phase 4)
- ✓ Stale docs (Phase 5)
- ✓ Performance regression vs claimed targets (Phase 6)
- ✓ Known follow-ups (Phase 7)
- ✓ Release prep (Phase 8)
- ✓ Merge mechanics (Phase 9)

Risks NOT covered (explicit non-goals):
- ABI / FFI compatibility (no breaking changes per spec; trusted)
- Security audit (no security-sensitive code in this branch)
- Cross-compilation matrix beyond x86 + ARM (mac/windows-arm64 are CI's job)
- Backward compat with prior v1.x users (no API removed; trusted)

**Placeholder scan:** None.

**Type consistency:** N/A — this is an audit plan, no type signatures.

Plan complete and ready to execute.
