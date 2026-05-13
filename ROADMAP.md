# Roadmap

Forward-looking notes captured after the v1.11.0 audit. Ordered by leverage, not by effort.

## Tooling

### Self-hosted Pi 5 / Cortex-A76 CI runner

Generic GitHub Actions `aarch64` runners don't exercise `--fp16` or `--i8mm` codegen on real silicon. v1.11.0 had to gate `f32x8` tests in `phase14_arm_neon.rs` and `phase14_exp_poly.rs` to `target_arch = "x86_64"` because the runner's plain aarch64 NEON is 128-bit; the actual Pi 5 path (`--target=cortex-a76`) isn't covered. The headline 2.23× `gemma4_gelu` perf number and the f16 KV path ship uncorroborated by automation. One physical Pi 5 + a self-hosted runner closes the loop and stops Pi-specific regressions from shipping silently.

### `ea bench` subcommand (or `cargo bench` harness)

Phase 6 of the v1.11.0 audit wrote `.ea` kernels + C harnesses + Makefile from scratch. A standing benchmark suite re-running per release would have surfaced the "~10× → 2.93×" calibration before the audit and would catch perf regressions from upstream LLVM/glibc.

Two paths: extend `ea` with a `bench` subcommand that takes an `.ea` + C harness and reports timing; or wire `benchmarks/v1.11.0/` into a Rust harness invoked by `cargo bench`. The second is cheaper to start.

### `cargo public-api` check + `docs/migrations/`

v1.11.0 had one breaking change (`maddubs_i32` → `madd_i16`). Caught by manual diff inspection in audit Phase 3, not by tooling. Future breaking changes should be impossible to ship silently.

- CI gate: `cargo public-api` (or equivalent) fails builds where the export surface changes without explicit acknowledgement.
- `docs/migrations/v1.11.0.md`, `v1.12.0.md`, … one file per breaking release, with the migration recipe in a stable canonical location.

### Single canonical `Specification.md`

Today the language spec is spread across `docs/src/reference/*.md` (types, intrinsics, CLI, ARM). The README has no single doc to link as "the spec." The dead `1.6.md` and the removed-from-remote `Specification.md` both hinted at this gap. Pick one form (tracked single-file at repo root, or `docs/src/reference/index.md` umbrella) and make it the source of truth.

## API consistency (v1.12.0 batch)

Carried forward from the v1.11.0 audit findings (PR #2 history, since-removed `audit-findings.md`):

- **`tanh_approx_f32`, `log_approx_f32`, `sin_cos_approx_f32`** — polynomial approximations following the `exp_poly_f32` pattern. `tanh_approx_f32` is most-requested (currently expressed via `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)` over `exp_poly_f32`).
- **`lo_i16x8` / `hi_i16x8` lane extractors** — the lane family currently exposes `lo*`/`hi*` for i8/u8/i32/f32 but not i16. The Phase 3 `madd_i16` ARM-recipe is awkward without these.
- **`cvt_f32_f16` width-16 form** — closes the API asymmetry: `cvt_f16_f32` accepts widths {4, 8, 16} but `cvt_f32_f16` only {4, 8}. Round-trip at 16-wide via `llvm.x86.avx512.mask.vcvtps2ph.512` on x86.
- **Monomorphic rename for `sat_add` / `sat_sub` / `abs_diff`** — these remain polymorphic (i8/u8/i16/u16/i32/u32 element-type variants share one name) in violation of "Concrete, not generic." Rename to `sat_add_i8x16`, etc., with a `#[deprecated]` migration window.

## Additions

### `#[deprecated(since = "X.Y.Z", note = "use Z")]` on intrinsics

A migration-window attribute on removed intrinsics would have made `maddubs_i32` → `madd_i16` painless: callers would compile with a warning during deprecation rather than fail with "unknown intrinsic" at the breaking release. Required for the v1.12.0 `sat_add_*` / `abs_diff_*` rename batch.

### Autoresearch ↔ perf-regression feedback

The 28 `autoresearch/kernels/<name>/best_kernel.ea` files are valuable as examples but aren't wired into the compiler's perf regression suite. An upstream LLVM bump or glibc change that regresses them goes unnoticed until someone reruns autoresearch manually. Wiring them into `ea bench` (above) gives cross-release performance tracking for free, and surfaces regressions caused by Eä's own codegen changes.

**Note on the existing autoresearch archive:** the kernels under `autoresearch/kernels/` were optimized in earlier Eä releases when the intrinsic surface was much smaller. Their `best_kernel.ea` files are mostly scalar / basic-SIMD and don't use the v1.10+ intrinsics (`smmla_*`, `exp_poly_f32`, native f16, AVX-512 lane family, `f32x{4,8}_from_scalars`). Reruning autoresearch on the modern intrinsic set is itself a v1.12.0 task — the existing archive is best read as historical baselines, not current best-known kernels.

## Observation, not a recommendation

The project's discipline mostly survives because the maintainer holds the line — at the spec level, in code review, in the audit. That's a single point of failure. The tooling items above (CI gates, migrations directory, public-api check, standing benchmarks) shift discipline from vigilance-dependent to mechanical, which is what scales.
