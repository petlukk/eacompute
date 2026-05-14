# Roadmap

Forward-looking notes. Ordered by leverage, not by effort.

## Shipped in v1.12.0 (2026-05-13)

- **Deprecation-warning infrastructure** + `docs/migrations/` directory + `cargo public-api` CI gate (PR #6).
- **`u64x{2,4,8}` vector types** (PR #7).
- **`wmul_u64_lo` / `wmul_u64_hi`** — u32×u32 → u64 widening multiply, first cross-platform `wmul_*` family member, Poly1305 unblocker (PR #8).
- **Typed sat_add / sat_sub / abs_diff spellings** (`sat_add_i8x16`, etc.) — polymorphic parents deprecated, removal v2.0.0. Plus an x86 codegen bug fix caught by the new end-to-end test discipline (PR #9).
- **Lane extractors filling the i16/u16 gap** — `lo128_i16x16` / `hi128_i16x16` / `lo128_u16x16` / `hi128_u16x16` / `lo256_i16x32` / `hi256_i16x32` (PR #10).
- **`widen_u8_u16_hi`** — partner to `widen_u8_u16` (PR #11).
- **`cvt_f32_f16` width-16 form** — closes the AVX-512 asymmetry with `cvt_f16_f32` (PR #12).

The runtime deprecation-warning mechanism (PR #6) supersedes the originally-planned `#[deprecated]` attribute approach.

## Tooling

### Self-hosted Pi 5 / Cortex-A76 CI runner

Generic GitHub Actions `aarch64` runners don't exercise `--fp16` or `--i8mm` codegen on real silicon. v1.11.0 had to gate `f32x8` tests in `phase14_arm_neon.rs` and `phase14_exp_poly.rs` to `target_arch = "x86_64"` because the runner's plain aarch64 NEON is 128-bit; the actual Pi 5 path (`--target=cortex-a76`) isn't covered. The headline 2.23× `gemma4_gelu` perf number and the f16 KV path ship uncorroborated by automation. One physical Pi 5 + a self-hosted runner closes the loop and stops Pi-specific regressions from shipping silently.

### `ea bench` subcommand (or `cargo bench` harness)

**SHIPPED in v1.13.0.** See `docs/src/reference/bench.md` for usage. The subcommand path was chosen over the `cargo bench` harness; the original observations below are retained for historical context.

Phase 6 of the v1.11.0 audit wrote `.ea` kernels + C harnesses + Makefile from scratch. A standing benchmark suite re-running per release would have surfaced the "~10× → 2.93×" `exp_poly_f32` calibration before the audit and would catch perf regressions from upstream LLVM/glibc.

Two paths: extend `ea` with a `bench` subcommand that takes an `.ea` + C harness and reports timing; or wire `benchmarks/v1.11.0/` into a Rust harness invoked by `cargo bench`. The second is cheaper to start. Both v1.11.0 and v1.12.0 shipped without this; v1.13.0 should not.

### Single canonical `Specification.md`

Today the language spec is spread across `docs/src/reference/*.md` (types, intrinsics, CLI, ARM). The README has no single doc to link as "the spec." The dead `1.6.md` and the removed-from-remote `Specification.md` both hinted at this gap. Pick one form (tracked single-file at repo root, or `docs/src/reference/index.md` umbrella) and make it the source of truth.

## Future API consistency

- **`tanh_approx_f32`, `log_approx_f32`, `sin_cos_approx_f32`** — polynomial approximations following the `exp_poly_f32` pattern. `tanh_approx_f32` is most-requested (currently expressed via `tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)` over `exp_poly_f32`).
- **Scalar `f16` conversion** — `to_f16(f32)` and partner scalar variants of the cvt family. The current intrinsic surface is vector-only.
- **`u16x32` token + sibling lane extractors** (`lo256_u16x32` / `hi256_u16x32`). Skipped in PR #10 because `u16x32` itself doesn't exist yet; add when a consumer asks.
- **Wider `wmul_u64`** — `wmul_u64(u32x4, u32x4) -> u64x4` full widening via paired pmuludq + interleave, or AVX2/AVX-512 widths (`u32x8`/`u32x16` inputs). The current `_lo`/`_hi` pair is sufficient for Poly1305; add wider forms when a real consumer benchmarks the savings.

## Future additions

### Autoresearch ↔ perf-regression feedback

The `autoresearch/kernels/<name>/best_kernel.ea` files are valuable as examples but aren't wired into the compiler's perf regression suite. An upstream LLVM bump or glibc change that regresses them goes unnoticed until someone reruns autoresearch manually. Wiring them into `ea bench` (above) gives cross-release performance tracking for free, and surfaces regressions caused by Eä's own codegen changes.

**Note on the existing autoresearch archive:** the kernels under `autoresearch/kernels/` were optimized in earlier Eä releases when the intrinsic surface was much smaller. Their `best_kernel.ea` files are mostly scalar / basic-SIMD and don't use the v1.10+ intrinsics. Re-running autoresearch on the modern (v1.12.0) intrinsic set is a v1.13.0 task — the existing archive is best read as historical baselines.

## Observation, not a recommendation

The project's discipline mostly survives because the maintainer holds the line — at the spec level, in code review, in the audit. That's a single point of failure. The tooling items above (CI gates, migrations directory, public-api check, standing benchmarks) shift discipline from vigilance-dependent to mechanical, which is what scales. v1.12.0 closed the migration + public-api items; `ea bench` and the Pi runner are still vigilance-dependent.
