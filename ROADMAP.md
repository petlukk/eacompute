# Roadmap

Forward-looking notes. Ordered by leverage, not by effort.

## Shipped in v1.14.0 (UNRELEASED)

### Runtime SIMD permute

`permute_runtime(f32x8, i32x8) -> f32x8` and `permute_runtime(i32x8, i32x8) -> i32x8`. Lowers to `vpermps` / `vpermd` on AVX2. ARM rejected with `codegen_error` referencing the NEON idiom doc. Shipped in v1.14.0.

### Prefetch hint variants

`prefetch_write(ptr, offset)` (rw=1, locality=3) and `prefetch_nta(ptr, offset)` (rw=0, locality=0), both cross-platform via `llvm.prefetch.p0`. Existing `prefetch` is unchanged. The original "prefetch as a statement" roadmap entry (commit `c879b45`) was based on a stale premise — `prefetch` already worked in function bodies; the real gap was hint-flavor coverage. See `docs/superpowers/specs/2026-05-18-prefetch-hints-design.md` for the post-mortem.

### shuffle two-source form

`shuffle(a, b, [indices])` overloads the existing `shuffle` intrinsic by arity. Two-source mask semantics match LLVM `shufflevector`: indices in `[0, width)` select from `a`, indices in `[width, 2 * width)` select from `b`. Single-source form unchanged. The original "Compile-time `shuffle` for width-8" Future entry (commit `c879b45`) was based on a stale premise — single-source width-8 shuffle already worked end-to-end; the real gap was the two-source mask form. See `docs/superpowers/specs/2026-05-18-shuffle-two-source-design.md` for the post-mortem.

### tanh_approx_f32

`tanh_approx_f32(v: f32xN) -> f32xN`. Rational `P(x²) · x / Q(x²)` approximation in the Eigen / TensorFlow / JAX fast-tanh family — degree-13 numerator (odd in x), degree-6 denominator (even in x²), one fdiv per call. Clamped to [-9, 9] for saturation. Max absolute error ~3e-7 across the body. Replaces the `(exp_poly_f32(2x) - 1) / (exp_poly_f32(2x) + 1)` workaround that the cookbook previously documented for tanh-GELU; the workaround suffers catastrophic cancellation near zero and is now obsolete. See `docs/superpowers/specs/2026-05-19-tanh-approx-f32-design.md`.

### u16x32 token + lo256/hi256 lane extractors

`u16x32` vector token (lexer, parser, type-annotation list, vector-literal suffix list) plus `lo256_u16x32(u16x32) -> u16x16` and `hi256_u16x32(u16x32) -> u16x16` lane extractors completing the i16/u16 symmetry. Dispatch-only additions — typeck reuses `check_lo_extract` / `check_hi_extract`, codegen reuses width-generic `compile_lo_extract` / `compile_hi_extract`. ARM rejection inherits from the existing >128-bit guard at the codegen vector-type validation site. PR #10 (v1.12.0) deferred this pair pending the `u16x32` token; v1.14.0 closes both at once.

### Fused wmul_u64 full-width form

`wmul_u64(u32x4, u32x4) -> u64x4` widens all four lanes in a single intrinsic call, replacing the manual `wmul_u64_lo` + `wmul_u64_hi` + concat dance. Lowers to two `vpmuludq` + interleave via LLVM's `mul(zext, zext)` pattern-match. x86-only: the `u64x4` return type is 256-bit and rejected by the existing ARM >128-bit guard before the intrinsic dispatcher runs. ARM callers continue to use the lo/hi pair (each returning the NEON-fitting `u64x2`). Wider-input variants (`wmul_u64(u32x8, ...) -> u64x8`) explicitly deferred — they require new `u32x8` / `u32x16` lexer tokens and have no documented consumer yet. See `docs/superpowers/specs/2026-05-19-wmul-u64-fused-design.md`.

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

- **`log_approx_f32`, `sin_cos_approx_f32`** — polynomial approximations following the `exp_poly_f32` pattern. `tanh_approx_f32` shipped in v1.14.0; the remaining two are speculative until a real consumer asks.
- **Wider-input `wmul_u64` variants** — AVX2/AVX-512 widths (`wmul_u64(u32x8, u32x8) -> u64x8` on AVX-512, etc.). Requires `u32x8` / `u32x16` lexer tokens which don't exist yet. The fused `wmul_u64(u32x4, u32x4) -> u64x4` shipped in v1.14.0; wider widths gated on a consumer asking *and* providing the input tokens.

## Future additions

### Multi-core / `parallel_for` primitive

Eä is single-thread SIMD today; concurrency comes from outer-loop threading in the Rust / Python / Go caller. A `parallel_for(range, body)` primitive that spawns SIMD work across cores would change the per-call performance model from "kernel uses one core" to "kernel uses the machine." Pi 5 has 4 A76 cores; Zen 4 has 16+. With single-thread SIMD perf increasingly memory-bound (chacha20 hits 3.6 GB/s on Zen 4 = DRAM ceiling, not compute), multi-core is the unused dimension where the next 2–10× lives.

Open design questions:
- **Thread-pool model.** Static partition is simple but loses to imbalanced workloads; work-stealing (rayon-style) is robust but adds runtime dependency. Eä's "no implicit runtime" stance suggests caller-supplied pool injection.
- **C ABI interaction.** Does the kernel signature change? `(thread_id, num_threads)` extra args, or hidden global?
- **Determinism.** Reductions become non-associative under parallel execution; `reduce_add` semantics across threads need a documented contract.
- **NUMA / cache discipline.** First-touch placement matters on Zen 4 and on multi-socket. Probably out-of-scope for v1.x, but the API shape shouldn't preclude it.

Likely a v1.15+ initiative — too large for v1.14.0, but worth scoping early so the smaller carry-overs don't constrain the design space.

### Autoresearch ↔ perf-regression feedback

The `autoresearch/kernels/<name>/best_kernel.ea` files are valuable as examples but aren't wired into the compiler's perf regression suite. An upstream LLVM bump or glibc change that regresses them goes unnoticed until someone reruns autoresearch manually. Wiring them into `ea bench` (above) gives cross-release performance tracking for free, and surfaces regressions caused by Eä's own codegen changes.

**Note on the existing autoresearch archive:** the kernels under `autoresearch/kernels/` were optimized in earlier Eä releases when the intrinsic surface was much smaller. Their `best_kernel.ea` files are mostly scalar / basic-SIMD and don't use the v1.10+ intrinsics. Re-running autoresearch on the modern (v1.12.0) intrinsic set is a v1.13.0 task — the existing archive is best read as historical baselines.

## Observation, not a recommendation

The project's discipline mostly survives because the maintainer holds the line — at the spec level, in code review, in the audit. That's a single point of failure. The tooling items above (CI gates, migrations directory, public-api check, standing benchmarks) shift discipline from vigilance-dependent to mechanical, which is what scales. v1.12.0 closed the migration + public-api items; `ea bench` and the Pi runner are still vigilance-dependent.
