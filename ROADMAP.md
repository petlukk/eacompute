# Roadmap

Forward-looking notes. Ordered by leverage, not by effort.

## Shipped in v1.15.0

### Scalar `stream_store` overloads

`stream_store(*mut i16, offset, i16)`, `stream_store(*mut u16, offset, u16)`,
`stream_store(*mut i32, offset, i32)`, `stream_store(*mut u32, offset, u32)`,
`stream_store(*mut i64, offset, i64)`, `stream_store(*mut u64, offset, u64)`.
Same `!nontemporal` metadata path as the existing vector form. Lowers to
`movnti` on x86 SSE2 for i32/i64. Concrete consumer pulling: Olorin's
`q4k_repack.ea` (pure-streaming block-layout repack blocked on scalar surface).

### `fence_nt()` intrinsic

Zero-argument store-store memory barrier. Lowers to `sfence` on x86 (via
`@llvm.x86.sse.sfence`) and `dmb ishst` on aarch64 (via `@llvm.aarch64.dmb`
with operand 10). Completes the `prefetch_nta` + `stream_store` + `fence_nt`
non-temporal memory-hint family. Most callers will not need it — host-side
sync primitives provide cross-thread release semantics — but for the rare
intra-kernel ordering case it's the documented expression. Note: does NOT
provide store-to-load ordering; a full barrier (`mfence`/`dmb sy`) is needed
for write-then-read-back patterns.

### Vector `stream_store` alignment fix

Pre-existing bug surfaced by the new objdump test discipline: the
`set_alignment` call on the store instruction passed element-width alignment
rather than vector-width. LLVM 18 took this conservatively and decomposed
NT vector stores into element-wise scalar `movntsd` sequences, defeating
the entire point of the intrinsic. Fix: set alignment to
`element_size * lane_count`. Behavior change visible to callers — fast path
for aligned buffers, SIGSEGV per the documented contract for misaligned.
Caught and pinned by the new alignment-failure crash test.

### `stream_store` documentation upgrade

Reference docs upgraded to `prefetch_nta` parity: target-specific lowering
table, alignment contract, ordering contract, and "When NOT to use" anti-
pattern guidance. The last item is most important — prevents adoption
regressions in working-buffer kernels (softmax accumulators, FWHT scratch)
where blanket-substituting `store → stream_store` would degrade by forcing
DRAM round-trips on cache-resident data.

## Shipped in v1.14.0

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

### log_approx_f32

`log_approx_f32(v: f32xN) -> f32xN`. Bit-level decomposition of `x = m · 2^e` (frexp convention, `m ∈ [0.5, 1)`), √2/2 rebalance to center the polynomial range, degree-8 Eigen-coefficient Horner in `(m - 1)`, then Cody-Waite recombine with `e · ln(2)`. Avoids `@llvm.log.v*f32`, which LLVM scalarizes to per-lane libm `logf`. Max absolute error ~3e-6 across `(0, +∞)`; matches `exp_poly_f32`'s 2⁻¹⁸ relative target. Composes cleanly with `exp_poly_f32` — pin-tested via a 4-input roundtrip kernel (`exp_poly_f32(log_approx_f32(x)) ≈ x` to ~1e-4 relative). See `docs/superpowers/specs/2026-05-19-log-approx-f32-design.md`.

### sin_approx_f32 + cos_approx_f32

`sin_approx_f32(v: f32xN) -> f32xN` and `cos_approx_f32(v: f32xN) -> f32xN`. Shared core: reduce mod π/2 via 2-piece Cody-Waite split with FMA-preserved precision, compute both sin and cos polynomials over the reduced argument `d' ∈ [-π/4, π/4]` (degree 3 in `s = d'²` for sin, degree 4 for cos), then quadrant-blend at the end. The `cos` variant reuses the same core with `q += 1` — a precision-free integer shift expressing the mathematical identity `cos(x) = sin(x + π/2)`. Max abs error ~3e-6 across `[-1e7, 1e7]`. Closes the original "Future API consistency" trio entry. Spec at `docs/superpowers/specs/2026-05-19-sin-cos-approx-f32-design.md`.

The roadmap entry was titled "sin_cos_approx_f32" suggesting a pair-return, but Eä has no precedent for multi-return intrinsics. Shipping two separate intrinsics matches the established pattern (`exp_poly_f32` / `tanh_approx_f32` / `log_approx_f32`) and lets LLVM CSE handle any caller-side range-reduction redundancy.

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

- **Wider-input `wmul_u64` variants** — AVX2/AVX-512 widths (`wmul_u64(u32x8, u32x8) -> u64x8` on AVX-512, etc.). Requires `u32x8` / `u32x16` lexer tokens which don't exist yet. The fused `wmul_u64(u32x4, u32x4) -> u64x4` shipped in v1.14.0; wider widths gated on a consumer asking *and* providing the input tokens.

## Future additions

### Multi-core / `parallel_for` primitive — deferred indefinitely (v1.15 audit)

Eä remains single-thread SIMD. Multi-core orchestration is the host's
responsibility. The v1.15 brainstorm evaluated two candidate forms — an
in-language `parallel_for` keyword and a binding-layer `parallel: true`
metadata flag generating per-language wrappers — and dropped both after a
consumer audit:

- **Olorin** ships `src/inference/threadpool.rs`, a custom SpinBarrier-based
  pool with `run_graph<F: Fn(usize, usize, &SpinBarrier, &AtomicI32) + Send
  + Sync>`. Designed for ggml-style inference graphs with cross-kernel
  barriers and atomic-int dynamic work counters. Strictly more capable than
  anything generic Eä could provide; a binding-layer wrapper cannot express
  cross-kernel barriers.
- **Cougar** invokes Eä kernels (`q4k_4row_dot` — 4 rows per call) inside
  `pool.run(n_threads, |tid, _| { ... })`, looping many times per thread
  with per-thread accumulator state across calls. A "wrap in N threads,
  one call each" binding wrapper would replace this with strictly worse
  work distribution.
- **eakv** has no parallelism today but the workaround for the dequant case
  is 3 lines of `concurrent.futures.ThreadPoolExecutor` in eakv's own code
  — not enough pull to justify permanent surface area in eacompute.

Criteria for revisiting: a new consumer profile that custom Olorin/Cougar
pools don't already serve — most likely a Python-first consumer with
non-trivial host-side parallelism cost, or a consumer with simpler graph
structure than ggml-style inference. None has surfaced.

### Autoresearch ↔ perf-regression feedback

The `autoresearch/kernels/<name>/best_kernel.ea` files are valuable as examples but aren't wired into the compiler's perf regression suite. An upstream LLVM bump or glibc change that regresses them goes unnoticed until someone reruns autoresearch manually. Wiring them into `ea bench` (above) gives cross-release performance tracking for free, and surfaces regressions caused by Eä's own codegen changes.

**Note on the existing autoresearch archive:** the kernels under `autoresearch/kernels/` were optimized in earlier Eä releases when the intrinsic surface was much smaller. Their `best_kernel.ea` files are mostly scalar / basic-SIMD and don't use the v1.10+ intrinsics. Re-running autoresearch on the modern (v1.12.0) intrinsic set is a v1.13.0 task — the existing archive is best read as historical baselines.

## Observation, not a recommendation

The project's discipline mostly survives because the maintainer holds the line — at the spec level, in code review, in the audit. That's a single point of failure. The tooling items above (CI gates, migrations directory, public-api check, standing benchmarks) shift discipline from vigilance-dependent to mechanical, which is what scales. v1.12.0 closed the migration + public-api items; `ea bench` and the Pi runner are still vigilance-dependent.
