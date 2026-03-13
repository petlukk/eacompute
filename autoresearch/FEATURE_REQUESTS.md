# Autoresearch Feature Requests

Feature requests discovered by the autoresearch agent during optimization runs.
These are concrete limitations the agent hit that prevented it from generating faster code.

## Vector min/max intrinsics

**Source:** Clamp benchmark, iterations 1-2
**Error:** `max expects (i32,i32), (f32,f32), or (f64,f64), got (f32x4, f32x4)`
**Impact:** Agent can't use `min(max(v, lo), hi)` for clamp — forced to use `select` which generates 4 instructions (`vcmpps` + `vblendvps` × 2) instead of C's 2-instruction `vmaxps`/`vminps`. Measured ~2x instruction count overhead in the hot loop.
**Fix:** Extend `min`/`max` intrinsics to accept all SIMD float types (f32x4, f32x8, f32x16, f64x2, f64x4). Lower to `llvm.maxnum`/`llvm.minnum` vector variants.
**Loop:** A (compiler) — first Loop A target
**Status:** DONE (commit 6ca5767). Implemented vector min/max, updated clamp kernel from select to min(max(v,lo),hi). Result: 91.4 → 81.2 µs = 11% improvement. First complete A→B feedback cycle.

## load() type inference for overloaded widths

**Source:** Dot product benchmark, iterations 1 and 4
**Error:** `vector width mismatch: 4 vs 8` on `fma(load(a, i), load(b, i), acc0)`
**Impact:** Agent wrote `load(a, i)` expecting f32x8 inference from the FMA context, but load inferred f32x4. Agent hit this twice with the same mistake — a UX signal.
**Fix:** Consider inferring load width from usage context, or provide clearer error message suggesting explicit type annotation: `load::<f32x8>(a, i)` or `let v: f32x8 = load(a, i)`.
**Loop:** A (compiler) or C (language design)
**Status:** DONE. Named typed load intrinsics implemented: `load_f32x4`, `load_f32x8`, `load_i32x8`, etc. for all vector types. No type annotation needed — `fma(load_f32x8(a, i), load_f32x8(b, i), acc)` just works. Error hint updated to reference new intrinsics.
**History:** PARTIAL (commit 6f90da2) added hint message. Escalated from 2/10 to 13/25 iterations in matmul. Resolved with named intrinsics (option b) — most aligned with Eä's explicit-over-implicit philosophy.

## Unordered (fast-math) reduce_add

**Source:** Loop A generell exploration, iterations 1-5
**Observation:** Agent attempted tree reduction for `reduce_add` in all 5 iterations. Currently `reduce_add` on floats uses `llvm.vector.reduce.fadd` with an ordered start value (0.0), forcing LLVM to emit width-many sequential dependent `fadd` instructions. The agent proposed replacing this with log2(width) shuffle+add pairs (tree reduction), where each level's additions are independent and can be pipelined.
**Blocked by:** `test_reduce_add_ir` asserts the vector reduce intrinsic appears in IR. Iteration 3 passed clippy + compilation but failed this test. Iterations 1,2,4 failed clippy. Iteration 5 timed out.
**Impact:** Reduction kernel (39.7µs) and dot product (95.9µs) are both latency-bound by the sequential reduce. Tree reduction would enable parallel execution of adds across their ~4-cycle latency.
**Design question:** Ordered reduce guarantees deterministic floating-point results. Unordered reduce is faster but non-deterministic across runs. Options: (a) default to fast-math unordered reduce, (b) add `reduce_add_fast()` as separate intrinsic, (c) compiler flag `--fast-math` that switches all reduces to unordered.
**Loop:** C (language design) — requires a design decision about floating-point semantics
**Status:** NEW
