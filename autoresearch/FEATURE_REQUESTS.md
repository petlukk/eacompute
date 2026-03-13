# Autoresearch Feature Requests

Feature requests discovered by the autoresearch agent during optimization runs.
These are concrete limitations the agent hit that prevented it from generating faster code.

## Vector min/max intrinsics

**Source:** Clamp benchmark, iterations 1-2
**Error:** `max expects (i32,i32), (f32,f32), or (f64,f64), got (f32x4, f32x4)`
**Impact:** Agent can't use `min(max(v, lo), hi)` for clamp — forced to use `select` which generates 4 instructions (`vcmpps` + `vblendvps` × 2) instead of C's 2-instruction `vmaxps`/`vminps`. Measured ~2x instruction count overhead in the hot loop.
**Fix:** Extend `min`/`max` intrinsics to accept all SIMD float types (f32x4, f32x8, f32x16, f64x2, f64x4). Lower to `llvm.maxnum`/`llvm.minnum` vector variants.
**Loop:** C (language design)

## load() type inference for overloaded widths

**Source:** Dot product benchmark, iterations 1 and 4
**Error:** `vector width mismatch: 4 vs 8` on `fma(load(a, i), load(b, i), acc0)`
**Impact:** Agent wrote `load(a, i)` expecting f32x8 inference from the FMA context, but load inferred f32x4. Agent hit this twice with the same mistake — a UX signal.
**Fix:** Consider inferring load width from usage context, or provide clearer error message suggesting explicit type annotation: `load::<f32x8>(a, i)` or `let v: f32x8 = load(a, i)`.
**Loop:** A (compiler) or C (language design)
