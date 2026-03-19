# Eä Kernel Optimization — Cornell Box Ray Tracer

You are optimizing an Eä ray tracing kernel: a Cornell Box renderer with 5 walls, 2 spheres (diffuse + mirror), direct lighting with hard shadows, and single-bounce reflection.
Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit the kernel to improve render performance. You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Output must match the C reference within rtol=1e-4, atol=1e-5.
3. One change per iteration. State your hypothesis clearly.
4. The `render` function signature must not change: `(out: *mut f32, width: i32, height: i32)`.
5. No dead code. No comments longer than one line.

## Strategy Space

You are free to choose any implementation strategy. The benchmark decides — not convention.

**Per-pixel work:** Each pixel traces a primary ray through `closest_hit` (5 plane tests + 2 sphere tests), then shades with a shadow ray (another `closest_hit`). Mirror sphere pixels recurse once. The hot function is `closest_hit`.

**Dimensions to explore:**

- **Algebraic simplification**: `hit_sphere` computes `a = dot(rd,rd)` but rd is normalized so `a = 1.0`. Eliminate the division by `2a` entirely.
- **rsqrt vs sqrt**: `v3_normalize` already uses rsqrt. Can `hit_sphere` use rsqrt where sqrt is currently used?
- **Precompute sphere constants**: sphere centers/radii are compile-time constants. Precompute `radius*radius`, `center` as Vec3, etc. inside the function rather than passing scalar args.
- **Early rejection**: in `closest_hit`, skip sphere tests if the ray is clearly pointing away. Quick dot-product test can reject before full quadratic.
- **Reduce divisions**: the 5 plane tests each do a division. Some share `1/rd.y` or `1/rd.x` — precompute reciprocals.
- **Inline control**: helper functions like `v3_add`, `v3_sub` should inline at -O3 but verify the generated code isn't bloated.
- **Shadow ray optimization**: shadow rays only need any-hit, not closest-hit. A specialized `any_hit` function can early-exit on first intersection.
- **Loop over pixels**: the outer loops are simple `for py in 0..height, for px in 0..width`. Consider processing multiple pixels at once if there's a SIMD opportunity.

**What probably won't help:**
- SIMD vectorization of individual rays: the branchy per-pixel logic (conditionals in closest_hit, recursion for mirror) makes per-ray SIMD very hard. Focus on scalar optimizations.
- Changing the scene: the scene is fixed and must produce identical output.

## Architecture Notes

- **243 lines** total, ~170 non-trivial.
- `closest_hit` is called 2-3 times per pixel (primary + shadow + optional reflection).
- `hit_sphere` does `sqrt(discriminant)` — the most expensive intrinsic per call.
- `v3_normalize` uses `rsqrt` (approximate) — this is faster than `1/sqrt` but introduces small errors. The C reference uses exact `1/sqrtf`.
- `trace` recurses with depth limit 1 — max 2 levels.
- Structs `Vec3` and `HitInfo` are returned by value — LLVM handles this via registers or stack.

## Available Eä Features

**SIMD types:**

| 128-bit | 256-bit | 512-bit |
|---------|---------|---------|
| f32x4   | f32x8   | f32x16  |
| f64x2   | f64x4   |         |
| i32x4   | i32x8   | i32x16  |
| i16x8   | i16x16  |         |
| i8x16   | i8x32   |         |
| u8x16   | u8x32   |         |

**Intrinsics:**
- Memory: load, load_f32x4, load_f32x8, load_i32x8 (typed variants for all vector types), store, stream_store, gather, scatter, prefetch(ptr, offset)
- Arithmetic: fma, sqrt, rsqrt, exp, min, max
- Reduction: reduce_add, reduce_add_fast, reduce_max, reduce_min (reduce_add_fast is float-only, uses unordered tree reduction — faster than reduce_add but non-deterministic FP order)
- Construction: splat, select
- Conversion: widen_i8_f32x{4,8,16}, widen_u8_f32x{4,8,16}, widen_u8_i32x{4,8,16}
- Integer: maddubs_i16, maddubs_i32

**Loop constructs:** while, for, unroll(N)

**Other:** const, static_assert, *restrict, *restrict mut

**Dot operators for SIMD element-wise ops:** `.+`, `.-`, `.*`, `./`, `.==`, `.<`, `.>`, `.&`, `.|`, `.^`, `.<<`, `.>>`|`.&`, `.|`, `.^`, `.<<`, `.>>`

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
