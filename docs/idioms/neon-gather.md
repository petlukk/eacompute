# NEON Gather Pattern

`gather(lut: *f32, idx: i32xN) -> f32xN` is an x86 AVX2 intrinsic with
no direct NEON equivalent. On ARM, eacompute errors at compile time
with a pointer to this document. The canonical workaround uses scalar
loads composed via `f32x{4,8}_from_scalars`.

## Pattern (i32x4 indices into f32 LUT)

```ea
// Replaces: let v: f32x4 = gather(lut, idx)
let v0: f32 = lut[idx[0]]
let v1: f32 = lut[idx[1]]
let v2: f32 = lut[idx[2]]
let v3: f32 = lut[idx[3]]
let v: f32x4 = f32x4_from_scalars(v0, v1, v2, v3)
```

The four scalar reads stay in source — the programmer sees the cost.
The compose is one line instead of stack-buffer round-tripping.

## Pattern (i32x8 indices)

```ea
let v: f32x8 = f32x8_from_scalars(
    lut[idx[0]], lut[idx[1]], lut[idx[2]], lut[idx[3]],
    lut[idx[4]], lut[idx[5]], lut[idx[6]], lut[idx[7]]
)
```

## Why no silent fallback

Eä's design philosophy: explicit cost, no hidden performance cliffs.
A silent scalar fallback for `gather` would mask a 4-8x slowdown
behind syntax that looks SIMD. The hard error + named compose pattern
keeps the cost visible.

## See also

- llama.cpp uses the same scalar-compose pattern in its NEON IQ
  dequant kernels (no SVE2 gather available on Cortex-A76).
- Hardware gather lands on AArch64 SVE2 (Apple M4, Graviton 3+,
  Snapdragon X), not on Pi 5's Cortex-A76.
