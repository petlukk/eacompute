# ARM / NEON

Eä supports AArch64 with NEON vector instructions. This page documents the differences from x86 and how to write portable kernels.

## Vector Width

ARM NEON provides 128-bit vector registers, plus 64-bit D-registers for narrower operations. Available vector types on ARM:

| 128-bit (standard) | 64-bit (NEON D-registers) | Not Supported |
|---------------------|---------------------------|---------------|
| `f32x4` | `i8x8`, `u8x8` | `f32x8`, `f32x16` |
| `f64x2` | `i16x4`, `u16x4` | `f64x4` |
| `i32x4`, `u32x4` | `i32x2` | `i32x8`, `i32x16` |
| `i16x8`, `u16x8` | | `i16x16` |
| `i8x16`, `u8x16` | | `i8x32`, `u8x32` |

64-bit vector types are ARM-only. Using them on x86 produces a compile error. 256-bit and 512-bit types are x86-only and produce a compile error on ARM. This is intentional -- Eä does not silently fall back to scalar code.

## Unavailable Intrinsics

The following intrinsics are x86-only and produce a compile error on ARM:

| Intrinsic | Reason |
|-----------|--------|
| `movemask(v)` | No ARM equivalent for extracting lane sign bits to a bitmask |
| `gather(ptr, indices)` | No hardware gather support in NEON |
| `scatter(ptr, indices, values)` | AVX-512 only |
| `maddubs_i16(a, b)` | x86 PMADDUBSW instruction, no NEON equivalent |
| `maddubs_i32(a, b)` | x86 specific |

All other intrinsics (loads, stores, math, reductions, splat, select, shuffle, conversions) work on ARM.

## ARM-Specific Intrinsics

### Dot Product (ARMv8.2-A, `--dotprod`)

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `vdot_i32(a, b)` | `(i8x16, i8x16) -> i32x4` | Signed dot product, groups of 4 per lane. Maps to NEON `sdot`. |

### I8MM Matrix Multiply (ARMv8.6-A, `--i8mm`)

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `smmla_i32(acc, a, b)` | `(i32x4, i8x16, i8x16) -> i32x4` | Signed x signed 2x8 x 8x2 matrix multiply-accumulate |
| `ummla_i32(acc, a, b)` | `(i32x4, u8x16, u8x16) -> i32x4` | Unsigned x unsigned matrix multiply-accumulate |
| `usmmla_i32(acc, a, b)` | `(i32x4, u8x16, i8x16) -> i32x4` | Unsigned x signed matrix multiply-accumulate |

The accumulator is explicit. First call uses `splat(0)` for zero-init. Available on Cortex-A78+, Apple M1+.

### Absolute Difference (base NEON)

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `abs_diff(a, b)` | `(T, T) -> T` | Element-wise absolute difference. Returns `|a - b|` per lane. |

Supported types: `i8x16`, `u8x16`, `i16x8`, `u16x8`, `i32x4`, `u32x4`. Maps to NEON `sabd`/`uabd` (one instruction). No x86 equivalent -- use `max(a .- b, b .- a)` explicitly on x86.

### Widening Multiply (base NEON)

| Intrinsic | Signature | Description |
|-----------|-----------|-------------|
| `wmul_i16(a, b)` | `(i8x8, i8x8) -> i16x8` | Signed widening multiply |
| `wmul_u16(a, b)` | `(u8x8, u8x8) -> u16x8` | Unsigned widening multiply |
| `wmul_i32(a, b)` | `(i16x4, i16x4) -> i32x4` | Signed widening multiply |
| `wmul_u32(a, b)` | `(u16x4, u16x4) -> u32x4` | Unsigned widening multiply |

Input types are 64-bit NEON vectors (D-registers). Output is 128-bit. Maps to NEON `smull`/`umull`. No x86 equivalent.

The `--dotprod` and `--i8mm` flags enable their respective extensions. Using an intrinsic without its flag produces a compile error with a hint to add it.

## Cross-Platform Intrinsics

These intrinsics work on both x86 and ARM with identical semantics:

| Intrinsic | Signature | x86 instruction | ARM instruction |
|-----------|-----------|-----------------|-----------------|
| `shuffle_bytes(table, idx)` | `(u8x16, u8x16) -> u8x16` | SSSE3 `pshufb` | NEON `tbl` |
| `sat_add(a, b)` | `(T, T) -> T` | SSE2 `padds`/`paddus` | NEON `sqadd`/`uqadd` |
| `sat_sub(a, b)` | `(T, T) -> T` | SSE2 `psubs`/`psubus` | NEON `sqsub`/`uqsub` |

`sat_add`/`sat_sub` support `i8x16`, `u8x16`, `i16x8`, `u16x8`. Signed vs unsigned saturation is determined by the element type. No feature flags required (base SSE2 and NEON).

## Cross-Compilation

Compile for ARM from an x86 host:

```bash
ea kernel.ea --lib --target-triple=aarch64-unknown-linux-gnu
```

For intrinsics requiring hardware extensions, add the appropriate flag:

```bash
ea kernel.ea --lib --target-triple=aarch64-unknown-linux-gnu --dotprod
```

When compiling natively on an ARM machine, no special flags are needed (except `--dotprod` for dot product intrinsics):

```bash
ea kernel.ea --lib
ea kernel.ea --lib --dotprod   # for vdot_i32
ea kernel.ea --lib --i8mm     # for smmla_i32, ummla_i32, usmmla_i32
```

The `--avx512` flag is rejected on ARM targets with a compile error. The `--i8mm` and `--dotprod` flags are rejected on x86 targets.

## Writing Portable Kernels

### Strategy 1: Use 128-bit types everywhere

Use `f32x4`, `i32x4`, etc. These work on both x86 (SSE) and ARM (NEON).

```
kernel scale(data: *mut f32, factor: f32) range(n) step(4) {
    let v: f32x4 = load(data, i);
    let f: f32x4 = splat(factor);
    store(data, i, v .* f);
}
```

This sacrifices throughput on x86 (which could use `f32x8` with AVX2) but runs everywhere.

### Strategy 2: Separate kernel files

Write platform-specific kernels in separate files and load the right one at runtime:

```
# kernel_x86.ea  -- uses f32x8 for AVX2
# kernel_arm.ea  -- uses f32x4 for NEON
```

```python
import platform
if platform.machine() == "aarch64":
    k = ea.load("kernel_arm.ea")
else:
    k = ea.load("kernel_x86.ea")
```

Both files export the same function signatures, so the calling code does not change.

### Strategy 3: Use the kernel construct

The `kernel` construct with `step(N)` lets you pick the vector width per file while keeping the loop logic identical. Write two `.ea` files with different step sizes and vector types, but the same exported function name and parameters.

### 256-bit Operations on NEON

NEON registers are 128-bit. Intrinsics that accept 256-bit vectors (`f32x8`, `i32x8`, `i16x16`, `i8x32`) automatically split inputs into 128-bit halves, operate on each half with NEON instructions, and concatenate the result. This applies to:

- `round_f32x8_i32x8`: 2x `fcvtns.4s`
- `pack_sat_i32x8`: 4x `sqxtn` (narrow each half of each argument)
- `pack_sat_i16x16`: 4x `sqxtn` (narrow each half of each argument)
