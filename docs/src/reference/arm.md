# ARM / NEON

Eä supports AArch64 with NEON vector instructions. This page documents the differences from x86 and how to write portable kernels.

## Vector Width

ARM NEON provides 128-bit vector registers. Only 128-bit vector types are available:

| Supported | Not Supported |
|-----------|---------------|
| `f32x4` | `f32x8`, `f32x16` |
| `f64x2` | `f64x4`, `f64x8` |
| `i32x4` | `i32x8`, `i32x16` |
| `i16x8` | `i16x16` |
| `i8x16` | `i8x32` |
| `u8x16` | `u8x32` |
| `u16x8` | `u16x16` |

Using a 256-bit or 512-bit vector type on an ARM target produces a compile error. This is intentional -- Eä does not silently fall back to scalar code.

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

| Intrinsic | Signature | Requires | Description |
|-----------|-----------|----------|-------------|
| `vdot_i32(a, b)` | `(i8x16, i8x16) -> i32x4` | `--dotprod` | Signed dot product, groups of 4 per lane. Maps to NEON `sdot`. |

The `--dotprod` flag enables the ARMv8.2-A dot product extension. Using `vdot_i32` without it produces a compile error.

## Cross-Platform Intrinsics

These intrinsics work on both x86 and ARM with identical semantics:

| Intrinsic | Signature | x86 instruction | ARM instruction |
|-----------|-----------|-----------------|-----------------|
| `shuffle_bytes(table, idx)` | `(u8x16, u8x16) -> u8x16` | SSSE3 `pshufb` | NEON `tbl` |

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
```

The `--avx512` flag is rejected on ARM targets with a compile error.

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
