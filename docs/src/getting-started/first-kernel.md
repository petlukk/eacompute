# Your First Kernel

Let's write a kernel that scales an array by a constant factor, then upgrade it to use SIMD.

## Step 1: Scalar version

Create `scale.ea`:

```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let mut i: i32 = 0
    while i < n {
        dst[i] = src[i] * factor
        i = i + 1
    }
}
```

Key things to notice:
- `export` makes the function callable from Python (C ABI)
- `*f32` is an immutable pointer to float32 — your input array
- `*mut f32` is a mutable pointer — your output array
- `n: i32` is the array length — the caller provides this
- All types are explicit. No inference, no ambiguity.

## Step 2: Call from Python

```python
import ea
import numpy as np

kernel = ea.load("scale.ea")

src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
dst = np.empty_like(src)
kernel.scale(src, dst, factor=2.0)
print(dst)  # [2. 4. 6. 8.]
```

`ea.load()` compiles `scale.ea` the first time, then caches the result in `__eacache__/`. Subsequent calls load from cache instantly.

## Step 3: SIMD version

Now let's process 8 floats at a time:

```
export func scale_simd(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let s: f32x8 = splat(factor)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        store(dst, i, v .* s)
        i = i + 8
    }
}
```

What changed:
- `f32x8` — a vector of 8 floats (256-bit, uses AVX2 on x86)
- `splat(factor)` — broadcasts the scalar to all 8 lanes
- `load(src, i)` — loads 8 consecutive floats starting at index `i`
- `v .* s` — element-wise multiply (the dot prefix `.` means "vector operation")
- `store(dst, i, ...)` — writes 8 floats back
- We increment by 8, not 1

> **Important:** This only works when `n` is a multiple of 8. For arbitrary lengths, see [Kernels](../guide/kernels.md) for tail-handling strategies.

## Step 4: Compare performance

```python
import ea
import numpy as np
import time

kernel = ea.load("scale.ea")
src = np.random.randn(10_000_000).astype(np.float32)
dst = np.empty_like(src)

# Eä scalar
start = time.perf_counter()
for _ in range(100):
    kernel.scale(src, dst, factor=2.0)
ea_scalar = (time.perf_counter() - start) / 100

# Eä SIMD
start = time.perf_counter()
for _ in range(100):
    kernel.scale_simd(src, dst, factor=2.0)
ea_simd = (time.perf_counter() - start) / 100

# NumPy
start = time.perf_counter()
for _ in range(100):
    np.multiply(src, 2.0, out=dst)
numpy_time = (time.perf_counter() - start) / 100

print(f"Eä scalar: {ea_scalar*1000:.2f} ms")
print(f"Eä SIMD:   {ea_simd*1000:.2f} ms")
print(f"NumPy:     {numpy_time*1000:.2f} ms")
```

For this simple operation, all three will be similar — it's bandwidth-bound (one operation per element loaded). Eä shines on compute-bound workloads where there are multiple operations per element. See the [Cookbook](../cookbook/numpy-comparison.md) for real-world comparisons.
