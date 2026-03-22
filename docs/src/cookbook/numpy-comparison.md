# Eä vs NumPy

When does writing a kernel in Eä actually beat NumPy? The answer comes down to one thing: **arithmetic intensity** -- how many operations you perform per byte loaded from memory.

## The memory bandwidth wall

Modern CPUs can process arithmetic far faster than they can load data from DRAM. A single DDR4 channel delivers ~30-40 GB/s. NumPy's ufuncs are already compiled C with SIMD -- for simple operations, they saturate this bandwidth just like hand-written SIMD would.

**Rule of thumb:** if your operation does fewer than 2 arithmetic ops per element loaded, it is bandwidth-bound. Eä will match NumPy but not beat it.

## Bandwidth-bound: Eä matches, no win

### Array scaling

```
export func scale(src: *f32, dst: *mut f32, factor: f32, n: i32) {
    let f: f32x8 = splat(factor)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        store(dst, i, v .* f)
        i = i + 8
    }
}
```

```python
dst = src * factor  # NumPy: one SIMD multiply per element, same speed
```

One multiply per element loaded. Both Eä and NumPy hit ~35 GB/s on typical hardware. No winner.

### Simple element-wise ops

Any operation that loads an element, does one thing, and stores the result is bandwidth-bound:

- `dst[i] = src[i] + offset`
- `dst[i] = abs(src[i])`
- `dst[i] = src_a[i] + src_b[i]`

NumPy already handles these at memory bandwidth speed. Writing an Eä kernel gains you nothing.

## Compute-bound: Eä wins

### Fused scale + bias + clamp

NumPy must make three separate passes over memory:

```python
dst = np.clip(src * scale + bias, 0.0, 1.0)  # 3 temporaries, 3 passes
```

Each pass loads the full array, computes one operation, and writes a temporary. For a 100 MB array, that is 600 MB of memory traffic.

Eä fuses everything into a single pass:

```
export func fused_scale_bias_clamp(src: *f32, dst: *mut f32, scale: f32, bias: f32, n: i32) {
    let s: f32x8 = splat(scale)
    let b: f32x8 = splat(bias)
    let zero: f32x8 = splat(0.0)
    let one: f32x8 = splat(1.0)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        let result: f32x8 = fma(v, s, b)
        let clamped: f32x8 = min(max(result, zero), one)
        store(dst, i, clamped)
        i = i + 8
    }
}
```

One load, one FMA, two comparisons, one store. The data stays in registers the entire time. This is 3-5x faster than the NumPy version on large arrays because it reads memory once instead of three times.

### Stencil operations

Convolutions, Sobel filters, and any operation that reads multiple neighboring elements per output pixel have high arithmetic intensity. A 3x3 Sobel kernel reads 9 values and performs 9 multiplications plus 8 additions per output -- well above the compute-bound threshold.

See [Image Processing](image-processing.md) for stencil patterns.

### Custom reductions with branching

NumPy cannot express per-element branching in vectorized form. Operations like "accumulate values, but skip negatives and double values above a threshold" require Python-level loops or awkward `np.where` chains.

Eä gives you SIMD comparisons and masked operations in a single loop body, keeping the pipeline full.

### Dot products and FMA chains

Any reduction that multiplies and accumulates benefits from FMA (fused multiply-add) and register-level accumulation:

```
let acc: f32x8 = fma(a, b, acc)  // a * b + acc in one instruction
```

NumPy's `np.dot` is fast for large matrices (it calls BLAS), but for custom reductions, per-row operations, or non-standard accumulation patterns, Eä's explicit FMA beats NumPy's element-wise approach.

See [ML Preprocessing](ml-preprocessing.md) for dot product and similarity patterns.

## Decision checklist

Before writing an Eä kernel, ask:

1. **How many ops per element?** If just one (scale, offset, abs), stay with NumPy.
2. **Does NumPy need multiple passes?** If your expression chains 3+ operations, Eä's fusion wins.
3. **Is there a stencil or neighbor access?** High arithmetic intensity -- Eä wins.
4. **Is there branching logic per element?** NumPy cannot vectorize this -- Eä wins.
5. **Is it a standard BLAS operation?** Use NumPy/SciPy -- they call optimized libraries.

## Real-world packages

These packages demonstrate Eä beating NumPy on compute-bound workloads:

- [easobel](https://github.com/petlukk/easobel) -- Sobel edge detection (stencil, ~9 ops/pixel)
- [eastat](https://github.com/petlukk/eastat) -- CSV parsing (branching SIMD scan)
- [eavec](https://github.com/petlukk/eavec) -- Vector similarity search (FMA-heavy dot products)
