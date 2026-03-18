# How 30 Lines of Eä Beat NumPy by 6×

*And why your framework is probably slower than a for-loop.*

---

## The Pitch

Here's a fused multiply-add: `out[i] = a[i] * b[i] + c[i]`. Sixteen million times.

NumPy does it in 46 milliseconds. Eä does it in 7 milliseconds. That's **6.6× faster**.

Here's the kicker: the Eä kernel is 30 lines. The NumPy version is one line. And the one-liner loses because it's actually *two lines pretending to be one*.

## The NumPy Version

```python
out = a * b + c
```

Simple. Elegant. Two full scans of 64 MB of data.

NumPy computes `a * b` first, writes 64 MB to a temporary array, then reads it back to add `c`. That's 256 MB of memory traffic for 192 MB of actual data. Every element gets loaded, stored, loaded again, and stored again.

On a modern CPU at ~35 GB/s memory bandwidth, that's a floor of about 7 milliseconds just for memory. NumPy hits 46 ms because it also has Python dispatch overhead, array allocation, and — crucially — it can't fuse the multiply and add into a single instruction.

## The Eä Version

```
export func fma_f32x8(
    a: *restrict f32,
    b: *restrict f32,
    c: *restrict f32,
    out: *mut f32,
    n: i32
) {
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= n {
        let va0: f32x8 = load(a, i)
        let vb0: f32x8 = load(b, i)
        let vc0: f32x8 = load(c, i)
        store(out, i, fma(va0, vb0, vc0))

        let va1: f32x8 = load(a, i + 8)
        let vb1: f32x8 = load(b, i + 8)
        let vc1: f32x8 = load(c, i + 8)
        store(out, i + 8, fma(va1, vb1, vc1))

        i = i + 16
    }
    while i < n {
        out[i] = a[i] * b[i] + c[i]
        i = i + 1
    }
}
```

One pass. Load a, b, c, fuse multiply-add in a single `vfmadd` instruction, store result. Each element is touched once. Total memory traffic: 256 MB (same 4 arrays), but only one scan instead of two.

The `*restrict` tells LLVM the arrays don't alias, enabling aggressive optimization. The `f32x8` processes 8 floats per instruction on AVX2. The 2× unroll (processing 16 elements per iteration) hides memory latency.

That's it. No magic. Just not doing twice the work.

## "But I'll Just Use Ray"

We benchmarked that too. For science.

| Method | Time | Throughput | vs NumPy |
|--------|------|------------|----------|
| NumPy (multiply + add) | 46,000 µs | 5.6 GB/s | baseline |
| **Eä (1 thread)** | **6,900 µs** | **37.0 GB/s** | **6.6×** |
| Eä (2 threads) | 6,500 µs | 39.1 GB/s | 7.0× |
| Dask (2 chunks) | 56,000 µs | 4.6 GB/s | 0.8× |
| Ray (2 workers) | 89,000 µs | 2.9 GB/s | 0.5× |

Ray is **twice as slow as NumPy**. For FMA. On two cores. On the same machine.

Why? Serialization. Ray pickles your 64 MB arrays, sends them to worker processes, unpickles them, runs NumPy (which does two passes), pickles the results, and sends them back. The actual compute takes 46 ms. The overhead takes another 43 ms.

Dask is better — it chunks lazily and uses NumPy under the hood — but it still can't fuse operations or control SIMD width. It's paying for an abstraction layer over code that's already fast enough.

The uncomfortable truth: for single-machine numerical work, **a for-loop that touches memory once will beat a distributed framework that touches memory twice**.

## The Dot Product Story

We weren't always winning. Our first benchmark had Eä's dot product at 0.27× of BLAS. Embarrassing.

```
// The naive version — DO NOT ship this
export func dot_naive(a: *f32, b: *f32, n: i32) -> f32 {
    let mut acc: f32 = 0.0
    let mut i: i32 = 0
    while i < n {
        acc = acc + a[i] * b[i]
        i = i + 1
    }
    return acc
}
```

Single scalar accumulator. Each iteration depends on the previous one. The CPU stalls waiting for the addition to complete before it can start the next multiply. Classic dependency chain bottleneck.

BLAS uses the trick every HPC programmer knows: **multiple independent accumulators**.

```
// The optimized version — ships in autoresearch/kernels/dot_product/
export func dot_f32x8(a: *restrict f32, b: *restrict f32, len: i32) -> f32 {
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 32 <= len {
        acc0 = fma(load(a, i),      load(b, i),      acc0)
        acc1 = fma(load(a, i + 8),  load(b, i + 8),  acc1)
        acc0 = fma(load(a, i + 16), load(b, i + 16), acc0)
        acc1 = fma(load(a, i + 24), load(b, i + 24), acc1)
        i = i + 32
    }
    // ... tail handling ...
    return reduce_add_fast(acc0 .+ acc1)
}
```

Two f32x8 accumulators (`acc0`, `acc1`) with 4× unroll. The CPU has 16 FMA operations in flight that don't depend on each other. The result:

| Method | Time | GB/s | vs BLAS |
|--------|------|------|---------|
| NumPy BLAS sdot | 3,535 µs | 35.9 | baseline |
| Eä naive (scalar) | 13,222 µs | 9.7 | 0.27× |
| Eä f32x4 (1 acc) | 4,474 µs | 28.6 | 0.79× |
| **Eä f32x8 (dual acc, 4× unroll)** | **3,500 µs** | **36.6** | **1.01×** |

From 0.27× to 1.01× — by changing the loop structure, not the algorithm. Both versions compute the same dot product. The fast one just asks the CPU to think about 32 elements instead of 1.

## What You're Actually Seeing

This isn't "Eä is fast." Eä is a thin wrapper around LLVM. The FMA kernel compiles to about 12 assembly instructions in the inner loop. Eä's job is to make writing those 12 instructions feel like writing 30 lines of readable code instead of 200 lines of intrinsics.

The real insight is about NumPy's cost model:

| Operation | NumPy | Eä |
|-----------|-------|----|
| `a * b + c` | 2 passes over data | 1 pass (fused FMA) |
| Temporary arrays | 1 allocation (64 MB) | 0 allocations |
| SIMD width | "whatever the compiler picks" | explicit f32x8 |
| Memory round-trips | 2 (multiply result → RAM → add) | 0 |

NumPy is fast *per operation*. But real workloads aren't one operation — they're pipelines. And every `np.` call is a full scan of your data.

## When Eä Doesn't Win

Bandwidth-bound operations with a single arithmetic op per element. Scaling an array:

```python
out = src * 2.5
```

NumPy: 5,631 µs. Eä: 6,138 µs. **NumPy wins** (by 8%).

Both saturate memory bandwidth at ~22 GB/s. There's nothing to fuse. One multiply per element loaded. No amount of SIMD cleverness makes DRAM faster.

This is the honest dividing line: if your inner loop does **fewer than ~4 operations per element**, NumPy is fast enough. If it does more, you're leaving performance on the table.

## Getting Started

```bash
pip install ea-compiler
```

```python
import ea

kernel = ea.load("fma.ea")

import numpy as np
a = np.random.rand(16_000_000).astype(np.float32)
b = np.random.rand(16_000_000).astype(np.float32)
c = np.random.rand(16_000_000).astype(np.float32)
out = np.zeros(16_000_000, dtype=np.float32)

kernel.fma_f32x8(a, b, c, out)  # 7 ms instead of 46 ms
```

No Cython. No Numba. No C compiler. No JIT warmup. Write a `.ea` file, call `ea.load()`, pass NumPy arrays.

The compiler handles SIMD, the binding handles types, and the hardware handles the rest.

---

*All benchmarks measured on a 2-core machine. Your numbers will be different, probably better (more cores = more bandwidth). The ratios hold. [Source code and methodology on GitHub.](https://github.com/petlukk/eacompute)*

*Eä is open-source (Apache 2.0). The compiler is ~12,000 lines of Rust with 475+ tests. [Documentation.](https://petlukk.github.io/eacompute/) [GitHub.](https://github.com/petlukk/eacompute)*
