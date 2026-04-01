# ML Preprocessing

ML preprocessing pipelines often apply the same sequence of operations to every element in large arrays: normalize, scale, compute similarities. These are natural fits for Eä kernels because they fuse multiple operations into single-pass loops.

## Normalize: zero-mean, unit-variance

The standard normalization `(x - mean) / std` requires two operations per element. NumPy computes them as separate passes:

```python
normalized = (data - mean) / std  # 2 passes, 2 temporaries
```

Eä fuses this into one pass using FMA. Rewrite `(x - mean) / std` as `x * (1/std) + (-mean/std)`:

```
export func normalize(
    src: *f32, dst: *mut f32,
    inv_std: f32, neg_mean_div_std: f32,
    n: i32
) {
    let s: f32x8 = splat(inv_std)
    let b: f32x8 = splat(neg_mean_div_std)
    let mut i: i32 = 0
    while i < n {
        let v: f32x8 = load(src, i)
        let result: f32x8 = fma(v, s, b)
        store(dst, i, result)
        i = i + 8
    }
}
```

The caller precomputes `inv_std = 1.0 / std` and `neg_mean_div_std = -mean / std` in Python. The kernel then does a single FMA per 8 elements -- one instruction that multiplies and adds simultaneously.

## Dot product: dual-accumulator pattern

A naive dot product uses one accumulator:

```
let mut acc: f32x8 = splat(0.0)
while i < n {
    let a: f32x8 = load(x, i)
    let b: f32x8 = load(y, i)
    acc = fma(a, b, acc)
    i = i + 8
}
```

This leaves performance on the table. FMA has a latency of 4-5 cycles but a throughput of 1 per cycle. With one accumulator, each FMA waits for the previous one to finish.

**Dual accumulators** hide this latency by interleaving independent FMA chains:

```
export func dot_product(x: *f32, y: *f32, out: *mut f32, n: i32) {
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i < n - 15 {
        let a0: f32x8 = load(x, i)
        let b0: f32x8 = load(y, i)
        acc0 = fma(a0, b0, acc0)
        let a1: f32x8 = load(x, i + 8)
        let b1: f32x8 = load(y, i + 8)
        acc1 = fma(a1, b1, acc1)
        i = i + 16
    }
    while i < n {
        let a: f32x8 = load(x, i)
        let b: f32x8 = load(y, i)
        acc0 = fma(a, b, acc0)
        i = i + 8
    }
    let combined: f32x8 = acc0 .+ acc1
    let result: f32 = horizontal_sum(combined)
    store(out, 0, splat(result))
}
```

The CPU can execute `fma(a0, b0, acc0)` and `fma(a1, b1, acc1)` in parallel because they use different accumulators. This typically doubles throughput on modern CPUs.

## Cosine similarity

Cosine similarity needs three reductions in one pass: `dot(a, b)`, `norm(a)`, and `norm(b)`. Computing these separately means three passes over the data. Eä fuses all three:

```
export func cosine_similarity(a: *f32, b: *f32, out: *mut f32, n: i32) {
    let mut dot_acc: f32x8 = splat(0.0)
    let mut norm_a_acc: f32x8 = splat(0.0)
    let mut norm_b_acc: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i < n {
        let va: f32x8 = load(a, i)
        let vb: f32x8 = load(b, i)
        dot_acc = fma(va, vb, dot_acc)
        norm_a_acc = fma(va, va, norm_a_acc)
        norm_b_acc = fma(vb, vb, norm_b_acc)
        i = i + 8
    }
    let dot_val: f32 = horizontal_sum(dot_acc)
    let norm_a_val: f32 = horizontal_sum(norm_a_acc)
    let norm_b_val: f32 = horizontal_sum(norm_b_acc)
    let result: f32 = dot_val / sqrt(norm_a_val * norm_b_val)
    store(out, 0, splat(result))
}
```

Three FMAs per loop iteration, all operating on the same loaded vectors. The data passes through cache once. NumPy would need `np.dot(a, b)`, `np.linalg.norm(a)`, and `np.linalg.norm(b)` -- three separate passes.

## Int8 quantized inference (ARM)

For quantized ML models using int8 weights, ARM provides dedicated matrix multiply instructions. On ARM with `--i8mm` (ARMv8.6-A, Cortex-A78+, Apple M1+):

```
export func matmul_i8_block(
    acc: *mut i32, activations: *i8, weights: *i8, n: i32
) {
    let mut a: i32x4 = splat(0)
    let mut i: i32 = 0
    while i < n {
        let act: i8x16 = load(activations, i)
        let wgt: i8x16 = load(weights, i)
        a = smmla_i32(a, act, wgt)
        i = i + 16
    }
    store(acc, 0, a)
}
```

`smmla_i32` performs a 2x8 x 8x2 signed matrix multiply-accumulate in one instruction. Also available: `ummla_i32` (unsigned x unsigned) and `usmmla_i32` (unsigned activations x signed weights, the most common ML pattern).

For older ARM chips without I8MM, use `vdot_i32` (requires `--dotprod`, ARMv8.2-A) for 4-way dot products, or `wmul_i16`/`wmul_i32` for widening multiplies.

## Fused Quantization Pipeline

Convert 32 float activations to int8 in 7 instructions, then feed directly to `maddubs_i32`:

```
kernel quantize_dot(activations: *const f32, weights: *const u8, out: *mut i32, inv_scale: f32, n: i32) {
    let f0: f32x8 = load(activations, i * 32);
    let f1: f32x8 = load(activations, i * 32 + 8);
    let f2: f32x8 = load(activations, i * 32 + 16);
    let f3: f32x8 = load(activations, i * 32 + 24);

    let scale: f32x8 = splat(inv_scale);
    let i0: i32x8 = round_f32x8_i32x8(f0 .* scale);
    let i1: i32x8 = round_f32x8_i32x8(f1 .* scale);
    let i2: i32x8 = round_f32x8_i32x8(f2 .* scale);
    let i3: i32x8 = round_f32x8_i32x8(f3 .* scale);

    let s01: i16x16 = pack_sat_i32x8(i0, i1);
    let s23: i16x16 = pack_sat_i32x8(i2, i3);
    let quant: i8x32 = pack_sat_i16x16(s01, s23);

    let w: u8x32 = load(weights, i * 32);
    let dot: i32x8 = maddubs_i32(w, quant);
    store(out, i * 8, dot);
}
```

Pipeline: 4x round + 2x pack_i32 + 1x pack_i16 = 7 instructions for 32 floats to 32 int8. Then 1x maddubs_i32 = 8 total.

## Batch operations

For ML workloads, you often apply the same operation to many rows. The Python side handles the loop over rows, calling the Eä kernel for each:

```python
import ea

lib = ea.load("similarity.ea")
for i in range(num_queries):
    lib.cosine_similarity(query[i], database[j], result, dim)
```

The kernel handles the inner loop (over vector dimensions) with SIMD. The outer loop (over queries or rows) stays in Python. This is the right split -- the inner loop is where SIMD matters.

## Real-world package

[eavec](https://github.com/petlukk/eavec) implements vector similarity search using the dual-accumulator FMA pattern. It computes cosine similarity across a database of vectors, returning the top-k most similar results.
