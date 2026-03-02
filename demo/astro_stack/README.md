# Astronomy Frame Stacking — Eä Demo

Stacks N noisy exposures to reduce noise using real telescope data
from NASA SkyView (M31 / Andromeda galaxy). Demonstrates Eä's
auto-generated Python bindings (`ea bind --python`) and streaming
kernel architecture.

This is a **binding-generation and correctness demo**, not a performance
showcase. Frame stacking is a bandwidth-bound workload (load, add, store)
where NumPy's `np.add` ufunc — already SIMD-optimized C — hits the same
memory bandwidth ceiling as explicit SIMD. There is no speed advantage
to claim here. See "Why not faster?" below.

## Results

1024×1024, 16 frames. Single-threaded, all buffers pre-allocated,
30 runs, median.

### Performance

```
NumPy stream (np.add)   :   5.6 ms
Ea stream    (SIMD)     :   6.2 ms
```

Ea matches NumPy across all tested sizes (1024² to 4096², 3–20 frames).
The ratio stays 1.00–1.10x. Both are bandwidth-bound.

### Signal-to-noise ratio

```
SNR gain: +12.03 dB from stacking 16 frames
```

## Why not faster?

Frame accumulation is `acc[i] += frame[i]` — one add per element loaded.
This is purely memory-bandwidth-bound: there's no compute intensity to
exploit, no fusion opportunity, no branching logic that NumPy can't express.
NumPy's ufuncs are already SIMD-optimized C hitting the same bandwidth.

Eä wins on workloads with higher arithmetic intensity per byte loaded:
- **Sobel**: stencil pattern with coefficient multiplication across neighbors
- **Vector search**: dual-accumulator FMA with high arithmetic intensity
- **CSV structural scan**: branching logic (quote-state tracking)
- **Cosine similarity**: three fused reductions in one pass

Simple element-wise operations aren't one of those. That's fine — it's
a confirmed non-goal, not a weakness.

## What this demo does show

1. **Auto-generated bindings work**: `ea bind --python` produces a
   zero-manual-ctypes Python module that handles type checking, length
   inference, and pointer marshalling.
2. **Correctness**: exact bit-for-bit match with NumPy (max diff = 0.0).
3. **Multi-kernel composition**: three kernels (`accumulate`, `scale`,
   `frame_stats`) composed from Python, each compiled to native SIMD code.
4. **Streaming memory model**: O(pixels) vs O(N × pixels) for batch.
   16 frames at 1024²: 4 MB streaming vs 64 MB batch.

## The kernel

```
// In-place accumulation: acc[i] += frame[i]
export func accumulate_f32x8(acc: *mut f32, frame: *restrict f32, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        prefetch(acc, i + 64)
        prefetch(frame, i + 64)
        let va0: f32x8 = load(acc, i)
        let vf0: f32x8 = load(frame, i)
        let va1: f32x8 = load(acc, i + 8)
        let vf1: f32x8 = load(frame, i + 8)
        store(acc, i, va0 .+ vf0)
        store(acc, i + 8, va1 .+ vf1)
        i = i + 16
    }
    // ... residual vector + scalar tail
}

// Single-pass min/max/sum via dual-accumulator reductions.
export func frame_stats(
    data: *restrict f32, len: i32,
    out_min: *mut f32, out_max: *mut f32, out_sum: *mut f32
) {
    // dual f32x8 accumulators with select for min/max, prefetch
    // merge + horizontal: reduce_add, reduce_min, reduce_max
}
```

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Build kernel + generate bindings
cd demo/astro_stack && bash build.sh

# Run demo
python run.py

# Run HST data benchmark
python bench_hst.py

# Run scale benchmark (1024² to 4096², confirms bandwidth-bound)
python bench_scale.py
```

## How it works

On first run, downloads a DSS telescope image of M31 from NASA SkyView,
normalizes it, and generates N noisy exposures. Falls back to synthetic
starfield if download fails.

```bash
ea stack.ea --lib -o libstack.so   # → libstack.so + stack.ea.json
ea bind stack.ea --python          # → stack.py (auto-generated)
```

```python
import stack as _stack

for frame in noisy_frames:
    _stack.accumulate_f32x8(acc, frame)
_stack.scale_f32x8(acc, out, 1.0 / n_frames)
_stack.frame_stats(result, out_min, out_max, out_sum)
```
